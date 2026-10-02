"""Image and multi-frame TIFF loading."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageSequence

SUPPORTED_IMAGES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_WORKING_PIXELS = 16_000_000
_PIL_LIMIT_LOCK = threading.Lock()


@dataclass(frozen=True)
class LoadedFrame:
    source: Path
    frame_number: int
    frame_count: int
    image: Image.Image
    source_size: tuple[int, int] | None = None

    @property
    def original_size(self) -> tuple[int, int]:
        return self.source_size or self.image.size

    @property
    def is_optimized(self) -> bool:
        return self.original_size != self.image.size

    @property
    def requires_tiled_inference(self) -> bool:
        return self.is_optimized

    @property
    def display_name(self) -> str:
        if self.frame_count == 1:
            name = self.source.name
        else:
            width = max(3, len(str(self.frame_count)))
            name = f"{self.source.name}  |  Frame {self.frame_number:0{width}d}/{self.frame_count}"
        if self.is_optimized:
            name += f"  |  Preview {self.image.width}x{self.image.height} | Full-resolution tiled inference {self.original_size[0]}x{self.original_size[1]}"
        return name


@dataclass(frozen=True)
class ImageTile:
    image: Image.Image
    box: tuple[int, int, int, int]


def iter_frame_tiles(
    frame: LoadedFrame, tile_size: tuple[int, int], overlap: float = 0.15
) -> Iterator[ImageTile]:
    """Yield full-detail image tiles without materializing an oversized source."""
    tile_width, tile_height = tile_size
    if tile_width < 1 or tile_height < 1:
        raise ValueError("Tile dimensions must be positive.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Tile overlap must be between 0 and 1.")
    positions = tuple(_tile_boxes(frame.original_size, tile_size, overlap))
    if not frame.requires_tiled_inference:
        for box in positions:
            yield ImageTile(frame.image.crop(box), box)
        return

    with _open_for_bounded_loading(frame.source) as source:
        if source.format != "TIFF" or int(getattr(source, "n_frames", 1)) != 1:
            raise ValueError(
                "Full-resolution tiled inference currently requires a single-frame TIFF source."
            )
        pixels, lookup = _raw_tiff_pixels(frame.source, source)
        try:
            for box in positions:
                left, top, right, bottom = box
                values = np.asarray(pixels[top:bottom, left:right])
                if lookup is not None:
                    values = lookup[values]
                yield ImageTile(Image.fromarray(values, mode="L").convert("RGB"), box)
        finally:
            del pixels


def _tile_boxes(
    image_size: tuple[int, int],
    tile_size: tuple[int, int],
    overlap: float,
) -> Iterator[tuple[int, int, int, int]]:
    image_width, image_height = image_size
    tile_width = min(tile_size[0], image_width)
    tile_height = min(tile_size[1], image_height)
    step_x = max(1, round(tile_width * (1.0 - overlap)))
    step_y = max(1, round(tile_height * (1.0 - overlap)))
    x_positions = list(range(0, max(1, image_width - tile_width + 1), step_x))
    y_positions = list(range(0, max(1, image_height - tile_height + 1), step_y))
    if x_positions[-1] != image_width - tile_width:
        x_positions.append(image_width - tile_width)
    if y_positions[-1] != image_height - tile_height:
        y_positions.append(image_height - tile_height)
    for top in y_positions:
        for left in x_positions:
            yield left, top, left + tile_width, top + tile_height


def load_frames(
    path: Path, progress: Callable[[int, int], None] | None = None
) -> list[LoadedFrame]:
    if path.suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError(f"Unsupported image format: {path.suffix or '<none>'}")

    with _open_for_bounded_loading(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count < 1:
            raise ValueError(f"No frames found in image: {path}")
        if image.width * image.height > MAX_WORKING_PIXELS:
            frame = _load_oversized_frame(path, image, frame_count)
            if progress is not None:
                progress(1, frame_count)
            return [frame]
        frames = []
        for index, frame in enumerate(ImageSequence.Iterator(image), start=1):
            frames.append(
                LoadedFrame(path, index, frame_count, frame.convert("RGB").copy())
            )
            if progress is not None:
                progress(index, frame_count)
        return frames


@contextmanager
def _open_for_bounded_loading(path: Path) -> Iterator[Image.Image]:
    # Pillow's global limit is bypassed only while reading dimensions. EchoSight
    # applies its own lower working-image ceiling before decoding any pixels.
    with _PIL_LIMIT_LOCK:
        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            image = Image.open(path)
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit
    try:
        yield image
    finally:
        image.close()


def _load_oversized_frame(
    path: Path, image: Image.Image, frame_count: int
) -> LoadedFrame:
    source_size = image.size
    if image.format == "TIFF" and frame_count == 1:
        working_image = _reduce_raw_tiff(path, image)
    else:
        raise ValueError(
            f"Image is {source_size[0]} x {source_size[1]} ({source_size[0] * source_size[1]:,} pixels). "
            "Full-resolution bounded inference requires a single-frame, uncompressed 8-bit grayscale or palette TIFF. "
            "Convert this source to that lossless TIFF format or split it into smaller lossless images."
        )
    return LoadedFrame(path, 1, frame_count, working_image, source_size)


def _reduce_raw_tiff(path: Path, image: Image.Image) -> Image.Image:
    reduction = max(
        2, math.ceil(math.sqrt((image.width * image.height) / MAX_WORKING_PIXELS))
    )
    target_width = math.ceil(image.width / reduction)
    target_height = math.ceil(image.height / reduction)
    output = Image.new("L", (target_width, target_height))
    pixels, lookup = _raw_tiff_pixels(path, image)
    output_rows_per_chunk = 128

    for output_top in range(0, target_height, output_rows_per_chunk):
        output_bottom = min(target_height, output_top + output_rows_per_chunk)
        source_top = output_top * reduction
        source_bottom = min(image.height, output_bottom * reduction)
        values = np.asarray(pixels[source_top:source_bottom])
        if lookup is not None:
            values = lookup[values]
        chunk = Image.fromarray(values, mode="L")
        reduced = chunk.resize(
            (target_width, output_bottom - output_top), Image.Resampling.BOX
        )
        output.paste(reduced, (0, output_top))

    del pixels
    return output.convert("RGB")


def _raw_tiff_pixels(
    path: Path, image: Image.Image
) -> tuple[np.memmap, np.ndarray | None]:
    if image.mode not in {"L", "P"} or len(image.tile) != 1:
        raise ValueError(
            "Oversized TIFF must be a single uncompressed 8-bit grayscale or palette image."
        )
    decoder, bounds, offset, arguments = image.tile[0]
    if decoder != "raw" or bounds != (0, 0, image.width, image.height):
        raise ValueError("Oversized TIFF must use one uncompressed contiguous strip.")
    raw_mode = arguments[0] if arguments else image.mode
    if raw_mode not in {"L", "P"}:
        raise ValueError(f"Unsupported oversized TIFF pixel layout: {raw_mode}")
    pixels = np.memmap(
        path, dtype=np.uint8, mode="r", offset=offset, shape=(image.height, image.width)
    )
    palette = image.getpalette()
    lookup = (
        np.asarray(palette[0::3], dtype=np.uint8)
        if image.mode == "P" and palette
        else None
    )
    return pixels, lookup
