"""Image and multi-frame TIFF loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageSequence

SUPPORTED_IMAGES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class LoadedFrame:
    source: Path
    frame_number: int
    frame_count: int
    image: Image.Image

    @property
    def display_name(self) -> str:
        if self.frame_count == 1:
            return self.source.name
        width = max(3, len(str(self.frame_count)))
        return f"{self.source.name}  |  Frame {self.frame_number:0{width}d}/{self.frame_count}"


def load_frames(path: Path, progress: Callable[[int, int], None] | None = None) -> list[LoadedFrame]:
    if path.suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError(f"Unsupported image format: {path.suffix or '<none>'}")

    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count < 1:
            raise ValueError(f"No frames found in image: {path}")
        frames = []
        for index, frame in enumerate(ImageSequence.Iterator(image), start=1):
            frames.append(LoadedFrame(path, index, frame_count, frame.convert("RGB").copy()))
            if progress is not None:
                progress(index, frame_count)
        return frames
