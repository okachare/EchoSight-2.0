"""Render normalized inference results onto source images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .inference import Detection, InferenceResult, TaskType


@dataclass(frozen=True)
class RenderOptions:
    show_boxes: bool = True
    show_labels: bool = True
    show_masks: bool = True
    show_heatmap: bool = True
    overlay_opacity: float = 0.38
    font_family: str = "Calibri"
    font_size: int = 10
    annotation_color: tuple[int, int, int] | None = None
    annotation_thickness: int = 3
    annotation_opacity: float = 1.0
    label_opacity: float = 1.0


COLORS = (
    (37, 185, 167),
    (255, 176, 76),
    (83, 154, 255),
    (235, 92, 116),
    (170, 126, 240),
)
DETECTION_COLOR = COLORS[0]


def render_result(
    source: Image.Image,
    result: InferenceResult,
    options: RenderOptions | None = None,
    hidden_annotations: set[int] | None = None,
) -> Image.Image:
    options = options or RenderOptions()
    hidden = hidden_annotations or set()
    rendered = source.convert("RGB").copy()
    occupied_labels: list[tuple[int, int, int, int]] = []

    if result.task_type is TaskType.ANOMALY and result.anomaly_map is not None and options.show_heatmap and 0 not in hidden:
        rendered = _render_anomaly_map(rendered, result.anomaly_map, options.overlay_opacity)

    for index, detection in enumerate(result.detections):
        if index in hidden:
            continue
        rendered = _render_detection(rendered, detection, options, occupied_labels)

    if options.show_labels and result.task_type is TaskType.ANOMALY and result.anomaly_score is not None and 0 not in hidden:
        color = options.annotation_color or COLORS[3]
        rendered = _draw_label(rendered, (8, 8), f"Anomaly {result.anomaly_score:.1%}", color, options)

    return rendered


def _render_detection(
    image: Image.Image,
    detection: Detection,
    options: RenderOptions,
    occupied_labels: list[tuple[int, int, int, int]] | None = None,
) -> Image.Image:
    rendered = image
    if options.annotation_color is not None:
        color = options.annotation_color
    elif detection.stage == "classification":
        color = COLORS[(detection.label_id + 1) % len(COLORS)]
    else:
        color = DETECTION_COLOR
    x1, y1, x2, y2 = _clamp_box(detection.box, rendered.size)
    if detection.stage == "classification":
        inset = (detection.stage_index + 1) * max(2, options.annotation_thickness)
        x1, y1 = min(x2, x1 + inset), min(y2, y1 + inset)
        x2, y2 = max(x1, x2 - inset), max(y1, y2 - inset)

    if detection.mask is not None and options.show_masks and x2 > x1 and y2 > y1:
        rendered = _blend_box_mask(
            rendered,
            detection.mask,
            (x1, y1, x2, y2),
            color,
            options.overlay_opacity * options.annotation_opacity,
        )

    if options.show_boxes:
        rendered = _draw_box(rendered, (x1, y1, x2, y2), color, options.annotation_thickness, options.annotation_opacity)
    if options.show_labels:
        stage = "Classification" if detection.stage == "classification" else "Detection"
        label = f"{stage} | {detection.label} | {_format_confidence(detection.confidence)}"
        rendered = _draw_label(
            rendered,
            (x1, y1 + detection.stage_index * (options.font_size + 10)),
            label,
            color,
            options,
            above=detection.stage != "classification",
            occupied=occupied_labels,
        )
    return rendered


def _format_confidence(value: float) -> str:
    if value >= 0.99995:
        return ">=99.99%"
    return f"{value:.2%}"


def _draw_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int,
    opacity: float,
) -> Image.Image:
    alpha = round(255 * float(np.clip(opacity, 0.0, 1.0)))
    if alpha == 0:
        return image
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(box, outline=(*color, alpha), width=max(1, int(thickness)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _blend_box_mask(
    image: Image.Image,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    opacity: float,
) -> Image.Image:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    mask_image = Image.fromarray(np.asarray(mask, dtype=np.float32), mode="F")
    mask_values = np.asarray(mask_image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    alpha = np.clip(mask_values, 0.0, 1.0) * float(np.clip(opacity, 0.0, 1.0))

    pixels = np.asarray(image, dtype=np.float32).copy()
    region = pixels[y1:y2, x1:x2]
    if region.size == 0:
        return image
    color_array = np.asarray(color, dtype=np.float32)
    region[:] = region * (1.0 - alpha[..., None]) + color_array * alpha[..., None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


def _render_anomaly_map(image: Image.Image, anomaly_map: np.ndarray, opacity: float) -> Image.Image:
    values = np.nan_to_num(np.asarray(anomaly_map, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    minimum = float(values.min())
    maximum = float(values.max())
    if (minimum < 0 or maximum > 1) and maximum > minimum:
        values = (values - minimum) / (maximum - minimum)
    elif maximum <= minimum:
        values = np.zeros_like(values)
    resized = Image.fromarray(values, mode="F").resize(image.size, Image.Resampling.BILINEAR)
    intensity = np.asarray(resized, dtype=np.float32)
    heatmap = np.empty((*intensity.shape, 3), dtype=np.float32)
    heatmap[..., 0] = 255
    heatmap[..., 1] = np.clip(255 * (1.0 - np.abs(intensity - 0.5) * 2.0), 0, 255)
    heatmap[..., 2] = np.clip(90 * (1.0 - intensity), 0, 255)
    alpha = np.clip(intensity * opacity, 0.0, 1.0)[..., None]
    source = np.asarray(image, dtype=np.float32)
    blended = source * (1.0 - alpha) + heatmap * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _clamp_box(box: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, round(x1))),
        max(0, min(height - 1, round(y1))),
        max(0, min(width, round(x2))),
        max(0, min(height, round(y2))),
    )


def _draw_label(
    image: Image.Image,
    origin: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    options: RenderOptions,
    above: bool = False,
    occupied: list[tuple[int, int, int, int]] | None = None,
) -> Image.Image:
    alpha = round(255 * float(np.clip(options.label_opacity, 0.0, 1.0)))
    if alpha == 0:
        return image
    font = _load_label_font(options.font_family, options.font_size)
    measure = ImageDraw.Draw(image)
    left, top, right, bottom = measure.textbbox((0, 0), text, font=font)
    width = right - left + 8
    height = bottom - top + 8
    x = min(max(0, origin[0]), max(0, image.size[0] - width))
    requested_y = origin[1] - height if above else origin[1]
    y = min(max(0, requested_y), max(0, image.size[1] - height))
    if occupied is not None:
        y = _available_label_y(x, y, width, height, image.size[1], occupied)
        occupied.append((x, y, x + width, y + height))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((x, y, x + width, y + height), fill=(12, 16, 20, alpha), outline=(*color, alpha))
    draw.text((x + 4, y + 4), text, fill=(240, 245, 248, alpha), font=font)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _available_label_y(
    x: int,
    requested_y: int,
    width: int,
    height: int,
    image_height: int,
    occupied: list[tuple[int, int, int, int]],
) -> int:
    maximum_y = max(0, image_height - height)
    step = height + 2
    candidates = range(requested_y, maximum_y + 1, step)
    for y in (*candidates, *range(requested_y - step, -1, -step)):
        candidate = (x, y, x + width, y + height)
        if not any(_rectangles_overlap(candidate, existing) for existing in occupied):
            return y
    return requested_y


def _rectangles_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]


def _load_label_font(family: str, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    requested_size = max(1, int(size))
    candidates = (f"{family}.ttf", "calibri.ttf", "arial.ttf")
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, requested_size)
        except OSError:
            continue
    return ImageFont.load_default()
