"""Memory-bounded full-resolution inference for oversized images."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from time import perf_counter
from typing import Union

import numpy as np
from PIL import Image

from ..frames import LoadedFrame, iter_frame_tiles
from .engine import Classification, Detection, InferenceEngine, InferenceResult
from .model_loader import TaskType
from .pipeline import ChainedInferenceEngine

InferenceBackend = Union[InferenceEngine, ChainedInferenceEngine]


def infer_frame(
    engine: InferenceBackend,
    frame: LoadedFrame,
    transform: Callable[[Image.Image], Image.Image] | None = None,
    progress: Callable[[int, int], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    finalizing: Callable[[], None] | None = None,
    overlap: float = 0.15,
) -> InferenceResult:
    """Infer one frame, streaming lossless source tiles when its preview is reduced."""
    apply_transform = transform or (lambda image: image)
    if not frame.requires_tiled_inference:
        if should_continue is not None and not should_continue():
            raise InterruptedError("Inference cancelled")
        if progress is not None:
            progress(1, 1)
        return engine.infer(apply_transform(frame.image), frame.source)

    input_height, input_width = _input_size(engine)
    tile_size = (input_width, input_height)
    tiles = iter_frame_tiles(frame, tile_size, overlap)
    total = _tile_count(frame.original_size, tile_size, overlap)
    started = perf_counter()
    first: InferenceResult | None = None
    detections: list[Detection] = []
    classifications: dict[tuple[int, str], Classification] = {}
    anomaly_score: float | None = None
    anomaly_map: np.ndarray | None = None
    output_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()
    for position, tile in enumerate(tiles, start=1):
        if should_continue is not None and not should_continue():
            raise InterruptedError("Inference cancelled")
        result = engine.infer(apply_transform(tile.image), frame.source)
        if first is None:
            first = result
            output_shapes = tuple((f"tile_output/{name}", shape) for name, shape in result.output_shapes)
            if result.task_type is TaskType.ANOMALY:
                anomaly_map = np.zeros((frame.image.height, frame.image.width), dtype=np.float32)
        left, top, _right, _bottom = tile.box
        roi_offset = position * 100_000
        detections.extend(
            replace(
                detection,
                box=(
                    detection.box[0] + left,
                    detection.box[1] + top,
                    detection.box[2] + left,
                    detection.box[3] + top,
                ),
                roi_index=(detection.roi_index + roi_offset) if detection.roi_index is not None else None,
            )
            for detection in result.detections
        )
        for classification in result.classifications:
            key = classification.label_id, classification.label
            if key not in classifications or classification.confidence > classifications[key].confidence:
                classifications[key] = classification
        if result.anomaly_score is not None:
            anomaly_score = max(anomaly_score or 0.0, result.anomaly_score)
        if anomaly_map is not None and result.anomaly_map is not None:
            preview_box = _preview_box(tile.box, frame.original_size, frame.image.size)
            width, height = preview_box[2] - preview_box[0], preview_box[3] - preview_box[1]
            resized = Image.fromarray(np.asarray(result.anomaly_map, dtype=np.float32), mode="F").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            region = np.asarray(resized, dtype=np.float32)
            target = anomaly_map[preview_box[1] : preview_box[3], preview_box[0] : preview_box[2]]
            np.maximum(target, region, out=target)
        if progress is not None:
            progress(position, total)
    if first is None:
        raise ValueError("Tiled inference produced no tile results.")
    if should_continue is not None and not should_continue():
        raise InterruptedError("Inference cancelled")
    if finalizing is not None:
        finalizing()
    return InferenceResult(
        source=frame.source,
        task_type=first.task_type,
        image_size=frame.original_size,
        input_size=first.input_size,
        duration_ms=(perf_counter() - started) * 1000.0,
        detections=tuple(_non_maximum_suppression(detections)),
        classifications=tuple(sorted(classifications.values(), key=lambda item: item.confidence, reverse=True)),
        anomaly_score=anomaly_score,
        anomaly_map=anomaly_map,
        output_shapes=output_shapes,
    )


def _input_size(engine: InferenceBackend) -> tuple[int, int]:
    size = engine.model_info.input_size
    if size is None:
        raise ValueError("Full-resolution tiled inference requires a fixed model input size.")
    return size


def _tile_count(image_size: tuple[int, int], tile_size: tuple[int, int], overlap: float) -> int:
    image_width, image_height = image_size
    tile_width, tile_height = min(tile_size[0], image_width), min(tile_size[1], image_height)
    step_x = max(1, round(tile_width * (1.0 - overlap)))
    step_y = max(1, round(tile_height * (1.0 - overlap)))
    columns = 1 + max(0, (image_width - tile_width + step_x - 1) // step_x)
    rows = 1 + max(0, (image_height - tile_height + step_y - 1) // step_y)
    return columns * rows


def _preview_box(
    box: tuple[int, int, int, int],
    source_size: tuple[int, int],
    preview_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    scale_x, scale_y = preview_size[0] / source_size[0], preview_size[1] / source_size[1]
    return (
        max(0, round(box[0] * scale_x)),
        max(0, round(box[1] * scale_y)),
        min(preview_size[0], max(1, round(box[2] * scale_x))),
        min(preview_size[1], max(1, round(box[3] * scale_y))),
    )


def _non_maximum_suppression(detections: list[Detection], threshold: float = 0.5) -> list[Detection]:
    """Apply exact NMS using spatial buckets instead of an all-pairs scan."""
    selected: list[Detection] = []
    buckets: dict[tuple[int, str, int, int], list[int]] = defaultdict(list)
    cell_size = _nms_cell_size(detections)
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        cells = tuple(_box_cells(candidate.box, cell_size))
        nearby = {
            selected_index
            for cell_x, cell_y in cells
            for selected_index in buckets[(candidate.label_id, candidate.stage, cell_x, cell_y)]
        }
        if any(_intersection_over_union(selected[index].box, candidate.box) >= threshold for index in nearby):
            continue
        selected_index = len(selected)
        selected.append(candidate)
        for cell_x, cell_y in cells:
            buckets[(candidate.label_id, candidate.stage, cell_x, cell_y)].append(selected_index)
    return selected


def _nms_cell_size(detections: list[Detection]) -> float:
    if not detections:
        return 64.0
    dimensions = [max(item.box[2] - item.box[0], item.box[3] - item.box[1], 1.0) for item in detections]
    return max(32.0, float(np.median(dimensions)) * 2.0)


def _box_cells(box: tuple[float, float, float, float], cell_size: float) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    first_x, first_y = math.floor(left / cell_size), math.floor(top / cell_size)
    last_x = math.floor(max(left, right - 1e-9) / cell_size)
    last_y = math.floor(max(top, bottom - 1e-9) / cell_size)
    return [(cell_x, cell_y) for cell_y in range(first_y, last_y + 1) for cell_x in range(first_x, last_x + 1)]


def _intersection_over_union(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0