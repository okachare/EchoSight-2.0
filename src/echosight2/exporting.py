"""Reproducible export of EchoSight inference runs."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from . import __version__
from .frames import LoadedFrame
from .inference import InferenceResult, ModelInfo
from .rendering import RenderOptions, render_result


@dataclass(frozen=True)
class ExportReport:
    directory: Path
    manifest: Path
    results_csv: Path
    annotated_images: tuple[Path, ...]
    exported_frames: int
    failed_frames: int


def export_run(
    destination: Path,
    frames: list[LoadedFrame],
    results: dict[int, InferenceResult],
    model_info: ModelInfo,
    model_parent: Path | None,
    preprocessing: dict[str, Any],
    render_options: RenderOptions | None = None,
    hidden_annotations: dict[int, set[int]] | None = None,
    failures: dict[int, str] | None = None,
    exported_at: datetime | None = None,
    image_transform: Callable[[Image.Image], Image.Image] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ExportReport:
    """Write annotated images, flat CSV results, and a reproducible JSON manifest."""
    if not results and not failures:
        raise ValueError("There are no inference results or failures to export.")

    timestamp = exported_at or datetime.now().astimezone()
    if timestamp.tzinfo is None:
        timestamp = timestamp.astimezone()
    run_directory = _unique_run_directory(destination, timestamp)
    image_directory = run_directory / "annotated"
    image_directory.mkdir(parents=True)
    options = render_options or RenderOptions()
    hidden = hidden_annotations or {}
    failed = failures or {}

    frame_records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    annotated_images: list[Path] = []
    durations: list[float] = []
    source_hashes: dict[Path, str | None] = {}

    frame_indexes = sorted(set(results) | set(failed))
    for position, frame_index in enumerate(frame_indexes, start=1):
        frame = frames[frame_index]
        source_hash = source_hashes.setdefault(frame.source, _sha256_optional(frame.source))
        if progress is not None:
            progress(position, len(frame_indexes), frame.display_name)
        if frame_index in failed:
            message = failed[frame_index]
            frame_records.append(_failed_frame_record(frame_index, frame, source_hash, message))
            csv_rows.append(_csv_row(frame_index, frame, source_hash, status="failed", error=message))
            continue

        result = results[frame_index]
        durations.append(result.duration_ms)
        output_name = _annotated_name(frame_index, frame)
        output_path = image_directory / output_name
        source_image = image_transform(frame.image) if image_transform is not None else frame.image
        rendered = render_result(source_image, result, options, hidden.get(frame_index, set()))
        rendered.save(output_path, format="PNG")
        annotated_images.append(output_path)
        frame_records.append(_result_record(frame_index, frame, source_hash, result, output_path.relative_to(run_directory)))
        csv_rows.extend(_result_rows(frame_index, frame, source_hash, result, output_path.relative_to(run_directory)))

    results_path = run_directory / "results.csv"
    _write_csv(results_path, csv_rows)
    manifest_path = run_directory / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "application": {"name": "EchoSight", "version": __version__},
        "exported_at": timestamp.isoformat(),
        "model": _model_record(model_info, model_parent),
        "processing": {
            "user_preprocessing": preprocessing,
            "model_preprocessing": _model_preprocessing(model_info),
            "render_options": asdict(options),
        },
        "summary": _summary(frames, results, failed, durations),
        "frames": frame_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    return ExportReport(
        directory=run_directory,
        manifest=manifest_path,
        results_csv=results_path,
        annotated_images=tuple(annotated_images),
        exported_frames=len(results),
        failed_frames=len(failed),
    )


def _unique_run_directory(destination: Path, timestamp: datetime) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / f"EchoSight_Run_{timestamp:%Y%m%d_%H%M%S}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = destination / f"{base.name}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _model_record(info: ModelInfo, model_parent: Path | None) -> dict[str, Any]:
    weights_path = info.path.with_suffix(".bin") if info.path.suffix.lower() == ".xml" else None
    return {
        "package_folder": str(model_parent.resolve()) if model_parent else None,
        "artifact": str(info.path),
        "artifact_sha256": _sha256(info.path),
        "weights_sha256": _sha256(weights_path) if weights_path and weights_path.is_file() else None,
        "format": info.format,
        "device": info.device,
        "task": info.task_type.value,
        "output_mode": info.output_mode,
        "model_type": info.model_type,
        "labels": list(info.labels),
        "confidence_threshold": info.confidence_threshold,
        "training_date": info.training_date,
        "artifact_date": info.artifact_date,
        "inputs": [_tensor_record(item) for item in info.inputs],
        "outputs": [_tensor_record(item) for item in info.outputs],
        "metadata": dict(info.metadata),
        "metrics": dict(info.metrics),
        "collaterals": list(info.collaterals),
    }


def _tensor_record(tensor: Any) -> dict[str, Any]:
    return {"name": tensor.name, "shape": list(tensor.shape), "element_type": tensor.element_type}


def _model_preprocessing(info: ModelInfo) -> dict[str, str]:
    metadata = dict(info.metadata)
    keys = (
        "resize_type",
        "pad_value",
        "intensity_mode",
        "intensity_min_value",
        "intensity_max_value",
        "intensity_scale_factor",
        "mean_values",
        "scale_values",
        "reverse_input_channels",
        "image_threshold",
        "pixel_threshold",
        "normalization_scale",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _summary(
    frames: list[LoadedFrame],
    results: dict[int, InferenceResult],
    failures: dict[int, str],
    durations: list[float],
) -> dict[str, Any]:
    annotation_count = sum(len(result.detections) + len(result.classifications) for result in results.values())
    anomaly_count = sum(result.anomaly_score is not None for result in results.values())
    return {
        "loaded_frames": len(frames),
        "exported_frames": len(results),
        "failed_frames": len(failures),
        "annotations": annotation_count,
        "anomaly_results": anomaly_count,
        "total_inference_ms": sum(durations),
        "average_inference_ms": sum(durations) / len(durations) if durations else 0.0,
    }


def _result_record(
    frame_index: int,
    frame: LoadedFrame,
    source_sha256: str | None,
    result: InferenceResult,
    annotated_path: Path,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "source": str(frame.source),
        "source_sha256": source_sha256,
        "source_frame": frame.frame_number,
        "source_frame_count": frame.frame_count,
        "status": "complete",
        "task": result.task_type.value,
        "summary": result.summary,
        "image_size": list(result.image_size),
        "model_input_size": list(result.input_size),
        "inference_ms": result.duration_ms,
        "anomaly_score": result.anomaly_score,
        "detections": [
            {
                "label_id": item.label_id,
                "label": item.label,
                "confidence": item.confidence,
                "box": list(item.box),
                "mask_shape": list(item.mask.shape) if item.mask is not None else None,
            }
            for item in result.detections
        ],
        "classifications": [asdict(item) for item in result.classifications],
        "output_shapes": [{"name": name, "shape": list(shape)} for name, shape in result.output_shapes],
        "annotated_image": annotated_path.as_posix(),
    }


def _failed_frame_record(
    frame_index: int,
    frame: LoadedFrame,
    source_sha256: str | None,
    message: str,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "source": str(frame.source),
        "source_sha256": source_sha256,
        "source_frame": frame.frame_number,
        "source_frame_count": frame.frame_count,
        "status": "failed",
        "error": message,
    }


def _result_rows(
    frame_index: int,
    frame: LoadedFrame,
    source_sha256: str | None,
    result: InferenceResult,
    annotated_path: Path,
) -> list[dict[str, Any]]:
    common = {
        "frame_index": frame_index,
        "source": str(frame.source),
        "source_sha256": source_sha256,
        "source_frame": frame.frame_number,
        "status": "complete",
        "task": result.task_type.value,
        "inference_ms": result.duration_ms,
        "annotated_image": annotated_path.as_posix(),
    }
    rows = []
    if result.anomaly_score is not None:
        rows.append({**common, "result_type": "anomaly", "label": "Anomaly", "confidence": result.anomaly_score})
    rows.extend(
        {
            **common,
            "result_type": "detection",
            "label": item.label,
            "label_id": item.label_id,
            "confidence": item.confidence,
            "x1": item.box[0],
            "y1": item.box[1],
            "x2": item.box[2],
            "y2": item.box[3],
            "has_mask": item.mask is not None,
        }
        for item in result.detections
    )
    rows.extend(
        {
            **common,
            "result_type": "classification",
            "label": item.label,
            "label_id": item.label_id,
            "confidence": item.confidence,
        }
        for item in result.classifications
    )
    return rows or [{**common, "result_type": "none"}]


def _csv_row(
    frame_index: int,
    frame: LoadedFrame,
    source_sha256: str | None,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "source": str(frame.source),
        "source_sha256": source_sha256,
        "source_frame": frame.frame_number,
        "status": status,
        "error": error,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "frame_index",
        "source",
        "source_sha256",
        "source_frame",
        "status",
        "task",
        "result_type",
        "label",
        "label_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "has_mask",
        "inference_ms",
        "annotated_image",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _annotated_name(frame_index: int, frame: LoadedFrame) -> str:
    stem = re_safe_name(frame.source.stem)
    return f"{frame_index + 1:05d}_{stem}_frame_{frame.frame_number:04d}.png"


def re_safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return safe.strip("_") or "image"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_optional(path: Path) -> str | None:
    try:
        return _sha256(path)
    except OSError:
        return None
