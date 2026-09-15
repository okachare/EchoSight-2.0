"""Export reviewed frames for model retraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .frames import LoadedFrame


@dataclass(frozen=True)
class TrainingExport:
    directory: Path
    images: tuple[Path, ...]


def export_training_frames(
    destination: Path,
    frames: list[tuple[int, LoadedFrame]],
    category: str,
    model_folder_name: str,
) -> TrainingExport:
    if category not in {"False_Hits", "Misses"}:
        raise ValueError(f"Unsupported training category: {category}")
    if not frames:
        raise ValueError("Select at least one result frame for training.")

    folder_name = f"Mark_for_Training_{category}_({_safe_name(model_folder_name)})"
    output_directory = destination / folder_name
    output_directory.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for index, frame in frames:
        stem = _safe_name(frame.source.stem)
        filename = f"{index + 1:05d}_{stem}_frame_{frame.frame_number:04d}.png"
        output_path = _available_path(output_directory / filename)
        frame.image.convert("RGB").save(output_path, format="PNG")
        exported.append(output_path)
    return TrainingExport(output_directory, tuple(exported))


def _available_path(path: Path) -> Path:
    candidate = path
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        suffix += 1
    return candidate


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return safe.strip("_") or "model"