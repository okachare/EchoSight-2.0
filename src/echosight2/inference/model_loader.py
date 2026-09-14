"""Model discovery, validation, and tensor introspection."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from openvino import Core, Model


class TaskType(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    ANOMALY = "anomaly"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int | None, ...]
    element_type: str


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    format: str
    task_type: TaskType
    inputs: tuple[TensorInfo, ...]
    outputs: tuple[TensorInfo, ...]
    labels: tuple[str, ...]
    model_type: str | None
    device: str
    metadata: tuple[tuple[str, str], ...] = ()
    metrics: tuple[tuple[str, float], ...] = ()
    training_date: str | None = None
    artifact_date: str | None = None
    collaterals: tuple[str, ...] = ()

    @property
    def confidence_threshold(self) -> float:
        values = dict(self.metadata)
        try:
            return float(values.get("confidence_threshold", "0.25"))
        except ValueError:
            return 0.25

    @property
    def input_size(self) -> tuple[int, int] | None:
        if not self.inputs or len(self.inputs[0].shape) != 4:
            return None
        height, width = self.inputs[0].shape[-2:]
        if height is None or width is None:
            return None
        return height, width

    @property
    def output_mode(self) -> str:
        return {
            TaskType.DETECTION: "Bounding boxes, labels, and confidence scores",
            TaskType.INSTANCE_SEGMENTATION: "Bounding boxes, labels, confidence scores, and instance masks",
            TaskType.ANOMALY: "Calibrated anomaly score and pixel heatmap",
            TaskType.CLASSIFICATION: "Ranked labels and confidence scores",
            TaskType.UNKNOWN: "Raw model outputs",
        }[self.task_type]


class ModelLoader:
    def __init__(self, device: str = "CPU", core: Core | None = None) -> None:
        self.device = device
        self.core = core or Core()

    def inspect(self, path: Path, package_root: Path | None = None) -> ModelInfo:
        model_path = path.resolve()
        self._validate_path(model_path)
        model = self.core.read_model(model_path)
        config = self._load_config(model_path, package_root)
        runtime_metadata = self._runtime_metadata(model)
        inputs = tuple(self._tensor_info(port) for port in model.inputs)
        outputs = tuple(self._tensor_info(port) for port in model.outputs)
        merged_config = {**config, **runtime_metadata}
        task_type = self._identify_task(model_path, outputs, merged_config)
        parameters = config.get("model_parameters", {}) if config else {}
        labels = self._labels_from_metadata(runtime_metadata, parameters.get("labels"))
        metrics, training_date = self._load_companion_metrics(model_path)
        return ModelInfo(
            path=model_path,
            format="OpenVINO IR" if model_path.suffix.lower() == ".xml" else "ONNX",
            task_type=task_type,
            inputs=inputs,
            outputs=outputs,
            labels=tuple(label for label in labels if label != "getitune_empty_lbl"),
            model_type=runtime_metadata.get("model_type") or (str(config["model_type"]) if config and config.get("model_type") else None),
            device=self.device,
            metadata=tuple(sorted(runtime_metadata.items())),
            metrics=metrics,
            training_date=training_date,
            artifact_date=datetime.fromtimestamp(model_path.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M %Z"),
            collaterals=self._inventory_collaterals(package_root),
        )

    def compile(self, path: Path, package_root: Path | None = None) -> tuple[ModelInfo, Any]:
        info = self.inspect(path, package_root)
        model = self.core.read_model(info.path)
        compiled_model = self.core.compile_model(model, self.device)
        return info, compiled_model

    @classmethod
    def discover(cls, parent: Path) -> tuple[Path, ...]:
        model_parent = parent.resolve()
        if not model_parent.is_dir():
            raise NotADirectoryError(f"Model parent folder does not exist: {model_parent}")

        candidates: dict[Path, Path] = {}
        ignored_directories = {".git", ".venv", "__pycache__", "build", "dist"}
        for root, directories, files in os.walk(model_parent, onerror=lambda _: None):
            directories[:] = [name for name in directories if name.lower() not in ignored_directories]
            file_names = {name.lower(): name for name in files}
            directory = Path(root)
            for lower_name, actual_name in file_names.items():
                path = directory / actual_name
                if lower_name.endswith(".xml") and path.with_suffix(".bin").is_file():
                    candidates[path.with_suffix("")] = path
                elif lower_name.endswith(".onnx"):
                    candidates.setdefault(path.with_suffix(""), path)

        return tuple(sorted(candidates.values(), key=cls._discovery_rank))

    @staticmethod
    def _discovery_rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        parts = {part.lower() for part in path.parts}
        if name == "model.xml" and "deployment" in parts:
            priority = 0
        elif name == "exported_model.xml":
            priority = 1
        elif path.suffix.lower() == ".xml":
            priority = 2
        else:
            priority = 3
        return priority, str(path).lower()

    @staticmethod
    def _validate_path(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {path}")
        if path.suffix.lower() not in {".xml", ".onnx"}:
            raise ValueError(f"Unsupported model format: {path.suffix or '<none>'}")
        if path.suffix.lower() == ".xml" and not path.with_suffix(".bin").is_file():
            raise FileNotFoundError(f"OpenVINO weights file does not exist: {path.with_suffix('.bin')}")

    @staticmethod
    def _tensor_info(port: Any) -> TensorInfo:
        shape: list[int | None] = []
        for dimension in port.partial_shape:
            shape.append(int(dimension.get_length()) if dimension.is_static else None)
        return TensorInfo(name=port.any_name, shape=tuple(shape), element_type=str(port.element_type))

    @staticmethod
    def _load_config(model_path: Path, package_root: Path | None = None) -> dict[str, Any]:
        candidates = [
            model_path.with_name("config.json"),
            model_path.parent.parent / "python" / "demo_package" / "config.json",
        ]
        if package_root is not None and package_root.is_dir():
            candidates.extend(package_root.rglob("config.json"))
        for candidate in candidates:
            if candidate.is_file():
                with candidate.open(encoding="utf-8") as stream:
                    value = json.load(stream)
                return value if isinstance(value, dict) else {}
        return {}

    @staticmethod
    def _inventory_collaterals(package_root: Path | None) -> tuple[str, ...]:
        if package_root is None or not package_root.is_dir():
            return ()
        counts = {"configuration": 0, "sample image": 0, "Python wrapper": 0}
        image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
        for root, directories, files in os.walk(package_root, onerror=lambda _: None):
            directories[:] = [name for name in directories if name.lower() not in {".git", ".venv", "build", "dist"}]
            for name in files:
                suffix = Path(name).suffix.lower()
                if suffix in {".json", ".yaml", ".yml"}:
                    counts["configuration"] += 1
                elif suffix in image_suffixes:
                    counts["sample image"] += 1
                elif suffix in {".py", ".pyc"}:
                    counts["Python wrapper"] += 1
        return tuple(f"{name}: {count}" for name, count in counts.items() if count)

    @classmethod
    def _labels_from_metadata(cls, metadata: dict[str, str], fallback: object) -> tuple[str, ...]:
        label_info = metadata.get("label_info")
        if label_info:
            try:
                values = json.loads(label_info).get("label_names", [])
                if isinstance(values, list):
                    return tuple(str(value) for value in values)
            except (AttributeError, json.JSONDecodeError):
                pass
        return cls._parse_labels(metadata.get("labels") or fallback)

    @staticmethod
    def _runtime_metadata(model: Model) -> dict[str, str]:
        try:
            runtime_info = model.get_rt_info()
            values = runtime_info["model_info"] if "model_info" in runtime_info else {}  # noqa: SIM401
        except (KeyError, RuntimeError):
            return {}
        return {
            str(key): str(getattr(value, "value", value))
            for key, value in values.items()
        } if isinstance(values, dict) else {}

    @classmethod
    def _load_companion_metrics(cls, model_path: Path) -> tuple[tuple[tuple[str, float], ...], str | None]:
        model_id = model_path.parent.name.removeprefix("getitune-workspace-")
        model_root = next((parent for parent in model_path.parents if parent.name == model_id), None)
        metric_files: list[Path] = []
        if model_root is not None:
            metric_files.extend(model_root.glob("metrics/version_*/*.csv"))
        for parent in list(model_path.parents)[:4]:
            projects = parent / "projects"
            if projects.is_dir():
                metric_files.extend(projects.glob(f"*/models/{model_id}/metrics/version_*/*.csv"))

        metrics: dict[str, float] = {}
        for metric_file in metric_files:
            with metric_file.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for prefix in ("val/", "test/"):
                for key in (name for name in (rows[0].keys() if rows else ()) if name.startswith(prefix)):
                    values = [cls._finite_float(row.get(key)) for row in rows]
                    valid = [value for value in values if value is not None and value >= 0]
                    if valid and not key.endswith(("data_time", "iter_time", "classes")):
                        metrics[key] = max(valid) if prefix == "val/" else valid[-1]

        companion_roots = {metric_file.parents[2] for metric_file in metric_files}
        if model_root is not None:
            companion_roots.add(model_root)
        training_date = None
        log_candidates = [root / "training.log" for root in companion_roots]
        for log_path in log_candidates:
            if not log_path.is_file():
                continue
            match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Model training completed", log_path.read_text(encoding="utf-8", errors="replace"))
            if match:
                training_date = match.group(1)
                break
        return tuple(sorted(metrics.items())), training_date

    @staticmethod
    def _finite_float(value: str | None) -> float | None:
        try:
            number = float(value) if value not in (None, "") else None
            return number if number is not None and math.isfinite(number) else None
        except ValueError:
            return None

    @classmethod
    def _identify_task(
        cls,
        model_path: Path,
        outputs: Iterable[TensorInfo],
        config: dict[str, Any],
    ) -> TaskType:
        configured = str(config.get("task_type", "")).lower().replace(" ", "_")
        aliases = {
            "detection": TaskType.DETECTION,
            "classification": TaskType.CLASSIFICATION,
            "instance_segmentation": TaskType.INSTANCE_SEGMENTATION,
            "anomaly": TaskType.ANOMALY,
            "anomaly_classification": TaskType.ANOMALY,
            "anomaly_segmentation": TaskType.ANOMALY,
        }
        if configured in aliases:
            return aliases[configured]

        output_list = tuple(outputs)
        output_names = {output.name.lower() for output in output_list}
        if any("mask" in name for name in output_names) and ("boxes" in output_names or "labels" in output_names):
            return TaskType.INSTANCE_SEGMENTATION
        if {"boxes", "labels"}.issubset(output_names):
            return TaskType.DETECTION

        path_hint = model_path.parent.parent.name.lower()
        if "instance segmentation" in path_hint or "instance_segmentation" in path_hint:
            return TaskType.INSTANCE_SEGMENTATION
        if "anomaly" in path_hint:
            return TaskType.ANOMALY
        if "classification" in path_hint:
            return TaskType.CLASSIFICATION
        if "detection" in path_hint:
            return TaskType.DETECTION

        if len(output_list) == 1:
            rank = len(output_list[0].shape)
            if rank >= 3:
                return TaskType.ANOMALY
            if rank <= 2:
                return TaskType.CLASSIFICATION
        return TaskType.UNKNOWN

    @staticmethod
    def _parse_labels(value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            return tuple(value.split())
        if isinstance(value, dict):
            return tuple(str(item) for item in value.values())
        if isinstance(value, list):
            return tuple(str(item) for item in value)
        return ()
