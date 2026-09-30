"""Inference orchestration for supported Geti task chains."""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from .engine import Detection, InferenceEngine, InferenceResult
from .model_loader import ModelLoader, TaskType


def discover_detection_classification_chain(parent: Path) -> tuple[Path, Path] | None:
    """Return model paths for an explicit Detection -> Crop -> Classification graph."""
    project_path = parent / "deployment" / "project.json"
    if not project_path.is_file():
        project_path = parent / "project.json"
    if not project_path.is_file():
        return None
    try:
        project = json.loads(project_path.read_text(encoding="utf-8"))
        tasks = project["pipeline"]["tasks"]
        task_types = [str(task["task_type"]).lower() for task in tasks]
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    if task_types != ["dataset", "detection", "crop", "classification"]:
        return None

    deployment_root = project_path.parent
    by_type = {str(task["task_type"]).lower(): str(task["title"]) for task in tasks if task["task_type"] in {"detection", "classification"}}
    paths = tuple(deployment_root / by_type[task_type] / "model" / "model.xml" for task_type in ("detection", "classification"))
    if all(path.is_file() and path.with_suffix(".bin").is_file() for path in paths):
        return paths
    return None


class ChainedInferenceEngine:
    """Run detection, crop each detected ROI, then classify each crop."""

    def __init__(self, detector: InferenceEngine, classifier: InferenceEngine) -> None:
        if detector.model_info.task_type is not TaskType.DETECTION:
            raise ValueError("The first chained model must be a detection model.")
        if classifier.model_info.task_type is not TaskType.CLASSIFICATION:
            raise ValueError("The second chained model must be a classification model.")
        self.detector = detector
        self.classifier = classifier
        self.model_infos = (detector.model_info, classifier.model_info)
        self.model_info = detector.model_info

    @classmethod
    def compile(cls, paths: tuple[Path, Path], package_root: Path) -> ChainedInferenceEngine:
        loader = ModelLoader()
        engines = []
        for path in paths:
            info = loader.inspect(path, package_root)
            model = loader.core.read_model(info.path)
            compiled = loader.core.compile_model(model, loader.device)
            engines.append(InferenceEngine(info, compiled))
        return cls(engines[0], engines[1])

    def infer(self, image: Image.Image, source: Path | None = None) -> InferenceResult:
        started = perf_counter()
        detected = self.detector.infer(image, source)
        classified_detections = []
        output_shapes = [(f"detection/{name}", shape) for name, shape in detected.output_shapes]

        for position, detection in enumerate(detected.detections, start=1):
            classified_detections.append(
                Detection(
                    label_id=detection.label_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    box=detection.box,
                    mask=detection.mask,
                    detector_confidence=detection.confidence,
                    stage="detection",
                    roi_index=position,
                )
            )
            crop = self._crop(image, detection.box)
            if crop is None:
                continue
            classified = self.classifier.infer(crop, source)
            output_shapes.extend((f"classification_{position}/{name}", shape) for name, shape in classified.output_shapes)
            selected = self._selected_classifications(classified)
            for classification_index, classification in enumerate(selected):
                classified_detections.append(
                    Detection(
                        label_id=classification.label_id,
                        label=classification.label,
                        confidence=classification.confidence,
                        box=detection.box,
                        detector_confidence=detection.confidence,
                        stage="classification",
                        roi_index=position,
                        stage_index=classification_index,
                    )
                )

        return InferenceResult(
            source=source,
            task_type=TaskType.DETECTION,
            image_size=image.size,
            input_size=detected.input_size,
            duration_ms=(perf_counter() - started) * 1000.0,
            detections=tuple(classified_detections),
            output_shapes=tuple(output_shapes),
        )

    def _selected_classifications(self, result: InferenceResult) -> tuple[Any, ...]:
        threshold = self.classifier.confidence_threshold
        metadata = dict(self.classifier.model_info.metadata)
        if metadata.get("hierarchical", "false").lower() != "true":
            return tuple(item for item in result.classifications[:1] if item.confidence >= threshold)
        return tuple(item for item in result.classifications if item.confidence >= threshold)

    @staticmethod
    def _crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image | None:
        left = max(0, min(image.width, math.floor(box[0])))
        top = max(0, min(image.height, math.floor(box[1])))
        right = max(0, min(image.width, math.ceil(box[2])))
        bottom = max(0, min(image.height, math.ceil(box[3])))
        if right <= left or bottom <= top:
            return None
        return image.crop((left, top, right, bottom))