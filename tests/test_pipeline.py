from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from echosight2.inference import (
    ChainedInferenceEngine,
    Classification,
    Detection,
    InferenceResult,
    ModelInfo,
    TaskType,
    TensorInfo,
    discover_detection_classification_chain,
)


def _info(task: TaskType, threshold: float = 0.5) -> ModelInfo:
    return ModelInfo(
        path=Path(f"{task.value}.xml"),
        format="OpenVINO IR",
        task_type=task,
        inputs=(TensorInfo("image", (1, 3, 32, 32), "f32"),),
        outputs=(),
        labels=("Anomaly",) if task is TaskType.DETECTION else ("Delamination", "Voids"),
        model_type=None,
        device="CPU",
        metadata=(("confidence_threshold", str(threshold)),),
    )


def test_chained_engine_classifies_each_detected_crop() -> None:
    detector = Mock(model_info=_info(TaskType.DETECTION))
    detector.infer.return_value = InferenceResult(
        source=Path("scan.tiff"),
        task_type=TaskType.DETECTION,
        image_size=(100, 80),
        input_size=(32, 32),
        duration_ms=4.0,
        detections=(Detection(0, "Anomaly", 0.91, (10.2, 20.4, 50.1, 60.9)),),
        output_shapes=(("bboxes", (1, 1, 4)),),
    )
    classifier = Mock(model_info=_info(TaskType.CLASSIFICATION), confidence_threshold=0.5)
    classifier.infer.return_value = InferenceResult(
        source=Path("scan.tiff"),
        task_type=TaskType.CLASSIFICATION,
        image_size=(41, 42),
        input_size=(32, 32),
        duration_ms=2.0,
        classifications=(Classification(1, "Voids", 0.82), Classification(0, "Delamination", 0.18)),
        output_shapes=(("output1", (1, 2)),),
    )
    engine = ChainedInferenceEngine(detector, classifier)

    result = engine.infer(Image.new("RGB", (100, 80), "white"), Path("scan.tiff"))

    assert len(result.detections) == 2
    detection, classification = result.detections
    assert detection.stage == "detection"
    assert detection.label == "Anomaly"
    assert detection.confidence == pytest.approx(0.91)
    assert classification.stage == "classification"
    assert classification.label == "Voids"
    assert classification.confidence == pytest.approx(0.82)
    assert classification.detector_confidence == pytest.approx(0.91)
    assert detection.roi_index == classification.roi_index == 1
    assert classifier.infer.call_args.args[0].size == (41, 41)


def test_discovers_only_explicit_detection_crop_classification_chain(tmp_path: Path) -> None:
    deployment = tmp_path / "deployment"
    for title in ("Detection", "Classification"):
        model = deployment / title / "model" / "model.xml"
        model.parent.mkdir(parents=True)
        model.write_text("xml", encoding="utf-8")
        model.with_suffix(".bin").write_bytes(b"bin")
    (deployment / "project.json").write_text(
        '{"pipeline":{"tasks":['
        '{"task_type":"dataset","title":"Dataset"},'
        '{"task_type":"detection","title":"Detection"},'
        '{"task_type":"crop","title":"Crop"},'
        '{"task_type":"classification","title":"Classification"}]}}',
        encoding="utf-8",
    )

    paths = discover_detection_classification_chain(tmp_path)

    assert paths is not None
    assert [path.parent.parent.name for path in paths] == ["Detection", "Classification"]