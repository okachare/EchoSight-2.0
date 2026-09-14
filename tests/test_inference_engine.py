from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from echosight2.inference import (
    InferenceEngine,
    ModelInfo,
    ModelLoader,
    TaskType,
    TensorInfo,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DETECTION_MODEL = WORKSPACE_ROOT / "Deployment" / "Test_Run_Detect" / "deployment" / "Detection" / "model" / "model.xml"
ANOMALY_MODEL = WORKSPACE_ROOT / "Deployment" / "Test_Run_Instance_Segmentation" / "deployment" / "Anomaly classification" / "model" / "model.xml"
SAMPLE_IMAGE = WORKSPACE_ROOT / "Debug" / "Evaluation" / "Test Data" / "00790_058.png"


def _model_info(task_type: TaskType) -> ModelInfo:
    return ModelInfo(
        path=Path("model.xml"),
        format="OpenVINO IR",
        task_type=task_type,
        inputs=(TensorInfo("image", (1, 3, 100, 200), "<Type: 'float32'>"),),
        outputs=(),
        labels=("defect",),
        model_type=None,
        device="CPU",
    )


def test_prepare_image_creates_nchw_float_tensor() -> None:
    engine = InferenceEngine(_model_info(TaskType.DETECTION), compiled_model=None)

    prepared = engine.prepare_image(Image.new("RGB", (400, 300), "white"))

    assert prepared.tensor.shape == (1, 3, 100, 200)
    assert prepared.tensor.dtype == np.float32
    assert prepared.original_size == (400, 300)
    assert prepared.input_size == (200, 100)


def test_uses_confidence_threshold_from_model_metadata() -> None:
    info = _model_info(TaskType.DETECTION)
    info = ModelInfo(
        path=info.path,
        format=info.format,
        task_type=info.task_type,
        inputs=info.inputs,
        outputs=info.outputs,
        labels=info.labels,
        model_type=info.model_type,
        device=info.device,
        metadata=(("confidence_threshold", "0.1"),),
    )

    engine = InferenceEngine(info, compiled_model=None)

    assert engine.confidence_threshold == pytest.approx(0.1)


def test_applies_model_metadata_preprocessing() -> None:
    info = _model_info(TaskType.ANOMALY)
    info = ModelInfo(
        path=info.path,
        format=info.format,
        task_type=info.task_type,
        inputs=(TensorInfo("image", (1, 3, 1, 1), "<Type: 'float32'>"),),
        outputs=info.outputs,
        labels=info.labels,
        model_type=info.model_type,
        device=info.device,
        metadata=(("mean_values", "10 20 30"), ("scale_values", "2 4 5")),
    )

    prepared = InferenceEngine(info, compiled_model=None).prepare_image(Image.new("RGB", (1, 1), (20, 40, 80)))

    assert prepared.tensor[0, :, 0, 0] == pytest.approx((5.0, 5.0, 10.0))


def test_normalizes_anomaly_output_from_model_metadata() -> None:
    info = _model_info(TaskType.ANOMALY)
    info = ModelInfo(
        path=info.path,
        format=info.format,
        task_type=info.task_type,
        inputs=info.inputs,
        outputs=info.outputs,
        labels=info.labels,
        model_type=info.model_type,
        device=info.device,
        metadata=(("image_threshold", "0.25"), ("pixel_threshold", "0.25"), ("normalization_scale", "0.5")),
    )
    engine = InferenceEngine(info, compiled_model=None)
    prepared = engine.prepare_image(Image.new("RGB", (200, 100), "black"))

    result = engine.normalize({"output": np.array([[[[0.0, 0.5]]]], dtype=np.float32)}, prepared, 1.0)

    assert result.anomaly_score == pytest.approx(1.0)
    assert result.anomaly_map.flat[0] == pytest.approx(0.0)
    assert result.anomaly_map.flat[1] == pytest.approx(1.0)


def test_normalizes_detection_boxes_to_source_coordinates() -> None:
    engine = InferenceEngine(_model_info(TaskType.DETECTION), compiled_model=None, confidence_threshold=0.25)
    prepared = engine.prepare_image(Image.new("RGB", (400, 300), "white"))
    outputs = {
        "boxes": np.array([[[10, 20, 100, 80, 0.9], [1, 2, 3, 4, 0.1]]], dtype=np.float32),
        "labels": np.array([[0, 0]], dtype=np.int64),
    }

    result = engine.normalize(outputs, prepared, duration_ms=12.0)

    assert len(result.detections) == 1
    assert result.detections[0].box == pytest.approx((20.0, 60.0, 200.0, 240.0))
    assert result.detections[0].label == "defect"


@pytest.mark.skipif(not (DETECTION_MODEL.exists() and SAMPLE_IMAGE.exists()), reason="Detection fixtures are unavailable")
def test_runs_real_detection_inference() -> None:
    info, compiled = ModelLoader().compile(DETECTION_MODEL)
    engine = InferenceEngine(info, compiled, confidence_threshold=0.0)

    with Image.open(SAMPLE_IMAGE) as image:
        result = engine.infer(image, SAMPLE_IMAGE)

    assert result.task_type is TaskType.DETECTION
    assert result.duration_ms > 0
    assert result.output_shapes
    assert len(result.detections) > 0


@pytest.mark.skipif(not (ANOMALY_MODEL.exists() and SAMPLE_IMAGE.exists()), reason="Anomaly fixtures are unavailable")
def test_runs_real_anomaly_inference() -> None:
    info, compiled = ModelLoader().compile(ANOMALY_MODEL)
    engine = InferenceEngine(info, compiled)

    with Image.open(SAMPLE_IMAGE) as image:
        result = engine.infer(image, SAMPLE_IMAGE)

    assert result.task_type is TaskType.ANOMALY
    assert result.anomaly_map is not None
    assert result.anomaly_map.shape == (448, 448)
    assert 0 <= result.anomaly_score <= 1