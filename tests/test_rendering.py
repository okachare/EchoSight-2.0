from pathlib import Path

import numpy as np
from PIL import Image

from echosight2.inference import Detection, InferenceResult, TaskType
from echosight2.rendering import render_result


def _result(task_type: TaskType, **values: object) -> InferenceResult:
    return InferenceResult(
        source=Path("sample.png"),
        task_type=task_type,
        image_size=(100, 80),
        input_size=(100, 80),
        duration_ms=10.0,
        **values,
    )


def test_renders_detection_box_and_mask() -> None:
    source = Image.new("RGB", (100, 80), "black")
    detection = Detection(0, "defect", 0.9, (20, 15, 70, 60), np.ones((8, 8), dtype=np.float32))

    rendered = render_result(source, _result(TaskType.INSTANCE_SEGMENTATION, detections=(detection,)))

    assert rendered.size == source.size
    assert np.any(np.asarray(rendered) != np.asarray(source))


def test_renders_anomaly_heatmap() -> None:
    source = Image.new("RGB", (100, 80), "black")
    anomaly_map = np.linspace(0, 1, 80 * 100, dtype=np.float32).reshape(80, 100)

    rendered = render_result(source, _result(TaskType.ANOMALY, anomaly_score=1.0, anomaly_map=anomaly_map))

    assert rendered.size == source.size
    assert np.any(np.asarray(rendered) != np.asarray(source))
