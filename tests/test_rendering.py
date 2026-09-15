from pathlib import Path

import numpy as np
from echosight2.inference import Detection, InferenceResult, TaskType
from echosight2.rendering import RenderOptions, render_result
from PIL import Image


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


def test_render_options_preserve_current_defaults_with_calibri_labels() -> None:
    options = RenderOptions()

    assert options.font_family == "Calibri"
    assert options.font_size == 10
    assert options.annotation_color is None
    assert options.annotation_thickness == 3
    assert options.annotation_opacity == 1.0
    assert options.label_opacity == 1.0


def test_custom_annotation_color_thickness_and_opacity_are_rendered() -> None:
    source = Image.new("RGB", (100, 80), "black")
    detection = Detection(0, "defect", 0.9, (20, 15, 70, 60))
    options = RenderOptions(
        show_labels=False,
        annotation_color=(255, 0, 0),
        annotation_thickness=5,
        annotation_opacity=0.5,
    )

    rendered = np.asarray(render_result(source, _result(TaskType.DETECTION, detections=(detection,)), options))

    assert 120 <= rendered[15, 20, 0] <= 135
    assert rendered[15, 23, 0] == rendered[15, 20, 0]
    assert rendered[15, 20, 1] == 0


def test_zero_label_opacity_hides_label_without_hiding_detection() -> None:
    source = Image.new("RGB", (100, 80), "black")
    detection = Detection(0, "defect", 0.9, (20, 30, 70, 60))
    options = RenderOptions(show_boxes=False, label_opacity=0.0)

    rendered = render_result(source, _result(TaskType.DETECTION, detections=(detection,)), options)

    assert np.array_equal(np.asarray(rendered), np.asarray(source))
