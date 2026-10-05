from pathlib import Path

import numpy as np
from PIL import Image

from echosight2 import rendering
from echosight2.inference import Detection, InferenceResult, TaskType
from echosight2.rendering import RenderOptions, render_result


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


def test_chained_stages_use_distinct_default_box_colors() -> None:
    source = Image.new("RGB", (100, 80), "black")
    detection = Detection(0, "Anomaly", 0.91, (20, 20, 80, 70), stage="detection", roi_index=1)
    classification = Detection(1, "Voids", 0.82, (20, 20, 80, 70), stage="classification", roi_index=1)

    rendered = np.asarray(
        render_result(
            source,
            _result(TaskType.DETECTION, detections=(detection, classification)),
            RenderOptions(show_labels=False, annotation_thickness=3),
        )
    )

    assert tuple(rendered[20, 20]) == (37, 185, 167)
    assert tuple(rendered[23, 23]) == (83, 154, 255)


def test_selected_annotation_receives_high_contrast_highlight() -> None:
    source = Image.new("RGB", (100, 80), "black")
    detection = Detection(0, "defect", 0.9, (20, 15, 70, 60))

    rendered = np.asarray(
        render_result(
            source,
            _result(TaskType.DETECTION, detections=(detection,)),
            RenderOptions(show_labels=False),
            selected_annotations={0},
        )
    )

    assert tuple(rendered[15, 20]) == rendering.SELECTION_COLOR


def test_label_layout_avoids_overlapping_stage_labels() -> None:
    occupied = [(20, 20, 90, 35)]

    y = rendering._available_label_y(25, 22, 70, 15, 100, occupied)

    candidate = (25, y, 95, y + 15)
    assert not rendering._rectangles_overlap(candidate, occupied[0])

def test_scales_full_resolution_detection_to_preview() -> None:
    source = Image.new("RGB", (100, 50), "black")
    detection = Detection(0, "Defect", 0.9, (500.0, 250.0, 1000.0, 500.0))
    result = InferenceResult(
        source=Path("wafer.tiff"),
        task_type=TaskType.DETECTION,
        image_size=(1000, 500),
        input_size=(100, 100),
        duration_ms=1.0,
        detections=(detection,),
    )

    rendered = np.asarray(render_result(source, result, RenderOptions(show_labels=False)))

    assert tuple(rendered[25, 50]) != (0, 0, 0)
