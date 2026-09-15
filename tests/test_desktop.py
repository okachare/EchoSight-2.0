import threading
import tkinter as tk
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from echosight2.desktop import EchoSightApp
from echosight2.frames import LoadedFrame
from echosight2.inference import Classification, Detection, InferenceResult, TaskType
from echosight2.rendering import RenderOptions
from echosight2.training import TrainingExport
from PIL import Image


def _result(confidence: float) -> InferenceResult:
    return InferenceResult(
        source=Path("inspection.tiff"),
        task_type=TaskType.DETECTION,
        image_size=(64, 48),
        input_size=(32, 32),
        duration_ms=4.0,
        detections=(Detection(0, "Defect", confidence, (1.0, 2.0, 20.0, 30.0)),),
        classifications=(Classification(0, "Review", confidence - 0.1),),
    )


def test_result_metrics_report_annotation_count_and_highest_confidence() -> None:
    annotations, confidence = EchoSightApp._result_metrics(_result(0.91))

    assert annotations == 2
    assert confidence == 0.91


def test_current_export_payload_excludes_other_results_and_failures() -> None:
    app = object.__new__(EchoSightApp)
    app.current_result_index = 2
    app.results = {1: _result(0.72), 2: _result(0.91)}
    app.inference_failures = {3: "failed"}

    results, failures = app._export_payload(current_only=True)

    assert results == {2: app.results[2]}
    assert failures == {}


def test_all_export_payload_preserves_results_and_failures() -> None:
    app = object.__new__(EchoSightApp)
    app.current_result_index = 2
    app.results = {1: _result(0.72), 2: _result(0.91)}
    app.inference_failures = {3: "failed"}

    results, failures = app._export_payload(current_only=False)

    assert results == app.results
    assert failures == app.inference_failures


def test_pause_gate_blocks_until_resume() -> None:
    app = object.__new__(EchoSightApp)
    app.resume_activity = threading.Event()
    app.cancel_inference = threading.Event()
    completed = threading.Event()

    worker = threading.Thread(target=lambda: (app._wait_if_paused(), completed.set()))
    worker.start()

    assert not completed.wait(0.2)
    app.resume_activity.set()
    assert completed.wait(1.0)
    worker.join()


def test_maximize_window_uses_zoomed_state_when_supported() -> None:
    app = object.__new__(EchoSightApp)
    app.state = Mock()
    app.attributes = Mock()

    app._maximize_window()

    app.state.assert_called_once_with("zoomed")
    app.attributes.assert_not_called()


def test_maximize_window_uses_fullscreen_fallback() -> None:
    app = object.__new__(EchoSightApp)
    app.state = Mock(side_effect=tk.TclError)
    app.attributes = Mock()

    app._maximize_window()

    app.attributes.assert_called_once_with("-fullscreen", True)


def test_annotation_draft_includes_control_values() -> None:
    app = object.__new__(EchoSightApp)
    app.show_boxes = Mock(get=Mock(return_value=True))
    app.show_labels = Mock(get=Mock(return_value=True))
    app.show_masks = Mock(get=Mock(return_value=False))
    app.show_heatmap = Mock(get=Mock(return_value=False))
    app.annotation_font_size = Mock(get=Mock(return_value=16.2))
    app.annotation_thickness = Mock(get=Mock(return_value=4.6))
    app.annotation_transparency = Mock(get=Mock(return_value=0.25))
    app.label_transparency = Mock(get=Mock(return_value=0.4))
    app.annotation_color = (12, 34, 56)

    options = app._annotation_draft()

    assert options.font_family == "Calibri"
    assert options.font_size == 16
    assert options.annotation_color == (12, 34, 56)
    assert options.annotation_thickness == 5
    assert options.annotation_opacity == 0.75
    assert options.label_opacity == 0.6


def test_render_options_use_applied_frame_profile() -> None:
    app = object.__new__(EchoSightApp)
    app.show_boxes = Mock(get=Mock(return_value=True))
    app.show_labels = Mock(get=Mock(return_value=False))
    app.show_masks = Mock(get=Mock(return_value=True))
    app.show_heatmap = Mock(get=Mock(return_value=True))
    app.annotation_popup = None
    app.current_result_index = 1
    app.annotation_profiles = {1: RenderOptions(font_size=18, annotation_color=(12, 34, 56), annotation_thickness=5)}

    options = app._render_options(1)

    assert options.font_size == 18
    assert options.annotation_color == (12, 34, 56)
    assert options.annotation_thickness == 5
    assert options.show_labels is False


def test_apply_preprocessing_to_all_frames_stores_same_profile() -> None:
    app = object.__new__(EchoSightApp)
    app.frames = [Mock(), Mock(), Mock()]
    app.preprocess_profiles = {}
    app._preprocess_draft = Mock(return_value=(1.1, 0.9, 1.4, 2.5))
    app._refresh_analysis = Mock()
    app._refresh_result = Mock()

    app._apply_preprocessing_all()

    assert app.preprocess_profiles == {index: (1.1, 0.9, 1.4, 2.5) for index in range(3)}


def test_apply_preprocessing_to_current_frame_only() -> None:
    app = object.__new__(EchoSightApp)
    app.current_analysis_index = 1
    app.preprocess_profiles = {0: (1.0, 1.0, 1.0, 0.0)}
    app._preprocess_draft = Mock(return_value=(1.2, 0.8, 1.3, 2.0))
    app._refresh_analysis = Mock()
    app._refresh_result = Mock()

    app._apply_preprocessing_current()

    assert app.preprocess_profiles == {0: (1.0, 1.0, 1.0, 0.0), 1: (1.2, 0.8, 1.3, 2.0)}


def test_apply_annotation_to_all_frames_stores_same_profile() -> None:
    app = object.__new__(EchoSightApp)
    app.frames = [Mock(), Mock()]
    app.annotation_profiles = {}
    options = RenderOptions(font_size=14, annotation_thickness=4)
    app._annotation_draft = Mock(return_value=options)
    app._refresh_analysis = Mock()
    app._refresh_result = Mock()

    app._apply_annotation_all()

    assert app.annotation_profiles == {0: options, 1: options}


def test_denoise_strength_controls_blur_amount() -> None:
    pixels = np.zeros((21, 21, 3), dtype=np.uint8)
    pixels[10, 10] = 255
    source = Image.fromarray(pixels)

    unchanged = EchoSightApp._process_image(source, (1.0, 1.0, 1.0, 0.0))
    denoised = EchoSightApp._process_image(source, (1.0, 1.0, 1.0, 3.0))

    assert np.array_equal(np.asarray(unchanged), pixels)
    assert np.asarray(denoised)[10, 10, 0] < 255


def test_analysis_preview_never_renders_result_annotations() -> None:
    app = object.__new__(EchoSightApp)
    source = Image.new("RGB", (64, 48), "black")
    app.current_analysis_index = 0
    app.frames = [Mock(image=source)]
    app.results = {0: _result(0.91)}
    app.analysis_canvas = Mock()
    app._processed_image = Mock(return_value=source)

    with patch("echosight2.desktop.render_result") as render:
        app._refresh_analysis()

    render.assert_not_called()
    app.analysis_canvas.set_image.assert_called_once_with(source)


def test_mark_for_training_exports_all_selected_false_hit_frames(tmp_path: Path) -> None:
    app = object.__new__(EchoSightApp)
    app.result_list = Mock(selection=Mock(return_value=("1", "3")))
    app.results = {1: _result(0.72), 3: _result(0.91)}
    app.frames = [
        LoadedFrame(tmp_path / f"frame_{index}.png", 1, 1, Image.new("RGB", (8, 6), "black"))
        for index in range(4)
    ]
    app.model_parent = Path("C:/models/Detection Model V17")
    app.model_info = None
    app.status_text = Mock()
    app._log = Mock()
    report = TrainingExport(tmp_path / "Mark_for_Training_False_Hits_(Detection_Model_V17)", (tmp_path / "one.png", tmp_path / "two.png"))

    with (
        patch("echosight2.desktop.filedialog.askdirectory", return_value=str(tmp_path)),
        patch("echosight2.desktop.export_training_frames", return_value=report) as export,
        patch("echosight2.desktop.messagebox.showinfo"),
    ):
        app._mark_for_training("False_Hits")

    export.assert_called_once_with(
        tmp_path,
        [(1, app.frames[1]), (3, app.frames[3])],
        "False_Hits",
        "Detection Model V17",
    )
    app.status_text.set.assert_called_once_with("Marked 2 frame(s) for training: False Hits")