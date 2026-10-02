import queue
import threading
import tkinter as tk
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from echosight2.desktop import MAX_CANVAS_ZOOM, EchoSightApp, FitImageCanvas
from echosight2.exporting import ExportReport
from echosight2.frames import LoadedFrame
from echosight2.inference import (
    ChainedInferenceEngine,
    Classification,
    Detection,
    InferenceEngine,
    InferenceResult,
    TaskType,
)
from echosight2.rendering import RenderOptions
from echosight2.training import TrainingExport


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


def test_chained_result_metrics_use_upstream_detection_confidence() -> None:
    result = InferenceResult(
        source=Path("inspection.tiff"),
        task_type=TaskType.DETECTION,
        image_size=(64, 48),
        input_size=(32, 32),
        duration_ms=4.0,
        detections=(
            Detection(0, "Anomaly", 0.73, (1.0, 2.0, 20.0, 30.0), stage="detection", roi_index=1),
            Detection(1, "Voids", 1.0, (1.0, 2.0, 20.0, 30.0), stage="classification", roi_index=1),
        ),
    )

    annotations, confidence = EchoSightApp._result_metrics(result)

    assert annotations == 2
    assert confidence == 0.73
    assert EchoSightApp._confidence_text(1.0) == ">=99.99%"


def test_threshold_override_applies_per_stage_and_resets_to_model_defaults() -> None:
    app = object.__new__(EchoSightApp)
    detector_info = Mock(confidence_threshold=0.6, task_type=TaskType.DETECTION)
    classifier_info = Mock(confidence_threshold=0.5, task_type=TaskType.CLASSIFICATION)
    detector = Mock(spec=InferenceEngine, confidence_threshold=0.6)
    classifier = Mock(spec=InferenceEngine, confidence_threshold=0.5)
    app.inference_engine = Mock(spec=ChainedInferenceEngine, detector=detector, classifier=classifier)
    app.inference_engines = (app.inference_engine,)
    app.model_infos = (detector_info, classifier_info)
    app.results = {0: _result(0.91)}
    app.secondary_results = {}
    app.inference_failures = {1: "failed"}
    app.secondary_inference_failures = {}
    app.hidden_annotations = {0: {1}}
    app.secondary_hidden_annotations = {}
    app._clear_results_ui = Mock()
    app._refresh_model_detail = Mock()
    app._log = Mock()
    app._close_threshold_popup = Mock()
    app._update_run_state = Mock()

    app._set_effective_thresholds((0.1, 0.25))

    assert detector.confidence_threshold == 0.1
    assert classifier.confidence_threshold == 0.25
    assert app._threshold_override_active()
    assert app.results == {}
    assert app.inference_failures == {}
    assert app.hidden_annotations == {}

    app._set_effective_thresholds((0.6, 0.5))

    assert not app._threshold_override_active()


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


def test_secondary_export_payload_is_independent_from_primary_results() -> None:
    app = object.__new__(EchoSightApp)
    app.current_result_index = 2
    app.secondary_results = {2: _result(0.81), 4: _result(0.67)}
    app.secondary_inference_failures = {5: "secondary failed"}

    current, current_failures = app._secondary_export_payload(current_only=True)
    all_results, all_failures = app._secondary_export_payload(current_only=False)

    assert current == {2: app.secondary_results[2]}
    assert current_failures == {}
    assert all_results == app.secondary_results
    assert all_failures == app.secondary_inference_failures


def test_current_export_does_not_include_other_frames_from_missing_model() -> None:
    app = object.__new__(EchoSightApp)
    app.current_result_index = 2
    app.results = {1: _result(0.91)}
    app.secondary_results = {2: _result(0.81), 4: _result(0.67)}
    app.inference_failures = {3: "primary failed"}
    app.secondary_inference_failures = {}

    primary, primary_failures = app._export_payload(current_only=True)
    secondary, secondary_failures = app._secondary_export_payload(current_only=True)

    assert primary == {}
    assert primary_failures == {}
    assert secondary == {2: app.secondary_results[2]}
    assert secondary_failures == {}


def test_dual_export_worker_keeps_pipeline_metadata_separate(tmp_path: Path) -> None:
    app = object.__new__(EchoSightApp)
    primary_info = Mock()
    secondary_info = Mock()
    app.pipeline_model_infos = ((primary_info,), (secondary_info,))
    app.model_parents = (tmp_path / "primary", tmp_path / "secondary")
    app.inference_engines = (Mock(confidence_threshold=0.4), Mock(confidence_threshold=0.7))
    app.export_events = queue.Queue()
    app._wait_if_paused = Mock(return_value=True)
    frame = LoadedFrame(tmp_path / "frame.png", 1, 1, Image.new("RGB", (8, 6), "black"))
    reports = (
        ExportReport(tmp_path / "run1", tmp_path / "m1.json", tmp_path / "r1.csv", (), 1, 0),
        ExportReport(tmp_path / "run2", tmp_path / "m2.json", tmp_path / "r2.csv", (), 1, 0),
    )

    with patch("echosight2.desktop.export_run", side_effect=reports) as export:
        app._export_worker(
            tmp_path,
            [frame],
            {0: _result(0.91)},
            primary_info,
            (primary_info,),
            app.model_parents[0],
            {},
            {0: app._default_preprocess_settings()},
            {0: RenderOptions()},
            {},
            {},
            (0.4,),
            {0: _result(0.72)},
            {},
            {},
        )

    assert export.call_args_list[0].kwargs["pipeline_model_infos"] == (primary_info,)
    assert export.call_args_list[1].kwargs["pipeline_model_infos"] == (secondary_info,)
    assert app.export_events.get_nowait() == ("complete", reports)


def test_inference_worker_runs_chain_and_independent_model_sequentially(tmp_path: Path) -> None:
    app = object.__new__(EchoSightApp)
    chain = object.__new__(ChainedInferenceEngine)
    independent = Mock(spec=InferenceEngine)
    app.inference_engines = (chain, independent)
    app.model_parents = (Path("chain"), Path("independent"))
    app.inference_events = queue.Queue()
    app._wait_if_paused = Mock(return_value=True)
    frame = LoadedFrame(tmp_path / "frame.png", 1, 1, Image.new("RGB", (8, 6), "black"))
    chain_result = _result(0.91)
    independent_result = _result(0.72)

    with patch("echosight2.desktop.infer_frame", side_effect=(chain_result, independent_result)) as infer:
        app._inference_worker([(0, frame)], {0: app._default_preprocess_settings()})

    events = list(app.inference_events.queue)
    assert [call.args[0] for call in infer.call_args_list] == [chain, independent]
    assert [(payload[3], payload[4]) for event, payload in events if event == "result"] == [
        (0, chain_result),
        (1, independent_result),
    ]
    assert events[-1] == ("complete", (2, 0))


def test_annotation_visibility_is_isolated_per_model() -> None:
    app = object.__new__(EchoSightApp)
    app.current_result_index = 3
    app.hidden_annotations = {}
    app.secondary_hidden_annotations = {}
    app._refresh_result = Mock()

    app._toggle_annotation(1, 2, Mock(get=Mock(return_value=False)))

    assert app.hidden_annotations == {}
    assert app.secondary_hidden_annotations == {3: {2}}
    app._refresh_result.assert_called_once_with()


def test_canvas_zoom_reaches_high_magnification_without_exceeding_bound() -> None:
    canvas = object.__new__(FitImageCanvas)
    canvas.source_image = Image.new("RGB", (100, 100))
    canvas.zoom = MAX_CANVAS_ZOOM / 1.2
    canvas.pan_x = 0.0
    canvas.pan_y = 0.0
    canvas.winfo_width = Mock(return_value=500)
    canvas.winfo_height = Mock(return_value=400)
    canvas._draw = Mock()
    event = Mock(delta=120, x=250, y=200)

    canvas._zoom_image(event)
    canvas._zoom_image(event)

    assert canvas.zoom == MAX_CANVAS_ZOOM
    canvas._draw.assert_called()


def test_result_redraw_coalesces_to_latest_request() -> None:
    app = object.__new__(EchoSightApp)
    app.current_result_index = 0
    app.frames = [Mock(image=Image.new("RGB", (8, 6), "black"))]
    app.results = {0: _result(0.91)}
    app.secondary_results = {}
    app.hidden_annotations = {}
    app.secondary_hidden_annotations = {}
    app.result_render_generation = 0
    app.result_render_running = True
    app.pending_result_render = None
    app._preprocess_settings = Mock(return_value=app._default_preprocess_settings())
    app._render_options = Mock(return_value=RenderOptions())
    app._start_pending_result_render = Mock()

    app._refresh_result()
    first = app.pending_result_render
    app.hidden_annotations = {0: {0}}
    app._refresh_result()

    assert first is not app.pending_result_render
    assert app.pending_result_render[0] == 2
    assert app.pending_result_render[6] == {0}
    app._start_pending_result_render.assert_not_called()


def test_clear_model_retains_images_and_clears_model_results() -> None:
    app = object.__new__(EchoSightApp)
    app.inference_running = False
    app.export_running = False
    app.model_info = Mock()
    app.model_path = Path("model.xml")
    app.model_parent = Path("model")
    app.model_infos = (Mock(),)
    app.inference_engine = Mock()
    app.inference_engines = (app.inference_engine,)
    app.pipeline_model_infos = ((app.model_infos[0],),)
    app.model_parents = (app.model_parent,)
    app.model_detail_base = "details"
    app.frames = [Mock()]
    app.results = {0: _result(0.91)}
    app.secondary_results = {}
    app.inference_failures = {1: "failed"}
    app.secondary_inference_failures = {}
    app.hidden_annotations = {0: {1}}
    app.secondary_hidden_annotations = {}
    app.threshold_override_enabled = Mock()
    app.threshold_variables = [Mock()]
    app.model_name = Mock()
    app.status_text = Mock()
    app._close_threshold_popup = Mock()
    app._set_model_detail = Mock()
    app._clear_results_ui = Mock()
    app._set_secondary_view_visible = Mock()
    app._log = Mock()
    app._update_run_state = Mock()

    app._clear_model()

    assert app.frames
    assert app.model_info is None
    assert app.inference_engine is None
    assert app.results == {}
    assert app.inference_failures == {}
    app.model_name.set.assert_called_once_with("No model loaded")


def test_clear_images_retains_model_and_clears_frame_state() -> None:
    app = object.__new__(EchoSightApp)
    app.inference_running = False
    app.export_running = False
    app.model_info = Mock()
    app.frames = [Mock()]
    app.results = {0: _result(0.91)}
    app.secondary_results = {}
    app.inference_failures = {1: "failed"}
    app.secondary_inference_failures = {}
    app.hidden_annotations = {0: {1}}
    app.secondary_hidden_annotations = {}
    app.preprocess_profiles = {0: (1.0, 1.0, 1.0, 0.0)}
    app.annotation_profiles = {0: RenderOptions()}
    app.annotation_variables = [Mock()]
    app.current_analysis_index = 0
    app.current_result_index = 0
    app.image_list = Mock()
    app.analysis_canvas = Mock()
    app.status_text = Mock()
    app._clear_results_ui = Mock()
    app._load_preprocess_profile = Mock()
    app._log = Mock()
    app._update_run_state = Mock()

    app._clear_images()

    assert app.model_info is not None
    assert app.frames == []
    assert app.results == {}
    assert app.preprocess_profiles == {}
    app.analysis_canvas.set_image.assert_called_once_with(None)


def test_save_terminal_log_writes_visible_session_text(tmp_path: Path) -> None:
    app = object.__new__(EchoSightApp)
    destination = tmp_path / "terminal.log"
    app.terminal = Mock(get=Mock(return_value="12:00:00 | session started"))
    app.status_text = Mock()
    app._log = Mock()

    with patch("echosight2.desktop.filedialog.asksaveasfilename", return_value=str(destination)):
        app._save_terminal_log()

    assert destination.read_text(encoding="utf-8") == "12:00:00 | session started\n"
    app._log.assert_called_once_with(f"Terminal log saved | {destination}")


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


def test_cancelled_inference_resets_progress_bar() -> None:
    app = object.__new__(EchoSightApp)
    app.inference_events = queue.Queue()
    app.inference_events.put(("cancelled", 3))
    app.progress = Mock()
    app._finish_inference = Mock()

    app._poll_inference()

    app.progress.configure.assert_called_once_with(value=0)
    app._finish_inference.assert_called_once_with("Cancelled after 3 frame(s)")


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


def test_dark_title_bar_uses_modern_windows_attribute() -> None:
    app = object.__new__(EchoSightApp)
    app.winfo_id = Mock(return_value=1234)
    set_attribute = Mock(return_value=0)

    with patch("echosight2.desktop.sys.platform", "win32"), patch(
        "echosight2.desktop.ctypes.windll", create=True
    ) as windll:
        windll.user32.GetParent.return_value = 5678
        windll.dwmapi.DwmSetWindowAttribute = set_attribute
        app._enable_dark_title_bar()

    assert set_attribute.call_count == 1
    assert set_attribute.call_args.args[:2] == (5678, 20)


def test_dark_title_bar_falls_back_for_older_windows() -> None:
    app = object.__new__(EchoSightApp)
    app.winfo_id = Mock(return_value=1234)
    set_attribute = Mock(side_effect=(1, 0))

    with patch("echosight2.desktop.sys.platform", "win32"), patch(
        "echosight2.desktop.ctypes.windll", create=True
    ) as windll:
        windll.user32.GetParent.return_value = 5678
        windll.dwmapi.DwmSetWindowAttribute = set_attribute
        app._enable_dark_title_bar()

    assert [call.args[1] for call in set_attribute.call_args_list] == [20, 19]


def test_dark_title_bar_is_skipped_outside_windows() -> None:
    app = object.__new__(EchoSightApp)
    app.winfo_id = Mock()

    with patch("echosight2.desktop.sys.platform", "linux"):
        app._enable_dark_title_bar()

    app.winfo_id.assert_not_called()


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
    app.secondary_results = {}
    app.frames = [
        LoadedFrame(tmp_path / f"frame_{index}.png", 1, 1, Image.new("RGB", (8, 6), "black"))
        for index in range(4)
    ]
    app.model_parent = Path("C:/models/Detection Model V17")
    app.model_parents = (app.model_parent,)
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