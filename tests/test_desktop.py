import threading
from pathlib import Path

from echosight2.desktop import EchoSightApp
from echosight2.inference import Classification, Detection, InferenceResult, TaskType


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