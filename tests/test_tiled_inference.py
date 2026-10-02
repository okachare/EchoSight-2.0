from pathlib import Path
from unittest.mock import Mock

from PIL import Image

import echosight2.frames as frame_module
from echosight2.frames import load_frames
from echosight2.inference import Detection, InferenceResult, TaskType, infer_frame
from echosight2.inference.tiled import _non_maximum_suppression


def test_tiled_inference_remaps_and_deduplicates_detections(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "wafer.tiff"
    Image.new("L", (10, 6), 120).save(path, compression="raw")
    monkeypatch.setattr(frame_module, "MAX_WORKING_PIXELS", 20)
    frame = load_frames(path)[0]
    info = Mock(input_size=(4, 4))
    engine = Mock(model_info=info)
    engine.infer.side_effect = lambda image, source: InferenceResult(
        source=source,
        task_type=TaskType.DETECTION,
        image_size=image.size,
        input_size=image.size,
        duration_ms=1.0,
        detections=(Detection(0, "Defect", 0.9, (1.0, 1.0, 3.0, 3.0)),),
    )
    updates = []
    finalized = []

    result = infer_frame(
        engine,
        frame,
        progress=lambda current, total: updates.append((current, total)),
        finalizing=lambda: finalized.append(True),
        overlap=0.0,
    )

    assert engine.infer.call_count == 6
    assert result.image_size == (10, 6)
    assert result.detections[0].box == (1.0, 1.0, 3.0, 3.0)
    assert any(item.box == (7.0, 3.0, 9.0, 5.0) for item in result.detections)
    assert updates[-1] == (6, 6)
    assert finalized == [True]


def test_tiled_inference_honors_cancellation_before_next_tile(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "wafer.tiff"
    Image.new("L", (8, 4), 120).save(path, compression="raw")
    monkeypatch.setattr(frame_module, "MAX_WORKING_PIXELS", 16)
    frame = load_frames(path)[0]
    engine = Mock(model_info=Mock(input_size=(4, 4)))
    engine.infer.return_value = InferenceResult(None, TaskType.DETECTION, (4, 4), (4, 4), 1.0)
    checks = iter((True, False))

    try:
        infer_frame(engine, frame, should_continue=lambda: next(checks), overlap=0.0)
    except InterruptedError:
        pass
    else:
        raise AssertionError("Expected tiled inference cancellation")

    assert engine.infer.call_count == 1


def test_spatial_nms_preserves_labels_and_stages_while_removing_duplicates() -> None:
    detections = [
        Detection(0, "Defect", 0.9, (10.0, 10.0, 30.0, 30.0), stage="detection"),
        Detection(0, "Defect", 0.8, (11.0, 11.0, 31.0, 31.0), stage="detection"),
        Detection(1, "Other", 0.7, (11.0, 11.0, 31.0, 31.0), stage="detection"),
        Detection(0, "Defect", 0.6, (11.0, 11.0, 31.0, 31.0), stage="classification"),
    ]

    selected = _non_maximum_suppression(detections)

    assert selected == [detections[0], detections[2], detections[3]]