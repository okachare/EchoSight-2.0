import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from echosight2.exporting import export_run
from echosight2.frames import LoadedFrame
from echosight2.inference import (
    Detection,
    InferenceResult,
    ModelInfo,
    TaskType,
    TensorInfo,
)
from echosight2.rendering import RenderOptions
from PIL import Image


def _model_info(model_path: Path, task: TaskType = TaskType.DETECTION) -> ModelInfo:
    return ModelInfo(
        path=model_path,
        format="OpenVINO IR",
        task_type=task,
        inputs=(TensorInfo("image", (1, 3, 32, 32), "f32"),),
        outputs=(TensorInfo("boxes", (1, 1, 5), "f32"),),
        labels=("Delamination",),
        model_type="YOLOX",
        device="CPU",
        metadata=(
            ("confidence_threshold", "0.1"),
            ("mean_values", "[123.675, 116.28, 103.53]"),
            ("scale_values", "[58.395, 57.12, 57.375]"),
        ),
        metrics=(("f1-score", 0.91),),
        training_date="2026-09-01",
        artifact_date="2026-09-02",
        collaterals=("config.json", "sample.jpg"),
    )


def test_export_run_writes_reproducible_detection_artifacts(tmp_path: Path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text("model artifact", encoding="utf-8")
    model_path.with_suffix(".bin").write_bytes(b"weights")
    source = tmp_path / "inspection.tiff"
    source.write_bytes(b"source image identity")
    frames = [
        LoadedFrame(source, 1, 2, Image.new("RGB", (64, 48), "white")),
        LoadedFrame(source, 2, 2, Image.new("RGB", (64, 48), "gray")),
    ]
    result = InferenceResult(
        source=source,
        task_type=TaskType.DETECTION,
        image_size=(64, 48),
        input_size=(32, 32),
        duration_ms=12.5,
        detections=(Detection(0, "Delamination", 0.82, (4.0, 5.0, 30.0, 35.0)),),
        output_shapes=(("boxes", (1, 1, 5)),),
    )

    report = export_run(
        destination=tmp_path / "exports",
        frames=frames,
        results={0: result},
        model_info=_model_info(model_path),
        model_parent=tmp_path,
        preprocessing={"brightness": 1.0, "contrast": 1.0},
        render_options=RenderOptions(
            show_labels=False,
            font_size=16,
            annotation_color=(12, 34, 56),
            annotation_thickness=5,
            annotation_opacity=0.75,
            label_opacity=0.6,
        ),
        render_options_by_frame={0: RenderOptions(font_size=20, annotation_color=(90, 80, 70))},
        failures={1: "Unsupported frame data"},
        exported_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
    )

    assert report.directory.name == "EchoSight_Run_20260910_123000"
    assert report.exported_frames == 1
    assert report.failed_frames == 1
    assert [path.name for path in report.annotated_images] == ["00001_inspection_frame_0001.png"]
    assert report.annotated_images[0].is_file()

    manifest = json.loads(report.manifest.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["model"]["artifact_sha256"]
    assert manifest["model"]["weights_sha256"]
    assert manifest["model"]["confidence_threshold"] == 0.1
    assert manifest["processing"]["user_preprocessing"]["brightness"] == 1.0
    assert manifest["processing"]["render_options"]["show_labels"] is False
    assert manifest["processing"]["render_options"]["font_family"] == "Calibri"
    assert manifest["processing"]["render_options"]["font_size"] == 16
    assert manifest["processing"]["render_options"]["annotation_color"] == [12, 34, 56]
    assert manifest["processing"]["render_options"]["annotation_thickness"] == 5
    assert manifest["processing"]["render_options"]["annotation_opacity"] == 0.75
    assert manifest["processing"]["render_options"]["label_opacity"] == 0.6
    assert manifest["processing"]["render_options_by_frame"]["0"]["font_size"] == 20
    assert manifest["processing"]["render_options_by_frame"]["0"]["annotation_color"] == [90, 80, 70]
    assert manifest["summary"]["exported_frames"] == 1
    assert manifest["summary"]["failed_frames"] == 1
    assert manifest["frames"][0]["detections"][0]["label"] == "Delamination"
    assert manifest["frames"][0]["source_sha256"]
    assert manifest["frames"][0]["source_sha256"] == manifest["frames"][1]["source_sha256"]
    assert manifest["frames"][1]["error"] == "Unsupported frame data"

    with report.results_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["result_type"] == "detection"
    assert rows[0]["source_sha256"] == manifest["frames"][0]["source_sha256"]
    assert rows[0]["confidence"] == "0.82"
    assert rows[1]["status"] == "failed"


def test_export_run_serializes_anomaly_result_and_unique_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "anomaly.xml"
    model_path.write_text("anomaly model", encoding="utf-8")
    source = tmp_path / "wafer image.png"
    frame = LoadedFrame(source, 1, 1, Image.new("RGB", (20, 10), "black"))
    result = InferenceResult(
        source=source,
        task_type=TaskType.ANOMALY,
        image_size=(20, 10),
        input_size=(32, 32),
        duration_ms=4.0,
        anomaly_score=0.767,
        anomaly_map=np.full((10, 20), 0.5, dtype=np.float32),
    )
    timestamp = datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc)
    arguments = {
        "destination": tmp_path / "exports",
        "frames": [frame],
        "results": {0: result},
        "model_info": _model_info(model_path, TaskType.ANOMALY),
        "model_parent": tmp_path,
        "preprocessing": {},
        "exported_at": timestamp,
    }

    first = export_run(**arguments)
    second = export_run(**arguments)

    assert second.directory.name == f"{first.directory.name}_2"
    manifest = json.loads(first.manifest.read_text(encoding="utf-8"))
    assert manifest["frames"][0]["anomaly_score"] == 0.767
    assert manifest["summary"]["anomaly_results"] == 1
    assert first.annotated_images[0].name == "00001_wafer_image_frame_0001.png"
