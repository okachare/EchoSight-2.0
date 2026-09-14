"""Run Phase 4 acceptance against the real workspace models and source data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from echosight2.exporting import ExportReport, export_run
from echosight2.frames import LoadedFrame, load_frames
from echosight2.inference import InferenceEngine, ModelLoader, TaskType


def _workspace_paths() -> dict[str, Path]:
    workspace = Path(__file__).resolve().parents[3]
    return {
        "workspace": workspace,
        "detection_package": workspace / "Deployment" / "NVL_S28C_Detect_09092026_V17",
        "anomaly_package": workspace / "Deployment" / "Test_Run_Instance_Segmentation",
        "detection_model": workspace
        / "Deployment"
        / "NVL_S28C_Detect_09092026_V17"
        / "deployment"
        / "Detection"
        / "model"
        / "model.xml",
        "anomaly_model": workspace
        / "Deployment"
        / "Test_Run_Instance_Segmentation"
        / "deployment"
        / "Anomaly classification"
        / "model"
        / "model.xml",
        "sample_image": workspace / "Debug" / "Evaluation" / "Test Data" / "00790_058.png",
        "tiff": workspace / "TiffSplitter" / "samples" / "ARL_Test.TIFF",
    }


def _require_files(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if name != "workspace" and not path.exists()]
    if missing:
        raise FileNotFoundError("Required Phase 4 fixtures are unavailable:\n" + "\n".join(missing))


def _compile_engine(model_path: Path, package_path: Path) -> InferenceEngine:
    info, compiled = ModelLoader().compile(model_path, package_path)
    return InferenceEngine(info, compiled)


def _run_detection(paths: dict[str, Path], destination: Path) -> tuple[ExportReport, dict[str, Any]]:
    frames = load_frames(paths["tiff"])
    if len(frames) != 66:
        raise AssertionError(f"Expected 66 TIFF frames, found {len(frames)}")

    engine = _compile_engine(paths["detection_model"], paths["detection_package"])
    if engine.model_info.task_type is not TaskType.DETECTION:
        raise AssertionError(f"Expected detection task, found {engine.model_info.task_type.value}")

    started = perf_counter()
    results = {}
    failures = {}
    for index, frame in enumerate(frames):
        print(f"Detection inference {index + 1}/{len(frames)}: {frame.display_name}", flush=True)
        try:
            results[index] = engine.infer(frame.image, frame.source)
        except Exception as error:  # noqa: BLE001
            failures[index] = f"{type(error).__name__}: {error}"

    report = export_run(
        destination,
        frames,
        results,
        engine.model_info,
        paths["detection_package"],
        preprocessing={"brightness": 1.0, "contrast": 1.0, "sharpness": 1.0, "denoise": False},
        failures=failures,
        progress=lambda position, total, name: print(f"Detection export {position}/{total}: {name}", flush=True),
    )
    elapsed = perf_counter() - started
    _validate_export(report, len(frames))
    return report, {
        "task": engine.model_info.task_type.value,
        "source_frames": len(frames),
        "successful_frames": len(results),
        "failed_frames": len(failures),
        "detections": sum(len(result.detections) for result in results.values()),
        "elapsed_seconds": round(elapsed, 3),
    }


def _run_anomaly(paths: dict[str, Path], destination: Path) -> tuple[ExportReport, dict[str, Any]]:
    with Image.open(paths["sample_image"]) as image:
        frame = LoadedFrame(paths["sample_image"], 1, 1, image.convert("RGB").copy())

    engine = _compile_engine(paths["anomaly_model"], paths["anomaly_package"])
    if engine.model_info.task_type is not TaskType.ANOMALY:
        raise AssertionError(f"Expected anomaly task, found {engine.model_info.task_type.value}")

    started = perf_counter()
    result = engine.infer(frame.image, frame.source)
    report = export_run(
        destination,
        [frame],
        {0: result},
        engine.model_info,
        paths["anomaly_package"],
        preprocessing={"brightness": 1.0, "contrast": 1.0, "sharpness": 1.0, "denoise": False},
    )
    elapsed = perf_counter() - started
    _validate_export(report, 1)
    if result.anomaly_map is None or result.anomaly_score is None:
        raise AssertionError("Anomaly acceptance result is missing its score or heatmap")
    return report, {
        "task": engine.model_info.task_type.value,
        "successful_frames": 1,
        "failed_frames": 0,
        "anomaly_score": result.anomaly_score,
        "elapsed_seconds": round(elapsed, 3),
    }


def _validate_export(report: ExportReport, expected_frames: int) -> None:
    if not report.manifest.is_file() or not report.results_csv.is_file():
        raise AssertionError(f"Export records are missing from {report.directory}")
    if report.exported_frames + report.failed_frames != expected_frames:
        raise AssertionError("Exported and failed frame counts do not match the acceptance input")
    if len(report.annotated_images) != report.exported_frames:
        raise AssertionError("An annotated image was not written for every successful result")

    manifest = json.loads(report.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise AssertionError("Unexpected export manifest schema")
    if not manifest["model"].get("artifact_sha256"):
        raise AssertionError("Model artifact hash is missing")
    if any(not frame.get("source_sha256") for frame in manifest["frames"]):
        raise AssertionError("One or more source hashes are missing")


def run(destination: Path) -> Path:
    paths = _workspace_paths()
    _require_files(paths)
    destination.mkdir(parents=True, exist_ok=True)

    detection_report, detection_summary = _run_detection(paths, destination)
    anomaly_report, anomaly_summary = _run_anomaly(paths, destination)
    acceptance_report = destination / "phase4_acceptance_report.json"
    acceptance_report.write_text(
        json.dumps(
            {
                "status": "passed",
                "completed_at": datetime.now().astimezone().isoformat(),
                "detection": {**detection_summary, "export": str(detection_report.directory)},
                "anomaly": {**anomaly_summary, "export": str(anomaly_report.directory)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return acceptance_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase4-acceptance",
        help="Directory for generated acceptance artifacts",
    )
    arguments = parser.parse_args()
    try:
        report = run(arguments.output.resolve())
    except Exception as error:  # noqa: BLE001
        print(f"PHASE 4 ACCEPTANCE FAILED: {type(error).__name__}: {error}")
        return 1
    print(f"PHASE 4 ACCEPTANCE PASSED: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
