# EchoSight 2.0

Model-agnostic, offline image inference and inspection for Windows.

EchoSight 2.0 is a clean rebuild of the EchoSight deployment tool. The application is Python based, debug-friendly, and designed to support the OpenVINO and ONNX models used by the GeTi CSAM workflow.

## Current Status

Phase 4 is complete as of 2026-09-14. EchoSight 2.0 provides parent-folder model discovery, package-collateral inspection, model-driven preprocessing and thresholds, CPU inference, normalized task-specific results, responsive background execution, a resizable Analysis/Results interface, reproducible run export, and real-model acceptance automation.

Multi-frame TIFF pages are expanded into independent in-memory frames. Each frame can be selected and inferred individually, or all frames can be processed with Run All. No duplicate TIFF files are written to disk.

Current validation: all 28 automated tests pass in the full development workspace. Phase 4 acceptance processed all 66 TIFF pages with the real `NVL_S28C_Detect_09092026_V17` package, producing 93 detections with no failed frames. `Test_Run_Instance_Segmentation` follows its embedded anomaly contract and also completed inference and export successfully.

## Shared Network Setup

EchoSight runs directly from the shared network folder while keeping dependencies and logs private to each Windows user. Python 3.9 is required. The setup script creates `%LOCALAPPDATA%\EchoSight\2.0\.venv`; it does not modify the shared source folder.

From the shared EchoSight directory, each user runs this once:

```powershell
.\SETUP.ps1
```

After setup, launch from the same shared directory:

```text
Launch_EchoSight.bat
```

For diagnostics, run:

```powershell
.\DIAGNOSE.ps1
```

Application and setup logs are stored under `%LOCALAPPDATA%\EchoSight\2.0\logs`. See [OPERATOR_GUIDE.md](OPERATOR_GUIDE.md) for operating instructions.

## Model Support

- OpenVINO IR: `.xml` with matching `.bin`
- ONNX: `.onnx`
- Detection
- Classification
- Instance segmentation
- Anomaly classification and segmentation

Select the trained model's parent folder. EchoSight recursively finds the deployable artifact and uses embedded metadata for task type, labels, thresholds, resize behavior, mean/scale preprocessing, channel order, and anomaly calibration. Folders containing multiple distinct models must be narrowed to one trained-model package.

## Result Visualization

- Detection boxes and confidence labels
- Instance mask overlays
- Calibrated anomaly heatmaps and scores
- Global overlay visibility controls
- Per-frame annotation visibility controls
- Mouse-wheel zoom, drag-to-pan, and double-click fit reset in both previews
- Live brightness, contrast, sharpness, and denoising controls in the Analysis preview
- Sortable frame, task, annotation-count, and highest-confidence results

Results are generated from the embedded model task contract. A package whose name says instance segmentation but whose model metadata says anomaly will produce anomaly output.

## Result Export

Use **Save All** or **Save Current** in the Results tab to choose a destination folder. EchoSight creates a unique timestamped run folder with:

- annotated PNG images using the active preprocessing and visibility settings;
- `results.csv` for filtering and downstream analysis;
- `run_manifest.json` with model hashes, package metadata, preprocessing, overlays, per-frame results, timings, summaries, and failures.

Inference continues after an individual frame failure so successful frames and failure details remain available in the same run record.

## Principles

- Keep inference, rendering, and UI independent.
- Normalize all model outputs into one result schema.
- Log enough context to diagnose failures without opening the source.
- Keep runtime dependencies minimal and development dependencies separate.
- Validate against the actual models stored in this workspace.

## Directory Layout

```text
EchoSight2.0/
  src/echosight2/       Python application package
  SETUP.ps1              Per-user environment setup
  Launch_EchoSight.bat   Windows launcher
  DIAGNOSE.ps1           Per-user runtime diagnostics
  requirements.txt       Runtime dependencies
```
