# Changelog

All notable changes to EchoSight are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - 2026-09-15

### Added
- Per-frame or all-frame Image Adjustments for brightness, contrast, sharpness, and continuous denoise strength.
- Per-frame or all-frame Annotation Control for Calibri label size, annotation color, line thickness, annotation transparency, and label transparency.
- Multi-selection Results actions for exporting original frames as model-specific False Hits or Misses retraining sets without overwriting existing files.
- CSAM Helper and EchoSight operator-training guidance covering the complete data flow, current controls, retraining review, confidence behavior, and troubleshooting.

### Changed
- Renamed **Load Model Folder** to **Load Model**.
- Kept Analysis previews unannotated and restricted model overlays to Results.
- Made image and annotation popup changes live previews that persist only when applied to the current frame or all frames.
- Gave the analyzed-image panel all surplus Results-tab width while the Results table and detail panel autosize to their contents.
- Reduced the initial Results table column widths while preserving manual resizing and horizontal scrolling.
- Updated exports and manifests to retain each frame's applied preprocessing and annotation-rendering settings.

### Fixed
- Preserved multi-row Results selections during list refreshes so batch False Hit and Miss exports remain reliable.
- Kept original, unannotated image pixels and TIFF-safe frame names in retraining exports.

### Validated
- All 43 automated tests pass, including desktop workflow, rendering pixels, per-frame exports, and retraining safeguards.
- The Results image panel receives the dominant width in a real 3840-pixel Tk layout check.

## [2.0.0] - 2026-09-14

### Phase 4 - Complete - 2026-09-14

#### Added
- Background Results-tab export with per-frame progress.
- Unique timestamped run directories containing annotated PNG images, `results.csv`, and a versioned `run_manifest.json`.
- Model and weights SHA-256 hashes, package contract, metrics, preprocessing, overlay state, output shapes, summaries, and timing in each run manifest.
- Structured detection, classification, anomaly, empty-result, and failed-frame CSV records.
- Per-frame inference failure capture so remaining batch frames continue processing.
- Real-model acceptance command for detection, anomaly, and exported run verification.
- Standalone Windows x64 directory and ZIP distributions with a SHA-256 build manifest.
- Operator setup, inference, export, troubleshooting, and acceptance guide.

#### Changed
- Adopted a shared-network source deployment for current use. Each user receives a local runtime under `%LOCALAPPDATA%\EchoSight\2.0`; the shared source is updated only by explicit publication from the working directory.

#### Validated
- 24 automated tests pass, including shared-runtime paths, detection and anomaly export contracts, TIFF-safe artifact names, failure records, and non-overwriting run directories.
- Ruff and VS Code diagnostics are clean for the Phase 4 export changes.
- Detection acceptance processed 66/66 TIFF frames with 93 detections and no failures.
- Anomaly acceptance completed inference and verified score, heatmap, and export artifacts.
- Packaged EchoSight 2.0.0 dependency diagnostics passed.

### Phase 3.1 - 2026-09-14

#### Added
- Clean Python 3.9 rebuild under `Deployment/EchoSight2.0`.
- Parent-folder model selection with recursive OpenVINO and ONNX discovery.
- Package collateral inventory and rich model information.
- Model-driven labels, thresholds, resize, intensity, mean/scale, channel, and anomaly calibration handling.
- Separate resizable Analysis and Results tabs.
- Animated activity state, granular per-frame progress, and session Terminal panel.
- Task-specific detection, classification, instance-mask, and anomaly-heatmap results.
- Per-annotation visibility controls and fit-to-pane result rendering.

#### Fixed
- Detection results were incorrectly filtered by a hardcoded 25% threshold; model-provided thresholds are now used.
- Raw pixels were previously passed to models that require normalized input; package metadata now drives preprocessing.
- Structured labels containing spaces are preserved.
- Selected tab styling now clearly identifies the active workspace.

#### Validated
- 20 automated tests pass.
- `NVL_S28C_Detect_09092026_V17` generates a rendered Delamination detection.
- `Test_Run_Instance_Segmentation` follows its embedded anomaly model contract and generates a calibrated heatmap.
- A 66-page TIFF expands and processes as independent in-memory frames.

---

## [1.0.0] - 2026-09-08

### Initial Release ✨

**First public release of EchoSight.**

#### Added
- **Core Features**
  - Standalone GUI application for running Geti OpenVINO model deployments
  - Support for Object Detection, Instance Segmentation, and Anomaly Detection models
  - Real-time model inference with background threading
  - Multi-format image import (PNG, JPG, BMP, WebP, TIFF)
  - Multi-frame TIFF support with automatic frame extraction

- **UI & Interaction**
  - Dark theme with professional high-contrast design
  - Soft-rounded button controls and customizable styling
  - Three-tab interface: Inference, Analysis, Results Review
  - Frame-by-frame navigation with Previous/Next controls
  - Canvas zoom and pan with scroll wheel and click-drag
  - Keyboard shortcuts (arrow keys, scroll navigation)

- **Image Processing**
  - Preprocessing filters: brightness, contrast, sharpness, denoising
  - Per-frame profile storage with applied-value badges
  - Scope control: apply to all frames, current frame, or selected frames
  - Original images preserved; processed copies used for inference
  - Interactive preview with zoom/pan preservation

- **Results Analysis**
  - Confidence threshold filtering (1-100%)
  - Independent annotation visibility toggles:
    - Show/hide detection boxes
    - Show/hide labels
    - Show/hide instance segmentation masks
  - Frame-by-frame confidence score display
  - Overall highest-confidence frame highlighting

- **Model Support**
  - Automatic Geti deployment structure discovery
  - Support for model.xml + model.bin architecture
  - Config.json parameter loading
  - Multi-task model detection (Detection/Segmentation/Anomaly)

- **Distribution & Installation**
  - Pre-built Windows portable package with bundled Python 3.9 runtime
  - Windows installer (.exe) for traditional installation
  - Pip-installable Python package for development
  - PyInstaller configuration for custom builds
  - Smart launcher batch script (auto-detects runtime)

- **Documentation**
  - Comprehensive README.md with features and quick start
  - Installation guide (4 methods: portable, installer, Python, custom build)
  - Detailed user guide with workflows and troubleshooting
  - Development guide for extending and contributing
  - Inline code documentation and docstrings

- **Development**
  - GitHub repository structure
  - .gitignore for Python/build artifacts
  - setup.py for pip installation
  - MIT License for open-source sharing

#### Technical Stack
- **UI Framework**: Tkinter (Python 3.9 standard library)
- **Image Processing**: OpenCV, Pillow, NumPy
- **AI Inference**: OpenVINO 2024.5
- **Build**: PyInstaller
- **Python Version**: 3.9 (exact requirement for OpenVINO 2024.5)

#### Known Limitations
- Windows only (Tkinter cross-platform possible in future)
- CPU inference only (GPU support can be enabled in code)
- Maximum TIFF file size depends on available RAM
- Model must be exported from Intel Geti platform

#### Testing
- ✓ Detection model inference verified
- ✓ Instance Segmentation model inference verified
- ✓ Anomaly Detection model inference verified
- ✓ Image preprocessing pipeline verified
- ✓ Multi-frame TIFF import verified
- ✓ Portable build and execution verified
- ✓ Installer build and execution verified

---

## Post-2.0 Roadmap

- Clean-workstation operator qualification.
- Automated batch and command-line workflows.
- Performance and memory optimization.
- Model comparison and review feedback capture.
- Production rollout, support ownership, and model update policy.

---

## Version History Format

```
## [Semantic Version] - YYYY-MM-DD

### Category (Added/Fixed/Changed/Removed/Deprecated/Security)

- Brief description of change
- Related issues/PRs if applicable
```

---

**Legend**:
- ✨ New feature
- 🐛 Bug fix
- 📝 Documentation
- 🔧 Configuration/Build
- ⚡ Performance improvement
- 🎨 UI/UX improvement

---

For detailed commit history, see [GitHub Commits](https://github.com/okachare/EchoSight/commits/main)

For issues and feature requests, see [GitHub Issues](https://github.com/okachare/EchoSight/issues)
