# EchoSight 2.0

Model-agnostic, offline image inference and inspection for Windows.

EchoSight 2.0 is a clean rebuild of the EchoSight deployment tool. The application is Python based, debug-friendly, and designed to support the OpenVINO and ONNX models used by the GeTi CSAM workflow.

## Current Status

EchoSight 2.0.4 was published on 2026-10-29. It provides parent-folder model discovery, package-collateral inspection, model-driven preprocessing and thresholds, CPU inference, normalized task-specific results, responsive background execution, a resizable Analysis/Results interface, reproducible run export, retraining feedback, and adaptive PowerPoint reporting.

Multi-frame TIFF pages are expanded into independent in-memory frames. Each frame can be selected and inferred individually, or all frames can be processed with Run All. No duplicate TIFF files are written to disk.

Oversized single-frame, contiguous, uncompressed 8-bit grayscale/palette TIFFs use separate display and inference paths. EchoSight retains a preview of at most 16 megapixels, then reopens the unchanged source in the worker and streams overlapping model-sized tiles at full resolution. Tile results remain in source coordinates and are merged before preview rendering, so display scaling does not remove defects from inference. Tiles and loaded model packages run sequentially with tile-level progress, pause, and cancellation to bound RAM and keep the UI responsive. Oversized formats that cannot be decoded by bounded random access, including JPEG, are rejected during loading instead of being inferred from a reduced image.

Tile outputs are merged incrementally and released before the next tile, followed by a visible **Consolidating** phase with spatially indexed duplicate suppression. Result labels use spatial collision indexing, and annotation visibility controls are paged at 100 entries, preventing high-count results from blocking the Tk interface at completion.

Current validation: 93 automated tests pass and 3 unavailable private-fixture tests skip as expected. Packaged diagnostics pass, and the packaged Intel PowerPoint template is byte-identical to its source asset. Phase 4 acceptance processed all 66 TIFF pages with the real `NVL_S28C_Detect_09092026_V17` package, producing 93 detections with no failed frames. `Test_Run_Instance_Segmentation` follows its embedded anomaly contract and also completed inference and export successfully. The real `NCL_S_ChainedModel_Test` package compiled and ran both stages, produced three detector ROIs and 12 ordered stage annotations, and preserved both model hashes in exports.

## Shared Network Setup

EchoSight runs directly from the shared network folder while keeping dependencies and logs private to each Windows user. Python 3.9 is required. The setup script creates `%LOCALAPPDATA%\EchoSight\2.0\.venv`; it does not modify the shared source folder.

This source-based setup runs `pip install` and therefore requires access to a configured Python package source, such as PyPI or an internal mirror. It is not the fully offline option.

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

## Fully Offline Portable Distribution

For a production machine without internet, Git, Python, or package-repository access, use the verified Windows x64 portable package:

- Archive: `artifacts/EchoSight2-2.0.4-windows-x64.zip`
- Size: 64,288,321 bytes
- SHA-256: `f68525d956cf7e6de7020b821195f58afe5e058a344829dd8c0ef07b26fd53e4`
- Use: copy the archive, verify the hash, extract it locally, and run `EchoSight2.exe`

The portable package bundles Python and runtime dependencies. It needs no installation or internet connection. A source deployment can also be installed offline by preparing a Python 3.9 wheelhouse on a connected staging machine and installing with `pip --no-index --find-links`.

## Model Support

- OpenVINO IR: `.xml` with matching `.bin`
- ONNX: `.onnx`
- Explicit Geti `Detection -> Crop -> Classification` chained deployments
- Up to two independent model packages in one session
- Detection
- Classification
- Instance segmentation
- Anomaly classification and segmentation

Select the trained model's parent folder. EchoSight recursively finds the deployable artifact and uses embedded metadata for task type, labels, thresholds, resize behavior, mean/scale preprocessing, channel order, and anomaly calibration. Folders containing multiple distinct models must be narrowed to one trained-model package.

For a supported chained deployment, select the package folder containing `deployment/project.json`. EchoSight verifies the declared task graph, compiles both models, detects regions on the full image, crops each detected region, and applies the classification model to each crop. Results list detection first in teal, followed by each classification in its own class color with an independent confidence and visibility checkbox. The frame table ranks chained frames by upstream detection confidence so saturated classifier scores do not hide detector variation. Run manifests record both model artifacts and each annotation's stage and ROI. Other multi-model graph shapes remain unsupported and are rejected rather than ordered by folder name.

Use **Add Model** after the first package is ready to load a second unrelated package. A declared Detection -> Crop -> Classification graph remains one chained slot; unrelated packages remain independent slots and each analyzes the same complete source frame. EchoSight executes slots sequentially to avoid CPU and memory spikes. Results displays one pane per package with independent overlays and annotation checkboxes. Saving results creates one timestamped evidence folder per package so hashes, thresholds, failures, and hidden annotations remain unambiguous.

The model's embedded confidence threshold remains the default. Experienced reviewers can open the small dropdown at the lower-right of Model Information, explicitly enable the advanced threshold override, and set stage-specific thresholds. Applying or resetting an override clears existing results and requires inference to be rerun; the compiled model is reused. Active overrides are identified in Model Information, the Terminal, and Results under Annotations. Exports record both embedded and effective thresholds. Overrides are session-only and reset when another model is loaded or EchoSight restarts.

## Result Visualization

- Side-by-side **Clear Model** and **Clear Images** actions that remove only the selected resource type and clear dependent results safely
- Native dark title-bar integration on supported Windows versions
- Detection boxes and confidence labels
- Instance mask overlays
- Calibrated anomaly heatmaps and scores
- Global overlay visibility controls
- Per-frame annotation visibility controls
- Per-frame, per-model annotation selection from either the annotation list or Results image, with a high-contrast highlight and **Show selected annotations** filtering for dense results
- Per-frame or all-frame Annotation Control for Calibri label size, annotation color, line thickness, and independent annotation/label transparency
- One-click 16:9 PowerPoint reports using the Intel master design and an editable `PQE SID JFDL FIFA` department textbox, with up to three selected Results frames per TIFF/VID, adaptive non-overlapping layouts, compact editable fields, preserved active annotation styling, and hidden-annotation exclusion
- Multi-select Results actions that save original frames as False Hits or Misses for model retraining
- Mouse-wheel zoom and drag-to-pan in both previews, preserving each pane's viewport while cycling frames; double-click resets fit-to-view
- Results zoom up to 1024x with viewport-only rendering for detailed wafer inspection
- Scrollable Results Details retaining every annotation-specific entry
- Latest-only background annotation redraws for responsive high-count checkbox and style changes
- Per-frame or all-frame brightness, contrast, sharpness, and denoise-strength controls with live Analysis preview
- Sortable frame, task, annotation-count, and highest-confidence results
- User-selected export of the visible Terminal session to a UTF-8 `.log` or `.txt` file

Results are generated from the embedded model task contract. A package whose name says instance segmentation but whose model metadata says anomaly will produce anomaly output.

## Result Export

Use **Save All** or **Save Current** in the Results tab to choose a destination folder. EchoSight creates a unique timestamped run folder with:

- annotated PNG images using the active preprocessing and visibility settings;
- `results.csv` for filtering and downstream analysis;
- `run_manifest.json` with model hashes, package metadata, preprocessing, overlays, per-frame results, timings, summaries, and failures.

Inference continues after an individual frame failure so successful frames and failure details remain available in the same run record.

## PowerPoint Reports

Ctrl-click or Shift-click the required Results rows, with up to three selected frames from each source TIFF, hide any annotations that should not be included, and select **Generate Report**. EchoSight groups the selected frames by source TIFF and creates one consistently formatted 16:9 slide per VID. Each image has a highlighted, editable Image Description placeholder so the author can identify the image without tying the report to a particular inspection domain. The VID is extracted from a `Z` identifier in the filename or from its trailing five digits.

Generated slides inherit the bundled Intel master background, logo, confidentiality label, and slide numbering. The `PQE SID JFDL FIFA` department name is a normal editable textbox on every slide so another group can replace it directly in PowerPoint. Generated report text uses the same minor-theme font and `TEXT_1` theme color as that footer by default. EchoSight keeps report content above the branded footer. If the template asset is missing or invalid, report generation falls back automatically to the established standard white design with the same editable department textbox and typography.

Each slide contains one compact, highlighted, editable text field for each entry, combining its label and value so formatting remains consistent while the author updates Sighting, Visual ID, and Comments directly in PowerPoint. Fields are sized to their initial text rather than stretched across unused slide area. Slides preserve active preprocessing, overlays, and annotation styling while excluding hidden annotations. Dual-model results are placed side by side within the corresponding frame image. Each slide includes an editable AI-verification notice at the lower right that the engineer can delete after validating the results.

The slide layout adapts to the number and aspect ratios of the selected images, including the wider output produced when two models are shown side by side. Sighting and Visual ID remain in a consistent header at the top of every slide. EchoSight evaluates horizontal, vertical, and mixed-grid arrangements with either side or bottom comments, then uses the non-overlapping layout that provides the greatest displayed image area.

The presentation author is read from the current Microsoft Office user profile, with the Windows user name as a fallback. PowerPoint is not required to generate or save the report. After saving, EchoSight asks Windows to open the file in the registered `.pptx` application; if none is installed, the report remains saved and EchoSight displays its location.

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
