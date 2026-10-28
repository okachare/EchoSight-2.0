# EchoSight 2.0 Operator Guide

## Start EchoSight

### First use

1. Open the shared EchoSight network folder.
2. Right-click `SETUP.ps1` and run it with PowerShell.
3. Wait for setup and diagnostics to complete.
4. Double-click `Launch_EchoSight.bat`.

Setup is required once for each Windows user profile and requires Python 3.9 plus access to the configured Python package source. Dependencies are installed under `%LOCALAPPDATA%\EchoSight\2.0`; the shared application files are not modified.

The source setup is not fully offline because `pip` must resolve the runtime packages. On an isolated work system, use the verified `EchoSight2-2.0.1-windows-x64.zip` portable package instead: verify SHA-256 `77c76b794130c564cac0661f2f76961429e17afc33c758e98b802575b415a37f`, extract it locally, and run `EchoSight2.exe`. The portable package requires neither Python nor internet access.

### Later use

Double-click `Launch_EchoSight.bat` in the shared folder. Run `SETUP.ps1` again only when setup reports a problem or the published requirements change.

Release validation: 90 workspace tests pass with 3 expected private-fixture skips. The shared 2.0.1 deployment contains 23 manifest-tracked files with zero hash mismatches, and packaged dependency diagnostics pass.

## Inspect Images

1. Select **Load Model** and choose the trained model's parent folder.
2. Optionally select **Add Model** to load a second independent package.
3. Confirm the tasks, labels, thresholds, and preprocessing in **Model Information**.
4. Select **Open Images** and choose one or more supported images or TIFF files.
5. Use **Run All** for the complete list, **Run Selected** for highlighted frames, or **Run Current** for the active frame.
6. Review generated output in the **Results** tab.
7. Use each model pane's annotation checkboxes to control exported visibility.

Use **Clear Model** to unload the current model while retaining loaded images. Use **Clear Images** to remove loaded images and their results while retaining the compiled model. Both actions clear results that can no longer be valid.

Supported inputs are PNG, JPEG, BMP, WebP, TIFF, and multi-frame TIFF.

For a Geti chained deployment, select the package folder above `deployment/project.json`. EchoSight currently supports the explicit **Detection -> Crop -> Classification** graph. Each detected box is cropped and passed to the classifier. Results shows the teal detection annotation first, followed by separate class-colored classification annotations with independent confidence values and visibility checkboxes. The frame list's highest-confidence column uses upstream detection confidence for chained results. Both stages and thresholds appear in Model Information, and both model artifacts are recorded in the run manifest.

**Add Model** loads a second package as an independent pipeline. Each package runs against the same full image; EchoSight does not feed one unrelated package's output into the other. A declared chained package still performs its Detection -> Crop -> Classification flow internally. With two package slots loaded, Results shows side-by-side panes and separate annotation controls. Packages run sequentially to keep resource use predictable.

### Advanced threshold review

The model's embedded threshold is used during normal operation. To investigate whether subtle defects produce weak candidates, select the small dropdown at the lower-right of **Model Information**, enable **Override model confidence threshold**, adjust the stage-specific slider, and select **Apply override**. Read and confirm the warning before proceeding. Existing results are cleared and inference must be rerun. An active override is shown in Model Information, the Terminal, and Results under **Annotations**. Use **Reset to model defaults** when the review is complete. Overrides are session-only and exports record both embedded and effective thresholds.

Lower thresholds can generate many false or duplicate detections. A weak candidate suggests the model noticed a feature with low confidence; it does not establish that the defect is real. If no useful candidate appears at a low threshold, the model may have missed or internally suppressed it.

For oversized single-frame, contiguous, uncompressed 8-bit grayscale/palette TIFFs, EchoSight creates a display preview of at most 16 megapixels but performs inference on overlapping full-resolution source tiles. The source remains unchanged. Coordinates are retained at source resolution and scaled only for display. Tile-level progress, pause, and cancellation remain active throughout the run. Oversized JPEGs and unsupported TIFF layouts are rejected during loading; convert them to the supported lossless TIFF layout or split them into smaller lossless images. EchoSight never silently substitutes the reduced preview for full-resolution inference.

During a multi-frame run, select **Pause** to stop before the next frame. The control changes to **Resume** and continues from the same position when selected again. **Cancel** stops the remaining frames while retaining completed results.

After the final full-resolution tile, the status changes to **Consolidating** while EchoSight removes duplicate overlap detections and prepares the bounded Results preview. Large annotation sets are shown 100 at a time under **Annotations**; use the arrow controls to reach every annotation and toggle its visibility independently.

Select an annotation's text in **Annotations**, or select its box in the Results image, to add or remove it from the current frame's selection. Selected annotations use a high-contrast gold highlight in both places. Under **Overlays**, choose **Show all annotations** for the complete enabled set or **Show selected annotations** to isolate only the selected items. Selection is independent for every frame and model pane, so a small review set can be retained while navigating dense results.

Use the mouse wheel over either image preview to zoom and drag the image to pan. Results supports up to 1024x fit-relative zoom for close wafer inspection while rendering only the visible viewport. Double-click the preview to restore fit-to-view. In Analysis, select the settings icon at the lower-right of the preview to adjust brightness, contrast, sharpness, and exact denoise strength. Changes preview immediately. Select **Apply to current frame** or **Apply to all frames** to choose the processing scope used by inference and exports.

In Results, select the color-wheel icon at the lower-right of the preview to open **Annotation Control**. Adjust label font size, annotation color, line thickness, annotation transparency, or label transparency. Changes preview live in Results only; Analysis always shows the unannotated image with its active image adjustments. Select **Apply to current frame** or **Apply to all frames** to choose the export scope. Select **Reset** to restore the standard per-class palette and default styling.

Both popups close when you select their icon again or click elsewhere in EchoSight. Unapplied preview changes are discarded when the popup closes.

The Results table can be sorted by frame, task type, annotation count, or highest confidence. Select a column header to alternate ascending and descending order.

**Result Details** retains every annotation-specific line in a fixed-height scrollable area. Use its scrollbar or mouse wheel to review entries beyond the visible field of view. Annotation checkbox changes and style adjustments render in the background; when several changes are made quickly, EchoSight applies the newest state instead of processing obsolete redraws.

## Generate a PowerPoint Report

1. In Results, Ctrl-click or Shift-click the required frame rows, selecting no more than three frames from each source TIFF.
2. Open each selected frame and clear the checkbox for every annotation that should be hidden.
3. Confirm the active overlays and annotation styling show the intended evidence.
4. Select **Generate Report**.
5. Choose the `.pptx` destination and review the presentation when it opens in PowerPoint.
6. Replace the highlighted Sighting and Comments placeholders directly in each slide.

EchoSight creates one 16:9 slide for each source TIFF/VID and places up to three selected frames on that slide. Replace the highlighted Image Description placeholder beneath each image with the appropriate description. The VID is read from a `Z` identifier in the TIFF filename; when no explicit identifier is present, the trailing five digits become `Z#####`. Hidden annotations are omitted, while visible annotations retain their active boxes, masks, labels, confidence values, colors, opacity, and line styling. If two model packages are loaded, their visible outputs are shown side by side for that frame.

Each slide uses one compact, highlighted, editable field for each entry: Sighting, Visual ID, Comments, and each Image Description. The label and value remain together when the field is edited or moved, and the initial field does not stretch across unused slide area. Delete the AI-verification notice only after an engineer confirms the generated results. The presentation author comes from the current Microsoft Office user profile, or the Windows user name when an Office profile is unavailable.

PowerPoint is not required to generate the `.pptx`. If no compatible presentation application is registered, EchoSight saves the report and displays its path instead of opening it automatically.

The report keeps Sighting and Visual ID together at the top of every slide. It automatically chooses a horizontal, vertical, or mixed image arrangement according to the number and shape of the images, placing Comments at the side or bottom to maximize image area without overlapping descriptions or report fields.

## Mark Frames for Retraining

1. In Results, select one or more frames. Use Ctrl or Shift to select multiple rows.
2. Select **Mark for Training (False Hits)** for incorrect model detections, or **Mark for Training (Misses)** for defects the model failed to detect.
3. Choose the parent destination folder.

EchoSight saves lossless, original unannotated PNG frames under `Mark_for_Training_False_Hits_(Model_folder_name)` or `Mark_for_Training_Misses_(Model_folder_name)`. Existing files are not overwritten. These folders can be reviewed and added to the model's next labeled training dataset.

## Export Run Evidence

1. Complete inference for the required frames.
2. In **Results**, select **Save All** for every completed result or **Save Current** for the active result.
3. Choose an existing destination folder.
4. Wait for the completion message before closing EchoSight.

Each timestamped run folder contains:

- `annotated/`: one PNG for every successful frame;
- `results.csv`: flat records for detections, classifications, anomaly scores, empty results, and failures;
- `run_manifest.json`: application version, model and source hashes, package metadata, preprocessing, overlays, frame results, timings, and failures.

The export reflects the active preprocessing, overlay switches, and per-annotation visibility selections. Existing run folders are never overwritten.

When two independent package slots are loaded, EchoSight creates one run folder per package. Each folder contains only that package's results, failures, model hashes, effective thresholds, and visibility choices. **Save Current** exports only the selected frame from each package that produced a result for that frame.

## Interpret Failures

A failure on one frame does not stop **Run All**. Successful frames remain available and the failed frame is recorded in the exported CSV and manifest.

Review the on-screen **Terminal** and the dated file under `%LOCALAPPDATA%\EchoSight\2.0\logs` for details. Preserve the exported run folder when escalating a model or image issue.

Select **Save Terminal Log** at the lower-right of the Terminal panel to save the currently visible session text as a UTF-8 `.log` or `.txt` file in a chosen location.

## Diagnostics

```powershell
.\DIAGNOSE.ps1
```

A zero exit code confirms that the local Python runtime and required inference libraries loaded successfully.

## Updates

The shared folder is a published copy, not the development workspace. Updates are copied from the working directory only when explicitly requested. Users automatically run the published source on their next launch; rerun `SETUP.ps1` when dependency requirements change.
