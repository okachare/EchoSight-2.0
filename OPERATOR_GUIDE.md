# EchoSight 2.0 Operator Guide

## Start EchoSight

### First use

1. Open the shared EchoSight network folder.
2. Right-click `SETUP.ps1` and run it with PowerShell.
3. Wait for setup and diagnostics to complete.
4. Double-click `Launch_EchoSight.bat`.

Setup is required once for each Windows user profile and requires Python 3.9 plus access to the configured Python package source. Dependencies are installed under `%LOCALAPPDATA%\EchoSight\2.0`; the shared application files are not modified.

### Later use

Double-click `Launch_EchoSight.bat` in the shared folder. Run `SETUP.ps1` again only when setup reports a problem or the published requirements change.

## Inspect Images

1. Select **Load Model** and choose the trained model's parent folder.
2. Confirm the task, labels, thresholds, and preprocessing in **Model Information**.
3. Select **Open Images** and choose one or more supported images or TIFF files.
4. Use **Run All** for the complete list, **Run Selected** for highlighted frames, or **Run Current** for the active frame.
5. Review generated output in the **Results** tab.
6. Use the overlay and annotation checkboxes to control exported visibility.

Supported inputs are PNG, JPEG, BMP, WebP, TIFF, and multi-frame TIFF.

During a multi-frame run, select **Pause** to stop before the next frame. The control changes to **Resume** and continues from the same position when selected again. **Cancel** stops the remaining frames while retaining completed results.

Use the mouse wheel over either image preview to zoom and drag the image to pan. Double-click the preview to restore fit-to-view. In Analysis, select the settings icon at the lower-right of the preview to adjust brightness, contrast, sharpness, and exact denoise strength. Changes preview immediately. Select **Apply to current frame** or **Apply to all frames** to choose the processing scope used by inference and exports.

In Results, select the color-wheel icon at the lower-right of the preview to open **Annotation Control**. Adjust label font size, annotation color, line thickness, annotation transparency, or label transparency. Changes preview live in Results only; Analysis always shows the unannotated image with its active image adjustments. Select **Apply to current frame** or **Apply to all frames** to choose the export scope. Select **Reset** to restore the standard per-class palette and default styling.

Both popups close when you select their icon again or click elsewhere in EchoSight. Unapplied preview changes are discarded when the popup closes.

The Results table can be sorted by frame, task type, annotation count, or highest confidence. Select a column header to alternate ascending and descending order.

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

## Interpret Failures

A failure on one frame does not stop **Run All**. Successful frames remain available and the failed frame is recorded in the exported CSV and manifest.

Review the on-screen **Terminal** and the dated file under `%LOCALAPPDATA%\EchoSight\2.0\logs` for details. Preserve the exported run folder when escalating a model or image issue.

## Diagnostics

```powershell
.\DIAGNOSE.ps1
```

A zero exit code confirms that the local Python runtime and required inference libraries loaded successfully.

## Updates

The shared folder is a published copy, not the development workspace. Updates are copied from the working directory only when explicitly requested. Users automatically run the published source on their next launch; rerun `SETUP.ps1` when dependency requirements change.
