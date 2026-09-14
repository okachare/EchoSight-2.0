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

1. Select **Load Model Folder** and choose the trained model's parent folder.
2. Confirm the task, labels, thresholds, and preprocessing in **Model Information**.
3. Select **Open Images** and choose one or more supported images or TIFF files.
4. Select a frame and use **Run Current**, or use **Run All** for the complete list.
5. Review generated output in the **Results** tab.
6. Use the overlay and annotation checkboxes to control exported visibility.

Supported inputs are PNG, JPEG, BMP, WebP, TIFF, and multi-frame TIFF.

## Export Run Evidence

1. Complete inference for the required frames.
2. In **Results**, select **Export Results**.
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
