from pathlib import Path

import numpy as np
from echosight2.frames import LoadedFrame
from echosight2.training import export_training_frames
from PIL import Image


def test_exports_selected_original_frames_to_model_category_folder(tmp_path: Path) -> None:
    source = tmp_path / "inspection scan.tiff"
    first_pixels = np.full((8, 10, 3), 40, dtype=np.uint8)
    second_pixels = np.full((8, 10, 3), 180, dtype=np.uint8)
    frames = [
        (2, LoadedFrame(source, 3, 12, Image.fromarray(first_pixels))),
        (7, LoadedFrame(source, 8, 12, Image.fromarray(second_pixels))),
    ]

    report = export_training_frames(tmp_path, frames, "False_Hits", "Detection Model V17")

    assert report.directory.name == "Mark_for_Training_False_Hits_(Detection_Model_V17)"
    assert [path.name for path in report.images] == [
        "00003_inspection_scan_frame_0003.png",
        "00008_inspection_scan_frame_0008.png",
    ]
    assert np.array_equal(np.asarray(Image.open(report.images[0])), first_pixels)
    assert np.array_equal(np.asarray(Image.open(report.images[1])), second_pixels)


def test_misses_export_does_not_overwrite_existing_training_image(tmp_path: Path) -> None:
    frame = LoadedFrame(tmp_path / "sample.png", 1, 1, Image.new("RGB", (4, 4), "white"))

    first = export_training_frames(tmp_path, [(0, frame)], "Misses", "Model")
    second = export_training_frames(tmp_path, [(0, frame)], "Misses", "Model")

    assert first.directory.name == "Mark_for_Training_Misses_(Model)"
    assert first.images[0].name == "00001_sample_frame_0001.png"
    assert second.images[0].name == "00001_sample_frame_0001_2.png"