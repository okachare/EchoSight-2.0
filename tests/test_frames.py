from pathlib import Path

import pytest
from PIL import Image

from echosight2.frames import load_frames

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MULTIFRAME_TIFF = WORKSPACE_ROOT / "TiffSplitter" / "samples" / "ARL_Test.TIFF"


@pytest.mark.skipif(not MULTIFRAME_TIFF.exists(), reason="Multi-frame TIFF fixture is unavailable")
def test_loads_every_tiff_page_as_an_independent_frame() -> None:
    frames = load_frames(MULTIFRAME_TIFF)

    assert len(frames) == 66
    assert frames[0].frame_number == 1
    assert frames[-1].frame_number == 66
    assert frames[0].frame_count == 66
    assert frames[0].image.mode == "RGB"
    assert frames[0].image.size == (281, 517)
    assert "Frame 001/66" in frames[0].display_name


def test_loads_regular_image_as_one_frame(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), "black").save(path)

    frames = load_frames(path)

    assert len(frames) == 1
    assert frames[0].display_name == "sample.png"


def test_reports_frame_loading_progress(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), "black").save(path)
    updates: list[tuple[int, int]] = []

    load_frames(path, lambda current, total: updates.append((current, total)))

    assert updates == [(1, 1)]
