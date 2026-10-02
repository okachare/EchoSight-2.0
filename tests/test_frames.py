from pathlib import Path

import pytest
from PIL import Image

import echosight2.frames as frame_module
from echosight2.frames import iter_frame_tiles, load_frames

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


def test_optimizes_oversized_uncompressed_palette_tiff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "oversized.tiff"
    image = Image.new("P", (100, 100))
    image.putpalette([value for index in range(256) for value in (index, index, index)])
    image.save(path, compression="raw")
    monkeypatch.setattr(frame_module, "MAX_WORKING_PIXELS", 2_500)

    frames = load_frames(path)

    assert len(frames) == 1
    assert frames[0].original_size == (100, 100)
    assert frames[0].image.size == (50, 50)
    assert frames[0].image.mode == "RGB"
    assert frames[0].is_optimized
    assert "Full-resolution tiled inference 100x100" in frames[0].display_name


def test_oversized_tiff_tiles_preserve_every_source_pixel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "oversized.tiff"
    values = bytes(range(100))
    Image.frombytes("L", (10, 10), values).save(path, compression="raw")
    monkeypatch.setattr(frame_module, "MAX_WORKING_PIXELS", 25)
    frame = load_frames(path)[0]

    tiles = list(iter_frame_tiles(frame, (4, 4), overlap=0.25))
    coverage = Image.new("L", frame.original_size, 0)
    reconstructed = Image.new("L", frame.original_size)
    for tile in tiles:
        reconstructed.paste(tile.image.convert("L"), tile.box[:2])
        coverage.paste(1, tile.box)

    assert frame.image.size == (5, 5)
    assert min(coverage.getdata()) == 1
    assert reconstructed.tobytes() == values


def test_oversized_jpeg_is_rejected_before_lossy_preview_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized.jpg"
    Image.new("RGB", (100, 100), "black").save(path)
    monkeypatch.setattr(frame_module, "MAX_WORKING_PIXELS", 2_500)

    with pytest.raises(ValueError, match="Full-resolution bounded inference requires"):
        load_frames(path)
