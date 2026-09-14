from pathlib import Path

from echosight2.diagnostics import inventory_models


def test_inventory_models_finds_supported_artifacts(tmp_path: Path) -> None:
    model_directory = tmp_path / "Debug" / "models"
    model_directory.mkdir(parents=True)
    (model_directory / "detector.xml").write_text("xml", encoding="utf-8")
    (model_directory / "detector.bin").write_bytes(b"bin")
    (model_directory / "classifier.onnx").write_bytes(b"onnx")
    (model_directory / "notes.txt").write_text("ignored", encoding="utf-8")

    artifacts = inventory_models(tmp_path)

    assert [artifact.kind for artifact in artifacts] == ["onnx", "bin", "xml"]
    xml_artifact = next(artifact for artifact in artifacts if artifact.kind == "xml")
    assert xml_artifact.paired_binary is True
