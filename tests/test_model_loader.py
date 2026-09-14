from pathlib import Path

import pytest

from echosight2.inference import ModelLoader, TaskType

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DETECTION_MODEL = WORKSPACE_ROOT / "Deployment" / "Test_Run_Detect" / "deployment" / "Detection" / "model" / "model.xml"
ANOMALY_MODEL = WORKSPACE_ROOT / "Deployment" / "Test_Run_Instance_Segmentation" / "deployment" / "Anomaly classification" / "model" / "model.xml"
ONNX_MODEL = WORKSPACE_ROOT / "Debug" / "NVL_Geti_Run" / "getitune-workspace-aa5aafdc-1098-4fdc-99a2-e762d61a85e6" / "exported_model.onnx"


@pytest.mark.skipif(not DETECTION_MODEL.exists(), reason="Detection fixture is not available")
def test_inspects_detection_model() -> None:
    info = ModelLoader().inspect(DETECTION_MODEL)

    assert info.task_type is TaskType.DETECTION
    assert info.input_size == (800, 992)
    assert {output.name for output in info.outputs} == {"boxes", "labels"}


@pytest.mark.skipif(not ANOMALY_MODEL.exists(), reason="Anomaly fixture is not available")
def test_inspects_anomaly_model() -> None:
    info = ModelLoader().inspect(ANOMALY_MODEL)

    assert info.task_type is TaskType.ANOMALY
    assert info.input_size == (448, 448)
    assert info.outputs[0].shape == (1, 1, 448, 448)


@pytest.mark.skipif(not ONNX_MODEL.exists(), reason="ONNX fixture is not available")
def test_inspects_instance_segmentation_onnx_model() -> None:
    info = ModelLoader().inspect(ONNX_MODEL)

    assert info.format == "ONNX"
    assert info.task_type is TaskType.INSTANCE_SEGMENTATION
    assert info.input_size == (1344, 1344)
    assert {output.name for output in info.outputs} == {"boxes", "labels", "masks"}


def test_rejects_unsupported_model_format(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")

    with pytest.raises(ValueError, match="Unsupported model format"):
        ModelLoader().inspect(model_path)


def test_discovers_model_from_parent_folder(tmp_path: Path) -> None:
    model_directory = tmp_path / "deployment" / "Detection" / "model"
    model_directory.mkdir(parents=True)
    (model_directory / "model.xml").write_text("<net/>", encoding="utf-8")
    (model_directory / "model.bin").write_bytes(b"weights")

    models = ModelLoader.discover(tmp_path)

    assert models == ((model_directory / "model.xml").resolve(),)


def test_discovery_prefers_xml_over_matching_onnx(tmp_path: Path) -> None:
    (tmp_path / "exported_model.xml").write_text("<net/>", encoding="utf-8")
    (tmp_path / "exported_model.bin").write_bytes(b"weights")
    (tmp_path / "exported_model.onnx").write_bytes(b"model")

    models = ModelLoader.discover(tmp_path)

    assert models == ((tmp_path / "exported_model.xml").resolve(),)


def test_uses_structured_label_metadata() -> None:
    metadata = {"label_info": '{"label_names": ["Delamination", "No object"]}', "labels": "ignored values"}

    labels = ModelLoader._labels_from_metadata(metadata, None)

    assert labels == ("Delamination", "No object")
