from pathlib import Path

from echosight2.runtime import log_directory, user_data_directory


def test_user_data_directory_uses_local_app_data(monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(Path("C:/Users/operator/AppData/Local")))

    assert user_data_directory() == Path("C:/Users/operator/AppData/Local/EchoSight/2.0")
    assert log_directory() == Path("C:/Users/operator/AppData/Local/EchoSight/2.0/logs")


def test_user_data_directory_has_home_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert user_data_directory() == tmp_path / "AppData" / "Local" / "EchoSight" / "2.0"
