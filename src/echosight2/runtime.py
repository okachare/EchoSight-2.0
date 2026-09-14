"""Per-user runtime paths for shared EchoSight deployments."""

from __future__ import annotations

import os
from pathlib import Path


def user_data_directory() -> Path:
    """Return the writable per-user EchoSight runtime directory."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / "EchoSight" / "2.0"


def log_directory() -> Path:
    """Return the per-user application log directory."""
    return user_data_directory() / "logs"
