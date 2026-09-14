"""Environment and workspace diagnostics for EchoSight 2.0."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    path: str
    kind: str
    size_bytes: int
    paired_binary: bool | None = None


def _package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception as error:  # noqa: BLE001 - report binary/import failures verbatim.
        return f"unavailable: {type(error).__name__}: {error}"
    return str(getattr(module, "__version__", "installed"))


def collect_environment() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "packages": {
            "numpy": _package_version("numpy"),
            "Pillow": _package_version("PIL"),
            "openvino": _package_version("openvino"),
            "onnxruntime": _package_version("onnxruntime"),
        },
    }


def inventory_models(workspace_root: Path) -> list[ModelArtifact]:
    artifacts: list[ModelArtifact] = []
    search_roots = [workspace_root / "Debug", workspace_root / "Deployment"]
    seen: set[Path] = set()

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for directory, _, filenames in os.walk(search_root, onerror=lambda _: None):
            for filename in filenames:
                path = Path(directory) / filename
                if path in seen or path.suffix.lower() not in {".xml", ".onnx", ".bin", ".ckpt"}:
                    continue
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    continue
                seen.add(path)
                artifacts.append(
                    ModelArtifact(
                        path=str(path.relative_to(workspace_root)),
                        kind=path.suffix.lower().lstrip("."),
                        size_bytes=size_bytes,
                        paired_binary=(path.with_suffix(".bin").exists() if path.suffix.lower() == ".xml" else None),
                    )
                )

    return sorted(artifacts, key=lambda item: item.path.lower())


def collect_report(workspace_root: Path) -> dict[str, object]:
    return {
        "environment": collect_environment(),
        "model_artifacts": [asdict(item) for item in inventory_models(workspace_root)],
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    report = collect_report(project_root)
    print(json.dumps(report, indent=2))

    unavailable = [
        f"{name}: {version}"
        for name, version in report["environment"]["packages"].items()
        if str(version).startswith("unavailable:")
    ]
    if importlib.util.find_spec("openvino") is None:
        unavailable.append("openvino: module spec not found")
    if unavailable:
        print("ERROR: Required runtime dependencies are unavailable:", file=sys.stderr)
        print("\n".join(unavailable), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
