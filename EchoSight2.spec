from pathlib import Path

project_root = Path(SPECPATH)
datas = [
    (str(project_root / "README.md"), "."),
    (str(project_root / "OPERATOR_GUIDE.md"), "."),
]
hiddenimports = [
    "openvino",
    "openvino.runtime",
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
]

analysis = Analysis(
    [str(project_root / "tools" / "echosight2_entry.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "ruff"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="EchoSight2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EchoSight2",
)
