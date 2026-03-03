#!/usr/bin/env python3
"""PyInstaller spec generator for MOSAIC.

Generates a standalone executable for MOSAIC on Linux (Ubuntu/WSL) and Windows.

Usage:
    # Install packaging deps
    pip install -e ".[packaging]"

    # Build one-folder distribution (recommended)
    pyinstaller mosaic.spec

    # The output will be in dist/mosaic/
    # Launch with: dist/mosaic/mosaic (Linux) or dist\\mosaic\\mosaic.exe (Windows)

Notes:
    - This packages the core GUI only. Workers (CleanRL, XuanCe, etc.) run as
      separate subprocesses and are NOT bundled — they must be installed in the
      target Python environment or virtualenv.
    - The trainer daemon is bundled and launched by the app at startup.
    - PyQt6 plugins (platforms, imageformats) are collected automatically.
    - For WSL: same as Linux. Ensure an X server or WSLg is available.
"""
import sys
from pathlib import Path

from PyInstaller.building.api import COLLECT, EXE, PYZ
from PyInstaller.building.build_main import Analysis

ROOT = Path(SPECPATH)

# ---------------------------------------------------------------------------
# Data files to bundle (non-Python resources)
# ---------------------------------------------------------------------------
datas = [
    # Assets (logos, images, SVGs)
    (str(ROOT / "gym_gui" / "assets"), "gym_gui/assets"),
    # QML files
    (str(ROOT / "gym_gui" / "ui" / "qml"), "gym_gui/ui/qml"),
    # Qt stylesheets
    (str(ROOT / "gym_gui" / "ui" / "themes" / "dark.qss"), "gym_gui/ui/themes"),
    # YAML configs
    (str(ROOT / "gym_gui" / "config" / "storage_profiles.yaml"), "gym_gui/config"),
    # Proto definition (for reference; compiled stubs are in Python)
    (str(ROOT / "gym_gui" / "services" / "trainer" / "proto" / "trainer.proto"), "gym_gui/services/trainer/proto"),
    # .env.example as fallback config
    (str(ROOT / ".env.example"), "."),
]

# ---------------------------------------------------------------------------
# Hidden imports that PyInstaller can't auto-detect
# ---------------------------------------------------------------------------
hiddenimports = [
    # PyQt6 modules
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtWebEngineWidgets",
    "PyQt6.QtWebEngineCore",
    "PyQt6.QtQml",
    "PyQt6.QtQuick",
    # qasync event loop
    "qasync",
    # gRPC
    "grpc",
    "grpc.aio",
    "grpc._cython",
    "google.protobuf",
    # Pydantic
    "pydantic",
    "pydantic.deprecated",
    # Trainer daemon (launched as subprocess)
    "gym_gui.services.trainer_daemon",
    # Worker discovery entry points
    "pkg_resources",
    "importlib.metadata",
]

# ---------------------------------------------------------------------------
# Packages to collect entirely (including submodules)
# ---------------------------------------------------------------------------
collect_submodules = [
    "gym_gui",
    "PyQt6",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(ROOT / "gym_gui" / "__main__.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy optional deps not needed in packaged GUI
        "matplotlib",
        "scipy",
        "pandas",
        "tensorflow",
        "tensorboard",
        "wandb",
        "vllm",
        "jax",
        "jaxlib",
        "ray",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="mosaic",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for logging; set False for release
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="mosaic",
)
