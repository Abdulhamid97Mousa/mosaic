"""Bundled assets for the refactored SPADE-BDI stack."""

from __future__ import annotations

from pathlib import Path


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def asl_path(name: str = "main_agent.asl") -> Path:
    """Return the filesystem path for an ASL asset bundled with the package."""

    return _package_root() / "asl" / name


__all__ = ["asl_path"]
