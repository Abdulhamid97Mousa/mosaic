from __future__ import annotations

"""Centralized filesystem paths used across the Gym GUI application."""

from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_ROOT.parent

# Writable runtime artifacts (shared with worker stack). Prefer repo-level var/.
VAR_ROOT = (_REPO_ROOT / "var").resolve()
# LEGACY_VAR_ROOT = _PACKAGE_ROOT / "var"
VAR_RECORDS_DIR = VAR_ROOT / "records"
VAR_TELEMETRY_DIR = VAR_ROOT / "telemetry"
VAR_CACHE_DIR = VAR_ROOT / "cache"
VAR_TMP_DIR = VAR_ROOT / "tmp"
VAR_LOGS_DIR = VAR_ROOT / "logs"
VAR_TRAINER_DIR = VAR_ROOT / "trainer"
VAR_TRAINER_DB = VAR_TRAINER_DIR / "trainer.sqlite"
VAR_DATA_DIR = VAR_ROOT / "data"

# Legacy runtime assets that ship with the project (toy-text boards, etc.).
RUNTIME_ROOT = _PACKAGE_ROOT / "runtime"
RUNTIME_DATA_DIR = RUNTIME_ROOT / "data"


def ensure_var_directories() -> None:
    """Create the writable directory structure if it does not exist."""

    for path in (
        VAR_ROOT,
        VAR_RECORDS_DIR,
        VAR_TELEMETRY_DIR,
        VAR_CACHE_DIR,
        VAR_TMP_DIR,
        VAR_LOGS_DIR,
        VAR_TRAINER_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "VAR_ROOT",
    "VAR_RECORDS_DIR",
    "VAR_TELEMETRY_DIR",
    "VAR_CACHE_DIR",
    "VAR_TMP_DIR",
    "VAR_LOGS_DIR",
    "VAR_TRAINER_DIR",
    "VAR_TRAINER_DB",
    "VAR_DATA_DIR",
    # "RUNTIME_ROOT",
    # "RUNTIME_DATA_DIR",
    "ensure_var_directories",
    # "LEGACY_VAR_ROOT",
]
