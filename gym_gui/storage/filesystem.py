from __future__ import annotations

"""Filesystem-backed storage helpers."""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "var" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_json(name: str, payload: Any) -> Path:
    """Persist a JSON payload under the data directory."""

    path = DATA_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_json(name: str) -> Any:
    """Load a JSON payload if available, otherwise return ``None``."""

    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["write_json", "read_json", "DATA_DIR"]
