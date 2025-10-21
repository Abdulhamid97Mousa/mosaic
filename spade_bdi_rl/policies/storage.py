"""Policy persistence helpers for the refactored SPADE-BDI stack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class PolicySnapshot:
    """In-memory representation of a saved policy artifact."""

    q_table: np.ndarray
    metadata: Dict[str, Any]


class PolicyStorage:
    """CRUD helpers for JSON policy snapshots with accompanying metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Optional[PolicySnapshot]:
        if not self.exists():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != "q_table/v1":
            raise ValueError(f"Unsupported policy format: {payload.get('format')}")
        q_table = np.array(payload.get("q_table", []), dtype=float)
        metadata = payload.get("metadata", {})
        return PolicySnapshot(q_table=q_table, metadata=metadata)

    def save(self, q_table: np.ndarray, metadata: Dict[str, Any]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "q_table/v1",
            "metadata": metadata,
            "q_table": q_table.tolist(),
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return self.path

    def delete(self) -> None:
        if self.exists():
            self.path.unlink()
