from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from .paths import _PACKAGE_ROOT

_PRESETS_PATH = (_PACKAGE_ROOT.parent / "metadata" / "cleanrl" / "eval_presets.json").resolve()


@lru_cache(maxsize=1)
def _load_presets() -> Dict[str, Any]:
    if not _PRESETS_PATH.exists():
        return {"defaults": {}, "envs": {}}
    try:
        return json.loads(_PRESETS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"defaults": {}, "envs": {}}


def get_eval_preset(env_id: str | None) -> Dict[str, Any]:
    presets = _load_presets()
    defaults = presets.get("defaults", {})
    envs = presets.get("envs", {}) or {}
    env_preset = envs.get(env_id, {}) if env_id else {}
    merged = dict(defaults)
    merged.update(env_preset or {})
    return merged
