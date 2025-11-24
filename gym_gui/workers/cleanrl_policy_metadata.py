"""Metadata helpers for discovering CleanRL checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gym_gui.config.paths import VAR_TRAINER_DIR


@dataclass(frozen=True)
class CleanRlCheckpoint:
    policy_path: Path
    run_id: str
    cleanrl_run_name: Optional[str]
    env_id: Optional[str]
    algo: Optional[str]
    seed: Optional[int]
    num_envs: Optional[int]
    fastlane_only: bool
    fastlane_video_mode: Optional[str]
    fastlane_grid_limit: Optional[int]
    config_path: Optional[Path]


def _find_run_id(policy_path: Path) -> tuple[Optional[str], Optional[str]]:
    resolved = policy_path.resolve()
    parts = resolved.parts
    for idx in range(len(parts) - 5):
        if parts[idx : idx + 3] == ("var", "trainer", "runs"):
            run_id = parts[idx + 3] if idx + 3 < len(parts) else None
            cleanrl_run = parts[idx + 5] if idx + 5 < len(parts) else None
            return run_id, cleanrl_run
    return None, None


def _load_trainer_config(run_id: str) -> Optional[dict]:
    cfg_path = VAR_TRAINER_DIR / "configs" / f"config-{run_id}.json"
    if not cfg_path.exists():
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_metadata(config: dict) -> tuple[Optional[str], Optional[str], Optional[int], Optional[int], bool, Optional[str], Optional[int]]:
    worker_cfg = (
        config.get("metadata", {})
        .get("worker", {})
        .get("config", {})
    )
    env_id = worker_cfg.get("env_id")
    algo = worker_cfg.get("algo")
    seed = worker_cfg.get("seed")
    extras = worker_cfg.get("extras", {}) if isinstance(worker_cfg, dict) else {}
    algo_params = extras.get("algo_params", {}) if isinstance(extras, dict) else {}
    num_envs = algo_params.get("num_envs") if isinstance(algo_params, dict) else None
    try:
        num_envs = int(num_envs)
    except (TypeError, ValueError):
        num_envs = None
    fastlane_only = bool(extras.get("fastlane_only"))
    video_mode = extras.get("fastlane_video_mode")
    grid_limit = extras.get("fastlane_grid_limit")
    try:
        grid_limit = int(grid_limit)
    except (TypeError, ValueError):
        grid_limit = None
    try:
        seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed = None
    return env_id, algo, seed, num_envs, fastlane_only, video_mode, grid_limit


def load_metadata_for_policy(policy_path: Path) -> Optional[CleanRlCheckpoint]:
    run_id, cleanrl_run = _find_run_id(policy_path)
    if run_id is None:
        return None
    config = _load_trainer_config(run_id)
    if not config:
        return None
    env_id, algo, seed, num_envs, fastlane_only, video_mode, grid_limit = _extract_metadata(config)
    return CleanRlCheckpoint(
        policy_path=policy_path,
        run_id=run_id,
        cleanrl_run_name=cleanrl_run,
        env_id=env_id,
        algo=algo,
        seed=seed,
        num_envs=num_envs,
        fastlane_only=fastlane_only,
        fastlane_video_mode=video_mode,
        fastlane_grid_limit=grid_limit,
        config_path=VAR_TRAINER_DIR / "configs" / f"config-{run_id}.json",
    )


def discover_policies(root: Optional[Path] = None) -> list[CleanRlCheckpoint]:
    root = root or (VAR_TRAINER_DIR / "runs")
    results: list[CleanRlCheckpoint] = []
    if not root.exists():
        return results
    for policy in sorted(root.glob("*/runs/*/*.cleanrl_model")):
        meta = load_metadata_for_policy(policy)
        if meta is not None:
            results.append(meta)
    return results


__all__ = [
    "CleanRlCheckpoint",
    "load_metadata_for_policy",
    "discover_policies",
]
