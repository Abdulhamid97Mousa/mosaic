"""Metadata helpers for discovering PettingZoo multi-agent checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from gym_gui.config.paths import VAR_TRAINER_DIR


@dataclass(frozen=True)
class PettingZooCheckpoint:
    """Metadata for a discovered PettingZoo policy checkpoint."""

    policy_path: Path
    run_id: str
    env_id: Optional[str]
    family: Optional[str]
    algorithm: Optional[str]
    api_type: Optional[str]  # "aec" or "parallel"
    seed: Optional[int]
    num_agents: Optional[int]
    total_timesteps: Optional[int]
    config_path: Optional[Path]


def _find_run_id(policy_path: Path) -> Optional[str]:
    """Extract run_id from policy path structure.

    Expected structure: var/trainer/runs/{run_id}/...
    """
    resolved = policy_path.resolve()
    parts = resolved.parts
    for idx in range(len(parts) - 4):
        if parts[idx : idx + 3] == ("var", "trainer", "runs"):
            run_id = parts[idx + 3] if idx + 3 < len(parts) else None
            return run_id
    return None


def _load_trainer_config(run_id: str) -> Optional[dict]:
    """Load trainer config for a run."""
    cfg_path = VAR_TRAINER_DIR / "configs" / f"config-{run_id}.json"
    if not cfg_path.exists():
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_metadata(
    config: dict,
) -> tuple[
    Optional[str],  # env_id
    Optional[str],  # family
    Optional[str],  # algorithm
    Optional[str],  # api_type
    Optional[int],  # seed
    Optional[int],  # num_agents
    Optional[int],  # total_timesteps
]:
    """Extract PettingZoo-specific metadata from trainer config."""
    worker_cfg = (
        config.get("metadata", {}).get("worker", {}).get("config", {})
    )
    ui_cfg = config.get("metadata", {}).get("ui", {})

    # Try worker config first, fall back to UI metadata
    env_id = worker_cfg.get("env_id") or ui_cfg.get("env_id")
    family = ui_cfg.get("family")
    algorithm = worker_cfg.get("algorithm") or ui_cfg.get("algorithm")
    api_type = ui_cfg.get("api_type")

    # If not in UI, check environment variables
    if not env_id:
        env_vars = config.get("environment", {})
        env_id = env_vars.get("PETTINGZOO_ENV_ID")
    if not family:
        env_vars = config.get("environment", {})
        family = env_vars.get("PETTINGZOO_FAMILY")
    if not algorithm:
        env_vars = config.get("environment", {})
        algorithm = env_vars.get("ALGORITHM")
    if not api_type:
        env_vars = config.get("environment", {})
        api_type = env_vars.get("PETTINGZOO_API_TYPE")

    # Extract numeric values
    seed = worker_cfg.get("seed")
    num_agents = ui_cfg.get("num_agents")
    total_timesteps = worker_cfg.get("total_timesteps")

    # Safely convert to int
    def safe_int(val) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    return (
        env_id,
        family,
        algorithm,
        api_type,
        safe_int(seed),
        safe_int(num_agents),
        safe_int(total_timesteps),
    )


def load_metadata_for_policy(policy_path: Path) -> Optional[PettingZooCheckpoint]:
    """Load metadata for a PettingZoo policy file.

    Args:
        policy_path: Path to the policy file (.pt, .zip, .pkl)

    Returns:
        PettingZooCheckpoint if metadata found, None otherwise
    """
    run_id = _find_run_id(policy_path)
    if run_id is None:
        return None

    config = _load_trainer_config(run_id)
    if not config:
        # Create minimal checkpoint without config
        return PettingZooCheckpoint(
            policy_path=policy_path,
            run_id=run_id,
            env_id=None,
            family=None,
            algorithm=None,
            api_type=None,
            seed=None,
            num_agents=None,
            total_timesteps=None,
            config_path=None,
        )

    env_id, family, algorithm, api_type, seed, num_agents, total_timesteps = (
        _extract_metadata(config)
    )

    return PettingZooCheckpoint(
        policy_path=policy_path,
        run_id=run_id,
        env_id=env_id,
        family=family,
        algorithm=algorithm,
        api_type=api_type,
        seed=seed,
        num_agents=num_agents,
        total_timesteps=total_timesteps,
        config_path=VAR_TRAINER_DIR / "configs" / f"config-{run_id}.json",
    )


def discover_policies(root: Optional[Path] = None) -> List[PettingZooCheckpoint]:
    """Discover all PettingZoo policy checkpoints.

    Searches for common policy file extensions in the runs directory.

    Args:
        root: Root directory to search (defaults to var/trainer/runs)

    Returns:
        List of discovered checkpoints with metadata
    """
    root = root or (VAR_TRAINER_DIR / "runs")
    results: List[PettingZooCheckpoint] = []

    if not root.exists():
        return results

    # Search for PettingZoo-specific patterns
    # Common extensions: .pt (PyTorch), .zip (SB3), .pkl (pickle)
    patterns = [
        "*/pettingzoo*/**/*.pt",
        "*/pettingzoo*/**/*.zip",
        "*/pettingzoo*/**/*.pkl",
        "*/pettingzoo*/**/model*.pt",
        "*/pettingzoo*/**/policy*.pt",
        "*/pettingzoo*/**/agent*.pt",
    ]

    seen_paths: set[Path] = set()

    for pattern in patterns:
        for policy in sorted(root.glob(pattern)):
            if policy in seen_paths:
                continue
            seen_paths.add(policy)

            # Verify it's a PettingZoo run by checking config
            run_id = _find_run_id(policy)
            if run_id:
                config = _load_trainer_config(run_id)
                if config:
                    # Check if this is actually a PettingZoo run
                    ui_cfg = config.get("metadata", {}).get("ui", {})
                    env_vars = config.get("environment", {})
                    worker_id = ui_cfg.get("worker_id", "")
                    pz_env = env_vars.get("PETTINGZOO_ENV_ID", "")

                    if "pettingzoo" in worker_id.lower() or pz_env:
                        meta = load_metadata_for_policy(policy)
                        if meta is not None:
                            results.append(meta)

    return results


__all__ = [
    "PettingZooCheckpoint",
    "load_metadata_for_policy",
    "discover_policies",
]
