"""Metadata helpers for discovering Ray RLlib checkpoints.

Ray RLlib checkpoints are stored in:
    var/trainer/runs/{run_id}/checkpoints/

Structure:
    checkpoints/
    ├── rllib_checkpoint.json      # Algorithm-level metadata
    ├── algorithm_state.pkl        # Full algorithm state (cloudpickle)
    └── policies/
        └── {policy_id}/           # e.g., "shared" or "agent_0"
            ├── rllib_checkpoint.json
            └── policy_state.pkl

The analytics.json in the run directory contains:
    - ray_metadata: {algorithm, env_id, env_family, paradigm, num_agents}
    - worker_type: "ray_worker"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from gym_gui.config.paths import VAR_TRAINER_DIR

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RayRLlibCheckpoint:
    """Represents a discovered Ray RLlib checkpoint."""

    checkpoint_path: Path  # Path to checkpoints/ directory
    run_id: str  # ULID run identifier
    algorithm: str  # PPO, APPO, IMPALA, DQN, SAC
    env_id: str  # pursuit_v4, multiwalker_v9, etc.
    env_family: str  # sisl, mpe, etc.
    paradigm: str  # parameter_sharing, independent, self_play
    policy_ids: List[str] = field(default_factory=list)  # ["shared"] or ["agent_0", ...]
    ray_version: str = "unknown"
    checkpoint_version: str = "1.1"
    config_path: Optional[Path] = None

    @property
    def display_name(self) -> str:
        """Human-readable display name for the checkpoint."""
        parts = [self.run_id[:8], self.env_id, self.algorithm]
        if self.paradigm:
            parts.append(f"[{self.paradigm}]")
        return " - ".join(parts)

    @property
    def policies_dir(self) -> Path:
        """Path to the policies directory."""
        return self.checkpoint_path / "policies"

    def get_policy_path(self, policy_id: str = "shared") -> Optional[Path]:
        """Get path to a specific policy checkpoint.

        Args:
            policy_id: Policy identifier (default: "shared" for parameter sharing)

        Returns:
            Path to policy directory or None if not found
        """
        policy_path = self.policies_dir / policy_id
        if policy_path.exists():
            return policy_path
        return None


@dataclass(frozen=True)
class RayRLlibPolicy:
    """Represents a single policy within a Ray RLlib checkpoint."""

    policy_path: Path  # Path to policies/{policy_id}/ directory
    policy_id: str  # Policy identifier (e.g., "shared", "agent_0")
    checkpoint: RayRLlibCheckpoint  # Parent checkpoint reference
    state_file: Optional[Path] = None  # Path to policy_state.pkl

    @property
    def display_name(self) -> str:
        """Human-readable display name for the policy."""
        return f"{self.checkpoint.display_name} / {self.policy_id}"


def _load_analytics(run_dir: Path) -> Optional[dict]:
    """Load analytics.json from run directory."""
    analytics_path = run_dir / "analytics.json"
    if not analytics_path.exists():
        return None
    try:
        return json.loads(analytics_path.read_text(encoding="utf-8"))
    except Exception as e:
        _LOGGER.warning("Failed to load analytics.json from %s: %s", run_dir, e)
        return None


def _load_rllib_checkpoint_meta(checkpoint_dir: Path) -> Optional[dict]:
    """Load rllib_checkpoint.json metadata."""
    meta_path = checkpoint_dir / "rllib_checkpoint.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        _LOGGER.warning("Failed to load rllib_checkpoint.json from %s: %s", checkpoint_dir, e)
        return None


def _is_ray_worker_run(run_dir: Path) -> bool:
    """Check if run directory is from ray_worker."""
    analytics = _load_analytics(run_dir)
    if not analytics:
        return False
    return analytics.get("worker_type") == "ray_worker"


def _has_valid_checkpoint(checkpoint_dir: Path) -> bool:
    """Check if checkpoint directory has valid Ray RLlib checkpoint files."""
    if not checkpoint_dir.exists():
        return False

    # Check for rllib_checkpoint.json
    rllib_meta = checkpoint_dir / "rllib_checkpoint.json"
    if not rllib_meta.exists():
        return False

    # Check for either algorithm_state.pkl or policies/ directory
    has_algorithm_state = (checkpoint_dir / "algorithm_state.pkl").exists()
    has_policies = (checkpoint_dir / "policies").is_dir()

    return has_algorithm_state or has_policies


def load_checkpoint_metadata(checkpoint_dir: Path) -> Optional[RayRLlibCheckpoint]:
    """Load metadata for a Ray RLlib checkpoint.

    Args:
        checkpoint_dir: Path to checkpoints/ directory

    Returns:
        RayRLlibCheckpoint or None if invalid
    """
    run_dir = checkpoint_dir.parent
    run_id = run_dir.name

    # Load analytics.json
    analytics = _load_analytics(run_dir)
    if not analytics or analytics.get("worker_type") != "ray_worker":
        return None

    ray_metadata = analytics.get("ray_metadata", {})

    # Load rllib_checkpoint.json
    rllib_meta = _load_rllib_checkpoint_meta(checkpoint_dir)
    if not rllib_meta:
        return None

    # Get policy IDs from checkpoint metadata or scan policies/ directory
    policy_ids = rllib_meta.get("policy_ids", [])
    if not policy_ids:
        # Fallback: scan policies/ directory
        policies_dir = checkpoint_dir / "policies"
        if policies_dir.is_dir():
            policy_ids = [p.name for p in policies_dir.iterdir() if p.is_dir()]
        if not policy_ids:
            policy_ids = ["shared"]  # Default assumption

    # Check for config file
    config_path = VAR_TRAINER_DIR / "configs" / f"config-{run_id}.json"
    if not config_path.exists():
        config_path = None

    return RayRLlibCheckpoint(
        checkpoint_path=checkpoint_dir,
        run_id=run_id,
        algorithm=ray_metadata.get("algorithm", "unknown"),
        env_id=ray_metadata.get("env_id", "unknown"),
        env_family=ray_metadata.get("env_family", "unknown"),
        paradigm=ray_metadata.get("paradigm", "parameter_sharing"),
        policy_ids=policy_ids,
        ray_version=rllib_meta.get("ray_version", "unknown"),
        checkpoint_version=rllib_meta.get("checkpoint_version", "1.1"),
        config_path=config_path,
    )


def discover_ray_checkpoints(root: Optional[Path] = None) -> List[RayRLlibCheckpoint]:
    """Discover all Ray RLlib checkpoints in the runs directory.

    Args:
        root: Root directory to scan (default: VAR_TRAINER_DIR / "runs")

    Returns:
        List of discovered checkpoints, sorted by run_id (newest first)
    """
    root = root or (VAR_TRAINER_DIR / "runs")
    results: List[RayRLlibCheckpoint] = []

    if not root.exists():
        return results

    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        checkpoint_dir = run_dir / "checkpoints"
        if not _has_valid_checkpoint(checkpoint_dir):
            continue

        if not _is_ray_worker_run(run_dir):
            continue

        meta = load_checkpoint_metadata(checkpoint_dir)
        if meta is not None:
            results.append(meta)
            _LOGGER.debug(
                "Discovered Ray checkpoint: %s (%s/%s)",
                meta.run_id,
                meta.algorithm,
                meta.env_id,
            )

    _LOGGER.info("Discovered %d Ray RLlib checkpoint(s)", len(results))
    return results


def discover_ray_policies(root: Optional[Path] = None) -> List[RayRLlibPolicy]:
    """Discover all individual policies from Ray RLlib checkpoints.

    This flattens multi-policy checkpoints into individual policy entries.

    Args:
        root: Root directory to scan (default: VAR_TRAINER_DIR / "runs")

    Returns:
        List of discovered policies
    """
    checkpoints = discover_ray_checkpoints(root)
    policies: List[RayRLlibPolicy] = []

    for checkpoint in checkpoints:
        for policy_id in checkpoint.policy_ids:
            policy_path = checkpoint.get_policy_path(policy_id)
            if policy_path is None:
                # Policy ID exists in metadata but directory not found
                _LOGGER.warning(
                    "Policy %s listed but not found in %s",
                    policy_id,
                    checkpoint.checkpoint_path,
                )
                continue

            # Check for policy state file
            state_file = policy_path / "policy_state.pkl"
            if not state_file.exists():
                state_file = None

            policies.append(
                RayRLlibPolicy(
                    policy_path=policy_path,
                    policy_id=policy_id,
                    checkpoint=checkpoint,
                    state_file=state_file,
                )
            )

    return policies


def get_checkpoints_for_env(
    env_id: str,
    root: Optional[Path] = None,
) -> List[RayRLlibCheckpoint]:
    """Get checkpoints matching a specific environment.

    Args:
        env_id: Environment identifier (e.g., "pursuit_v4")
        root: Root directory to scan

    Returns:
        List of matching checkpoints
    """
    all_checkpoints = discover_ray_checkpoints(root)
    return [c for c in all_checkpoints if c.env_id == env_id]


def get_latest_checkpoint(
    env_id: Optional[str] = None,
    algorithm: Optional[str] = None,
    root: Optional[Path] = None,
) -> Optional[RayRLlibCheckpoint]:
    """Get the most recent checkpoint, optionally filtered.

    Args:
        env_id: Filter by environment (optional)
        algorithm: Filter by algorithm (optional)
        root: Root directory to scan

    Returns:
        Most recent matching checkpoint or None
    """
    checkpoints = discover_ray_checkpoints(root)

    if env_id:
        checkpoints = [c for c in checkpoints if c.env_id == env_id]
    if algorithm:
        checkpoints = [c for c in checkpoints if c.algorithm == algorithm]

    return checkpoints[0] if checkpoints else None


__all__ = [
    "RayRLlibCheckpoint",
    "RayRLlibPolicy",
    "load_checkpoint_metadata",
    "discover_ray_checkpoints",
    "discover_ray_policies",
    "get_checkpoints_for_env",
    "get_latest_checkpoint",
]
