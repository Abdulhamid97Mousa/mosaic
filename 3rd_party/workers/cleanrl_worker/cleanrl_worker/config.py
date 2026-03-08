"""Configuration parsing helpers for the CleanRL worker shim."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol


@dataclass(frozen=True)
class CleanRLWorkerConfig:
    """Structured configuration for launching a CleanRL worker run.

    Implements the MOSAIC WorkerConfig protocol for standardization.
    """

    run_id: str
    algo: str
    env_id: str
    total_timesteps: int
    seed: Optional[int] = None
    extras: dict[str, Any] = field(default_factory=dict)
    worker_id: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate required fields on initialization."""
        # Validate required fields
        if not self.run_id:
            raise ValueError("cleanrl_worker config missing required field 'run_id'")
        if not self.algo:
            raise ValueError("cleanrl_worker config missing required field 'algo'")
        if not self.env_id:
            raise ValueError("cleanrl_worker config missing required field 'env_id'")
        if self.total_timesteps is None:
            raise ValueError("cleanrl_worker config missing required field 'total_timesteps'")

        # Validate mode-specific requirements
        mode = self.extras.get("mode")
        if mode == "policy_eval" and not self.extras.get("policy_path"):
            raise ValueError("policy_eval mode requires a policy_path extra")

        # Assert protocol compliance (structural typing check)
        assert isinstance(self, WorkerConfigProtocol), (
            "CleanRLWorkerConfig must implement WorkerConfig protocol"
        )

    def with_overrides(
        self,
        *,
        algo: Optional[str] = None,
        env_id: Optional[str] = None,
        total_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        worker_id: Optional[str] = None,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> "CleanRLWorkerConfig":
        """Return a new config applying CLI overrides."""

        merged_extras = dict(self.extras)
        if extras:
            merged_extras.update(extras)

        return replace(
            self,
            algo=algo or self.algo,
            env_id=env_id or self.env_id,
            total_timesteps=total_timesteps or self.total_timesteps,
            seed=self.seed if seed is None else seed,
            worker_id=worker_id if worker_id is not None else self.worker_id,
            extras=merged_extras,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary (WorkerConfig protocol requirement).

        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CleanRLWorkerConfig":
        """Deserialize configuration from dictionary (WorkerConfig protocol requirement).

        Handles both flat and nested formats for backwards compatibility.

        Args:
            data: Dictionary containing configuration values

        Returns:
            CleanRLWorkerConfig instance
        """
        # Extract extras if present
        extras_field = data.get("extras")
        extras: dict[str, Any]
        if isinstance(extras_field, MutableMapping):
            extras = dict(extras_field)
        else:
            extras = {}

        # Handle total_timesteps special case (eval mode)
        total_timesteps_value = data.get("total_timesteps")
        total_timesteps = int(total_timesteps_value or 0)

        if total_timesteps <= 0:
            eval_episodes = extras.get("eval_episodes")
            try:
                total_timesteps = max(1, int(eval_episodes)) if eval_episodes is not None else 1
            except (TypeError, ValueError):
                total_timesteps = 1

        return cls(
            run_id=str(data["run_id"]),
            algo=str(data["algo"]),
            env_id=str(data["env_id"]),
            total_timesteps=total_timesteps,
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            extras=extras,
            worker_id=str(data["worker_id"]).strip() or None if data.get("worker_id") else None,
            raw=dict(data),
        )


def _extract_worker_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the worker config subtree from a trainer config payload.

    DEPRECATED: Use gym_gui.core.worker.extract_worker_config() instead.
    Kept for backwards compatibility.
    """

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata missing from trainer config payload")

    worker_section = metadata.get("worker")
    if not isinstance(worker_section, Mapping):
        raise ValueError("metadata.worker missing from trainer config payload")

    worker_config = worker_section.get("config")
    if not isinstance(worker_config, Mapping):
        raise ValueError("metadata.worker.config missing from trainer config payload")

    return worker_config


def _validate_required_fields(payload: Mapping[str, Any]) -> None:
    """Validate required fields in worker config payload.

    DEPRECATED: Validation moved to CleanRLWorkerConfig.__post_init__().
    Kept for backwards compatibility.
    """
    run_fields = ("run_id", "algo", "env_id")
    for field_name in run_fields:
        if not payload.get(field_name):
            raise ValueError(f"cleanrl_worker config missing required field '{field_name}'")

    if "total_timesteps" not in payload:
        raise ValueError("cleanrl_worker config missing required field 'total_timesteps'")


def parse_worker_config(payload: Mapping[str, Any]) -> CleanRLWorkerConfig:
    """Parse a dict payload into a CleanRLWorkerConfig.

    Args:
        payload: Dictionary containing worker configuration

    Returns:
        CleanRLWorkerConfig instance

    Note:
        Prefer using CleanRLWorkerConfig.from_dict() directly.
        This function is kept for backwards compatibility.
    """
    return CleanRLWorkerConfig.from_dict(dict(payload))


def load_worker_config(path: Path) -> CleanRLWorkerConfig:
    """Load and parse a trainer-generated worker config JSON file.

    Args:
        path: Path to JSON configuration file

    Returns:
        CleanRLWorkerConfig instance

    Note:
        For new code, use gym_gui.core.worker.load_worker_config_from_file() instead.
        This function is kept for backwards compatibility.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "metadata" in payload:
        worker_payload = _extract_worker_payload(payload)
    else:
        worker_payload = payload
    return parse_worker_config(worker_payload)


def load_worker_config_from_string(text: str) -> CleanRLWorkerConfig:
    """Convenience helper for tests to parse JSON content directly."""

    payload = json.loads(text)
    return parse_worker_config(payload)


# Type alias for backwards compatibility
WorkerConfig = CleanRLWorkerConfig
