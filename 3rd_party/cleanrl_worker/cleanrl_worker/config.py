"""Configuration parsing helpers for the CleanRL worker shim."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class WorkerConfig:
    """Structured configuration for launching a CleanRL worker run."""

    run_id: str
    algo: str
    env_id: str
    total_timesteps: int
    seed: Optional[int] = None
    extras: dict[str, Any] = field(default_factory=dict)
    worker_id: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def with_overrides(
        self,
        *,
        algo: Optional[str] = None,
        env_id: Optional[str] = None,
        total_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        worker_id: Optional[str] = None,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> "WorkerConfig":
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


def _extract_worker_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the worker config subtree from a trainer config payload."""

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
    run_fields = ("run_id", "algo", "env_id")
    for field_name in run_fields:
        if not payload.get(field_name):
            raise ValueError(f"cleanrl_worker config missing required field '{field_name}'")

    if "total_timesteps" not in payload:
        raise ValueError("cleanrl_worker config missing required field 'total_timesteps'")


def parse_worker_config(payload: Mapping[str, Any]) -> WorkerConfig:
    """Parse a dict payload into a WorkerConfig."""

    _validate_required_fields(payload)

    extras_field = payload.get("extras")
    extras: dict[str, Any]
    if isinstance(extras_field, MutableMapping):
        extras = dict(extras_field)
    else:
        extras = {}

    mode = extras.get("mode")
    if mode == "policy_eval" and not extras.get("policy_path"):
        raise ValueError("policy_eval mode requires a policy_path extra")

    total_timesteps_value = payload.get("total_timesteps")
    total_timesteps = int(total_timesteps_value or 0)

    if total_timesteps <= 0:
        eval_episodes = extras.get("eval_episodes")
        try:
            total_timesteps = max(1, int(eval_episodes)) if eval_episodes is not None else 1
        except (TypeError, ValueError):
            total_timesteps = 1

    return WorkerConfig(
        run_id=str(payload["run_id"]),
        algo=str(payload["algo"]),
        env_id=str(payload["env_id"]),
        total_timesteps=total_timesteps,
        seed=int(payload["seed"]) if payload.get("seed") is not None else None,
        extras=extras,
        worker_id=str(payload["worker_id"]).strip() or None if payload.get("worker_id") else None,
        raw=dict(payload),
    )


def load_worker_config(path: Path) -> WorkerConfig:
    """Load and parse a trainer-generated worker config JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "metadata" in payload:
        worker_payload = _extract_worker_payload(payload)
    else:
        worker_payload = payload
    return parse_worker_config(worker_payload)


def load_worker_config_from_string(text: str) -> WorkerConfig:
    """Convenience helper for tests to parse JSON content directly."""

    payload = json.loads(text)
    return parse_worker_config(payload)
