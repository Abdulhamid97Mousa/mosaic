"""Structured configuration objects for the refactored SPADE-BDI worker."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class PolicyStrategy(str, Enum):
    """Available strategies describing how the worker should handle policies."""

    TRAIN = "train"
    LOAD = "load"
    EVAL = "eval"
    TRAIN_AND_SAVE = "train_and_save"

    @classmethod
    def from_any(cls, value: Optional[str]) -> "PolicyStrategy":
        if value is None:
            return cls.EVAL
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported policy_strategy '{value}'. Expected one of {[v.value for v in cls]}"
            ) from exc


@dataclass(slots=True)
class RunConfig:
    """Runtime contract consumed by the JSON worker entrypoint."""

    run_id: str
    env_id: str
    seed: int = 0
    max_episodes: int = 1
    max_steps_per_episode: int = 200
    policy_strategy: PolicyStrategy = PolicyStrategy.EVAL
    policy_path: Optional[Path] = None
    agent_id: str = "bdi_rl"
    capture_video: bool = False
    headless: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunConfig":
        data = dict(payload)
        run_id = str(data.pop("run_id"))
        env_id = str(data.pop("env_id"))
        seed = int(data.pop("seed", 0))
        max_episodes = int(data.pop("max_episodes", 1))
        max_steps = int(data.pop("max_steps_per_episode", data.pop("max_steps", 200)))
        strategy = PolicyStrategy.from_any(data.pop("policy_strategy", None))

        policy_path_val = data.pop("policy_path", None)
        policy_path = Path(policy_path_val).expanduser().resolve() if policy_path_val else None

        agent_id = str(data.pop("agent_id", "bdi_rl"))
        capture_video = bool(data.pop("capture_video", False))
        headless = bool(data.pop("headless", True))

        return cls(
            run_id=run_id,
            env_id=env_id,
            seed=seed,
            max_episodes=max_episodes,
            max_steps_per_episode=max_steps,
            policy_strategy=strategy,
            policy_path=policy_path,
            agent_id=agent_id,
            capture_video=capture_video,
            headless=headless,
            extra=data,
        )

    def ensure_policy_path(self) -> Path:
        """Ensure a canonical policy path exists, creating parent dirs as needed."""

        if self.policy_path is None:
            base_dir = _default_policy_root()
            self.policy_path = base_dir / self.env_id / f"{self.agent_id}.json"
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        return self.policy_path


def _default_policy_root() -> Path:
    """Resolve the canonical policy directory relative to the repository root."""

    env_dir = os.environ.get("GYM_GUI_VAR_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve() / "policies"

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "var" / "trainer" / "policies").resolve()
