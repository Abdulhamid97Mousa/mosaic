"""Structured configuration objects for the refactored SPADE-BDI worker."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import logging
from functools import partial

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_CONFIG_EVENT,
    LOG_WORKER_CONFIG_WARNING,
    LOG_WORKER_CONFIG_UI_PATH,
    LOG_WORKER_CONFIG_DURABLE_PATH,
)
from spade_bdi_rl.constants import (
    DEFAULT_STEP_DELAY_S,
    DEFAULT_WORKER_TELEMETRY_BUFFER_SIZE,
    DEFAULT_WORKER_EPISODE_BUFFER_SIZE,
)

_LOGGER = logging.getLogger(__name__)
_log = partial(log_constant, _LOGGER)


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
    game_id: str
    seed: int = 1  # CRITICAL: Seed must never be 0, must start from 1
    max_episodes: int = 1
    max_steps_per_episode: int = 200
    policy_strategy: PolicyStrategy = PolicyStrategy.EVAL
    policy_path: Optional[Path] = None
    agent_id: str = "bdi_rl"
    worker_id: Optional[str] = None
    capture_video: bool = False
    headless: bool = True
    step_delay: float = DEFAULT_STEP_DELAY_S  # Delay in seconds between training steps (for real-time observation)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunConfig":
        data = dict(payload)
        run_id = str(data.pop("run_id"))
        game_id = str(data.pop("game_id"))
        # CRITICAL: Seed must never be 0, must start from 1
        seed = max(1, int(data.pop("seed", 1)))
        max_episodes = int(data.pop("max_episodes", 1))
        max_steps = int(data.pop("max_steps_per_episode", data.pop("max_steps", 200)))
        strategy = PolicyStrategy.from_any(data.pop("policy_strategy", None))

        policy_path_val = data.pop("policy_path", None)
        policy_path = Path(policy_path_val).expanduser().resolve() if policy_path_val else None

        agent_id = str(data.pop("agent_id", "bdi_rl"))
        worker_id_val = data.pop("worker_id", None)
        worker_id = str(worker_id_val) if worker_id_val is not None else None
        capture_video = bool(data.pop("capture_video", False))
        headless = bool(data.pop("headless", True))
        step_delay = float(data.pop("step_delay", DEFAULT_STEP_DELAY_S))

        return cls(
            run_id=run_id,
            game_id=game_id,
            seed=seed,
            max_episodes=max_episodes,
            max_steps_per_episode=max_steps,
            policy_strategy=strategy,
            policy_path=policy_path,
            agent_id=agent_id,
            worker_id=worker_id,
            capture_video=capture_video,
            headless=headless,
            step_delay=step_delay,
            extra=data,
        )

    def __post_init__(self) -> None:
        """Log configuration details after initialization."""
        # CRITICAL: Enforce that seed is never 0
        original_seed = self.seed
        if self.seed < 1:
            object.__setattr__(self, "seed", 1)
            _log(
                LOG_WORKER_CONFIG_WARNING,
                message="Seed coerced to 1 (must never be 0)",
                extra={"original_seed": original_seed},
            )

        _log(
            LOG_WORKER_CONFIG_EVENT,
            message="RUN_CONFIG_LOADED",
            extra={
                "run_id": self.run_id,
                "game_id": self.game_id,
                "worker_id": self.worker_id,
                "seed": self.seed,
                "max_episodes": self.max_episodes,
                "max_steps_per_episode": self.max_steps_per_episode,
                "policy_strategy": self.policy_strategy.value,
                "agent_id": self.agent_id,
                "worker_id": self.worker_id,
                "capture_video": self.capture_video,
                "headless": self.headless,
                "step_delay_s": self.step_delay,
                "policy_path": str(self.policy_path) if self.policy_path else None,
                "extra_keys": list(self.extra.keys()),
            },
        )

        path_config = self.extra.get("path_config")
        if not isinstance(path_config, dict):
            _log(
                LOG_WORKER_CONFIG_WARNING,
                message="Missing path_config overrides; falling back to defaults",
                extra={"run_id": self.run_id},
            )
            path_config = {}

        ui_only_config = path_config.get("ui_only") if isinstance(path_config, dict) else None
        telemetry_config = path_config.get("telemetry_durable") if isinstance(path_config, dict) else None

        if not isinstance(ui_only_config, dict):
            ui_only_config = {}
        if not isinstance(telemetry_config, dict):
            telemetry_config = {}

        requested_step_delay_ms = ui_only_config.get("step_delay_ms")
        applied_step_delay_ms = int(round(self.step_delay * 1000))
        step_delay_mismatch = (
            requested_step_delay_ms is not None
            and abs(int(requested_step_delay_ms) - applied_step_delay_ms) > 5
        )

        _log(
            LOG_WORKER_CONFIG_UI_PATH,
            message="UI path settings applied",
            extra={
                "run_id": self.run_id,
                "live_rendering_enabled": bool(ui_only_config.get("live_rendering_enabled", True)),
                "ui_rendering_throttle": ui_only_config.get("ui_rendering_throttle"),
                "render_delay_ms": ui_only_config.get("render_delay_ms"),
                "requested_step_delay_ms": requested_step_delay_ms,
                "applied_step_delay_ms": applied_step_delay_ms,
                "step_delay_mismatch": step_delay_mismatch,
            },
        )

        telemetry_buffer_config = telemetry_config.get("telemetry_buffer_size")
        telemetry_buffer_applied = self.extra.get(
            "telemetry_buffer_size",
            DEFAULT_WORKER_TELEMETRY_BUFFER_SIZE,
        )
        episode_buffer_config = telemetry_config.get("episode_buffer_size")
        episode_buffer_applied = self.extra.get(
            "episode_buffer_size",
            DEFAULT_WORKER_EPISODE_BUFFER_SIZE,
        )

        _log(
            LOG_WORKER_CONFIG_DURABLE_PATH,
            message="Telemetry durable path settings applied",
            extra={
                "run_id": self.run_id,
                "training_telemetry_throttle": telemetry_config.get("training_telemetry_throttle"),
                "telemetry_buffer_requested": telemetry_buffer_config,
                "telemetry_buffer_applied": telemetry_buffer_applied,
                "episode_buffer_requested": episode_buffer_config,
                "episode_buffer_applied": episode_buffer_applied,
            },
        )

        if step_delay_mismatch:
            _log(
                LOG_WORKER_CONFIG_WARNING,
                message="Step delay mismatch between UI request and applied configuration",
                extra={
                    "run_id": self.run_id,
                    "requested_step_delay_ms": requested_step_delay_ms,
                    "applied_step_delay_ms": applied_step_delay_ms,
                },
            )

    def ensure_policy_path(self) -> Path:
        """Ensure a canonical policy path exists, creating parent dirs as needed."""

        if self.policy_path is None:
            base_dir = _default_policy_root()
            self.policy_path = base_dir / self.game_id / f"{self.agent_id}.json"
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        return self.policy_path


def _default_policy_root() -> Path:
    """Resolve the canonical policy directory relative to the repository root."""

    env_dir = os.environ.get("GYM_GUI_VAR_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve() / "policies"

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "var" / "trainer" / "policies").resolve()
