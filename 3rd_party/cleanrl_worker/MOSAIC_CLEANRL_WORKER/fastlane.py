"""FastLane telemetry helpers for the CleanRL worker."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from itertools import count
from typing import Any, Optional

import numpy as np
import gymnasium as gym


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return False


def is_fastlane_enabled() -> bool:
    return _truthy(os.getenv("GYM_GUI_FASTLANE_ONLY"))


@dataclass(frozen=True)
class _FastLaneConfig:
    enabled: bool
    slot: int
    run_id: str
    agent_id: str
    worker_id: str
    seed: int


def _resolve_config() -> _FastLaneConfig:
    enabled = is_fastlane_enabled()
    slot = int(os.getenv("GYM_GUI_FASTLANE_SLOT", "0"))
    run_id = os.getenv("CLEANRL_RUN_ID") or os.getenv("RUN_ID") or "cleanrl-run"
    agent_id = os.getenv("CLEANRL_AGENT_ID") or os.getenv("AGENT_ID") or "cleanrl-agent"
    worker_id = (
        os.getenv("WORKER_ID")
        or os.getenv("CLEANRL_WORKER_ID")
        or os.getenv("CLEANRL_AGENT_ID")
        or "cleanrl-worker"
    )
    try:
        seed = int(os.getenv("CLEANRL_SEED", "0"))
    except ValueError:
        seed = 0
    return _FastLaneConfig(enabled=enabled, slot=slot, run_id=run_id, agent_id=agent_id, worker_id=worker_id, seed=seed)


_CONFIG = _resolve_config()
_ENV_SLOT_COUNTER = count()


def maybe_wrap_env(env: Any) -> Any:
    """Attach the FastLane wrapper if fastlane streaming is enabled."""

    if not _CONFIG.enabled:
        return env
    slot_index = next(_ENV_SLOT_COUNTER)
    return FastLaneTelemetryWrapper(env, _CONFIG, slot_index)


class FastLaneTelemetryWrapper(gym.Wrapper):
    """Gym wrapper that emits FastLane-ready telemetry on every step."""

    def __init__(self, env: gym.Env, config: _FastLaneConfig, slot_index: int) -> None:
        super().__init__(env)
        self._config = config
        self._slot = slot_index
        self._active = config.enabled and slot_index == config.slot
        self._episode_index = 0
        self._step_index = 0
        self._episode_return = 0.0
        self._stdout = sys.stdout

    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = self.env.reset(*args, **kwargs)
        self._step_index = 0
        self._episode_return = 0.0
        return result

    def step(self, action: Any):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._active:
            self._emit_step(float(reward), bool(terminated), bool(truncated))
            self._step_index += 1
            self._episode_return += float(reward)
            if terminated or truncated:
                self._emit_episode()
                self._episode_index += 1
                self._step_index = 0
                self._episode_return = 0.0
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _emit_step(self, reward: float, terminated: bool, truncated: bool) -> None:
        frame_payload = self._grab_frame()
        event: dict[str, Any] = {
            "type": "step",
            "run_id": self._config.run_id,
            "agent_id": self._config.agent_id,
            "worker_id": self._config.worker_id,
            "episode": int(self._episode_index + self._config.seed),
            "episode_index": int(self._episode_index),
            "step": int(self._step_index),
            "step_index": int(self._step_index),
            "reward": float(reward),
            "total_reward": float(self._episode_return + reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "metadata": {"episode_index": int(self._episode_index)},
            "ts_unix_ns": time.time_ns(),
        }
        if frame_payload is not None:
            event["render_payload"] = frame_payload
        self._write_event(event)

    def _emit_episode(self) -> None:
        event = {
            "type": "episode",
            "run_id": self._config.run_id,
            "agent_id": self._config.agent_id,
            "worker_id": self._config.worker_id,
            "episode": int(self._episode_index + self._config.seed),
            "episode_index": int(self._episode_index),
            "total_reward": float(self._episode_return),
            "steps": int(self._step_index),
            "metadata": {"episode_index": int(self._episode_index)},
            "ts_unix_ns": time.time_ns(),
        }
        self._write_event(event)

    def _grab_frame(self) -> Optional[dict[str, Any]]:
        try:
            frame = self.env.render()
        except Exception:
            return None
        if isinstance(frame, (tuple, list)) and frame:
            frame = frame[0]
        if frame is None:
            return None
        arr = None
        if isinstance(frame, np.ndarray):
            arr = frame
        elif hasattr(frame, "__array__"):
            try:
                arr = np.asarray(frame)
            except Exception:
                arr = None
        if arr is None:
            return None
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        if arr.ndim != 3:
            return None
        return {"rgb": arr.astype(np.uint8).tolist()}

    def _write_event(self, payload: dict[str, Any]) -> None:
        try:
            json.dump(payload, self._stdout, separators=(",", ":"))
            self._stdout.write("\n")
            self._stdout.flush()
        except Exception:
            pass


__all__ = ["is_fastlane_enabled", "maybe_wrap_env"]
