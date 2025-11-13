"""FastLane telemetry helpers for the CleanRL worker."""

from __future__ import annotations

import os
import time
import sys
from dataclasses import dataclass
from itertools import count
from multiprocessing import shared_memory
from time import perf_counter
from typing import Any, Optional

import numpy as np
import gymnasium as gym

try:  # pragma: no cover - relies on repo layout
    from gym_gui.fastlane import FastLaneWriter, FastLaneConfig, FastLaneMetrics
    from gym_gui.fastlane.buffer import create_fastlane_name
except ImportError:  # pragma: no cover - best effort fallback
    FastLaneWriter = None  # type: ignore
    FastLaneConfig = None  # type: ignore
    FastLaneMetrics = None  # type: ignore
    create_fastlane_name = None  # type: ignore


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
    try:
        slot = int(os.getenv("GYM_GUI_FASTLANE_SLOT", "0"))
    except ValueError:
        slot = 0
    try:
        total_envs = max(1, int(os.getenv("CLEANRL_NUM_ENVS", "1")))
    except ValueError:
        total_envs = 1
    slot = max(0, min(slot, total_envs - 1))
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
        self._last_emit_ns = 0
        self._throttle_interval_ns = int(float(os.getenv("CLEANRL_FASTLANE_INTERVAL_MS", "0")) * 1e6)
        self._downscale_max_dim = int(os.getenv("CLEANRL_FASTLANE_MAX_DIM", "0"))
        self._writer: Optional[FastLaneWriter] = None  # type: ignore[assignment]
        self._last_metrics_ts: Optional[float] = None
        self._debug_counter = 0
        self._log_debug(
            f"FastLane wrapper active={self._active} slot={slot_index}/{config.slot} run={config.run_id}"
        )

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

    def close(self):  # type: ignore[override]
        self._close_writer()
        return super().close()

    # ------------------------------------------------------------------
    def _emit_step(self, reward: float, terminated: bool, truncated: bool) -> None:
        frame_payload = self._grab_frame()
        now_ns = time.time_ns()
        if self._throttle_interval_ns and self._last_emit_ns:
            if now_ns - self._last_emit_ns < self._throttle_interval_ns:
                frame_payload = None
        if frame_payload is not None:
            self._last_emit_ns = now_ns
            self._publish_frame(frame_payload, reward)
        else:
            self._log_debug("FastLane frame skipped (throttled or render failed)", limit=5)

    def _emit_episode(self) -> None:
        # Metrics stream already tracks totals; no extra emission needed
        return None

    def _grab_frame(self) -> Optional[tuple[bytes, int, int, int]]:
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
        if self._downscale_max_dim and max(arr.shape[:2]) > self._downscale_max_dim:
            scale = self._downscale_max_dim / max(arr.shape[:2])
            new_h = max(1, int(arr.shape[0] * scale))
            new_w = max(1, int(arr.shape[1] * scale))
            arr = _resize_nearest(arr, new_h, new_w)
        arr = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        return (arr.tobytes(), arr.shape[1], arr.shape[0], arr.shape[2])

    def _publish_frame(self, frame: tuple[bytes, int, int, int], reward: float) -> None:
        if FastLaneWriter is None or FastLaneConfig is None or FastLaneMetrics is None:
            return
        frame_bytes, width, height, channels = frame
        writer = self._writer
        if writer is None:
            config = FastLaneConfig(
                width=width,
                height=height,
                channels=channels,
                pixel_format="RGB" if channels == 3 else "RGBA",
            )
            writer = self._create_writer(config)
            if writer is None:
                return
            self._writer = writer

        now = perf_counter()
        if self._last_metrics_ts is None:
            step_rate = 0.0
        else:
            delta = now - self._last_metrics_ts
            step_rate = 1.0 / delta if delta > 0 else 0.0
        self._last_metrics_ts = now

        metrics = FastLaneMetrics(
            last_reward=float(reward),
            rolling_return=float(self._episode_return + reward),
            step_rate_hz=step_rate,
        )
        try:
            writer.publish(frame_bytes, metrics=metrics)
        except Exception:
            self._close_writer()
        else:
            self._log_debug("FastLane frame published", limit=10)

    def _create_writer(self, config: Any) -> Optional[FastLaneWriter]:  # type: ignore[name-defined]
        try:
            return FastLaneWriter.create(self._config.run_id, config)  # type: ignore[union-attr]
        except FileExistsError:
            if create_fastlane_name is None:
                return None
            try:
                name = create_fastlane_name(self._config.run_id)
                shm = shared_memory.SharedMemory(name=name, create=False)
                return FastLaneWriter(shm, config)  # type: ignore[call-arg]
            except Exception:
                return None

    def _close_writer(self) -> None:
        writer = self._writer
        if writer is None:
            return
        try:
            writer.close()
        finally:
            self._writer = None
            self._log_debug("FastLane writer closed")

    def _log_debug(self, message: str, *, limit: int | None = None) -> None:
        if limit is not None and self._debug_counter >= limit:
            return
        self._debug_counter += 1
        print(f"[FASTLANE] {message}", file=sys.stderr, flush=True)


__all__ = ["is_fastlane_enabled", "maybe_wrap_env"]


def _resize_nearest(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize an HWC image via nearest-neighbour without pulling heavy deps."""

    h, w, c = arr.shape
    row_idx = (np.linspace(0, h - 1, new_h)).astype(np.int64)
    col_idx = (np.linspace(0, w - 1, new_w)).astype(np.int64)
    return arr[row_idx][:, col_idx]
