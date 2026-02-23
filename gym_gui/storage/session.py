from __future__ import annotations

"""Lightweight session recording utilities used by StorageRecorderService."""

from dataclasses import dataclass, field
from datetime import datetime
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import logging

from gym_gui.constants.constants_telemetry import (
    TELEMETRY_KEY_AUTORESET_MODE,
    TELEMETRY_KEY_SPACE_SIGNATURE,
    TELEMETRY_KEY_TIME_STEP,
    TELEMETRY_KEY_VECTOR_METADATA,
)
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_SESSION_NUMPY_SCALAR_COERCE_FAILED,
    LOG_SERVICE_SESSION_NDARRAY_SUMMARY_FAILED,
    LOG_SERVICE_SESSION_LAZYFRAMES_SUMMARY_FAILED,
    LOG_SERVICE_SESSION_TOLIST_COERCE_FAILED,
    LOG_SERVICE_SESSION_ITERABLE_COERCE_FAILED,
)


@dataclass(slots=True)
class EpisodeRecord:
    """Single step snapshot captured for persistence."""

    episode_id: str
    step_index: int
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_id: str | None = None
    frame_ref: str | None = None
    payload_version: int = 0
    run_id: str | None = None
    worker_id: str | None = None
    time_step: int | None = None
    space_signature: Mapping[str, Any] | None = None
    vector_metadata: Mapping[str, Any] | None = None


class SessionRecorder(LogConstantMixin):
    """Write episode data to disk while honoring retention rules."""

    def __init__(
        self,
        base_dir: Path,
        *,
        ring_buffer_limit: int,
        retain_episodes: int,
        compress_frames: bool = False,
        telemetry_only: bool = False,
    ) -> None:
        self._base_dir = base_dir
        self._ring_buffer_limit = max(1, int(ring_buffer_limit))
        self._retain_episodes = max(1, int(retain_episodes))
        self._compress_frames = compress_frames
        self._telemetry_only = telemetry_only
        self._steps: List[EpisodeRecord] = []
        self._current_episode_id: str | None = None
        # LogConstantMixin requires an instance logger
        self._logger = logging.getLogger(__name__)
        # One-time log flags to avoid flooding on repeated normalization paths
        self._log_once_flags: dict[str, bool] = {
            "np_scalar": False,
            "ndarray": False,
            "lazyframes": False,
            "tolist": False,
            "iterable": False,
        }

    # ------------------------------------------------------------------
    def write_step(self, record: EpisodeRecord) -> None:
        if self._current_episode_id is None:
            self._current_episode_id = record.episode_id
        elif self._current_episode_id != record.episode_id:
            # Episode changed, flush current buffer first
            self.finalize_episode()
            self._current_episode_id = record.episode_id
            self._steps.clear()

        self._steps.append(record)
        if len(self._steps) > self._ring_buffer_limit:
            self._steps.pop(0)

    # ------------------------------------------------------------------
    def finalize_episode(self) -> None:
        if self._current_episode_id is None:
            return

        # If buffer is empty, don't write anything (will be written from database)
        if not self._steps:
            return

        episode_dir = self._base_dir / self._current_episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        payload = [self._serialize_step(step) for step in self._steps]
        output_file = episode_dir / "episode.jsonl"
        with output_file.open("w", encoding="utf-8") as handle:
            for entry in payload:
                handle.write(json.dumps(entry) + "\n")

        self._prune_old_episodes()
        self._steps.clear()
        self._current_episode_id = None

    def write_episode_from_steps(self, episode_id: str, steps: list[EpisodeRecord]) -> None:
        """Write episode JSONL file from a list of steps (used when buffer is empty).

        This method is called when the in-memory buffer is empty but we have steps
        from the database that need to be written to the JSONL file.
        """
        if not steps:
            return

        episode_dir = self._base_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        payload = [self._serialize_step(step) for step in steps]
        output_file = episode_dir / "episode.jsonl"
        with output_file.open("w", encoding="utf-8") as handle:
            for entry in payload:
                handle.write(json.dumps(entry) + "\n")

        self._prune_old_episodes()

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.finalize_episode()

    # ------------------------------------------------------------------
    def _serialize_step(self, step: EpisodeRecord) -> dict[str, Any]:
        data = {
            "step_index": step.step_index,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "timestamp": step.timestamp.isoformat(),
            "info": self._json_safe(step.info),
        }
        if not self._telemetry_only:
            data["observation"] = self._json_safe(step.observation)
        if step.agent_id is not None:
            data["agent_id"] = step.agent_id
        if step.frame_ref is not None:
            data["frame_ref"] = step.frame_ref
        if step.payload_version:
            data["payload_version"] = int(step.payload_version)
        if step.run_id is not None:
            data["run_id"] = step.run_id
        if step.worker_id is not None:
            data["worker_id"] = step.worker_id
        if step.time_step is not None:
            data[TELEMETRY_KEY_TIME_STEP] = int(step.time_step)
        if step.space_signature is not None:
            data[TELEMETRY_KEY_SPACE_SIGNATURE] = self._json_safe(step.space_signature)
        if step.vector_metadata is not None:
            vector_payload = self._json_safe(step.vector_metadata)
            data[TELEMETRY_KEY_VECTOR_METADATA] = vector_payload
            if (
                isinstance(step.vector_metadata, Mapping)
                and TELEMETRY_KEY_AUTORESET_MODE in step.vector_metadata
                and TELEMETRY_KEY_AUTORESET_MODE not in data
            ):
                data[TELEMETRY_KEY_AUTORESET_MODE] = step.vector_metadata[TELEMETRY_KEY_AUTORESET_MODE]
        if self._compress_frames:
            # Placeholder hook for future compression
            data["compressed"] = False
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log_once(self, key: str, constant, *, message: str | None = None, extra: dict | None = None, exc: BaseException | None = None) -> None:
        """Emit a structured log once per key to avoid flooding."""
        if not self._log_once_flags.get(key):
            self.log_constant(constant, message=message, extra=extra or {}, exc_info=exc)
            self._log_once_flags[key] = True

    def _json_safe(self, value: Any, *, _depth: int = 0) -> Any:
        """Best-effort conversion of arbitrary objects to JSON serialisable data."""

        if _depth > 10:
            return repr(value)

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, datetime):
            return value.isoformat()

        # Normalize NumPy scalar values to Python primitives
        try:
            if isinstance(value, np.generic):  # np.integer / np.floating / np.bool_
                return value.item()
        except Exception as exc:  # pragma: no cover - defensive when numpy is absent
            self._log_once(
                "np_scalar",
                LOG_SERVICE_SESSION_NUMPY_SCALAR_COERCE_FAILED,
                extra={"value_type": getattr(type(value), "__name__", str(type(value)))},
                exc=exc,
            )

        # Summarize NumPy arrays to avoid exploding payload sizes in JSONL
        # We intentionally do not inline raw pixel data here; frames are stored via frame_ref when enabled.
        try:
            if isinstance(value, np.ndarray):
                try:
                    nbytes = int(value.nbytes)
                except Exception:
                    nbytes = None
                return {
                    "__type__": "ndarray_summary",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    **({"nbytes": nbytes} if nbytes is not None else {}),
                }
        except Exception as exc:  # pragma: no cover - defensive
            self._log_once(
                "ndarray",
                LOG_SERVICE_SESSION_NDARRAY_SUMMARY_FAILED,
                extra={"value_type": getattr(type(value), "__name__", str(type(value)))},
                exc=exc,
            )

        # Recognize common LazyFrames wrappers and summarize
        try:
            cls_name = getattr(value, "__class__", type(value)).__name__
            if cls_name == "LazyFrames":
                # Try to derive shape/dtype without materializing the stacked array
                frame_shape = getattr(value, "frame_shape", None)
                shape = getattr(value, "shape", None)
                dtype = getattr(value, "dtype", None)
                frames = getattr(value, "_frames", None)
                if shape is None and isinstance(frames, list) and frames:
                    first = frames[0]
                    try:
                        h, w = first.shape[:2]
                        # Assume stack on first axis if gymnasium's LazyFrames isn't present
                        shape = (len(frames),) + tuple(getattr(first, "shape", (h, w)))
                    except Exception:
                        shape = (len(frames),)
                    dtype = getattr(first, "dtype", dtype)
                return {
                    "__type__": "lazyframes_summary",
                    **({"frame_shape": list(frame_shape)} if isinstance(frame_shape, tuple) else {}),
                    **({"shape": list(shape)} if isinstance(shape, tuple) else {}),
                    **({"dtype": str(dtype)} if dtype is not None else {}),
                }
        except Exception as exc:  # pragma: no cover - defensive
            self._log_once(
                "lazyframes",
                LOG_SERVICE_SESSION_LAZYFRAMES_SUMMARY_FAILED,
                extra={"value_type": getattr(type(value), "__name__", str(type(value)))},
                exc=exc,
            )

        if isinstance(value, Mapping):
            return {
                str(key): self._json_safe(item, _depth=_depth + 1)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item, _depth=_depth + 1) for item in value]

        # Avoid value.tolist() for large arrays; handled by ndarray_summary above.
        if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
            try:
                return value.tolist()
            except TypeError as exc:
                self._log_once(
                    "tolist",
                    LOG_SERVICE_SESSION_TOLIST_COERCE_FAILED,
                    extra={"value_type": getattr(type(value), "__name__", str(type(value)))},
                    exc=exc,
                )

        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            try:
                return [self._json_safe(item, _depth=_depth + 1) for item in list(value)]
            except TypeError as exc:
                self._log_once(
                    "iterable",
                    LOG_SERVICE_SESSION_ITERABLE_COERCE_FAILED,
                    extra={"value_type": getattr(type(value), "__name__", str(type(value)))},
                    exc=exc,
                )

        return repr(value)

    def _prune_old_episodes(self) -> None:
        existing: Sequence[Path] = sorted(
            (path for path in self._base_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for folder in existing[self._retain_episodes :]:
            for child in folder.iterdir():
                child.unlink()
            folder.rmdir()


__all__ = ["EpisodeRecord", "SessionRecorder"]
