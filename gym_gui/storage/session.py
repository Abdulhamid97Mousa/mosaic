from __future__ import annotations

"""Lightweight session recording utilities used by StorageRecorderService."""

from dataclasses import dataclass, field
from datetime import datetime
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable, List


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



def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Best-effort conversion of arbitrary objects to JSON serialisable data."""

    if _depth > 10:
        return repr(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, _depth=_depth + 1)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, _depth=_depth + 1) for item in value]

    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return value.tolist()
        except TypeError:
            pass

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        try:
            return [_json_safe(item, _depth=_depth + 1) for item in list(value)]
        except TypeError:
            pass

    return repr(value)


class SessionRecorder:
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
            "info": _json_safe(step.info),
        }
        if not self._telemetry_only:
            data["observation"] = _json_safe(step.observation)
        if self._compress_frames:
            # Placeholder hook for future compression
            data["compressed"] = False
        return data

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
