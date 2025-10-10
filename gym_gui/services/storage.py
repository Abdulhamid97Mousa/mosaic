from __future__ import annotations

"""Storage recorder service wiring episode/session persistence."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gym_gui.config.settings import get_storage_profile_config
from gym_gui.storage.session import EpisodeRecord, SessionRecorder


@dataclass(slots=True)
class StorageProfile:
    name: str
    strategy: str = "ndjson"
    ring_buffer_size: int = 1024
    retain_episodes: int = 10
    compress_frames: bool = False
    telemetry_only: bool = False
    telemetry_store: str = "jsonl"
    extras: dict[str, Any] = field(default_factory=dict)


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_EPISODE_OUTPUT_DIR = _PACKAGE_ROOT / "runtime" / "data" / "episodes"


def _as_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


class StorageRecorderService:
    """Coordinates session recorders and storage profiles."""

    def __init__(self) -> None:
        self._profiles: Dict[str, StorageProfile] = {}
        self._active_profile: Optional[str] = None
        self._recorder: Optional[SessionRecorder] = None

    # ------------------------------------------------------------------
    def ensure_profiles_loaded(self) -> None:
        if self._profiles:
            return
        config = get_storage_profile_config()
        for name, payload in config.items():
            strategy = str(payload.get("strategy", "ndjson"))
            ring_buffer = _as_int(payload.get("ring_buffer"), default=1024)
            retain_source = payload.get("retain")
            if retain_source is None:
                retain_source = payload.get("max_sessions" if strategy == "sqlite_blob" else "retain")
            retain = _as_int(retain_source, default=10)
            telemetry_only = strategy == "telemetry_only" or _as_bool(payload.get("telemetry_only"))
            compress_frames = _as_bool(
                payload.get("compress_frames") or (payload.get("frame_compression") == "png")
            )
            telemetry_store = str(payload.get("telemetry_store", "jsonl"))
            tracked_keys = {
                "strategy",
                "ring_buffer",
                "retain",
                "max_sessions",
                "compress_frames",
                "frame_compression",
                "telemetry_only",
                "telemetry_store",
            }
            extras = {k: v for k, v in payload.items() if k not in tracked_keys}

            self._profiles[name] = StorageProfile(
                name=name,
                strategy=strategy,
                ring_buffer_size=ring_buffer,
                retain_episodes=retain,
                compress_frames=compress_frames,
                telemetry_only=telemetry_only,
                telemetry_store=telemetry_store,
                extras=extras,
            )

    def available_profiles(self) -> Iterable[str]:
        self.ensure_profiles_loaded()
        return self._profiles.keys()

    def set_active_profile(self, profile_name: str) -> None:
        self.ensure_profiles_loaded()
        if profile_name not in self._profiles:
            raise KeyError(f"Unknown storage profile '{profile_name}'")
        self._active_profile = profile_name

    def get_active_profile(self) -> StorageProfile:
        self.ensure_profiles_loaded()
        if self._active_profile is None:
            self._active_profile = next(iter(self._profiles))
        return self._profiles[self._active_profile]

    # ------------------------------------------------------------------
    def _ensure_recorder(self) -> SessionRecorder:
        if self._recorder is not None:
            return self._recorder
        base_dir = _EPISODE_OUTPUT_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        profile = self.get_active_profile()
        self._recorder = SessionRecorder(
            base_dir=base_dir,
            ring_buffer_limit=profile.ring_buffer_size,
            retain_episodes=profile.retain_episodes,
            compress_frames=profile.compress_frames,
            telemetry_only=profile.telemetry_only,
        )
        return self._recorder

    def record_step(self, record: EpisodeRecord) -> None:
        recorder = self._ensure_recorder()
        recorder.write_step(record)

    def flush_episode(self) -> None:
        recorder = self._ensure_recorder()
        recorder.finalize_episode()

    def reset_session(self) -> None:
        if self._recorder:
            self._recorder.close()
        self._recorder = None


__all__ = [
    "StorageRecorderService",
    "StorageProfile",
]
