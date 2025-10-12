from __future__ import annotations

"""Runtime configuration management for the Gym GUI project."""

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable, TYPE_CHECKING
import yaml  # type: ignore[import-not-found]


from dotenv import load_dotenv

from gym_gui.core.enums import ControlMode

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
_STORAGE_PROFILE_FILE = _PROJECT_ROOT / "gym_gui" / "config" / "storage_profiles.yaml"

# Load environment variables from the project root if present. We do this once at
# import time so that any module importing settings has access to the values.
load_dotenv(_ENV_FILE, override=False)


def _normalize_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_control_mode(raw: str | None) -> ControlMode:
    if not raw:
        return ControlMode.HUMAN_ONLY
    candidate = raw.strip()
    try:
        return ControlMode[candidate.upper()]
    except KeyError:
        try:
            return ControlMode(candidate)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                f"Unsupported control mode '{raw}'. Expected one of: "
                f"{', '.join(mode.name for mode in ControlMode)}"
            ) from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Typed view over the environment configuration.
    
    Note: Game-specific configurations (like FrozenLake's is_slippery or
    Taxi's is_raining) are now in gym_gui.config.game_configs module.
    """

    qt_api: str = "pyqt6"
    gym_default_env: str = "FrozenLake-v1"
    gym_video_dir: Path | None = None
    enable_agent_autostart: bool = False
    log_level: str = "INFO"
    use_gpu: bool = False
    default_control_mode: ControlMode = ControlMode.HUMAN_ONLY
    agent_ids: tuple[str, ...] = field(default_factory=tuple)
    default_seed: int = 1
    allow_seed_reuse: bool = False

    @property
    def video_dir(self) -> Path | None:
        """Return the video directory if configured and ensure it exists."""

        if self.gym_video_dir is None:
            return None
        self.gym_video_dir.mkdir(parents=True, exist_ok=True)
        return self.gym_video_dir


def _resolve_video_dir(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    return Path(raw_path).expanduser().resolve()


_DEFAULT_SETTINGS = Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment variables and cache the result."""

    defaults = _DEFAULT_SETTINGS

    qt_api = os.getenv("QT_API", defaults.qt_api)
    gym_default_env = os.getenv("GYM_DEFAULT_ENV", defaults.gym_default_env)
    gym_video_dir = _resolve_video_dir(os.getenv("GYM_VIDEO_DIR"))
    enable_agent_autostart = _normalize_bool(
        os.getenv("ENABLE_AGENT_AUTOSTART"), default=defaults.enable_agent_autostart
    )
    log_level = os.getenv("GYM_LOG_LEVEL", os.getenv("LOG_LEVEL", defaults.log_level))
    use_gpu = _normalize_bool(os.getenv("USE_GPU"), default=defaults.use_gpu)
    default_control_mode = _parse_control_mode(
        os.getenv("DEFAULT_CONTROL_MODE")
        or os.getenv("GYM_CONTROL_MODE")
    )
    agent_ids = _split_csv(os.getenv("AGENT_IDS"))
    default_seed_raw = os.getenv("GYM_DEFAULT_SEED") or os.getenv("DEFAULT_SEED")
    try:
        default_seed = int(default_seed_raw) if default_seed_raw is not None else defaults.default_seed
    except (TypeError, ValueError):
        default_seed = defaults.default_seed
    allow_seed_reuse = _normalize_bool(
        os.getenv("GYM_ALLOW_SEED_REUSE") or os.getenv("ALLOW_SEED_REUSE"),
        default=defaults.allow_seed_reuse,
    )

    return Settings(
        qt_api=qt_api,
        gym_default_env=gym_default_env,
        gym_video_dir=gym_video_dir,
        enable_agent_autostart=enable_agent_autostart,
        log_level=log_level,
        use_gpu=use_gpu,
        default_control_mode=default_control_mode,
        agent_ids=agent_ids,
        default_seed=max(1, default_seed),
        allow_seed_reuse=allow_seed_reuse,
    )


def reload_settings() -> Settings:
    """Clear the settings cache and reload from the environment."""

    get_settings.cache_clear()
    load_dotenv(_ENV_FILE, override=True)
    return get_settings()


__all__ = ["Settings", "get_settings", "reload_settings"]


@lru_cache(maxsize=1)
def get_storage_profile_config() -> dict[str, dict[str, object]]:
    """Load raw storage profile settings from YAML configuration."""

    if not _STORAGE_PROFILE_FILE.exists():
        return {}

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load storage profiles. Run 'pip install -r requirements.txt'."
        )

    with _STORAGE_PROFILE_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("storage_profiles.yaml must define a mapping of profiles")

    normalized: dict[str, dict[str, object]] = {}
    for name, raw in data.items():
        if isinstance(raw, dict):
            normalized[name] = raw

    return normalized


__all__.append("get_storage_profile_config")
