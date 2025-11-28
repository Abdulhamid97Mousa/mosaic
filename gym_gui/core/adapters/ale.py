"""ALE (Atari Learning Environment) adapters.

This module provides adapters for Atari 2600 environments via ALE. It supports
configurable observation types (rgb/ram/grayscale), frameskip, repeat-action
probability (RAP), and flavour parameters (difficulty/mode). The adapter emits
structured :class:`AdapterStep` payloads with enriched :class:`StepState` that
surface key runtime traits such as remaining lives and active configuration.

Initial coverage focuses on Adventure with the following variants:

- ``Adventure-v4`` (rgb, frameskip=(2,5), RAP=0.0)
- ``ALE/Adventure-v5`` (rgb, frameskip=4, RAP=0.25)

Other Atari titles can be added following this pattern.
"""

from __future__ import annotations

from typing import Any, Mapping

import logging

import numpy as np

from gym_gui.core.adapters.base import EnvironmentAdapter, AdapterStep, StepState
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
	LOG_ADAPTER_ALE_METADATA_PROBE_FAILED,
	LOG_ADAPTER_ALE_NAMESPACE_IMPORT_FAILED,
)

_LOGGER = logging.getLogger(__name__)
try:  # pragma: no cover - avoid hard dependency in non-ALE test runs
	from gym_gui.config.game_configs import ALEConfig  # type: ignore
except Exception:  # pragma: no cover - soft fallback if config not yet defined
	ALEConfig = None  # type: ignore[assignment]

# Ensure ALE namespace is registered with Gymnasium when this module is imported.
# Many installations of gymnasium + ale-py do not auto-register entry points,
# but importing ale_py.env wires up the "ALE/" namespace.
try:  # pragma: no cover - environment dependent
    import ale_py.env  # type: ignore
except Exception:
	# Leave registration to runtime if ALE isn't available; adapter will still raise clearly on load.
	_LOGGER = logging.getLogger(__name__)
	log_constant(
		_LOGGER,
		LOG_ADAPTER_ALE_NAMESPACE_IMPORT_FAILED,
		extra={
			"module": "ale_py.env",
		},
	)


class ALEAdapter(EnvironmentAdapter[np.ndarray, int]):
	"""Adapter for ALE environments with configurable observations.

	The adapter mirrors the xuance Atari wrapper's knobs where sensible for a GUI
	context. It does not apply training wrappers (e.g., frame-stacking) but forwards
	parameters to Gymnasium and exposes telemetry-friendly metadata.
	"""

	default_render_mode = RenderMode.RGB_ARRAY
	supported_render_modes = (RenderMode.RGB_ARRAY,)
	supported_control_modes = (ControlMode.AGENT_ONLY,)

	DEFAULT_ENV_ID = GameId.ADVENTURE_V4.value

	def __init__(
		self,
		context: Any | None = None,
		*,
		config: Any | None = None,
	) -> None:
		super().__init__(context)
		# Accept a typed ALEConfig when available; tolerate plain dict-like for flexibility
		self._config = config
		cfg_id = getattr(config, "env_id", None) if config is not None else None  # type: ignore[attr-defined]
		# Resolve the class-declared default id without invoking the instance property
		default_id = getattr(type(self), "id", None)
		if not isinstance(default_id, str) or not default_id:
			default_id = GameId.ALE_ADVENTURE_V5.value
		# Default to the adapter's declared id (variant) when config doesn't provide one
		self._env_id: str = cfg_id or default_id
		self._step_counter = 0
		self._episode_return = 0.0
		self._lives_probe_failed_once = False

	# ------------------------------------------------------------------
	# Lifecycle hooks
	# ------------------------------------------------------------------

	@property
	def id(self) -> str:  # type: ignore[override]
		return self._env_id

	def gym_kwargs(self) -> dict[str, Any]:
		kwargs: dict[str, Any] = {}
		# Pull values from config if present; otherwise supply sane variant defaults
		cfg = self._config

		def _get(name: str, default: Any = None) -> Any:
			if cfg is None:
				return default
			return getattr(cfg, name, default)

		obs_type = _get("obs_type", "rgb")
		frameskip = _get("frameskip", None)
		rap = _get("repeat_action_probability", None)
		difficulty = _get("difficulty", None)
		mode = _get("mode", None)
		full_action_space = _get("full_action_space", False)
		render_mode = _get("render_mode", self.default_render_mode.value)

		# Apply variant-specific defaults when values are unspecified
		if frameskip is None or rap is None:
			v_frameskip, v_rap = self._variant_defaults(self._env_id)
			frameskip = v_frameskip if frameskip is None else frameskip
			rap = v_rap if rap is None else rap

		# Construct kwargs
		kwargs["render_mode"] = render_mode
		kwargs["obs_type"] = obs_type
		if frameskip is not None:
			kwargs["frameskip"] = frameskip
		if rap is not None:
			kwargs["repeat_action_probability"] = float(rap)
		if difficulty is not None:
			kwargs["difficulty"] = int(difficulty)
		if mode is not None:
			kwargs["mode"] = int(mode)
		if bool(full_action_space):
			kwargs["full_action_space"] = True

		return kwargs

	def render(self) -> dict[str, Any]:
		frame = super().render()
		array = np.asarray(frame)
		return {
			"mode": RenderMode.RGB_ARRAY.value,
			"rgb": array,
			"game_id": self._env_id,
		}

	def reset(
		self,
		*,
		seed: int | None = None,
		options: dict[str, Any] | None = None,
	) -> AdapterStep[np.ndarray]:
		step = super().reset(seed=seed, options=options)
		self._step_counter = 0
		self._episode_return = 0.0
		return step

	def step(self, action: int) -> AdapterStep[np.ndarray]:
		step = super().step(action)
		self._step_counter += 1
		self._episode_return += float(step.reward)
		return step

	# ------------------------------------------------------------------
	# Adapter customisations
	# ------------------------------------------------------------------

	def build_step_state(self, observation: np.ndarray, info: Mapping[str, Any]) -> StepState:
		# Snapshot ALE-specific metadata when available
		env_meta: dict[str, Any] = {"env_id": self._env_id}

		# Include configuration echoes for transparency
		kwargs = self.gym_kwargs()
		env_meta.update({
			"obs_type": kwargs.get("obs_type"),
			"frameskip": kwargs.get("frameskip"),
			"repeat_action_probability": kwargs.get("repeat_action_probability"),
			"difficulty": kwargs.get("difficulty"),
			"mode": kwargs.get("mode"),
			"full_action_space": bool(kwargs.get("full_action_space", False)),
		})

		metrics: dict[str, Any] = {
			"step": self._step_counter,
			"episode_return": float(self._episode_return),
		}

		# Probe remaining lives via ALE API if present
		try:  # pragma: no cover - exercised in integration, guarded for portability
			env = self._require_env()
			base = getattr(env, "unwrapped", env)
			ale = getattr(base, "ale", None)
			if ale is not None and hasattr(ale, "lives"):
				lives = int(ale.lives())
				metrics["lives"] = lives
		except Exception as exc:
			# Best effort only: log once to avoid noise if underlying API doesn't expose lives
			if not getattr(self, "_lives_probe_failed_once", False):
				self._lives_probe_failed_once = True
				self.log_constant(
					LOG_ADAPTER_ALE_METADATA_PROBE_FAILED,
					exc_info=exc,
					extra={
						"env_id": self._env_id,
						"context": "lives_probe",
					},
				)

		# Attach raw info block for downstream consumers
		raw = dict(info) if isinstance(info, Mapping) else {}

		return StepState(metrics=metrics, environment=env_meta, raw=raw)

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------

	@staticmethod
	def _variant_defaults(env_id: str) -> tuple[Any, float]:
		"""Return (frameskip, repeat_action_probability) defaults by variant.

		Values follow ALE docs:
		- v4: frameskip=(2, 5), RAP=0.0
		- v5: frameskip=4, RAP=0.25
		"""
		if env_id.endswith("-v4") and not env_id.startswith("ALE/"):
			return (2, 5), 0.0
		if env_id.startswith("ALE/") and env_id.endswith("-v5"):
			return 4, 0.25
		# Fallback to ALE defaults
		return 4, 0.25


class AdventureV4Adapter(ALEAdapter):
	"""Adapter for Adventure-v4 (classic Atari v4 variant)."""

	id = GameId.ADVENTURE_V4.value
	supported_control_modes = (
		ControlMode.HUMAN_ONLY,
		ControlMode.AGENT_ONLY,
		ControlMode.HYBRID_TURN_BASED,
		ControlMode.HYBRID_HUMAN_AGENT,
	)


class AdventureV5Adapter(ALEAdapter):
	"""Adapter for ALE/Adventure-v5 (namespaced ALE v5 variant)."""

	id = GameId.ALE_ADVENTURE_V5.value
	supported_control_modes = (
		ControlMode.HUMAN_ONLY,
		ControlMode.AGENT_ONLY,
		ControlMode.HYBRID_TURN_BASED,
		ControlMode.HYBRID_HUMAN_AGENT,
	)


class AirRaidV4Adapter(ALEAdapter):
	"""Adapter for AirRaid-v4."""

	id = GameId.AIR_RAID_V4.value
	default_render_mode = RenderMode.RGB_ARRAY
	supported_control_modes = (
	    ControlMode.HUMAN_ONLY,
	    ControlMode.AGENT_ONLY,
	    ControlMode.HYBRID_TURN_BASED,
	    ControlMode.HYBRID_HUMAN_AGENT,
	)


class AirRaidV5Adapter(ALEAdapter):
	"""Adapter for ALE/AirRaid-v5."""

	id = GameId.ALE_AIR_RAID_V5.value
	default_render_mode = RenderMode.RGB_ARRAY
	supported_control_modes = (
	    ControlMode.HUMAN_ONLY,
	    ControlMode.AGENT_ONLY,
	    ControlMode.HYBRID_TURN_BASED,
	    ControlMode.HYBRID_HUMAN_AGENT,
	)


class AssaultV4Adapter(ALEAdapter):
	"""Adapter for Assault-v4."""

	id = GameId.ASSAULT_V4.value
	default_render_mode = RenderMode.RGB_ARRAY
	supported_control_modes = (
		ControlMode.HUMAN_ONLY,
		ControlMode.AGENT_ONLY,
		ControlMode.HYBRID_TURN_BASED,
		ControlMode.HYBRID_HUMAN_AGENT,
	)


class AssaultV5Adapter(ALEAdapter):
	"""Adapter for ALE/Assault-v5."""

	id = GameId.ALE_ASSAULT_V5.value
	default_render_mode = RenderMode.RGB_ARRAY
	supported_control_modes = (
		ControlMode.HUMAN_ONLY,
		ControlMode.AGENT_ONLY,
		ControlMode.HYBRID_TURN_BASED,
		ControlMode.HYBRID_HUMAN_AGENT,
	)


ALE_ADAPTERS: dict[GameId, type[ALEAdapter]] = {
	GameId.ADVENTURE_V4: AdventureV4Adapter,
	GameId.ALE_ADVENTURE_V5: AdventureV5Adapter,
	GameId.AIR_RAID_V4: AirRaidV4Adapter,
	GameId.ALE_AIR_RAID_V5: AirRaidV5Adapter,
	GameId.ASSAULT_V4: AssaultV4Adapter,
	GameId.ALE_ASSAULT_V5: AssaultV5Adapter,
}


__all__ = [
	"ALEAdapter",
	"AdventureV4Adapter",
	"AdventureV5Adapter",
	"AirRaidV4Adapter",
	"AirRaidV5Adapter",
    "AssaultV4Adapter",
    "AssaultV5Adapter",
	"ALE_ADAPTERS",
]
