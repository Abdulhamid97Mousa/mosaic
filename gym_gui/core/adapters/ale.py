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
# Modern ale-py (0.11+) requires explicit registration with gymnasium.
try:  # pragma: no cover - environment dependent
    import ale_py
    import gymnasium
    gymnasium.register_envs(ale_py)
except Exception:
	# Leave registration to runtime if ALE isn't available; adapter will still raise clearly on load.
	_LOGGER = logging.getLogger(__name__)
	log_constant(
		_LOGGER,
		LOG_ADAPTER_ALE_NAMESPACE_IMPORT_FAILED,
		extra={
			"module": "ale_py",
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


class AlienV5Adapter(ALEAdapter):
	"""Adapter for ALE/Alien-v5."""
	id = GameId.ALE_ALIEN_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class AmidarV5Adapter(ALEAdapter):
	"""Adapter for ALE/Amidar-v5."""
	id = GameId.ALE_AMIDAR_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class AsterixV5Adapter(ALEAdapter):
	"""Adapter for ALE/Asterix-v5."""
	id = GameId.ALE_ASTERIX_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class AsteroidsV5Adapter(ALEAdapter):
	"""Adapter for ALE/Asteroids-v5."""
	id = GameId.ALE_ASTEROIDS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class AtlantisV5Adapter(ALEAdapter):
	"""Adapter for ALE/Atlantis-v5."""
	id = GameId.ALE_ATLANTIS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BackgammonV5Adapter(ALEAdapter):
	"""Adapter for ALE/Backgammon-v5."""
	id = GameId.ALE_BACKGAMMON_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BankHeistV5Adapter(ALEAdapter):
	"""Adapter for ALE/BankHeist-v5."""
	id = GameId.ALE_BANK_HEIST_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BasicMathV5Adapter(ALEAdapter):
	"""Adapter for ALE/BasicMath-v5."""
	id = GameId.ALE_BASIC_MATH_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BattleZoneV5Adapter(ALEAdapter):
	"""Adapter for ALE/BattleZone-v5."""
	id = GameId.ALE_BATTLE_ZONE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BeamRiderV5Adapter(ALEAdapter):
	"""Adapter for ALE/BeamRider-v5."""
	id = GameId.ALE_BEAM_RIDER_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BerzerkV5Adapter(ALEAdapter):
	"""Adapter for ALE/Berzerk-v5."""
	id = GameId.ALE_BERZERK_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BlackjackV5Adapter(ALEAdapter):
	"""Adapter for ALE/Blackjack-v5."""
	id = GameId.ALE_BLACKJACK_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BowlingV5Adapter(ALEAdapter):
	"""Adapter for ALE/Bowling-v5."""
	id = GameId.ALE_BOWLING_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BoxingV5Adapter(ALEAdapter):
	"""Adapter for ALE/Boxing-v5."""
	id = GameId.ALE_BOXING_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class BreakoutV5Adapter(ALEAdapter):
	"""Adapter for ALE/Breakout-v5."""
	id = GameId.ALE_BREAKOUT_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class CarnivalV5Adapter(ALEAdapter):
	"""Adapter for ALE/Carnival-v5."""
	id = GameId.ALE_CARNIVAL_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class CasinoV5Adapter(ALEAdapter):
	"""Adapter for ALE/Casino-v5."""
	id = GameId.ALE_CASINO_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class CentipedeV5Adapter(ALEAdapter):
	"""Adapter for ALE/Centipede-v5."""
	id = GameId.ALE_CENTIPEDE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class ChopperCommandV5Adapter(ALEAdapter):
	"""Adapter for ALE/ChopperCommand-v5."""
	id = GameId.ALE_CHOPPER_COMMAND_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class CrazyClimberV5Adapter(ALEAdapter):
	"""Adapter for ALE/CrazyClimber-v5."""
	id = GameId.ALE_CRAZY_CLIMBER_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class CrossbowV5Adapter(ALEAdapter):
	"""Adapter for ALE/Crossbow-v5."""
	id = GameId.ALE_CROSSBOW_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class DarkchambersV5Adapter(ALEAdapter):
	"""Adapter for ALE/Darkchambers-v5."""
	id = GameId.ALE_DARKCHAMBERS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class DefenderV5Adapter(ALEAdapter):
	"""Adapter for ALE/Defender-v5."""
	id = GameId.ALE_DEFENDER_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class DemonAttackV5Adapter(ALEAdapter):
	"""Adapter for ALE/DemonAttack-v5."""
	id = GameId.ALE_DEMON_ATTACK_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class DonkeyKongV5Adapter(ALEAdapter):
	"""Adapter for ALE/DonkeyKong-v5."""
	id = GameId.ALE_DONKEY_KONG_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class DoubleDunkV5Adapter(ALEAdapter):
	"""Adapter for ALE/DoubleDunk-v5."""
	id = GameId.ALE_DOUBLE_DUNK_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class EarthworldV5Adapter(ALEAdapter):
	"""Adapter for ALE/Earthworld-v5."""
	id = GameId.ALE_EARTHWORLD_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class ElevatorActionV5Adapter(ALEAdapter):
	"""Adapter for ALE/ElevatorAction-v5."""
	id = GameId.ALE_ELEVATOR_ACTION_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class EnduroV5Adapter(ALEAdapter):
	"""Adapter for ALE/Enduro-v5."""
	id = GameId.ALE_ENDURO_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class EntombedV5Adapter(ALEAdapter):
	"""Adapter for ALE/Entombed-v5."""
	id = GameId.ALE_ENTOMBED_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class EtV5Adapter(ALEAdapter):
	"""Adapter for ALE/Et-v5."""
	id = GameId.ALE_ET_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class FishingDerbyV5Adapter(ALEAdapter):
	"""Adapter for ALE/FishingDerby-v5."""
	id = GameId.ALE_FISHING_DERBY_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class FlagCaptureV5Adapter(ALEAdapter):
	"""Adapter for ALE/FlagCapture-v5."""
	id = GameId.ALE_FLAG_CAPTURE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class FreewayV5Adapter(ALEAdapter):
	"""Adapter for ALE/Freeway-v5."""
	id = GameId.ALE_FREEWAY_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class FroggerV5Adapter(ALEAdapter):
	"""Adapter for ALE/Frogger-v5."""
	id = GameId.ALE_FROGGER_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class FrostbiteV5Adapter(ALEAdapter):
	"""Adapter for ALE/Frostbite-v5."""
	id = GameId.ALE_FROSTBITE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class GalaxianV5Adapter(ALEAdapter):
	"""Adapter for ALE/Galaxian-v5."""
	id = GameId.ALE_GALAXIAN_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class GopherV5Adapter(ALEAdapter):
	"""Adapter for ALE/Gopher-v5."""
	id = GameId.ALE_GOPHER_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class GravitarV5Adapter(ALEAdapter):
	"""Adapter for ALE/Gravitar-v5."""
	id = GameId.ALE_GRAVITAR_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class HangmanV5Adapter(ALEAdapter):
	"""Adapter for ALE/Hangman-v5."""
	id = GameId.ALE_HANGMAN_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class HauntedHouseV5Adapter(ALEAdapter):
	"""Adapter for ALE/HauntedHouse-v5."""
	id = GameId.ALE_HAUNTED_HOUSE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class HeroV5Adapter(ALEAdapter):
	"""Adapter for ALE/Hero-v5."""
	id = GameId.ALE_HERO_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class HumanCannonballV5Adapter(ALEAdapter):
	"""Adapter for ALE/HumanCannonball-v5."""
	id = GameId.ALE_HUMAN_CANNONBALL_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class IceHockeyV5Adapter(ALEAdapter):
	"""Adapter for ALE/IceHockey-v5."""
	id = GameId.ALE_ICE_HOCKEY_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class JamesbondV5Adapter(ALEAdapter):
	"""Adapter for ALE/Jamesbond-v5."""
	id = GameId.ALE_JAMESBOND_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class JourneyEscapeV5Adapter(ALEAdapter):
	"""Adapter for ALE/JourneyEscape-v5."""
	id = GameId.ALE_JOURNEY_ESCAPE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class KaboomV5Adapter(ALEAdapter):
	"""Adapter for ALE/Kaboom-v5."""
	id = GameId.ALE_KABOOM_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class KangarooV5Adapter(ALEAdapter):
	"""Adapter for ALE/Kangaroo-v5."""
	id = GameId.ALE_KANGAROO_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class KeystoneKapersV5Adapter(ALEAdapter):
	"""Adapter for ALE/KeystoneKapers-v5."""
	id = GameId.ALE_KEYSTONE_KAPERS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class KingKongV5Adapter(ALEAdapter):
	"""Adapter for ALE/KingKong-v5."""
	id = GameId.ALE_KING_KONG_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class KlaxV5Adapter(ALEAdapter):
	"""Adapter for ALE/Klax-v5."""
	id = GameId.ALE_KLAX_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class KoolaidV5Adapter(ALEAdapter):
	"""Adapter for ALE/Koolaid-v5."""
	id = GameId.ALE_KOOLAID_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class KrullV5Adapter(ALEAdapter):
	"""Adapter for ALE/Krull-v5."""
	id = GameId.ALE_KRULL_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class KungFuMasterV5Adapter(ALEAdapter):
	"""Adapter for ALE/KungFuMaster-v5."""
	id = GameId.ALE_KUNG_FU_MASTER_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class LaserGatesV5Adapter(ALEAdapter):
	"""Adapter for ALE/LaserGates-v5."""
	id = GameId.ALE_LASER_GATES_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class LostLuggageV5Adapter(ALEAdapter):
	"""Adapter for ALE/LostLuggage-v5."""
	id = GameId.ALE_LOST_LUGGAGE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class MarioBrosV5Adapter(ALEAdapter):
	"""Adapter for ALE/MarioBros-v5."""
	id = GameId.ALE_MARIO_BROS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class MiniatureGolfV5Adapter(ALEAdapter):
	"""Adapter for ALE/MiniatureGolf-v5."""
	id = GameId.ALE_MINIATURE_GOLF_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class MontezumaRevengeV5Adapter(ALEAdapter):
	"""Adapter for ALE/MontezumaRevenge-v5."""
	id = GameId.ALE_MONTEZUMA_REVENGE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class MrDoV5Adapter(ALEAdapter):
	"""Adapter for ALE/MrDo-v5."""
	id = GameId.ALE_MR_DO_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class MsPacmanV5Adapter(ALEAdapter):
	"""Adapter for ALE/MsPacman-v5."""
	id = GameId.ALE_MS_PACMAN_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class NameThisGameV5Adapter(ALEAdapter):
	"""Adapter for ALE/NameThisGame-v5."""
	id = GameId.ALE_NAME_THIS_GAME_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class OthelloV5Adapter(ALEAdapter):
	"""Adapter for ALE/Othello-v5."""
	id = GameId.ALE_OTHELLO_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class PacmanV5Adapter(ALEAdapter):
	"""Adapter for ALE/Pacman-v5."""
	id = GameId.ALE_PACMAN_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class PhoenixV5Adapter(ALEAdapter):
	"""Adapter for ALE/Phoenix-v5."""
	id = GameId.ALE_PHOENIX_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class PitfallV5Adapter(ALEAdapter):
	"""Adapter for ALE/Pitfall-v5."""
	id = GameId.ALE_PITFALL_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class PongV5Adapter(ALEAdapter):
	"""Adapter for ALE/Pong-v5."""
	id = GameId.ALE_PONG_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class PooyanV5Adapter(ALEAdapter):
	"""Adapter for ALE/Pooyan-v5."""
	id = GameId.ALE_POOYAN_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class PrivateEyeV5Adapter(ALEAdapter):
	"""Adapter for ALE/PrivateEye-v5."""
	id = GameId.ALE_PRIVATE_EYE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class QbertV5Adapter(ALEAdapter):
	"""Adapter for ALE/Qbert-v5."""
	id = GameId.ALE_QBERT_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class RiverraidV5Adapter(ALEAdapter):
	"""Adapter for ALE/Riverraid-v5."""
	id = GameId.ALE_RIVERRAID_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class RoadRunnerV5Adapter(ALEAdapter):
	"""Adapter for ALE/RoadRunner-v5."""
	id = GameId.ALE_ROAD_RUNNER_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class RobotankV5Adapter(ALEAdapter):
	"""Adapter for ALE/Robotank-v5."""
	id = GameId.ALE_ROBOTANK_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class SeaquestV5Adapter(ALEAdapter):
	"""Adapter for ALE/Seaquest-v5."""
	id = GameId.ALE_SEAQUEST_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class SirLancelotV5Adapter(ALEAdapter):
	"""Adapter for ALE/SirLancelot-v5."""
	id = GameId.ALE_SIR_LANCELOT_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class SkiingV5Adapter(ALEAdapter):
	"""Adapter for ALE/Skiing-v5."""
	id = GameId.ALE_SKIING_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class SolarisV5Adapter(ALEAdapter):
	"""Adapter for ALE/Solaris-v5."""
	id = GameId.ALE_SOLARIS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class SpaceInvadersV5Adapter(ALEAdapter):
	"""Adapter for ALE/SpaceInvaders-v5."""
	id = GameId.ALE_SPACE_INVADERS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class SpaceWarV5Adapter(ALEAdapter):
	"""Adapter for ALE/SpaceWar-v5."""
	id = GameId.ALE_SPACE_WAR_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class StarGunnerV5Adapter(ALEAdapter):
	"""Adapter for ALE/StarGunner-v5."""
	id = GameId.ALE_STAR_GUNNER_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class SupermanV5Adapter(ALEAdapter):
	"""Adapter for ALE/Superman-v5."""
	id = GameId.ALE_SUPERMAN_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class SurroundV5Adapter(ALEAdapter):
	"""Adapter for ALE/Surround-v5."""
	id = GameId.ALE_SURROUND_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class TennisV5Adapter(ALEAdapter):
	"""Adapter for ALE/Tennis-v5."""
	id = GameId.ALE_TENNIS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class TetrisV5Adapter(ALEAdapter):
	"""Adapter for ALE/Tetris-v5."""
	id = GameId.ALE_TETRIS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class TimePilotV5Adapter(ALEAdapter):
	"""Adapter for ALE/TimePilot-v5."""
	id = GameId.ALE_TIME_PILOT_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class TrondeadV5Adapter(ALEAdapter):
	"""Adapter for ALE/Trondead-v5."""
	id = GameId.ALE_TRONDEAD_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class TurmoilV5Adapter(ALEAdapter):
	"""Adapter for ALE/Turmoil-v5."""
	id = GameId.ALE_TURMOIL_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class TutankhamV5Adapter(ALEAdapter):
	"""Adapter for ALE/Tutankham-v5."""
	id = GameId.ALE_TUTANKHAM_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class UpNDownV5Adapter(ALEAdapter):
	"""Adapter for ALE/UpNDown-v5."""
	id = GameId.ALE_UP_N_DOWN_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class VentureV5Adapter(ALEAdapter):
	"""Adapter for ALE/Venture-v5."""
	id = GameId.ALE_VENTURE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class VideoCheckersV5Adapter(ALEAdapter):
	"""Adapter for ALE/VideoCheckers-v5."""
	id = GameId.ALE_VIDEO_CHECKERS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class VideoChessV5Adapter(ALEAdapter):
	"""Adapter for ALE/VideoChess-v5."""
	id = GameId.ALE_VIDEO_CHESS_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class VideoCubeV5Adapter(ALEAdapter):
	"""Adapter for ALE/VideoCube-v5."""
	id = GameId.ALE_VIDEO_CUBE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class VideoPinballV5Adapter(ALEAdapter):
	"""Adapter for ALE/VideoPinball-v5."""
	id = GameId.ALE_VIDEO_PINBALL_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class WizardOfWorV5Adapter(ALEAdapter):
	"""Adapter for ALE/WizardOfWor-v5."""
	id = GameId.ALE_WIZARD_OF_WOR_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class WordZapperV5Adapter(ALEAdapter):
	"""Adapter for ALE/WordZapper-v5."""
	id = GameId.ALE_WORD_ZAPPER_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class YarsRevengeV5Adapter(ALEAdapter):
	"""Adapter for ALE/YarsRevenge-v5."""
	id = GameId.ALE_YARS_REVENGE_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


class ZaxxonV5Adapter(ALEAdapter):
	"""Adapter for ALE/Zaxxon-v5."""
	id = GameId.ALE_ZAXXON_V5.value
	supported_control_modes = (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT)


ALE_ADAPTERS: dict[GameId, type[ALEAdapter]] = {
	GameId.ADVENTURE_V4: AdventureV4Adapter,
	GameId.ALE_ADVENTURE_V5: AdventureV5Adapter,
	GameId.AIR_RAID_V4: AirRaidV4Adapter,
	GameId.ALE_AIR_RAID_V5: AirRaidV5Adapter,
	GameId.ASSAULT_V4: AssaultV4Adapter,
	GameId.ALE_ASSAULT_V5: AssaultV5Adapter,
	GameId.ALE_ALIEN_V5: AlienV5Adapter,
	GameId.ALE_AMIDAR_V5: AmidarV5Adapter,
	GameId.ALE_ASTERIX_V5: AsterixV5Adapter,
	GameId.ALE_ASTEROIDS_V5: AsteroidsV5Adapter,
	GameId.ALE_ATLANTIS_V5: AtlantisV5Adapter,
	GameId.ALE_BACKGAMMON_V5: BackgammonV5Adapter,
	GameId.ALE_BANK_HEIST_V5: BankHeistV5Adapter,
	GameId.ALE_BASIC_MATH_V5: BasicMathV5Adapter,
	GameId.ALE_BATTLE_ZONE_V5: BattleZoneV5Adapter,
	GameId.ALE_BEAM_RIDER_V5: BeamRiderV5Adapter,
	GameId.ALE_BERZERK_V5: BerzerkV5Adapter,
	GameId.ALE_BLACKJACK_V5: BlackjackV5Adapter,
	GameId.ALE_BOWLING_V5: BowlingV5Adapter,
	GameId.ALE_BOXING_V5: BoxingV5Adapter,
	GameId.ALE_BREAKOUT_V5: BreakoutV5Adapter,
	GameId.ALE_CARNIVAL_V5: CarnivalV5Adapter,
	GameId.ALE_CASINO_V5: CasinoV5Adapter,
	GameId.ALE_CENTIPEDE_V5: CentipedeV5Adapter,
	GameId.ALE_CHOPPER_COMMAND_V5: ChopperCommandV5Adapter,
	GameId.ALE_CRAZY_CLIMBER_V5: CrazyClimberV5Adapter,
	GameId.ALE_CROSSBOW_V5: CrossbowV5Adapter,
	GameId.ALE_DARKCHAMBERS_V5: DarkchambersV5Adapter,
	GameId.ALE_DEFENDER_V5: DefenderV5Adapter,
	GameId.ALE_DEMON_ATTACK_V5: DemonAttackV5Adapter,
	GameId.ALE_DONKEY_KONG_V5: DonkeyKongV5Adapter,
	GameId.ALE_DOUBLE_DUNK_V5: DoubleDunkV5Adapter,
	GameId.ALE_EARTHWORLD_V5: EarthworldV5Adapter,
	GameId.ALE_ELEVATOR_ACTION_V5: ElevatorActionV5Adapter,
	GameId.ALE_ENDURO_V5: EnduroV5Adapter,
	GameId.ALE_ENTOMBED_V5: EntombedV5Adapter,
	GameId.ALE_ET_V5: EtV5Adapter,
	GameId.ALE_FISHING_DERBY_V5: FishingDerbyV5Adapter,
	GameId.ALE_FLAG_CAPTURE_V5: FlagCaptureV5Adapter,
	GameId.ALE_FREEWAY_V5: FreewayV5Adapter,
	GameId.ALE_FROGGER_V5: FroggerV5Adapter,
	GameId.ALE_FROSTBITE_V5: FrostbiteV5Adapter,
	GameId.ALE_GALAXIAN_V5: GalaxianV5Adapter,
	GameId.ALE_GOPHER_V5: GopherV5Adapter,
	GameId.ALE_GRAVITAR_V5: GravitarV5Adapter,
	GameId.ALE_HANGMAN_V5: HangmanV5Adapter,
	GameId.ALE_HAUNTED_HOUSE_V5: HauntedHouseV5Adapter,
	GameId.ALE_HERO_V5: HeroV5Adapter,
	GameId.ALE_HUMAN_CANNONBALL_V5: HumanCannonballV5Adapter,
	GameId.ALE_ICE_HOCKEY_V5: IceHockeyV5Adapter,
	GameId.ALE_JAMESBOND_V5: JamesbondV5Adapter,
	GameId.ALE_JOURNEY_ESCAPE_V5: JourneyEscapeV5Adapter,
	GameId.ALE_KABOOM_V5: KaboomV5Adapter,
	GameId.ALE_KANGAROO_V5: KangarooV5Adapter,
	GameId.ALE_KEYSTONE_KAPERS_V5: KeystoneKapersV5Adapter,
	GameId.ALE_KING_KONG_V5: KingKongV5Adapter,
	GameId.ALE_KLAX_V5: KlaxV5Adapter,
	GameId.ALE_KOOLAID_V5: KoolaidV5Adapter,
	GameId.ALE_KRULL_V5: KrullV5Adapter,
	GameId.ALE_KUNG_FU_MASTER_V5: KungFuMasterV5Adapter,
	GameId.ALE_LASER_GATES_V5: LaserGatesV5Adapter,
	GameId.ALE_LOST_LUGGAGE_V5: LostLuggageV5Adapter,
	GameId.ALE_MARIO_BROS_V5: MarioBrosV5Adapter,
	GameId.ALE_MINIATURE_GOLF_V5: MiniatureGolfV5Adapter,
	GameId.ALE_MONTEZUMA_REVENGE_V5: MontezumaRevengeV5Adapter,
	GameId.ALE_MR_DO_V5: MrDoV5Adapter,
	GameId.ALE_MS_PACMAN_V5: MsPacmanV5Adapter,
	GameId.ALE_NAME_THIS_GAME_V5: NameThisGameV5Adapter,
	GameId.ALE_OTHELLO_V5: OthelloV5Adapter,
	GameId.ALE_PACMAN_V5: PacmanV5Adapter,
	GameId.ALE_PHOENIX_V5: PhoenixV5Adapter,
	GameId.ALE_PITFALL_V5: PitfallV5Adapter,
	GameId.ALE_PONG_V5: PongV5Adapter,
	GameId.ALE_POOYAN_V5: PooyanV5Adapter,
	GameId.ALE_PRIVATE_EYE_V5: PrivateEyeV5Adapter,
	GameId.ALE_QBERT_V5: QbertV5Adapter,
	GameId.ALE_RIVERRAID_V5: RiverraidV5Adapter,
	GameId.ALE_ROAD_RUNNER_V5: RoadRunnerV5Adapter,
	GameId.ALE_ROBOTANK_V5: RobotankV5Adapter,
	GameId.ALE_SEAQUEST_V5: SeaquestV5Adapter,
	GameId.ALE_SIR_LANCELOT_V5: SirLancelotV5Adapter,
	GameId.ALE_SKIING_V5: SkiingV5Adapter,
	GameId.ALE_SOLARIS_V5: SolarisV5Adapter,
	GameId.ALE_SPACE_INVADERS_V5: SpaceInvadersV5Adapter,
	GameId.ALE_SPACE_WAR_V5: SpaceWarV5Adapter,
	GameId.ALE_STAR_GUNNER_V5: StarGunnerV5Adapter,
	GameId.ALE_SUPERMAN_V5: SupermanV5Adapter,
	GameId.ALE_SURROUND_V5: SurroundV5Adapter,
	GameId.ALE_TENNIS_V5: TennisV5Adapter,
	GameId.ALE_TETRIS_V5: TetrisV5Adapter,
	GameId.ALE_TIME_PILOT_V5: TimePilotV5Adapter,
	GameId.ALE_TRONDEAD_V5: TrondeadV5Adapter,
	GameId.ALE_TURMOIL_V5: TurmoilV5Adapter,
	GameId.ALE_TUTANKHAM_V5: TutankhamV5Adapter,
	GameId.ALE_UP_N_DOWN_V5: UpNDownV5Adapter,
	GameId.ALE_VENTURE_V5: VentureV5Adapter,
	GameId.ALE_VIDEO_CHECKERS_V5: VideoCheckersV5Adapter,
	GameId.ALE_VIDEO_CHESS_V5: VideoChessV5Adapter,
	GameId.ALE_VIDEO_CUBE_V5: VideoCubeV5Adapter,
	GameId.ALE_VIDEO_PINBALL_V5: VideoPinballV5Adapter,
	GameId.ALE_WIZARD_OF_WOR_V5: WizardOfWorV5Adapter,
	GameId.ALE_WORD_ZAPPER_V5: WordZapperV5Adapter,
	GameId.ALE_YARS_REVENGE_V5: YarsRevengeV5Adapter,
	GameId.ALE_ZAXXON_V5: ZaxxonV5Adapter,
}


__all__ = [
	"ALEAdapter",
	"AdventureV4Adapter",
	"AdventureV5Adapter",
	"AirRaidV4Adapter",
	"AirRaidV5Adapter",
	"AssaultV4Adapter",
	"AssaultV5Adapter",
	"AlienV5Adapter",
	"AmidarV5Adapter",
	"AsterixV5Adapter",
	"AsteroidsV5Adapter",
	"AtlantisV5Adapter",
	"BackgammonV5Adapter",
	"BankHeistV5Adapter",
	"BasicMathV5Adapter",
	"BattleZoneV5Adapter",
	"BeamRiderV5Adapter",
	"BerzerkV5Adapter",
	"BlackjackV5Adapter",
	"BowlingV5Adapter",
	"BoxingV5Adapter",
	"BreakoutV5Adapter",
	"CarnivalV5Adapter",
	"CasinoV5Adapter",
	"CentipedeV5Adapter",
	"ChopperCommandV5Adapter",
	"CrazyClimberV5Adapter",
	"CrossbowV5Adapter",
	"DarkchambersV5Adapter",
	"DefenderV5Adapter",
	"DemonAttackV5Adapter",
	"DonkeyKongV5Adapter",
	"DoubleDunkV5Adapter",
	"EarthworldV5Adapter",
	"ElevatorActionV5Adapter",
	"EnduroV5Adapter",
	"EntombedV5Adapter",
	"EtV5Adapter",
	"FishingDerbyV5Adapter",
	"FlagCaptureV5Adapter",
	"FreewayV5Adapter",
	"FroggerV5Adapter",
	"FrostbiteV5Adapter",
	"GalaxianV5Adapter",
	"GopherV5Adapter",
	"GravitarV5Adapter",
	"HangmanV5Adapter",
	"HauntedHouseV5Adapter",
	"HeroV5Adapter",
	"HumanCannonballV5Adapter",
	"IceHockeyV5Adapter",
	"JamesbondV5Adapter",
	"JourneyEscapeV5Adapter",
	"KaboomV5Adapter",
	"KangarooV5Adapter",
	"KeystoneKapersV5Adapter",
	"KingKongV5Adapter",
	"KlaxV5Adapter",
	"KoolaidV5Adapter",
	"KrullV5Adapter",
	"KungFuMasterV5Adapter",
	"LaserGatesV5Adapter",
	"LostLuggageV5Adapter",
	"MarioBrosV5Adapter",
	"MiniatureGolfV5Adapter",
	"MontezumaRevengeV5Adapter",
	"MrDoV5Adapter",
	"MsPacmanV5Adapter",
	"NameThisGameV5Adapter",
	"OthelloV5Adapter",
	"PacmanV5Adapter",
	"PhoenixV5Adapter",
	"PitfallV5Adapter",
	"PongV5Adapter",
	"PooyanV5Adapter",
	"PrivateEyeV5Adapter",
	"QbertV5Adapter",
	"RiverraidV5Adapter",
	"RoadRunnerV5Adapter",
	"RobotankV5Adapter",
	"SeaquestV5Adapter",
	"SirLancelotV5Adapter",
	"SkiingV5Adapter",
	"SolarisV5Adapter",
	"SpaceInvadersV5Adapter",
	"SpaceWarV5Adapter",
	"StarGunnerV5Adapter",
	"SupermanV5Adapter",
	"SurroundV5Adapter",
	"TennisV5Adapter",
	"TetrisV5Adapter",
	"TimePilotV5Adapter",
	"TrondeadV5Adapter",
	"TurmoilV5Adapter",
	"TutankhamV5Adapter",
	"UpNDownV5Adapter",
	"VentureV5Adapter",
	"VideoCheckersV5Adapter",
	"VideoChessV5Adapter",
	"VideoCubeV5Adapter",
	"VideoPinballV5Adapter",
	"WizardOfWorV5Adapter",
	"WordZapperV5Adapter",
	"YarsRevengeV5Adapter",
	"ZaxxonV5Adapter",
	"ALE_ADAPTERS",
]
