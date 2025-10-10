from __future__ import annotations

"""Qt session controller that bridges Gym adapters to the GUI."""

from dataclasses import dataclass, replace
import logging
from datetime import datetime
from typing import Any, Optional

import gymnasium.spaces as spaces
import numpy as np

from qtpy import QtCore

from gym_gui.config.game_configs import (
    CliffWalkingConfig,
    CarRacingConfig,
    BipedalWalkerConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    TaxiConfig,
)
from gym_gui.config.settings import Settings
from gym_gui.core.adapters.base import AdapterContext, AdapterStep, EnvironmentAdapter
from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.core.enums import ControlMode, GameId, EnvironmentFamily, ENVIRONMENT_FAMILY_BY_GAME
from gym_gui.core.factories.adapters import create_adapter, get_adapter_cls
from gym_gui.services.actor import ActorService, EpisodeSummary, StepSnapshot
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.telemetry import TelemetryService
from gym_gui.utils.timekeeping import SessionTimers


@dataclass(slots=True)
class SessionState:
    """Lightweight snapshot of the current episode state."""

    game_id: GameId
    control_mode: ControlMode
    step_index: int
    turn: str
    terminated: bool
    truncated: bool


class SessionController(QtCore.QObject):
    """Manage adapter lifecycle and emit Qt-friendly signals for the UI."""

    session_initialized = QtCore.Signal(str, str, object)  # type: ignore[attr-defined]
    step_processed = QtCore.Signal(object, int)  # type: ignore[attr-defined]
    episode_finished = QtCore.Signal(bool)  # type: ignore[attr-defined]
    status_message = QtCore.Signal(str)  # type: ignore[attr-defined]
    awaiting_human = QtCore.Signal(bool, str)  # type: ignore[attr-defined]
    turn_changed = QtCore.Signal(str)  # type: ignore[attr-defined]
    error_occurred = QtCore.Signal(str)  # type: ignore[attr-defined]
    auto_play_state_changed = QtCore.Signal(bool)  # type: ignore[attr-defined]

    def __init__(self, settings: Settings, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._adapter: EnvironmentAdapter | None = None
        self._game_id: GameId | None = None
        self._control_mode: ControlMode = settings.default_control_mode
        self._step_index: int = -1
        self._turn: str = "human"
        self._game_started: bool = False
        self._game_paused: bool = False
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setInterval(600)
        self._auto_timer.timeout.connect(self._auto_step)
        self._idle_timer = QtCore.QTimer(self)
        self._idle_timer.setInterval(600)
        self._idle_timer.timeout.connect(self._idle_step)
        self._user_idle_interval: int | None = None
        self._current_idle_interval = self._idle_timer.interval()
        self._logger = logging.getLogger("gym_gui.controllers.session")
        self._awaiting_human = False
        self._pending_input_label: str | None = None
        self._last_agent_position: tuple[int, int] | None = None
        self._timers = SessionTimers()
        self._settings_overrides: dict[str, Any] = {}
        self._effective_settings: Settings = settings
        locator = get_service_locator()
        self._telemetry = locator.resolve(TelemetryService)
        self._actor_service = locator.resolve(ActorService)
        self._episode_counter = 0
        self._episode_id: str | None = None
        self._episode_active = False
        self._episode_reward = 0.0
        self._episode_metadata: dict[str, Any] = {}
        self._last_step: AdapterStep | None = None
        self._passive_action: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def configure_interval(self, interval_ms: int) -> None:
        interval = max(50, interval_ms)
        self._auto_timer.setInterval(interval)
        self._user_idle_interval = interval
        self._update_idle_timer()

    def start_game(self) -> None:
        """Mark the game as started and enable input controls."""
        if self._adapter is None:
            self.status_message.emit("⚠ No environment loaded. Click 'Load Environment' first.")
            return

        self._game_started = True
        self._game_paused = False
        # Note: Human input controller is set up in main_window, not here
        self.status_message.emit("Game started! Use controls to play.")

    def pause_game(self) -> None:
        """Pause the game - stops all actions."""
        if not self._game_started:
            self.status_message.emit("⚠ No game to pause.")
            return
        self.stop_auto_play()
        self._stop_idle_tick()  # Stop the idle timer for Box2D games
        self._game_paused = True
        self.status_message.emit("Game paused.")

    def resume_game(self) -> None:
        """Resume the game from paused state."""
        if not self._game_started:
            self.status_message.emit("⚠ No game to resume.")
            return
        self._game_paused = False
        self._update_idle_timer()  # Restart idle timer if needed for Box2D games
        self.status_message.emit("Game resumed.")

    def terminate_game(self) -> None:
        """Stop the game and mark episode as finished."""
        self.stop_auto_play()
        self._game_started = False
        self._game_paused = False
        if self._episode_active and self._last_step is not None:
            self._finalize_episode(self._last_step, aborted=True)
        self.episode_finished.emit(True)
        self.status_message.emit("Game terminated. Click 'Start Game' to play again.")

    def load_environment(
        self,
        game_id: GameId,
        control_mode: ControlMode,
        *,
        seed: int | None = None,
        settings_overrides: dict[str, Any] | None = None,
        game_config: (
            FrozenLakeConfig
            | TaxiConfig
            | CliffWalkingConfig
            | LunarLanderConfig
            | CarRacingConfig
            | BipedalWalkerConfig
            | None
        ) = None,
    ) -> None:
        self.stop_auto_play()
        self._dispose_adapter()
        if settings_overrides is not None:
            self._settings_overrides = settings_overrides
        effective_settings = (
            replace(self._settings, **self._settings_overrides)
            if self._settings_overrides
            else self._settings
        )
        self._effective_settings = effective_settings
        context = AdapterContext(
            settings=effective_settings,
            control_mode=control_mode,
            logger_factory=logging.getLogger,
        )
        try:
            adapter = create_adapter(game_id, context, game_config=game_config)
            adapter.ensure_control_mode(control_mode)
            adapter.load()
            initial_step = adapter.reset(seed=seed)
        except Exception as exc:  # pragma: no cover - UI surfaces error
            self._logger.exception("Failed to load environment", exc_info=exc)
            self.error_occurred.emit(str(exc))
            return

        self._adapter = adapter
        self._game_id = game_id
        self._control_mode = control_mode
        self._step_index = 0
        self._turn = "human"
        self._game_started = False  # Game not started until user clicks "Start Game"
        self._game_paused = False
        self._awaiting_human = False  # Don't await input until game starts
        self._begin_episode(game_id, control_mode)
        self._timers.reset_episode()
        self.session_initialized.emit(game_id.value, control_mode.value, initial_step)
        self.step_processed.emit(initial_step, self._step_index)
        self._update_status(initial_step, prefix="Environment ready")
        self.awaiting_human.emit(False, "Click 'Start Game' to begin")
        self.turn_changed.emit(self._turn)
        self._last_agent_position = self._extract_agent_position(initial_step)
        self._last_step = initial_step
        self._passive_action = self._resolve_passive_action()
        self._pending_input_label = "environment_ready"
        self._record_step(initial_step, action=None, input_source=self._pending_input_label)
        self._update_idle_timer()

    def reset_environment(self, *, seed: int | None = None) -> None:
        if self._adapter is None or self._game_id is None:
            return
        self.stop_auto_play()
        try:
            step = self._adapter.reset(seed=seed)
        except Exception as exc:  # pragma: no cover - UI surfaces error
            self._logger.exception("Failed to reset environment", exc_info=exc)
            self.error_occurred.emit(str(exc))
            return
        self._step_index = 0
        self._turn = "human"
        self._game_started = False  # Reset requires "Start Game" again
        self._game_paused = False
        self._awaiting_human = False
        self._begin_episode(self._game_id, self._control_mode)
        self.step_processed.emit(step, self._step_index)
        self._update_status(step, prefix="Environment reset")
        self.awaiting_human.emit(False, "Click 'Start Game' to begin")
        self.turn_changed.emit(self._turn)
        self._last_agent_position = self._extract_agent_position(step)
        self._timers.reset_episode()
        self._last_step = step
        self._passive_action = self._resolve_passive_action()
        self._pending_input_label = "environment_reset"
        self._record_step(step, action=None, input_source=self._pending_input_label)
        self._update_idle_timer()

    def perform_human_action(self, action: int, *, key_label: str | None = None) -> None:
        if self._adapter is None:
            return
        if not self._game_started:
            self.status_message.emit("Click 'Start Game' before taking actions")
            return
        if self._game_paused:
            self.status_message.emit("Game is paused. Click 'Continue' to resume.")
            return
        if self._control_mode not in {
            ControlMode.HUMAN_ONLY,
            ControlMode.HYBRID_TURN_BASED,
            ControlMode.HYBRID_HUMAN_AGENT,
        }:
            self.status_message.emit("Current mode ignores human input")
            return
        if self._control_mode == ControlMode.HYBRID_TURN_BASED and self._turn != "human":
            self.status_message.emit("Hybrid mode: waiting for agent turn")
            return
        self._awaiting_human = False
        self.awaiting_human.emit(False, "")
        self._pending_input_label = key_label or "human"
        self._logger.debug(
            "Human action received label='%s' action=%s", self._pending_input_label, action
        )
        self._apply_action(action)

    def toggle_auto_play(self, *, enabled: bool) -> None:
        if enabled:
            self.start_auto_play()
        else:
            self.stop_auto_play()

    def start_auto_play(self) -> None:
        if self._adapter is None:
            return
        if not self._game_started:
            self.status_message.emit("Click 'Start Game' before auto-play")
            return
        if self._game_paused:
            self.status_message.emit("Game is paused. Click 'Continue' to resume.")
            return
        if self._auto_timer.isActive():
            return
        self._auto_timer.start()
        self.auto_play_state_changed.emit(True)
        self.status_message.emit("Auto-play started")

    def stop_auto_play(self) -> None:
        if self._auto_timer.isActive():
            self._auto_timer.stop()
            self.auto_play_state_changed.emit(False)
            self.status_message.emit("Auto-play paused")

    def perform_agent_step(self) -> None:
        if self._adapter is None:
            return
        if not self._game_started:
            self.status_message.emit("Click 'Start Game' before agent step")
            return
        if self._game_paused:
            self.status_message.emit("Game is paused. Click 'Continue' to resume.")
            return
        action = self._select_agent_action()
        if action is None:
            self._awaiting_human = True
            self.awaiting_human.emit(True, "Awaiting human action")
            return
        self._pending_input_label = "agent"
        self._apply_action(action)

    def shutdown(self) -> None:
        self.stop_auto_play()
        self._dispose_adapter()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    @staticmethod
    def supported_control_modes(game_id: GameId) -> tuple[ControlMode, ...]:
        adapter_cls = get_adapter_cls(game_id)
        return adapter_cls.supported_control_modes

    @property
    def action_space(self) -> Any | None:
        if self._adapter is None:
            return None
        return self._adapter.action_space

    @property
    def game_id(self) -> GameId | None:
        return self._game_id

    @property
    def timers(self) -> SessionTimers:
        return self._timers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _dispose_adapter(self) -> None:
        if self._episode_active and self._last_step is not None:
            self._finalize_episode(self._last_step, aborted=True, timestamp=datetime.utcnow())
        if self._adapter is not None:
            try:
                self._adapter.close()
            finally:
                self._adapter = None
                self._game_id = None
                self._step_index = -1
        self._episode_active = False
        self._episode_id = None
        self._stop_idle_tick()
        self._passive_action = None
        if self._telemetry is not None:
            self._telemetry.reset()

    def _auto_step(self) -> None:
        if self._adapter is None:
            self.stop_auto_play()
            return
        if self._game_paused:
            # Auto-play should be stopped when paused, but add safety check
            self.stop_auto_play()
            return
        action = self._select_agent_action()
        if action is None:
            # Waiting for human input in human or hybrid modes
            self.stop_auto_play()
            self._awaiting_human = True
            self.awaiting_human.emit(True, "Awaiting human action")
            return
        self._pending_input_label = "auto_play"
        self._apply_action(action)

    def _select_agent_action(self) -> int | None:
        if self._adapter is None:
            return None

        if self._control_mode == ControlMode.HUMAN_ONLY:
            return None

        if self._control_mode == ControlMode.HYBRID_TURN_BASED and self._turn != "agent":
            return None

        if self._actor_service is not None and self._last_step is not None:
            snapshot = self._build_step_snapshot(self._last_step)
            try:
                selected = self._actor_service.select_action(snapshot)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.exception("Actor service failed to select action", exc_info=exc)
            else:
                if selected is not None:
                    return selected

        try:
            space = self._adapter.action_space
            if hasattr(space, "sample"):
                action = space.sample()
                return int(action) if isinstance(action, (bool, int)) else action
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.exception("Failed to sample action", exc_info=exc)
            self.error_occurred.emit(str(exc))
        return None

    def _apply_action(self, action: int) -> None:
        if self._adapter is None:
            return
        try:
            step = self._adapter.step(action)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            self._logger.exception("Step failed", exc_info=exc)
            self.error_occurred.emit(str(exc))
            self.stop_auto_play()
            self._pending_input_label = None
            return

        self._step_index += 1
        self._last_step = step
        step_timestamp = datetime.utcnow()
        source_label = self._pending_input_label or "unknown"
        self._record_step(
            step,
            action=action,
            timestamp=step_timestamp,
            input_source=source_label,
        )
        self._timers.mark_first_move(when=step_timestamp)
        self.step_processed.emit(step, self._step_index)
        self._update_status(step)
        self._log_step_outcome(action, step)

        # Clear the label so future steps require explicit attribution
        self._pending_input_label = None

        finished = bool(step.terminated or step.truncated)
        if finished:
            self._timers.mark_outcome(when=step_timestamp)
            self._finalize_episode(step, timestamp=step_timestamp)
            self.episode_finished.emit(True)
            self.stop_auto_play()
            self._awaiting_human = False
            self.awaiting_human.emit(False, "Episode finished")
            self._stop_idle_tick()
        else:
            self._advance_turn()
            if self._control_mode == ControlMode.HUMAN_ONLY:
                self._awaiting_human = True
                self.awaiting_human.emit(True, "Awaiting human action")
            self._update_idle_timer()

    def _advance_turn(self) -> None:
        if self._control_mode != ControlMode.HYBRID_TURN_BASED:
            return
        self._turn = "agent" if self._turn == "human" else "human"
        self.turn_changed.emit(self._turn)
        if self._turn == "human":
            self._awaiting_human = True
            self.awaiting_human.emit(True, "Awaiting human action")
        else:
            self._awaiting_human = False
            self.awaiting_human.emit(False, "Agent turn")
            # If auto-play is active, immediately allow agent to act on next timeout
            if not self._auto_timer.isActive():
                self.start_auto_play()

    def _begin_episode(self, game_id: GameId | None, control_mode: ControlMode) -> None:
        if game_id is None:
            return
        if self._episode_active and self._last_step is not None:
            self._finalize_episode(self._last_step, aborted=True, timestamp=datetime.utcnow())
        self._episode_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self._episode_id = f"{game_id.value}-{timestamp}-{self._episode_counter:04d}"
        self._episode_reward = 0.0
        self._episode_metadata = {
            "game_id": game_id.value,
            "control_mode": control_mode.value,
        }
        self._episode_active = True
        self._last_step = None
        if self._telemetry is not None:
            self._telemetry.reset()

    def _record_step(
        self,
        step: AdapterStep,
        *,
        action: int | None,
        timestamp: datetime | None = None,
        input_source: str | None = None,
    ) -> StepRecord | None:
        if not self._episode_active or self._episode_id is None:
            return None
        snapshot = self._build_step_snapshot(step)
        self._episode_reward += step.reward
        if self._actor_service is not None and action is not None:
            try:
                self._actor_service.notify_step(snapshot)
            except Exception:  # pragma: no cover - defensive safeguard
                self._logger.exception("Actor service failed during notify_step")
        info_payload = dict(step.info)
        if input_source is not None:
            info_payload.setdefault("input_source", input_source)
        record = StepRecord(
            episode_id=self._episode_id,
            step_index=self._step_index,
            action=action,
            observation=step.observation,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            info=info_payload,
            timestamp=timestamp or datetime.utcnow(),
            render_payload=step.render_payload,
        )
        if self._telemetry is not None:
            self._telemetry.record_step(record)
        return record

    def _idle_step(self) -> None:
        if not self._should_idle_tick():
            self._stop_idle_tick()
            return
        if not self._awaiting_human:
            return
        if self._passive_action is None:
            self._logger.debug("Idle timer active but passive action unavailable")
            self._stop_idle_tick()
            return
        self._awaiting_human = False
        self.awaiting_human.emit(False, "Passive idle step")
        self._pending_input_label = "idle_tick"
        self._apply_action(self._passive_action)

    def _determine_idle_interval(self) -> int:
        if self._user_idle_interval is not None:
            return self._user_idle_interval
        return self._compute_idle_interval()

    def _set_idle_interval(self, interval: int) -> None:
        clamped = max(30, int(interval))
        if clamped != self._current_idle_interval:
            self._idle_timer.setInterval(clamped)
            self._current_idle_interval = clamped

    def _compute_idle_interval(self) -> int:
        adapter = self._adapter
        if adapter is None:
            return max(30, self._current_idle_interval)

        env = getattr(adapter, "_env", None)
        metadata = getattr(env, "metadata", None) if env is not None else None
        if isinstance(metadata, dict):
            fps_raw = metadata.get("render_fps") or metadata.get("video.frames_per_second")
            fps_value = 0.0
            if fps_raw is not None:
                try:
                    fps_value = float(fps_raw)
                except (TypeError, ValueError):
                    fps_value = 0.0
            if fps_value > 0:
                return max(30, int(1000 / fps_value))

        try:
            space = adapter.action_space
        except Exception:
            return 150

        if isinstance(space, spaces.Box):
            return 80
        if isinstance(space, (spaces.MultiBinary, spaces.MultiDiscrete, spaces.Discrete)):
            return 120
        if isinstance(space, spaces.Tuple):
            return 120
        return 180

    def _should_idle_tick(self) -> bool:
        if self._adapter is None or self._game_id is None:
            return False
        # Don't idle tick when game is paused
        if self._game_paused:
            return False
        # Only allow idle tick for Box2D games
        if ENVIRONMENT_FAMILY_BY_GAME.get(self._game_id) != EnvironmentFamily.BOX2D:
            return False
        if self._control_mode != ControlMode.HUMAN_ONLY:
            return False
        if self._last_step is not None and (self._last_step.terminated or self._last_step.truncated):
            return False
        return self._passive_action is not None

    def _start_idle_tick(self) -> None:
        if not self._idle_timer.isActive():
            self._idle_timer.start()

    def _stop_idle_tick(self) -> None:
        if self._idle_timer.isActive():
            self._idle_timer.stop()

    def _update_idle_timer(self) -> None:
        desired_interval = self._determine_idle_interval()
        self._set_idle_interval(desired_interval)
        if self._should_idle_tick():
            self._start_idle_tick()
        else:
            self._stop_idle_tick()

    def _resolve_passive_action(self) -> Any | None:
        if self._adapter is None or self._game_id is None:
            return None
        # Only provide a passive action for Box2D games
        if ENVIRONMENT_FAMILY_BY_GAME.get(self._game_id) != EnvironmentFamily.BOX2D:
            return None
        space = self._adapter.action_space
        if isinstance(space, spaces.Discrete):
            return 0
        if isinstance(space, spaces.MultiDiscrete):
            return np.zeros_like(space.nvec)
        if isinstance(space, spaces.MultiBinary):
            return np.zeros(space.n, dtype=int)
        if isinstance(space, spaces.Box):
            dtype = space.dtype if getattr(space, "dtype", None) is not None else np.float32
            return np.zeros(space.shape, dtype=dtype)
        if isinstance(space, spaces.Tuple):
            return tuple(self._resolve_passive_component(subspace) for subspace in space.spaces)
        return None

    def _resolve_passive_component(self, space: spaces.Space) -> Any:
        if isinstance(space, spaces.Discrete):
            return 0
        if isinstance(space, spaces.MultiDiscrete):
            return np.zeros_like(space.nvec)
        if isinstance(space, spaces.MultiBinary):
            return np.zeros(space.n, dtype=int)
        if isinstance(space, spaces.Box):
            dtype = space.dtype if getattr(space, "dtype", None) is not None else np.float32
            return np.zeros(space.shape, dtype=dtype)
        if isinstance(space, spaces.Tuple):
            return tuple(self._resolve_passive_component(sub) for sub in space.spaces)
        return None

    def _build_step_snapshot(self, step: AdapterStep) -> StepSnapshot:
        return StepSnapshot(
            step_index=self._step_index,
            observation=step.observation,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            info=dict(step.info),
        )

    def _finalize_episode(
        self,
        step: AdapterStep,
        *,
        aborted: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        if not self._episode_active or self._episode_id is None:
            return
        summary = EpisodeSummary(
            episode_index=self._episode_counter,
            total_reward=self._episode_reward,
            steps=self._step_index + 1,
            metadata=dict(self._episode_metadata),
        )
        if self._actor_service is not None:
            try:
                self._actor_service.notify_episode_end(summary)
            except Exception:  # pragma: no cover - defensive safeguard
                self._logger.exception("Actor service failed during notify_episode_end")
        if self._telemetry is not None:
            rollup = EpisodeRollup(
                episode_id=self._episode_id,
                total_reward=self._episode_reward,
                steps=self._step_index + 1,
                terminated=step.terminated if not aborted else False,
                truncated=step.truncated if not aborted else True,
                metadata=dict(self._episode_metadata),
                timestamp=timestamp or datetime.utcnow(),
            )
            self._telemetry.complete_episode(rollup)
        self._episode_active = False
        self._episode_id = None

    def _extract_agent_position(self, step: AdapterStep) -> tuple[int, int] | None:
        payload = getattr(step, "render_payload", None)
        if isinstance(payload, dict):
            position = payload.get("agent_position")
            if isinstance(position, tuple) and len(position) == 2:
                return position  # type: ignore[return-value]
        return None

    def _log_step_outcome(self, action: int, step: AdapterStep) -> None:
        input_label = self._pending_input_label or "system"
        self._pending_input_label = None

        prev_position = self._last_agent_position
        new_position = self._extract_agent_position(step)
        self._last_agent_position = new_position

        delta: tuple[int, int] | None = None
        if prev_position is not None and new_position is not None:
            delta = (new_position[0] - prev_position[0], new_position[1] - prev_position[1])

        self._logger.debug(
            "Step %s via input='%s' action=%s reward=%.2f terminated=%s truncated=%s position=%s delta=%s",
            self._step_index,
            input_label,
            action,
            step.reward,
            step.terminated,
            step.truncated,
            new_position,
            delta,
        )

    def _update_status(self, step: AdapterStep, *, prefix: str | None = None) -> None:
        # Avoid emitting routine per-step status messages to the global status bar.
        # Only emit when an explicit prefix is supplied (for notable events like
        # environment ready or environment reset). Emitting per-step details
        # causes noisy status text such as "reward=... terminated=... truncated=..."
        # to appear in the UI which is undesirable.
        if prefix is None:
            return

        message = (
            f"reward={step.reward:.2f} terminated={step.terminated} truncated={step.truncated}"
        )
        message = f"{prefix}: {message}"
        self.status_message.emit(message)


__all__ = ["SessionController", "SessionState"]
