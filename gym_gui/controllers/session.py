from __future__ import annotations

"""Qt session controller that bridges Gym adapters to the GUI."""

from dataclasses import dataclass, replace
import logging
from typing import Any, Optional

from qtpy import QtCore

from gym_gui.config.game_configs import CliffWalkingConfig, FrozenLakeConfig, TaxiConfig
from gym_gui.config.settings import Settings
from gym_gui.core.adapters.base import AdapterContext, AdapterStep, EnvironmentAdapter
from gym_gui.core.enums import ControlMode, GameId
from gym_gui.core.factories.adapters import create_adapter, get_adapter_cls
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
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setInterval(600)
        self._auto_timer.timeout.connect(self._auto_step)
        self._logger = logging.getLogger("gym_gui.controllers.session")
        self._awaiting_human = False
        self._pending_input_label: str | None = None
        self._last_agent_position: tuple[int, int] | None = None
        self._timers = SessionTimers()
        self._settings_overrides: dict[str, Any] = {}
        self._effective_settings: Settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def configure_interval(self, interval_ms: int) -> None:
        self._auto_timer.setInterval(max(50, interval_ms))

    def load_environment(
        self,
        game_id: GameId,
        control_mode: ControlMode,
        *,
        seed: int | None = None,
        settings_overrides: dict[str, Any] | None = None,
        game_config: FrozenLakeConfig | TaxiConfig | CliffWalkingConfig | None = None,
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
        self._awaiting_human = control_mode in {
            ControlMode.HUMAN_ONLY,
            ControlMode.HYBRID_TURN_BASED,
            ControlMode.HYBRID_HUMAN_AGENT,
        }
        self._timers.reset_episode()
        self.session_initialized.emit(game_id.value, control_mode.value, initial_step)
        self.step_processed.emit(initial_step, self._step_index)
        self._update_status(initial_step, prefix="Environment ready")
        self.awaiting_human.emit(self._awaiting_human, "Awaiting human action" if self._awaiting_human else "")
        self.turn_changed.emit(self._turn)
        self._last_agent_position = self._extract_agent_position(initial_step)

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
        self._awaiting_human = self._control_mode in {
            ControlMode.HUMAN_ONLY,
            ControlMode.HYBRID_TURN_BASED,
            ControlMode.HYBRID_HUMAN_AGENT,
        }
        self.step_processed.emit(step, self._step_index)
        self._update_status(step, prefix="Environment reset")
        self.awaiting_human.emit(self._awaiting_human, "Awaiting human action" if self._awaiting_human else "")
        self.turn_changed.emit(self._turn)
        self._last_agent_position = self._extract_agent_position(step)
        self._timers.reset_episode()

    def perform_human_action(self, action: int, *, key_label: str | None = None) -> None:
        if self._adapter is None:
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
        if self._adapter is not None:
            try:
                self._adapter.close()
            finally:
                self._adapter = None
                self._game_id = None
                self._step_index = -1

    def _auto_step(self) -> None:
        if self._adapter is None:
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
        self._timers.mark_first_move()
        try:
            step = self._adapter.step(action)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            self._logger.exception("Step failed", exc_info=exc)
            self.error_occurred.emit(str(exc))
            self.stop_auto_play()
            self._pending_input_label = None
            return

        self._step_index += 1
        self.step_processed.emit(step, self._step_index)
        self._update_status(step)
        self._log_step_outcome(action, step)

        finished = bool(step.terminated or step.truncated)
        if finished:
            self._timers.mark_outcome()
            self.episode_finished.emit(True)
            self.stop_auto_play()
            self._awaiting_human = False
            self.awaiting_human.emit(False, "Episode finished")
        else:
            self._advance_turn()

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
