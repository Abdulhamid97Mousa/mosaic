from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]

from gym_gui.config.game_configs import (
    CliffWalkingConfig,
    CarRacingConfig,
    BipedalWalkerConfig,
    ALEConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    MiniGridConfig,
    TaxiConfig,
    DEFAULT_FROZEN_LAKE_V2_CONFIG,
    DEFAULT_MINIGRID_EMPTY_5x5_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG,
    DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG,
)
from gym_gui.core.enums import ControlMode, GameId, get_game_display_name
from gym_gui.services.actor import ActorDescriptor
from gym_gui.ui.environments.single_agent_env.gym import (
    build_bipedal_controls,
    build_car_racing_controls,
    build_cliff_controls,
    build_frozenlake_controls,
    build_frozenlake_v2_controls,
    build_lunarlander_controls,
    build_taxi_controls,
)
from gym_gui.ui.environments.single_agent_env.minigrid.config_panel import (
    MINIGRID_GAME_IDS,
    ControlCallbacks as MinigridControlCallbacks,
    build_minigrid_controls,
    resolve_default_config as resolve_minigrid_default_config,
)
from gym_gui.ui.environments.single_agent_env.ale import (
    ALE_GAME_IDS,
    ControlCallbacks as ALEControlCallbacks,
    build_ale_controls,
)
from gym_gui.ui.workers import WorkerDefinition, get_worker_catalog


@dataclass(frozen=True)
class ControlPanelConfig:
    """Configuration for the control panel with game-specific configs."""
    
    available_modes: Dict[GameId, Iterable[ControlMode]]
    default_mode: ControlMode
    frozen_lake_config: FrozenLakeConfig
    taxi_config: TaxiConfig
    cliff_walking_config: CliffWalkingConfig
    lunar_lander_config: LunarLanderConfig
    car_racing_config: CarRacingConfig
    bipedal_walker_config: BipedalWalkerConfig
    minigrid_empty_config: MiniGridConfig
    minigrid_doorkey_5x5_config: MiniGridConfig
    minigrid_doorkey_6x6_config: MiniGridConfig
    minigrid_doorkey_8x8_config: MiniGridConfig
    minigrid_doorkey_16x16_config: MiniGridConfig
    minigrid_lavagap_config: MiniGridConfig
    default_seed: int
    allow_seed_reuse: bool
    actors: tuple[ActorDescriptor, ...]
    default_actor_id: Optional[str] = None
    workers: Tuple[WorkerDefinition, ...] = tuple()


class ControlPanelWidget(QtWidgets.QWidget):
    control_mode_changed = pyqtSignal(ControlMode)
    game_changed = pyqtSignal(GameId)
    load_requested = pyqtSignal(GameId, ControlMode, int)
    reset_requested = pyqtSignal(int)
    agent_form_requested = pyqtSignal(str)
    worker_changed = pyqtSignal(str)
    slippery_toggled = pyqtSignal(bool)
    frozen_v2_config_changed = pyqtSignal(str, object)  # (param_name, value)
    taxi_config_changed = pyqtSignal(str, bool)  # (param_name, value)
    cliff_config_changed = pyqtSignal(str, bool)  # (param_name, value)
    lunar_config_changed = pyqtSignal(str, object)  # (param_name, value)
    car_config_changed = pyqtSignal(str, object)  # (param_name, value)
    bipedal_config_changed = pyqtSignal(str, object)  # (param_name, value)
    start_game_requested = pyqtSignal()
    pause_game_requested = pyqtSignal()
    continue_game_requested = pyqtSignal()
    terminate_game_requested = pyqtSignal()
    agent_step_requested = pyqtSignal()
    actor_changed = pyqtSignal(str)
    train_agent_requested = pyqtSignal(str)  # New signal for headless training
    trained_agent_requested = pyqtSignal(str)  # Load trained policy/evaluation

    def __init__(
        self,
        *,
        config: ControlPanelConfig,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if not config.workers:
            config = ControlPanelConfig(
                available_modes=config.available_modes,
                default_mode=config.default_mode,
                frozen_lake_config=config.frozen_lake_config,
                taxi_config=config.taxi_config,
                cliff_walking_config=config.cliff_walking_config,
                lunar_lander_config=config.lunar_lander_config,
                car_racing_config=config.car_racing_config,
                bipedal_walker_config=config.bipedal_walker_config,
                minigrid_empty_config=config.minigrid_empty_config,
                minigrid_doorkey_5x5_config=config.minigrid_doorkey_5x5_config,
                minigrid_doorkey_6x6_config=config.minigrid_doorkey_6x6_config,
                minigrid_doorkey_8x8_config=config.minigrid_doorkey_8x8_config,
                minigrid_doorkey_16x16_config=config.minigrid_doorkey_16x16_config,
                minigrid_lavagap_config=config.minigrid_lavagap_config,
                default_seed=config.default_seed,
                allow_seed_reuse=config.allow_seed_reuse,
                actors=config.actors,
                default_actor_id=config.default_actor_id,
                workers=get_worker_catalog(),
            )

        self._config = config
        self._available_modes = config.available_modes
        self._default_seed = max(1, config.default_seed)
        self._allow_seed_reuse = config.allow_seed_reuse
        self._worker_definitions: Tuple[WorkerDefinition, ...] = config.workers
        self._minigrid_defaults: Dict[GameId, MiniGridConfig] = {
            GameId.MINIGRID_EMPTY_5x5: config.minigrid_empty_config,
            GameId.MINIGRID_DOORKEY_5x5: config.minigrid_doorkey_5x5_config,
            GameId.MINIGRID_DOORKEY_6x6: config.minigrid_doorkey_6x6_config,
            GameId.MINIGRID_DOORKEY_8x8: config.minigrid_doorkey_8x8_config,
            GameId.MINIGRID_DOORKEY_16x16: config.minigrid_doorkey_16x16_config,
            GameId.MINIGRID_LAVAGAP_S7: config.minigrid_lavagap_config,
        }
        self._game_overrides: Dict[GameId, Dict[str, object]] = {
            GameId.FROZEN_LAKE: {
                "is_slippery": config.frozen_lake_config.is_slippery,
                "success_rate": config.frozen_lake_config.success_rate,
                "reward_schedule": config.frozen_lake_config.reward_schedule,
            },
            GameId.FROZEN_LAKE_V2: {
                "is_slippery": DEFAULT_FROZEN_LAKE_V2_CONFIG.is_slippery,
                "success_rate": DEFAULT_FROZEN_LAKE_V2_CONFIG.success_rate,
                "reward_schedule": DEFAULT_FROZEN_LAKE_V2_CONFIG.reward_schedule,
                "grid_height": DEFAULT_FROZEN_LAKE_V2_CONFIG.grid_height,
                "grid_width": DEFAULT_FROZEN_LAKE_V2_CONFIG.grid_width,
                "start_position": DEFAULT_FROZEN_LAKE_V2_CONFIG.start_position,
                "goal_position": DEFAULT_FROZEN_LAKE_V2_CONFIG.goal_position,
                "hole_count": DEFAULT_FROZEN_LAKE_V2_CONFIG.hole_count,
                "random_holes": DEFAULT_FROZEN_LAKE_V2_CONFIG.random_holes,
            },
            GameId.TAXI: {
                "is_raining": config.taxi_config.is_raining,
                "fickle_passenger": config.taxi_config.fickle_passenger,
            },
            GameId.CLIFF_WALKING: {"is_slippery": config.cliff_walking_config.is_slippery},
            GameId.LUNAR_LANDER: {
                "continuous": config.lunar_lander_config.continuous,
                "gravity": config.lunar_lander_config.gravity,
                "enable_wind": config.lunar_lander_config.enable_wind,
                "wind_power": config.lunar_lander_config.wind_power,
                "turbulence_power": config.lunar_lander_config.turbulence_power,
                "max_episode_steps": config.lunar_lander_config.max_episode_steps,
            },
            GameId.CAR_RACING: {
                "continuous": config.car_racing_config.continuous,
                "domain_randomize": config.car_racing_config.domain_randomize,
                "lap_complete_percent": config.car_racing_config.lap_complete_percent,
                "max_episode_steps": config.car_racing_config.max_episode_steps,
                "max_episode_seconds": config.car_racing_config.max_episode_seconds,
            },
            GameId.BIPEDAL_WALKER: {
                "hardcore": config.bipedal_walker_config.hardcore,
                "max_episode_steps": config.bipedal_walker_config.max_episode_steps,
                "max_episode_seconds": config.bipedal_walker_config.max_episode_seconds,
            },
            GameId.MINIGRID_EMPTY_5x5: {
                "partial_observation": config.minigrid_empty_config.partial_observation,
                "image_observation": config.minigrid_empty_config.image_observation,
                "reward_multiplier": config.minigrid_empty_config.reward_multiplier,
                "agent_view_size": config.minigrid_empty_config.agent_view_size,
                "max_episode_steps": config.minigrid_empty_config.max_episode_steps,
                "seed": config.minigrid_empty_config.seed,
            },
            GameId.MINIGRID_DOORKEY_5x5: {
                "partial_observation": config.minigrid_doorkey_5x5_config.partial_observation,
                "image_observation": config.minigrid_doorkey_5x5_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_5x5_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_5x5_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_5x5_config.max_episode_steps,
                "seed": config.minigrid_doorkey_5x5_config.seed,
            },
            GameId.MINIGRID_DOORKEY_6x6: {
                "partial_observation": config.minigrid_doorkey_6x6_config.partial_observation,
                "image_observation": config.minigrid_doorkey_6x6_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_6x6_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_6x6_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_6x6_config.max_episode_steps,
                "seed": config.minigrid_doorkey_6x6_config.seed,
            },
            GameId.MINIGRID_DOORKEY_8x8: {
                "partial_observation": config.minigrid_doorkey_8x8_config.partial_observation,
                "image_observation": config.minigrid_doorkey_8x8_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_8x8_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_8x8_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_8x8_config.max_episode_steps,
                "seed": config.minigrid_doorkey_8x8_config.seed,
            },
            GameId.MINIGRID_DOORKEY_16x16: {
                "partial_observation": config.minigrid_doorkey_16x16_config.partial_observation,
                "image_observation": config.minigrid_doorkey_16x16_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_16x16_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_16x16_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_16x16_config.max_episode_steps,
                "seed": config.minigrid_doorkey_16x16_config.seed,
            },
            GameId.MINIGRID_LAVAGAP_S7: {
                "partial_observation": config.minigrid_lavagap_config.partial_observation,
                "image_observation": config.minigrid_lavagap_config.image_observation,
                "reward_multiplier": config.minigrid_lavagap_config.reward_multiplier,
                "agent_view_size": config.minigrid_lavagap_config.agent_view_size,
                "max_episode_steps": config.minigrid_lavagap_config.max_episode_steps,
                "seed": config.minigrid_lavagap_config.seed,
            },
        }

        self._current_game: Optional[GameId] = None
        self._current_mode: ControlMode = self._load_mode_preference(config.default_mode)
        self._awaiting_human: bool = False
        self._auto_running: bool = False
        self._game_started: bool = False
        self._game_paused: bool = False
        self._actor_descriptors: Dict[str, ActorDescriptor] = {
            descriptor.actor_id: descriptor for descriptor in config.actors
        }
        self._actor_order: tuple[ActorDescriptor, ...] = config.actors
        default_actor = config.default_actor_id
        if default_actor is None and self._actor_order:
            default_actor = self._actor_order[0].actor_id
        self._active_actor_id: Optional[str] = default_actor

        # Store game configurations
        self._game_configs: Dict[GameId, Dict[str, object]] = {
            GameId.FROZEN_LAKE: {
                "is_slippery": config.frozen_lake_config.is_slippery
            },
            GameId.TAXI: {
                "is_raining": config.taxi_config.is_raining,
                "fickle_passenger": config.taxi_config.fickle_passenger,
            },
            GameId.CLIFF_WALKING: {
                "is_slippery": config.cliff_walking_config.is_slippery,
            },
            GameId.LUNAR_LANDER: {
                "continuous": config.lunar_lander_config.continuous,
                "gravity": config.lunar_lander_config.gravity,
                "enable_wind": config.lunar_lander_config.enable_wind,
                "wind_power": config.lunar_lander_config.wind_power,
                "turbulence_power": config.lunar_lander_config.turbulence_power,
                "max_episode_steps": config.lunar_lander_config.max_episode_steps,
            },
            GameId.CAR_RACING: {
                "continuous": config.car_racing_config.continuous,
                "domain_randomize": config.car_racing_config.domain_randomize,
                "lap_complete_percent": config.car_racing_config.lap_complete_percent,
                "max_episode_steps": config.car_racing_config.max_episode_steps,
                "max_episode_seconds": config.car_racing_config.max_episode_seconds,
            },
            GameId.BIPEDAL_WALKER: {
                "hardcore": config.bipedal_walker_config.hardcore,
                "max_episode_steps": config.bipedal_walker_config.max_episode_steps,
                "max_episode_seconds": config.bipedal_walker_config.max_episode_seconds,
            },
            GameId.MINIGRID_EMPTY_5x5: {
                "partial_observation": config.minigrid_empty_config.partial_observation,
                "image_observation": config.minigrid_empty_config.image_observation,
                "reward_multiplier": config.minigrid_empty_config.reward_multiplier,
                "agent_view_size": config.minigrid_empty_config.agent_view_size,
                "max_episode_steps": config.minigrid_empty_config.max_episode_steps,
                "seed": config.minigrid_empty_config.seed,
            },
            GameId.MINIGRID_DOORKEY_5x5: {
                "partial_observation": config.minigrid_doorkey_5x5_config.partial_observation,
                "image_observation": config.minigrid_doorkey_5x5_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_5x5_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_5x5_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_5x5_config.max_episode_steps,
                "seed": config.minigrid_doorkey_5x5_config.seed,
            },
            GameId.MINIGRID_DOORKEY_6x6: {
                "partial_observation": config.minigrid_doorkey_6x6_config.partial_observation,
                "image_observation": config.minigrid_doorkey_6x6_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_6x6_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_6x6_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_6x6_config.max_episode_steps,
                "seed": config.minigrid_doorkey_6x6_config.seed,
            },
            GameId.MINIGRID_DOORKEY_8x8: {
                "partial_observation": config.minigrid_doorkey_8x8_config.partial_observation,
                "image_observation": config.minigrid_doorkey_8x8_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_8x8_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_8x8_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_8x8_config.max_episode_steps,
                "seed": config.minigrid_doorkey_8x8_config.seed,
            },
            GameId.MINIGRID_DOORKEY_16x16: {
                "partial_observation": config.minigrid_doorkey_16x16_config.partial_observation,
                "image_observation": config.minigrid_doorkey_16x16_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_16x16_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_16x16_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_16x16_config.max_episode_steps,
                "seed": config.minigrid_doorkey_16x16_config.seed,
            },
            GameId.MINIGRID_LAVAGAP_S7: {
                "partial_observation": config.minigrid_lavagap_config.partial_observation,
                "image_observation": config.minigrid_lavagap_config.image_observation,
                "reward_multiplier": config.minigrid_lavagap_config.reward_multiplier,
                "agent_view_size": config.minigrid_lavagap_config.agent_view_size,
                "max_episode_steps": config.minigrid_lavagap_config.max_episode_steps,
                "seed": config.minigrid_lavagap_config.seed,
            },
        }

        self._current_worker_id: Optional[str] = None
        self._build_ui()
        self._apply_current_mode_selection()
        self._connect_signals()
        self._update_control_states()
        self._populate_actor_combo()
        self._populate_worker_combo()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def populate_games(self, games: Iterable[GameId], *, default: Optional[GameId] = None) -> None:
        games_tuple = tuple(games)
        self._game_combo.blockSignals(True)
        self._game_combo.clear()
        for game in games_tuple:
            self._game_combo.addItem(get_game_display_name(game), game)
        # Ensure scrollbar is visible after populating items
        combo_view = self._game_combo.view()
        if combo_view is not None and isinstance(combo_view, QtWidgets.QAbstractItemView):
            combo_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self._game_combo.blockSignals(False)

        if not games_tuple:
            return

        chosen = default if (default is not None and default in games_tuple) else games_tuple[0]
        index = self._game_combo.findData(chosen)
        if index < 0:
            index = 0
        self._game_combo.setCurrentIndex(index)
        chosen_game = self._game_combo.itemData(index)
        if isinstance(chosen_game, GameId):
            self._emit_game_changed(chosen_game)

    def current_actor(self) -> Optional[str]:
        return self._active_actor_id

    def current_worker_id(self) -> Optional[str]:
        return self._current_worker_id

    def current_worker(self) -> Optional[WorkerDefinition]:
        return self._current_worker_definition()

    def set_active_actor(self, actor_id: str) -> None:
        if actor_id == self._active_actor_id:
            return
        index = self._actor_combo.findData(actor_id)
        if index < 0:
            return
        self._actor_combo.blockSignals(True)
        self._actor_combo.setCurrentIndex(index)
        self._actor_combo.blockSignals(False)
        self._active_actor_id = actor_id
        self._update_actor_description()

    def update_modes(self, game_id: GameId) -> None:
        supported = tuple(self._available_modes.get(game_id, ()))
        if not supported:
            self._mode_combo.setEnabled(False)
            return

        self._mode_combo.blockSignals(True)
        self._mode_combo.clear()
        for mode in supported:
            label = mode.value.replace("_", " ").title()
            self._mode_combo.addItem(label, mode)
        self._mode_combo.blockSignals(False)
        self._mode_combo.setEnabled(bool(supported))

        if self._current_mode not in supported:
            self._current_mode = supported[0]
            self._persist_mode_preference(self._current_mode)
            index = self._mode_combo.findData(self._current_mode)
            if index >= 0:
                self._mode_combo.blockSignals(True)
                self._mode_combo.setCurrentIndex(index)
                self._mode_combo.blockSignals(False)
            self._emit_mode_changed(self._current_mode)
            return

        index = self._mode_combo.findData(self._current_mode)
        if index >= 0:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(index)
            self._mode_combo.blockSignals(False)
        is_human_tab = self._tab_widget is not None and self._tab_widget.currentWidget() is self._human_tab
        self._mode_combo.setEnabled(is_human_tab and bool(supported))
        self._update_control_states()

    def set_status(
        self,
        *,
        step: int,
        reward: float,
        total_reward: float,
        terminated: bool,
        truncated: bool,
        turn: str,
        awaiting_human: bool,
        session_time: str,
        active_time: str,
        episode_duration: str,
        outcome_time: str = "—",
        outcome_wall_clock: str | None = None,
    ) -> None:
        self._step_label.setText(str(step))
        self._reward_label.setText(f"{reward:.2f}")
        self._total_reward_label.setText(f"{total_reward:.2f}")
        self._terminated_label.setText(self._format_bool(terminated))
        self._truncated_label.setText(self._format_bool(truncated))
        self._turn_label.setText(turn)
        self.set_awaiting_human(awaiting_human)
        self.set_time_labels(
            session_time,
            active_time,
            outcome_time,
            outcome_timestamp=outcome_wall_clock,
        )

    def set_turn(self, turn: str) -> None:
        self._turn_label.setText(turn)

    def set_mode(self, mode: ControlMode) -> None:
        index = self._mode_combo.findData(mode)
        if index < 0:
            return
        self._current_mode = mode
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(index)
        self._mode_combo.blockSignals(False)
        self._emit_mode_changed(mode)

    def set_game(self, game: GameId) -> None:
        index = self._game_combo.findData(game)
        if index >= 0:
            self._game_combo.blockSignals(True)
            self._game_combo.setCurrentIndex(index)
            self._game_combo.blockSignals(False)
            self._emit_game_changed(game)

    def override_slippery(self, enabled: bool) -> None:
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
        overrides["is_slippery"] = enabled
        if self._frozen_slippery_checkbox is not None:
            self._frozen_slippery_checkbox.blockSignals(True)
            self._frozen_slippery_checkbox.setChecked(enabled)
            self._frozen_slippery_checkbox.blockSignals(False)

    def current_seed(self) -> int:
        return int(self._seed_spin.value())

    def current_mode(self) -> ControlMode:
        return self._current_mode

    def current_game(self) -> Optional[GameId]:
        return self._current_game

    def get_overrides(self, game_id: GameId) -> Dict[str, object]:
        return dict(self._game_overrides.get(game_id, {}))

    def set_auto_running(self, running: bool) -> None:
        self._auto_running = running
        self._update_control_states()

    def set_game_started(self, started: bool) -> None:
        """Set whether the game has been started."""
        self._game_started = started
        if not started:
            self._game_paused = False
        self._update_control_states()

    def set_game_paused(self, paused: bool) -> None:
        """Set whether the game is paused."""
        self._game_paused = paused
        self._update_control_states()

    def set_slippery_visible(self, visible: bool) -> None:
        if self._frozen_slippery_checkbox is not None:
            self._frozen_slippery_checkbox.setVisible(visible)

    def set_awaiting_human(self, awaiting: bool) -> None:
        self._awaiting_human = awaiting
        self._awaiting_label.setText("Yes" if awaiting else "No")
        self._update_control_states()

    def set_time_labels(
        self,
        session_time: str,
        active_time: str,
        outcome_time: str,
        *,
        outcome_timestamp: str | None = None,
    ) -> None:
        self._session_time_label.setText(session_time)
        self._active_time_label.setText(active_time)
        self._outcome_time_label.setText(outcome_time)
        tooltip = "Elapsed time between the first move and the recorded outcome."
        if outcome_timestamp and outcome_timestamp != "—":
            tooltip += f"\nOutcome recorded at {outcome_timestamp}."
        elif outcome_timestamp == "—":
            tooltip += "\nOutcome not recorded yet."
        self._outcome_time_label.setToolTip(tooltip)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._tab_widget = QtWidgets.QTabWidget(self)
        self._tab_widget.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self._tab_widget.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tab_widget)

        self._tab_to_mode = {}

        self._human_tab = QtWidgets.QWidget(self)
        human_layout = QtWidgets.QVBoxLayout(self._human_tab)
        human_layout.setContentsMargins(0, 0, 0, 0)
        human_layout.setSpacing(12)
        human_layout.addWidget(self._create_environment_group(self._human_tab))
        # Wrap Game Configuration in a scroll area to ensure full visibility in Human mode
        cfg_group = self._create_config_group(self._human_tab)
        self._config_scroll = QtWidgets.QScrollArea(self._human_tab)
        self._config_scroll.setWidgetResizable(True)
        self._config_scroll.setStyleSheet("QScrollArea { border: none; }")
        self._config_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._config_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self._config_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        # Ensure the scroll area prefers to expand vertically
        sp = self._config_scroll.sizePolicy()
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)
        self._config_scroll.setSizePolicy(sp)
        self._config_scroll.setWidget(cfg_group)
        human_layout.addWidget(self._config_scroll)
        human_layout.addWidget(self._create_mode_group(self._human_tab))
        self._control_buttons_widget = self._create_control_group(self._human_tab)
        human_layout.addWidget(self._control_buttons_widget)
        human_layout.addWidget(self._create_status_group(self._human_tab))
        # Give the Game Configuration scroll area more room compared to trailing groups
        # so that it remains visible and scrollable on smaller screens.
        # Indices (after adds): 0=env, 1=config_scroll, 2=mode, 3=control, 4=status, 5=stretch
        human_layout.setStretch(1, 1)
        human_layout.addStretch(0)
        human_index = self._tab_widget.addTab(self._human_tab, "Human Control")
        self._tab_to_mode[human_index] = ControlMode.HUMAN_ONLY

        self._single_agent_tab = QtWidgets.QWidget(self)
        single_layout = QtWidgets.QVBoxLayout(self._single_agent_tab)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(12)
        single_layout.addWidget(self._create_actor_group(self._single_agent_tab))
        single_layout.addWidget(self._create_worker_group(self._single_agent_tab))
        single_layout.addWidget(self._create_training_group(self._single_agent_tab))
        single_layout.addStretch(1)
        single_index = self._tab_widget.addTab(self._single_agent_tab, "Single-Agent Mode")
        self._tab_to_mode[single_index] = ControlMode.AGENT_ONLY

        self._multi_agent_tab = QtWidgets.QWidget(self)
        multi_layout = QtWidgets.QVBoxLayout(self._multi_agent_tab)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.setSpacing(12)
        placeholder = QtWidgets.QLabel("Multi-agent configuration coming soon.", self._multi_agent_tab)
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        multi_layout.addWidget(placeholder)
        multi_layout.addStretch(1)
        multi_index = self._tab_widget.addTab(self._multi_agent_tab, "Multi-Agent Mode")
        self._tab_to_mode[multi_index] = ControlMode.MULTI_AGENT_COOP

        self._on_tab_changed(self._tab_widget.currentIndex())

    def _create_environment_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Environment", parent)
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(1, 1)

        layout.addWidget(QtWidgets.QLabel("Environment", group), 0, 0)
        self._game_combo = QtWidgets.QComboBox(group)
        self._game_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._game_combo.setMaxVisibleItems(10)  # Show only 10 items with scrollbar
        # Force the combobox to use a scrollable list view
        self._game_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        layout.addWidget(self._game_combo, 0, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Seed", group), 1, 0)
        self._seed_spin = QtWidgets.QSpinBox(group)
        self._seed_spin.setRange(1, 10_000_000)
        self._seed_spin.setValue(self._default_seed)
        layout.addWidget(self._seed_spin, 1, 1)

        self._seed_reuse_checkbox = QtWidgets.QCheckBox("Allow seed reuse", group)
        self._seed_reuse_checkbox.setChecked(self._allow_seed_reuse)
        layout.addWidget(self._seed_reuse_checkbox, 1, 2)
        if self._allow_seed_reuse:
            self._seed_spin.setToolTip("Seeds auto-increment by default. Adjust before loading to reuse a previous seed.")
        else:
            self._seed_spin.setToolTip("Seed increments automatically after each episode.")
        self._seed_reuse_checkbox.stateChanged.connect(self._on_seed_reuse_changed)

        self._load_button = QtWidgets.QPushButton("Load Environment", group)
        layout.addWidget(self._load_button, 2, 0, 1, 3)
        return group

    def _create_config_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        self._config_group = QtWidgets.QGroupBox("Game Configuration", parent)
        self._config_layout = QtWidgets.QFormLayout(self._config_group)
        self._frozen_slippery_checkbox = None
        return self._config_group

    def _create_mode_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        self._mode_group = QtWidgets.QGroupBox("Control Mode", parent)
        layout = QtWidgets.QVBoxLayout(self._mode_group)
        self._mode_combo = QtWidgets.QComboBox(self._mode_group)
        self._mode_combo.setEnabled(False)
        for mode in ControlMode:
            label = mode.value.replace("_", " ").title()
            self._mode_combo.addItem(label, mode)
        layout.addWidget(self._mode_combo)
        return self._mode_group

    def _create_actor_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Active Actor", parent)
        layout = QtWidgets.QVBoxLayout(group)
        self._actor_combo = QtWidgets.QComboBox(group)
        self._actor_combo.setEnabled(bool(self._actor_order))
        layout.addWidget(self._actor_combo)
        self._actor_description = QtWidgets.QLabel("—", group)
        self._actor_description.setWordWrap(True)
        layout.addWidget(self._actor_description)
        return group

    def _create_worker_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Worker Integration", parent)
        layout = QtWidgets.QVBoxLayout(group)
        self._worker_combo = QtWidgets.QComboBox(group)
        self._worker_combo.setEnabled(bool(self._worker_definitions))
        layout.addWidget(self._worker_combo)
        self._worker_description = QtWidgets.QLabel("Select a worker to view capabilities.", group)
        self._worker_description.setWordWrap(True)
        layout.addWidget(self._worker_description)
        return group

    def _create_training_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Headless Training", parent)
        layout = QtWidgets.QVBoxLayout(group)
        self._configure_agent_button = QtWidgets.QPushButton("🚀 Configure Agent…", group)
        self._configure_agent_button.setToolTip(
            "Open the agent training form to configure the backend used for headless training."
        )
        self._configure_agent_button.setEnabled(False)
        self._configure_agent_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; background-color: #455a64; color: white; }"
            "QPushButton:hover { background-color: #37474f; }"
            "QPushButton:pressed { background-color: #263238; }"
            "QPushButton:disabled { background-color: #9ea7aa; color: #ECEFF1; }"
        )
        layout.addWidget(self._configure_agent_button)

        self._train_agent_button = QtWidgets.QPushButton("🤖 Train Agent", group)
        self._train_agent_button.setToolTip(
            "Submit a headless training run to the trainer daemon.\n"
            "Training will run in the background with live telemetry streaming."
        )
        self._train_agent_button.setEnabled(False)
        self._train_agent_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; background-color: #1976d2; color: white; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:pressed { background-color: #0d47a1; }"
            "QPushButton:disabled { background-color: #90caf9; color: #E3F2FD; }"
        )
        layout.addWidget(self._train_agent_button)

        self._trained_agent_button = QtWidgets.QPushButton("📦 Load Trained Policy", group)
        self._trained_agent_button.setToolTip(
            "Select an existing policy or checkpoint to evaluate inside the GUI."
        )
        self._trained_agent_button.setEnabled(False)
        self._trained_agent_button.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; background-color: #388e3c; color: white; }"
            "QPushButton:hover { background-color: #2e7d32; }"
            "QPushButton:pressed { background-color: #1b5e20; }"
            "QPushButton:disabled { background-color: #a5d6a7; color: #E8F5E9; }"
        )
        layout.addWidget(self._trained_agent_button)
        return group

    def _create_control_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Game Control Flow", parent)
        layout = QtWidgets.QVBoxLayout(group)

        row1 = QtWidgets.QHBoxLayout()
        self._start_button = QtWidgets.QPushButton("Start Game", group)
        self._pause_button = QtWidgets.QPushButton("Pause Game", group)
        self._continue_button = QtWidgets.QPushButton("Continue Game", group)
        row1.addWidget(self._start_button)
        row1.addWidget(self._pause_button)
        row1.addWidget(self._continue_button)
        layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self._terminate_button = QtWidgets.QPushButton("Terminate Game", group)
        self._step_button = QtWidgets.QPushButton("Agent Step", group)
        self._reset_button = QtWidgets.QPushButton("Reset", group)
        row2.addWidget(self._terminate_button)
        row2.addWidget(self._step_button)
        row2.addWidget(self._reset_button)
        layout.addLayout(row2)
        return group

    def _create_status_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        self._status_group = QtWidgets.QGroupBox("Status", parent)
        layout = QtWidgets.QGridLayout(self._status_group)
        self._step_label = QtWidgets.QLabel("0", self._status_group)
        self._reward_label = QtWidgets.QLabel("0.0", self._status_group)
        self._total_reward_label = QtWidgets.QLabel("0.00", self._status_group)
        self._terminated_label = QtWidgets.QLabel("No", self._status_group)
        self._truncated_label = QtWidgets.QLabel("No", self._status_group)
        self._turn_label = QtWidgets.QLabel("human", self._status_group)
        self._awaiting_label = QtWidgets.QLabel("–", self._status_group)
        self._session_time_label = QtWidgets.QLabel("00:00:00", self._status_group)
        self._active_time_label = QtWidgets.QLabel("—", self._status_group)
        self._outcome_time_label = QtWidgets.QLabel("—", self._status_group)

        fields = [
            ("Step", self._step_label),
            ("Reward", self._reward_label),
            ("Total Reward", self._total_reward_label),
            ("Episode Finished", self._terminated_label),
            ("Episode Aborted", self._truncated_label),
            ("Turn", self._turn_label),
            ("Awaiting Input", self._awaiting_label),
            ("Session Uptime", self._session_time_label),
            ("Active Play Time", self._active_time_label),
            ("Outcome Time", self._outcome_time_label),
        ]

        midpoint = (len(fields) + 1) // 2
        columns = [fields[:midpoint], fields[midpoint:]]
        for col_index, column_fields in enumerate(columns):
            for row_index, (title, value_label) in enumerate(column_fields):
                title_label = QtWidgets.QLabel(title, self._status_group)
                title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
                base_col = col_index * 2
                layout.addWidget(title_label, row_index, base_col)
                layout.addWidget(value_label, row_index, base_col + 1)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        return self._status_group

    def _on_tab_changed(self, index: int) -> None:
        if not hasattr(self, "_tab_to_mode"):
            return
        mode = self._tab_to_mode.get(index)
        if mode is not None:
            self._apply_mode_from_tab(mode)

        is_human_tab = self._tab_widget.widget(index) is self._human_tab
        if hasattr(self, "_mode_group") and self._mode_group is not None:
            self._mode_group.setVisible(is_human_tab)
        if hasattr(self, "_status_group") and self._status_group is not None:
            self._status_group.setVisible(is_human_tab)
        if hasattr(self, "_control_buttons_widget") and self._control_buttons_widget is not None:
            self._control_buttons_widget.setVisible(is_human_tab)

    def _apply_mode_from_tab(self, mode: ControlMode) -> None:
        index = self._mode_combo.findData(mode)
        if index < 0:
            current = self._current_game
            supported = tuple(self._available_modes.get(current, ())) if current is not None else ()
            if supported:
                mode = supported[0]
                index = self._mode_combo.findData(mode)
        if index >= 0:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(index)
            self._mode_combo.blockSignals(False)
        self._emit_mode_changed(mode)

    def _connect_signals(self) -> None:
        self._game_combo.currentIndexChanged.connect(self._on_game_changed)
        self._seed_spin.valueChanged.connect(lambda _: self._update_control_states())
        self._wire_mode_combo()
        self._worker_combo.currentIndexChanged.connect(self._on_worker_selection_changed)
        self._configure_agent_button.clicked.connect(self._emit_agent_form_requested)

        self._load_button.clicked.connect(self._on_load_clicked)
        self._train_agent_button.clicked.connect(self._emit_train_agent_requested)
        self._trained_agent_button.clicked.connect(self._emit_trained_agent_requested)
        self._start_button.clicked.connect(self._on_start_clicked)
        self._pause_button.clicked.connect(self._on_pause_clicked)
        self._continue_button.clicked.connect(self._on_continue_clicked)
        self._terminate_button.clicked.connect(self._on_terminate_clicked)
        self._step_button.clicked.connect(self._on_step_clicked)
        self._reset_button.clicked.connect(self._on_reset_clicked)
        self._actor_combo.currentIndexChanged.connect(self._on_actor_selection_changed)

    def set_seed_value(self, seed: int) -> None:
        clamped = max(1, min(seed, self._seed_spin.maximum()))
        if self._seed_spin.value() == clamped:
            return
        self._seed_spin.blockSignals(True)
        self._seed_spin.setValue(clamped)
        self._seed_spin.blockSignals(False)
        self._update_control_states()

    # ------------------------------------------------------------------
    # Signal emitters
    # ------------------------------------------------------------------
    def _emit_mode_changed(self, mode: ControlMode) -> None:
        if self._current_mode != mode:
            self._current_mode = mode
            self._persist_mode_preference(mode)
        self.control_mode_changed.emit(mode)
        self._update_control_states()

    def _emit_game_changed(self, game: GameId | None) -> None:
        if game is None:
            return
        if self._current_game != game:
            self._current_game = game
        self.game_changed.emit(game)
        self.update_modes(game)
        self._refresh_game_config_ui()
        self.set_slippery_visible(game == GameId.FROZEN_LAKE)

    # ------------------------------------------------------------------
    # Qt slots
    # ------------------------------------------------------------------
    def _on_game_changed(self, index: int) -> None:
        game = self._game_combo.itemData(index)
        if isinstance(game, GameId):
            self._emit_game_changed(game)

    def _on_seed_reuse_changed(self, state: int) -> None:
        try:
            state_enum = QtCore.Qt.CheckState(state)
        except ValueError:
            state_enum = QtCore.Qt.CheckState.Unchecked
        enabled = state_enum == QtCore.Qt.CheckState.Checked
        self._allow_seed_reuse = enabled
        if enabled:
            hint = "Seeds auto-increment by default. Adjust before loading to reuse a previous seed."
        else:
            hint = "Seed increments automatically after each episode."
        self._seed_spin.setToolTip(hint)

    def _on_load_clicked(self) -> None:
        if self._current_game is None:
            return
        self.load_requested.emit(self._current_game, self._current_mode, self.current_seed())

    def _on_start_clicked(self) -> None:
        self._game_started = True
        self._game_paused = False
        self._update_control_states()
        self.start_game_requested.emit()

    def _on_pause_clicked(self) -> None:
        self._game_paused = True
        self._update_control_states()
        self.pause_game_requested.emit()

    def _on_continue_clicked(self) -> None:
        self._game_paused = False
        self._update_control_states()
        self.continue_game_requested.emit()

    def _on_terminate_clicked(self) -> None:
        self._game_started = False
        self._game_paused = False
        self._update_control_states()
        self.terminate_game_requested.emit()

    def _on_step_clicked(self) -> None:
        self.agent_step_requested.emit()

    def _on_reset_clicked(self) -> None:
        self._game_started = False
        self._game_paused = False
        self._update_control_states()
        self.reset_requested.emit(self.current_seed())

    def _on_slippery_toggled(self, state: int) -> None:
        try:
            state_enum = QtCore.Qt.CheckState(state)
        except ValueError:
            state_enum = QtCore.Qt.CheckState.Unchecked
        enabled = state_enum == QtCore.Qt.CheckState.Checked
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
        overrides["is_slippery"] = enabled
        self.slippery_toggled.emit(enabled)

    def _on_taxi_config_changed(self, param_name: str, value: bool) -> None:
        """Handle changes to Taxi configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.TAXI, {})
        overrides[param_name] = value
        self.taxi_config_changed.emit(param_name, value)

    def _on_cliff_config_changed(self, param_name: str, value: bool) -> None:
        """Handle changes to CliffWalking configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.CLIFF_WALKING, {})
        overrides[param_name] = value
        self.cliff_config_changed.emit(param_name, value)

    def _on_lunar_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to LunarLander configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.LUNAR_LANDER, {})
        overrides[param_name] = value
        self.lunar_config_changed.emit(param_name, value)

    def _on_car_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to CarRacing configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.CAR_RACING, {})
        overrides[param_name] = value
        self.car_config_changed.emit(param_name, value)

    def _on_bipedal_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to BipedalWalker configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.BIPEDAL_WALKER, {})
        overrides[param_name] = value
        self.bipedal_config_changed.emit(param_name, value)

    def _on_frozen_v2_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to FrozenLake-v2 configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE_V2, {})
        overrides[param_name] = value
        self.frozen_v2_config_changed.emit(param_name, value)

    def _resolve_minigrid_defaults(self, game_id: GameId) -> MiniGridConfig:
        stored = self._minigrid_defaults.get(game_id)
        if stored is not None:
            return stored
        return resolve_minigrid_default_config(game_id)

    def _on_minigrid_config_changed(self, param_name: str, value: object) -> None:
        if self._current_game not in MINIGRID_GAME_IDS:
            return
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value
        config_bucket = self._game_configs.setdefault(current_game, {})
        config_bucket[param_name] = value

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------
    def _wire_mode_combo(self) -> None:
        self._mode_combo.currentIndexChanged.connect(self._on_mode_selection_changed)

    def _on_mode_selection_changed(self, index: int) -> None:
        mode = self._mode_combo.itemData(index)
        if not isinstance(mode, ControlMode):
            return
        if mode == self._current_mode:
            return
        self._current_mode = mode
        self._persist_mode_preference(mode)
        self._emit_mode_changed(mode)

    def _update_control_states(self) -> None:
        """Update button states based on game flow."""
        is_human = self._current_mode == ControlMode.HUMAN_ONLY
        
        # Start button: enabled only if game not started and environment loaded
        self._start_button.setEnabled(not self._game_started)
        
        # Pause button: enabled only if game started and not paused
        self._pause_button.setEnabled(self._game_started and not self._game_paused)
        
        # Continue button: enabled only if game paused
        self._continue_button.setEnabled(self._game_paused)
        
        # Terminate button: enabled only if game started
        self._terminate_button.setEnabled(self._game_started)
        
        # Agent Step: enabled only if game started, not paused, not human-only, and not auto-running
        self._step_button.setEnabled(
            self._game_started and not self._game_paused and not is_human and not self._auto_running
        )
        
        # Reset: always enabled (can reset even during active game)
        self._reset_button.setEnabled(True)

        # Enable actor selector only when an agent can participate
        has_agent_component = self._current_mode != ControlMode.HUMAN_ONLY
        agent_only_mode = self._current_mode == ControlMode.AGENT_ONLY
        self._actor_combo.setEnabled(has_agent_component and self._actor_combo.count() > 0)
        
        worker_def = self._current_worker_definition()
        supports_training = bool(worker_def and worker_def.supports_training)
        supports_policy = bool(worker_def and worker_def.supports_policy_load)
        self._configure_agent_button.setEnabled(agent_only_mode and supports_training)
        self._train_agent_button.setEnabled(agent_only_mode and supports_training)
        self._trained_agent_button.setEnabled(agent_only_mode and supports_policy)

        self._update_actor_description()
        self._update_worker_description()

    def _apply_current_mode_selection(self) -> None:
        index = self._mode_combo.findData(self._current_mode)
        if index < 0:
            return
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(index)
        self._mode_combo.blockSignals(False)

    def _persist_mode_preference(self, mode: ControlMode) -> None:
        """Persist control mode preference to environment variable."""
        os.environ["GYM_CONTROL_MODE"] = mode.name.lower()

    def _load_mode_preference(self, fallback: ControlMode) -> ControlMode:
        """Load control mode preference from environment variable."""
        stored = os.environ.get("GYM_CONTROL_MODE")
        if stored is None:
            return fallback
        stored = stored.strip().upper()
        try:
            return ControlMode[stored]
        except KeyError:
            try:
                return ControlMode(stored.lower())
            except ValueError:
                return fallback

    def _populate_actor_combo(self) -> None:
        self._actor_combo.blockSignals(True)
        self._actor_combo.clear()
        for descriptor in self._actor_order:
            self._actor_combo.addItem(descriptor.display_name, descriptor.actor_id)
        self._actor_combo.blockSignals(False)

        if not self._actor_order:
            self._actor_description.setText("No actors registered")
            return

        default_id = self._active_actor_id or self._actor_order[0].actor_id
        index = self._actor_combo.findData(default_id)
        if index < 0:
            index = 0
        self._actor_combo.setCurrentIndex(index)
        current_data = self._actor_combo.currentData()
        self._active_actor_id = current_data if isinstance(current_data, str) else None
        self._update_actor_description()

    def _populate_worker_combo(self) -> None:
        self._worker_combo.blockSignals(True)
        self._worker_combo.clear()
        for definition in self._worker_definitions:
            self._worker_combo.addItem(definition.display_name, definition.worker_id)

        if not self._worker_definitions:
            self._worker_combo.setEnabled(False)
            self._worker_combo.blockSignals(False)
            self._worker_description.setText(
                "No worker integrations are registered. Configure a worker to enable training."
            )
            self._current_worker_id = None
            self._update_control_states()
            return

        # Select the first worker by default without emitting cascaded signals
        self._worker_combo.setEnabled(True)
        self._worker_combo.setCurrentIndex(0)
        self._worker_combo.blockSignals(False)
        self._on_worker_selection_changed(self._worker_combo.currentIndex())

    def _on_actor_selection_changed(self, index: int) -> None:
        actor_id = self._actor_combo.itemData(index)
        if not isinstance(actor_id, str):
            return
        if actor_id == self._active_actor_id:
            return
        self._active_actor_id = actor_id
        self._update_actor_description()
        self.actor_changed.emit(actor_id)

    def _on_worker_selection_changed(self, index: int) -> None:
        worker_id = self._worker_combo.itemData(index)
        if not isinstance(worker_id, str):
            worker_id = None
        if worker_id == self._current_worker_id:
            return
        self._current_worker_id = worker_id
        self._update_worker_description()
        self._update_control_states()
        if worker_id:
            self.worker_changed.emit(worker_id)

    def _emit_agent_form_requested(self) -> None:
        worker_id = self._current_worker_id
        if worker_id is None:
            return
        self.agent_form_requested.emit(worker_id)

    def _emit_train_agent_requested(self) -> None:
        worker_id = self._current_worker_id
        if worker_id is None:
            return
        self.train_agent_requested.emit(worker_id)

    def _emit_trained_agent_requested(self) -> None:
        worker_id = self._current_worker_id
        if worker_id is None:
            return
        self.trained_agent_requested.emit(worker_id)

    def _update_actor_description(self) -> None:
        if self._active_actor_id is None:
            self._actor_description.setText("—")
            return
        descriptor = self._actor_descriptors.get(self._active_actor_id)
        if descriptor is None or descriptor.description is None:
            self._actor_description.setText("—")
            return
        self._actor_description.setText(descriptor.description)

    def _current_worker_definition(self) -> Optional[WorkerDefinition]:
        if self._current_worker_id is None:
            return None
        for definition in self._worker_definitions:
            if definition.worker_id == self._current_worker_id:
                return definition
        return None

    def _update_worker_description(self) -> None:
        definition = self._current_worker_definition()
        if definition is None:
            self._worker_description.setText(
                "No worker selected. Please choose an integration to enable training."
            )
            return
        capabilities = ", ".join(definition.capabilities()) or "No declared capabilities"
        self._worker_description.setText(f"{definition.description}\nCapabilities: {capabilities}")

    @staticmethod
    def _format_bool(value: bool) -> str:
        return "Yes" if value else "No"

    def _clear_config_layout(self) -> None:
        while self._config_layout.count():
            item = self._config_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            layout = item.layout()
            if layout is not None:
                layout.deleteLater()

    def _refresh_game_config_ui(self) -> None:
        self._clear_config_layout()
        if self._frozen_slippery_checkbox is not None:
            try:
                self._frozen_slippery_checkbox.deleteLater()
            except RuntimeError:
                pass
            self._frozen_slippery_checkbox = None

        if self._current_game is None:
            label = QtWidgets.QLabel(
                "Select an environment to view configuration options.",
                self._config_group,
            )
            label.setWordWrap(True)
            self._config_layout.addRow("", label)
            return

        if self._current_game == GameId.FROZEN_LAKE:
            overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
            checkbox = build_frozenlake_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.frozen_lake_config,
                on_slippery_toggled=self._on_slippery_toggled,
            )
            self._frozen_slippery_checkbox = checkbox
        elif self._current_game is not None and self._current_game in MINIGRID_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            defaults = self._resolve_minigrid_defaults(current_game)
            callbacks = MinigridControlCallbacks(on_change=self._on_minigrid_config_changed)
            build_minigrid_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                defaults=defaults,
                callbacks=callbacks,
            )
        elif self._current_game is not None and self._current_game in ALE_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            callbacks = ALEControlCallbacks(on_change=self._on_ale_config_changed)
            # Provide a lightweight defaults object consistent with current selection
            defaults = ALEConfig(env_id=current_game.value)
            build_ale_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                defaults=defaults,
                callbacks=callbacks,
            )
        elif self._current_game == GameId.FROZEN_LAKE_V2:
            overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE_V2, {})
            build_frozenlake_v2_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                on_change=self._on_frozen_v2_config_changed,
            )
        elif self._current_game == GameId.LUNAR_LANDER:
            overrides = self._game_overrides.setdefault(GameId.LUNAR_LANDER, {})
            build_lunarlander_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.lunar_lander_config,
                on_change=self._on_lunar_config_changed,
            )
        elif self._current_game == GameId.TAXI:
            overrides = self._game_overrides.setdefault(GameId.TAXI, {})
            build_taxi_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.taxi_config,
                on_change=self._on_taxi_config_changed,
            )
        elif self._current_game == GameId.CLIFF_WALKING:
            overrides = self._game_overrides.setdefault(GameId.CLIFF_WALKING, {})
            build_cliff_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.cliff_walking_config,
                on_change=self._on_cliff_config_changed,
            )
        elif self._current_game == GameId.CAR_RACING:
            overrides = self._game_overrides.setdefault(GameId.CAR_RACING, {})
            build_car_racing_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.car_racing_config,
                on_change=self._on_car_config_changed,
            )
        elif self._current_game == GameId.BIPEDAL_WALKER:
            overrides = self._game_overrides.setdefault(GameId.BIPEDAL_WALKER, {})
            build_bipedal_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.bipedal_walker_config,
                on_change=self._on_bipedal_config_changed,
            )
        else:
            label = QtWidgets.QLabel(
                "No additional configuration options for this environment.",
                self._config_group,
            )
            label.setWordWrap(True)
            self._config_layout.addRow("", label)

    def _on_ale_config_changed(self, param_name: str, value: object) -> None:
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value
