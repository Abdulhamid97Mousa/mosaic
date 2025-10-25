"""Control panel container widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtWidgets

from gym_gui.config.game_configs import (
    CliffWalkingConfig,
    CarRacingConfig,
    BipedalWalkerConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    TaxiConfig,
)
from gym_gui.core.factories.adapters import available_games
from gym_gui.ui.widgets.control_panel import ControlPanelConfig, ControlPanelWidget

if TYPE_CHECKING:
    from gym_gui.config.settings import Settings
    from gym_gui.controllers.session import SessionController
    from gym_gui.services.actor import ActorService


class ControlPanelContainer(QtWidgets.QWidget):
    """Container for the control panel widget with all signal connections."""

    def __init__(
        self,
        settings: Settings,
        session: SessionController,
        actor_service: ActorService,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._session = session
        self._actor_service = actor_service

        # Build control panel config
        available_modes = {}
        for game in available_games():
            available_modes[game] = SessionController.supported_control_modes(game)

        actor_descriptors = self._actor_service.describe_actors()
        default_actor_id = self._actor_service.get_active_actor_id()

        control_config = ControlPanelConfig(
            available_modes=available_modes,
            default_mode=settings.default_control_mode,
            frozen_lake_config=FrozenLakeConfig(is_slippery=False),
            taxi_config=TaxiConfig(is_raining=False, fickle_passenger=False),
            cliff_walking_config=CliffWalkingConfig(is_slippery=False),
            lunar_lander_config=LunarLanderConfig(),
            car_racing_config=CarRacingConfig.from_env(),
            bipedal_walker_config=BipedalWalkerConfig.from_env(),
            default_seed=settings.default_seed,
            allow_seed_reuse=settings.allow_seed_reuse,
            actors=actor_descriptors,
            default_actor_id=default_actor_id,
        )

        self._control_panel = ControlPanelWidget(config=control_config, parent=self)
        if default_actor_id is not None:
            self._control_panel.set_active_actor(default_actor_id)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._control_panel)

    def get_control_panel(self) -> ControlPanelWidget:
        """Get the underlying control panel widget."""
        return self._control_panel

    def populate_games(self, games, default=None):
        """Populate games in the control panel."""
        self._control_panel.populate_games(games, default=default)

    def current_game(self):
        """Get currently selected game."""
        return self._control_panel.current_game()

    def current_mode(self):
        """Get currently selected control mode."""
        return self._control_panel.current_mode()

    def current_actor(self):
        """Get currently selected actor."""
        return self._control_panel.current_actor()

    def set_active_actor(self, actor_id: str) -> None:
        """Set the active actor."""
        self._control_panel.set_active_actor(actor_id)

    def set_turn(self, turn: str) -> None:
        """Set the turn label."""
        self._control_panel.set_turn(turn)

    # Delegate all signals from the control panel
    @property
    def load_requested(self):
        """Signal: load requested."""
        return self._control_panel.load_requested

    @property
    def reset_requested(self):
        """Signal: reset requested."""
        return self._control_panel.reset_requested

    @property
    def train_agent_requested(self):
        """Signal: train agent requested."""
        return self._control_panel.train_agent_requested

    @property
    def trained_agent_requested(self):
        """Signal: trained agent requested."""
        return self._control_panel.trained_agent_requested

    @property
    def start_game_requested(self):
        """Signal: start game requested."""
        return self._control_panel.start_game_requested

    @property
    def pause_game_requested(self):
        """Signal: pause game requested."""
        return self._control_panel.pause_game_requested

    @property
    def continue_game_requested(self):
        """Signal: continue game requested."""
        return self._control_panel.continue_game_requested

    @property
    def terminate_game_requested(self):
        """Signal: terminate game requested."""
        return self._control_panel.terminate_game_requested

    @property
    def agent_step_requested(self):
        """Signal: agent step requested."""
        return self._control_panel.agent_step_requested

    @property
    def game_changed(self):
        """Signal: game changed."""
        return self._control_panel.game_changed

    @property
    def control_mode_changed(self):
        """Signal: control mode changed."""
        return self._control_panel.control_mode_changed

    @property
    def actor_changed(self):
        """Signal: actor changed."""
        return self._control_panel.actor_changed

    @property
    def agent_form_requested(self):
        """Signal: agent form requested."""
        return self._control_panel.agent_form_requested

    @property
    def slippery_toggled(self):
        """Signal: slippery toggled."""
        return self._control_panel.slippery_toggled

    @property
    def frozen_v2_config_changed(self):
        """Signal: frozen v2 config changed."""
        return self._control_panel.frozen_v2_config_changed

    @property
    def taxi_config_changed(self):
        """Signal: taxi config changed."""
        return self._control_panel.taxi_config_changed

    @property
    def cliff_config_changed(self):
        """Signal: cliff config changed."""
        return self._control_panel.cliff_config_changed

    @property
    def lunar_config_changed(self):
        """Signal: lunar config changed."""
        return self._control_panel.lunar_config_changed

    @property
    def car_config_changed(self):
        """Signal: car config changed."""
        return self._control_panel.car_config_changed

    @property
    def bipedal_config_changed(self):
        """Signal: bipedal config changed."""
        return self._control_panel.bipedal_config_changed


__all__ = ["ControlPanelContainer"]
