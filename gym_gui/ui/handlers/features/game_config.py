"""Game configuration change handlers using composition pattern.

This module provides a handler class that manages environment configuration
changes. It receives its dependencies via constructor injection, making it
type-safe and testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gym_gui.config.game_config_builder import GameConfigBuilder
from gym_gui.core.enums import GameId
from gym_gui.ui.environments.single_agent_env.vizdoom import VIZDOOM_GAME_IDS

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget


class GameConfigHandler:
    """Handles game configuration changes from the control panel.

    This class encapsulates all environment configuration logic, delegating
    UI updates to the status bar and environment reloading to the session.

    Args:
        control_panel: The control panel widget for reading current settings.
        session: The session controller for reloading environments.
        status_bar: The status bar for showing feedback messages.
    """

    def __init__(
        self,
        control_panel: "ControlPanelWidget",
        session: "SessionController",
        status_bar: "QStatusBar",
    ) -> None:
        self._control_panel = control_panel
        self._session = session
        self._status_bar = status_bar

    def on_slippery_toggled(self, enabled: bool) -> None:
        """Handle slippery ice toggle from control panel."""
        status = "enabled" if enabled else "disabled"
        current_game = self._control_panel.current_game()

        if current_game == GameId.FROZEN_LAKE and self._session.game_id == GameId.FROZEN_LAKE:
            self._status_bar.showMessage(
                f"Frozen Lake slippery ice {status}. Reloading environment...",
                5000,
            )
            self._reload_environment(GameId.FROZEN_LAKE)
        else:
            self._status_bar.showMessage(
                f"Frozen Lake slippery ice {status}. Reload to apply.",
                5000,
            )

    def on_taxi_config_changed(self, param_name: str, value: bool) -> None:
        """Handle Taxi configuration changes from control panel."""
        status = "enabled" if value else "disabled"
        param_label = "rain" if param_name == "is_raining" else "fickle passenger"
        current_game = self._control_panel.current_game()

        if current_game == GameId.TAXI and self._session.game_id == GameId.TAXI:
            self._status_bar.showMessage(
                f"Taxi {param_label} {status}. Reloading environment...",
                5000,
            )
            self._reload_environment(GameId.TAXI)
        else:
            self._status_bar.showMessage(
                f"Taxi {param_label} {status}. Reload to apply.",
                5000,
            )

    def on_frozen_v2_config_changed(self, param_name: str, value: object) -> None:
        """Handle FrozenLake-v2 configuration changes from control panel."""
        label_map = {
            "is_slippery": "slippery ice",
            "grid_height": "grid height",
            "grid_width": "grid width",
            "start_position": "start position",
            "goal_position": "goal position",
            "hole_count": "hole count",
            "random_holes": "random hole placement",
        }
        current_game = self._control_panel.current_game()
        descriptor = label_map.get(param_name, param_name)
        value_str = self._format_value(value)

        reloading = (
            current_game == GameId.FROZEN_LAKE_V2
            and self._session.game_id == GameId.FROZEN_LAKE_V2
        )
        message = f"FrozenLake-v2 {descriptor} updated to {value_str}."
        if reloading:
            message += " Reloading to apply..."
        self._status_bar.showMessage(message, 5000 if reloading else 4000)

        if reloading:
            self._reload_environment(GameId.FROZEN_LAKE_V2)

    def on_cliff_config_changed(self, _param_name: str, value: bool) -> None:
        """Handle CliffWalking configuration changes from control panel."""
        status = "enabled" if value else "disabled"
        param_label = "slippery cliff"
        current_game = self._control_panel.current_game()

        if (
            current_game == GameId.CLIFF_WALKING
            and self._session.game_id == GameId.CLIFF_WALKING
        ):
            self._status_bar.showMessage(
                f"Cliff Walking {param_label} {status}. Reloading environment...",
                5000,
            )
            self._reload_environment(GameId.CLIFF_WALKING)
        else:
            self._status_bar.showMessage(
                f"Cliff Walking {param_label} {status}. Reload to apply.",
                5000,
            )

    def on_lunar_config_changed(self, param_name: str, value: object) -> None:
        """Handle LunarLander configuration changes from control panel."""
        label_map = {
            "continuous": "continuous control",
            "gravity": "gravity",
            "enable_wind": "wind",
            "wind_power": "wind power",
            "turbulence_power": "turbulence",
        }
        current_game = self._control_panel.current_game()
        descriptor = label_map.get(param_name, param_name)
        value_str = self._format_value(value)

        reloading = (
            current_game == GameId.LUNAR_LANDER
            and self._session.game_id == GameId.LUNAR_LANDER
        )
        message = f"Lunar Lander {descriptor} updated to {value_str}."
        if reloading:
            message += " Reloading to apply..."
        self._status_bar.showMessage(message, 5000 if reloading else 4000)

        if reloading:
            self._reload_environment(GameId.LUNAR_LANDER)

    def on_car_config_changed(self, param_name: str, value: object) -> None:
        """Handle CarRacing configuration changes from control panel."""
        label_map = {
            "continuous": "continuous control",
            "domain_randomize": "domain randomization",
            "lap_complete_percent": "lap completion requirement",
            "max_episode_steps": "episode step limit",
            "max_episode_seconds": "episode time limit",
        }
        current_game = self._control_panel.current_game()
        descriptor = label_map.get(param_name, param_name)
        value_str = self._format_value(value, allow_none=False)

        reloading = (
            current_game == GameId.CAR_RACING
            and self._session.game_id == GameId.CAR_RACING
        )
        message = f"Car Racing {descriptor} updated to {value_str}."
        if reloading:
            message += " Reloading to apply..."
        self._status_bar.showMessage(message, 5000 if reloading else 4000)

        if reloading:
            self._reload_environment(GameId.CAR_RACING)

    def on_bipedal_config_changed(self, param_name: str, value: object) -> None:
        """Handle BipedalWalker configuration changes from control panel."""
        label_map = {
            "hardcore": "hardcore terrain",
            "max_episode_steps": "episode step limit",
            "max_episode_seconds": "episode time limit",
        }
        current_game = self._control_panel.current_game()
        descriptor = label_map.get(param_name, param_name)
        value_str = self._format_value(value)

        reloading = (
            current_game == GameId.BIPEDAL_WALKER
            and self._session.game_id == GameId.BIPEDAL_WALKER
        )
        message = f"Bipedal Walker {descriptor} {value_str}."
        if reloading:
            message += " Reloading to apply..."
        self._status_bar.showMessage(message, 5000 if reloading else 4000)

        if reloading:
            self._reload_environment(GameId.BIPEDAL_WALKER)

    def on_vizdoom_config_changed(self, param_name: str, value: object) -> None:
        """Handle ViZDoom configuration changes from control panel."""
        # Note: sound_enabled is disabled in the UI (causes crashes in embedded mode)
        label_map = {
            "screen_resolution": "resolution",
            "screen_format": "screen format",
            "render_hud": "HUD",
            "render_weapon": "weapon",
            "render_crosshair": "crosshair",
            "render_particles": "particles",
            "render_decals": "decals",
            "episode_timeout": "episode timeout",
            "living_reward": "living reward",
            "death_penalty": "death penalty",
            "depth_buffer": "depth buffer",
            "labels_buffer": "labels buffer",
            "automap_buffer": "automap buffer",
        }
        current_game = self._control_panel.current_game()
        if current_game is None:
            return
        descriptor = label_map.get(param_name, param_name.replace("_", " "))
        value_str = self._format_value(value, allow_none=False)

        # Check if we're currently running a ViZDoom game
        is_vizdoom_current = current_game in VIZDOOM_GAME_IDS
        is_vizdoom_session = self._session.game_id in VIZDOOM_GAME_IDS
        reloading = is_vizdoom_current and is_vizdoom_session and current_game == self._session.game_id

        message = f"ViZDoom {descriptor} updated to {value_str}."
        if reloading:
            message += " Reloading to apply..."
        self._status_bar.showMessage(message, 5000 if reloading else 4000)

        if reloading:
            self._reload_environment(current_game)

    def _reload_environment(self, game_id: GameId) -> None:
        """Reload the environment with current control panel settings."""
        mode = self._control_panel.current_mode()
        seed = self._control_panel.current_seed()
        overrides = self._control_panel.get_overrides(game_id)
        game_config = GameConfigBuilder.build_config(game_id, overrides)
        self._session.load_environment(
            game_id,
            mode,
            seed=seed,
            game_config=game_config,
        )

    @staticmethod
    def _format_value(value: object, allow_none: bool = True) -> str:
        """Format a configuration value for display."""
        if value is None:
            return "default" if allow_none else str(value)
        elif isinstance(value, bool):
            return "enabled" if value else "disabled"
        elif isinstance(value, tuple):
            return f"({value[0]}, {value[1]})"
        elif isinstance(value, (int, float)):
            if isinstance(value, int):
                return str(value)
            return f"{value:.2f}"
        else:
            return str(value)


__all__ = ["GameConfigHandler"]
