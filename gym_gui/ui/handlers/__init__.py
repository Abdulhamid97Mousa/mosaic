"""Handler classes for MainWindow using composition pattern.

These handlers extract related functionality from MainWindow to improve
maintainability and testability. Each handler is a standalone class that
receives its dependencies via constructor injection.

- GameConfigHandler: Environment configuration change handlers
- MPCHandler: MuJoCo MPC launch/stop handlers
- LogHandler: Log filtering and display
- ChessHandler: Chess game move handling (Human Control Mode)
- ConnectFourHandler: Connect Four game move handling (Human Control Mode)
- GoHandler: Go game move handling (Human Control Mode)
- HumanVsAgentHandler: Human vs Agent AI setup and management

Usage:
    # In MainWindow.__init__:
    self._game_config_handler = GameConfigHandler(
        control_panel=self._control_panel,
        session=self._session,
        status_bar=self._status_bar,
    )

    # For board games, connect signals from BoardGameRendererStrategy:
    self._render_tabs.chess_move_made.connect(self._chess_handler.on_chess_move)
    self._render_tabs.connect_four_column_clicked.connect(
        self._connect_four_handler.on_column_clicked
    )
    self._render_tabs.go_intersection_clicked.connect(
        self._go_handler.on_intersection_clicked
    )

    # For Human vs Agent mode:
    self._human_vs_agent_handler = HumanVsAgentHandler(
        status_bar=self._status_bar,
    )
"""

from __future__ import annotations

from gym_gui.ui.handlers.game_config_handlers import GameConfigHandler
from gym_gui.ui.handlers.log_handlers import LogHandler
from gym_gui.ui.handlers.mpc_handlers import MPCHandler
from gym_gui.ui.handlers.chess_handlers import ChessHandler
from gym_gui.ui.handlers.connect_four_handlers import ConnectFourHandler
from gym_gui.ui.handlers.go_handlers import GoHandler
from gym_gui.ui.handlers.human_vs_agent_handlers import HumanVsAgentHandler

__all__ = [
    "GameConfigHandler",
    "LogHandler",
    "MPCHandler",
    "ChessHandler",
    "ConnectFourHandler",
    "GoHandler",
    "HumanVsAgentHandler",
]
