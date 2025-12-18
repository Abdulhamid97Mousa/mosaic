"""Multi-agent game routing handler for MainWindow.

Extracts multi-agent game loading and routing logic from MainWindow.
Routes game requests to appropriate environment loaders.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QStatusBar, QWidget

if TYPE_CHECKING:
    from gym_gui.core.enums import GameId
    from gym_gui.ui.handlers.env_loaders import (
        ChessEnvLoader,
        ConnectFourEnvLoader,
        GoEnvLoader,
        TicTacToeEnvLoader,
    )

_LOGGER = logging.getLogger(__name__)


class MultiAgentGameHandler:
    """Handler for multi-agent game routing.

    Routes load/start/reset requests to appropriate environment loaders
    for Chess, Connect Four, Go, and Tic-Tac-Toe games.
    """

    def __init__(
        self,
        status_bar: QStatusBar,
        chess_loader: "ChessEnvLoader",
        connect_four_loader: "ConnectFourEnvLoader",
        go_loader: "GoEnvLoader",
        tictactoe_loader: "TicTacToeEnvLoader",
        set_game_info: Callable[[str], None],
        get_game_info: Callable[["GameId"], Optional[str]],
        parent: Optional[QWidget] = None,
    ):
        """Initialize the handler.

        Args:
            status_bar: Status bar for messages.
            chess_loader: Chess environment loader.
            connect_four_loader: Connect Four environment loader.
            go_loader: Go environment loader.
            tictactoe_loader: Tic-Tac-Toe environment loader.
            set_game_info: Callback to set game info panel HTML.
            get_game_info: Callback to get game info HTML by GameId.
            parent: Optional parent widget.
        """
        self._status_bar = status_bar
        self._chess_loader = chess_loader
        self._connect_four_loader = connect_four_loader
        self._go_loader = go_loader
        self._tictactoe_loader = tictactoe_loader
        self._set_game_info = set_game_info
        self._get_game_info = get_game_info
        self._parent = parent

    def on_load_requested(self, env_id: str, seed: int) -> None:
        """Handle load request from Multi-Agent tab (Human vs Agent mode).

        Routes to appropriate loader based on environment ID.

        Args:
            env_id: Environment ID (e.g., "chess_v6", "connect_four_v3").
            seed: Random seed for game initialization.
        """
        _LOGGER.info(
            "Multi-agent env load requested | env_id=%s seed=%s",
            env_id,
            seed,
        )

        if env_id == "chess_v6":
            self._load_chess_game(seed)
        elif env_id == "connect_four_v3":
            self._load_connect_four_game(seed)
        elif env_id == "go_v5":
            self._load_go_game(seed)
        elif env_id == "tictactoe_v3":
            self._load_tictactoe_game(seed)
        else:
            # Other PettingZoo environments not yet implemented
            self._status_bar.showMessage(
                f"Multi-agent environment not yet supported: {env_id}",
                5000,
            )

    def _load_chess_game(self, seed: int) -> None:
        """Load and initialize the Chess game with interactive board."""
        from gym_gui.core.enums import GameId

        self._chess_loader.load(seed, parent=self._parent)
        desc = self._get_game_info(GameId.CHESS)
        if desc:
            self._set_game_info(desc)

    def _load_connect_four_game(self, seed: int) -> None:
        """Load and initialize the Connect Four game with interactive board."""
        from gym_gui.core.enums import GameId

        self._connect_four_loader.load(seed, parent=self._parent)
        desc = self._get_game_info(GameId.CONNECT_FOUR)
        if desc:
            self._set_game_info(desc)

    def _load_go_game(self, seed: int) -> None:
        """Load and initialize the Go game with interactive board."""
        from gym_gui.core.enums import GameId

        self._go_loader.load(seed, parent=self._parent)
        desc = self._get_game_info(GameId.GO)
        if desc:
            self._set_game_info(desc)

    def _load_tictactoe_game(self, seed: int) -> None:
        """Load and initialize the Tic-Tac-Toe game with interactive board."""
        from gym_gui.core.enums import GameId

        self._tictactoe_loader.load(seed, parent=self._parent)
        desc = self._get_game_info(GameId.TIC_TAC_TOE)
        if desc:
            self._set_game_info(desc)

    def on_start_requested(self, env_id: str, human_agent: str, seed: int) -> None:
        """Handle start game request from Multi-Agent tab.

        Args:
            env_id: Environment ID (e.g., "chess", "chess_v6").
            human_agent: Which agent the human plays ("player_0" or "player_1").
            seed: Random seed.
        """
        if env_id in ("chess", "chess_v6") and self._chess_loader.is_loaded:
            self._chess_loader.on_start_requested(human_agent, seed)
        elif env_id in ("connect_four", "connect_four_v3") and self._connect_four_loader.is_loaded:
            self._connect_four_loader.on_start_requested(human_agent, seed)
        elif env_id in ("go", "go_v5") and self._go_loader.is_loaded:
            self._go_loader.on_start_requested(human_agent, seed)
        elif env_id in ("tictactoe", "tictactoe_v3") and self._tictactoe_loader.is_loaded:
            self._tictactoe_loader.on_start_requested(human_agent, seed)
        else:
            self._status_bar.showMessage(f"Start game not supported for: {env_id}", 3000)

    def on_reset_requested(self, seed: int) -> None:
        """Handle reset game request from Multi-Agent tab.

        Resets the currently active game.

        Args:
            seed: New random seed for reset.
        """
        if self._chess_loader.is_loaded:
            self._chess_loader.on_reset_requested(seed)
        elif self._connect_four_loader.is_loaded:
            self._connect_four_loader.on_reset_requested(seed)
        elif self._go_loader.is_loaded:
            self._go_loader.on_reset_requested(seed)
        elif self._tictactoe_loader.is_loaded:
            self._tictactoe_loader.on_reset_requested(seed)
        else:
            self._status_bar.showMessage("No active game to reset", 3000)

    def on_ai_opponent_changed(self, opponent_type: str, difficulty: str) -> None:
        """Handle AI opponent selection change.

        Args:
            opponent_type: Type of AI opponent ("random", "stockfish", "custom").
            difficulty: Difficulty level for engines like Stockfish.
        """
        # Currently only chess supports AI config changes
        self._chess_loader.on_ai_config_changed(opponent_type, difficulty)
