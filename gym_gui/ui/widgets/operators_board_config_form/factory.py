"""Factory for creating game-specific board configuration dialogs.

This module provides the BoardConfigDialogFactory which creates the
appropriate configuration dialog for each supported board game.

New games can be registered dynamically using the register() method.

Example:
    from gym_gui.ui.widgets.operators_board_config_form import BoardConfigDialogFactory

    # Check if game is supported
    if BoardConfigDialogFactory.supports("chess_v6"):
        dialog = BoardConfigDialogFactory.create("chess_v6", initial_fen, parent)
        if dialog.exec() == QDialog.Accepted:
            custom_state = dialog.get_state()

    # Register a new game
    BoardConfigDialogFactory.register("my_game", MyGameConfigDialog)
"""

import logging
from functools import partial
from typing import Dict, Type, Optional, List

from PyQt6 import QtWidgets

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_UI_BOARD_CONFIG_FACTORY_CREATE,
    LOG_UI_BOARD_CONFIG_UNSUPPORTED_GAME,
)
from .base import BoardConfigDialog

_LOGGER = logging.getLogger(__name__)
_log = partial(log_constant, _LOGGER)


class BoardConfigDialogFactory:
    """Factory to create game-specific configuration dialogs.

    This factory maintains a registry of game IDs to dialog classes,
    allowing new games to be registered dynamically.

    Supported games:
    - chess_v6: PettingZoo Chess
    - open_spiel/checkers: OpenSpiel Checkers (8x8, American rules)
    - draughts/american_checkers: American Checkers (8x8)
    - draughts/russian_checkers: Russian Checkers (8x8, flying kings)
    - draughts/international_draughts: International Draughts (10x10)

    Planned games:
    - go_v5: Go (9x9, 13x13, 19x19)
    - connect_four_v3: Connect Four
    - tictactoe_v3: Tic-Tac-Toe
    """

    _registry: Dict[str, Type[BoardConfigDialog]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Lazy initialization of the registry with built-in games."""
        if cls._initialized:
            return

        # Mark as initialized first to prevent recursion
        cls._initialized = True

        # Import and register chess editor
        try:
            from .chess_editor import ChessConfigDialog
            cls._registry["chess_v6"] = ChessConfigDialog
            _LOGGER.debug("Registered ChessConfigDialog for chess_v6")
        except ImportError as e:
            _LOGGER.warning(f"Failed to register chess editor: {e}")

        # Import and register OpenSpiel Checkers editor (8x8)
        try:
            from .checkers_editor import OpenSpielCheckersConfigDialog
            cls._registry["open_spiel/checkers"] = OpenSpielCheckersConfigDialog
            _LOGGER.debug("Registered OpenSpielCheckersConfigDialog for open_spiel/checkers")
        except ImportError as e:
            _LOGGER.warning(f"Failed to register OpenSpiel checkers editor: {e}")

        # Import and register American Checkers editor (8x8)
        try:
            from .american_checkers_editor import AmericanCheckersConfigDialog
            cls._registry["draughts/american_checkers"] = AmericanCheckersConfigDialog
            _LOGGER.debug("Registered AmericanCheckersConfigDialog for draughts/american_checkers")
        except ImportError as e:
            _LOGGER.warning(f"Failed to register American checkers editor: {e}")

        # Import and register Russian Checkers editor (8x8)
        try:
            from .russian_checkers_editor import RussianCheckersConfigDialog
            cls._registry["draughts/russian_checkers"] = RussianCheckersConfigDialog
            _LOGGER.debug("Registered RussianCheckersConfigDialog for draughts/russian_checkers")
        except ImportError as e:
            _LOGGER.warning(f"Failed to register Russian checkers editor: {e}")

        # Import and register International Draughts editor (10x10)
        try:
            from .international_draughts_editor import InternationalDraughtsConfigDialog
            cls._registry["draughts/international_draughts"] = InternationalDraughtsConfigDialog
            _LOGGER.debug("Registered InternationalDraughtsConfigDialog for draughts/international_draughts")
        except ImportError as e:
            _LOGGER.warning(f"Failed to register International draughts editor: {e}")

    @classmethod
    def create(
        cls,
        game_id: str,
        initial_state: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None
    ) -> BoardConfigDialog:
        """Create the appropriate dialog for the given game.

        Args:
            game_id: Game identifier (e.g., "chess_v6", "go_v5")
            initial_state: Optional initial state notation string
            parent: Parent widget

        Returns:
            Game-specific configuration dialog instance

        Raises:
            ValueError: If game_id is not supported
        """
        cls._ensure_initialized()

        if game_id not in cls._registry:
            supported = ", ".join(cls._registry.keys()) or "none"
            _log(
                LOG_UI_BOARD_CONFIG_UNSUPPORTED_GAME,
                extra={"game_id": game_id, "supported_games": supported},
            )
            raise ValueError(
                f"No configuration dialog for game '{game_id}'. "
                f"Supported games: {supported}"
            )

        dialog_class = cls._registry[game_id]
        _log(
            LOG_UI_BOARD_CONFIG_FACTORY_CREATE,
            extra={"game_id": game_id, "dialog_class": dialog_class.__name__},
        )
        return dialog_class(initial_state, parent)

    @classmethod
    def supports(cls, game_id: str) -> bool:
        """Check if a game has a configuration dialog.

        Args:
            game_id: Game identifier to check

        Returns:
            True if the game is supported
        """
        cls._ensure_initialized()
        return game_id in cls._registry

    @classmethod
    def register(
        cls,
        game_id: str,
        dialog_class: Type[BoardConfigDialog]
    ) -> None:
        """Register a new game dialog.

        This allows external modules to add support for new games
        without modifying the factory code.

        Args:
            game_id: Game identifier (e.g., "checkers", "go_v5")
            dialog_class: Dialog class implementing BoardConfigDialog

        Example:
            class MyGameConfigDialog(BoardConfigDialog):
                # Implementation...
                pass

            BoardConfigDialogFactory.register("my_game", MyGameConfigDialog)
        """
        cls._ensure_initialized()

        if game_id in cls._registry:
            _LOGGER.warning(
                f"Overwriting existing dialog for {game_id}: "
                f"{cls._registry[game_id].__name__} -> {dialog_class.__name__}"
            )

        cls._registry[game_id] = dialog_class
        _LOGGER.info(f"Registered {dialog_class.__name__} for {game_id}")

    @classmethod
    def unregister(cls, game_id: str) -> bool:
        """Unregister a game dialog.

        Args:
            game_id: Game identifier to unregister

        Returns:
            True if the game was registered and removed
        """
        cls._ensure_initialized()

        if game_id in cls._registry:
            del cls._registry[game_id]
            _LOGGER.info(f"Unregistered dialog for {game_id}")
            return True
        return False

    @classmethod
    def get_supported_games(cls) -> List[str]:
        """Get list of supported game IDs.

        Returns:
            List of game identifiers that have configuration dialogs
        """
        cls._ensure_initialized()
        return list(cls._registry.keys())

    @classmethod
    def get_dialog_class(cls, game_id: str) -> Optional[Type[BoardConfigDialog]]:
        """Get the dialog class for a game without creating an instance.

        Args:
            game_id: Game identifier

        Returns:
            Dialog class, or None if not supported
        """
        cls._ensure_initialized()
        return cls._registry.get(game_id)
