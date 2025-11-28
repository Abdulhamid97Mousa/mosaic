"""Human vs Agent game handlers.

This module provides handler classes for Human vs Agent gameplay mode,
including AI opponent setup (Stockfish, custom policies, random).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.controllers.chess_game import ChessGameController
    from gym_gui.ui.widgets.human_vs_agent_config_form import HumanVsAgentConfig

from gym_gui.services.chess_ai import StockfishService
from gym_gui.services.chess_ai.stockfish_service import StockfishConfig

_LOG = logging.getLogger(__name__)


class HumanVsAgentHandler:
    """Handles Human vs Agent game setup and AI provider management.

    This handler manages:
    - AI opponent initialization (Stockfish, random, custom)
    - AI provider callbacks for ChessGameController
    - Cleanup of AI resources

    Args:
        status_bar: The status bar for showing feedback messages.
    """

    def __init__(self, status_bar: "QStatusBar") -> None:
        self._status_bar = status_bar
        self._stockfish_service: Optional[StockfishService] = None
        self._current_ai_opponent: str = "stockfish"
        self._current_ai_difficulty: str = "medium"

    @property
    def current_opponent(self) -> str:
        """Current AI opponent type."""
        return self._current_ai_opponent

    @property
    def current_difficulty(self) -> str:
        """Current AI difficulty level."""
        return self._current_ai_difficulty

    def setup_ai_provider(
        self,
        config: "HumanVsAgentConfig",
        chess_controller: "ChessGameController",
    ) -> str:
        """Set up the AI action provider for chess from configuration.

        Args:
            config: Full HumanVsAgentConfig object with all settings.
            chess_controller: The chess game controller to configure.

        Returns:
            Display name for the AI opponent.
        """
        # Store current settings
        self._current_ai_opponent = config.opponent_type
        self._current_ai_difficulty = config.difficulty

        # Clean up existing Stockfish service
        self.cleanup()

        if config.opponent_type == "stockfish":
            return self._setup_stockfish(config, chess_controller)

        # Default: random AI (no action provider = uses default random)
        chess_controller.set_ai_action_provider(None)
        return "Random AI"

    def _setup_stockfish(
        self,
        config: "HumanVsAgentConfig",
        chess_controller: "ChessGameController",
    ) -> str:
        """Set up Stockfish as the AI provider.

        Args:
            config: Configuration with Stockfish settings.
            chess_controller: The chess game controller.

        Returns:
            Display name for Stockfish or fallback.
        """
        # Create Stockfish config from dialog settings
        stockfish_config = StockfishConfig(
            skill_level=config.stockfish.skill_level,
            depth=config.stockfish.depth,
            time_limit_ms=config.stockfish.time_limit_ms,
            threads=config.stockfish.threads,
            hash_mb=config.stockfish.hash_mb,
        )
        self._stockfish_service = StockfishService(stockfish_config)

        if self._stockfish_service.is_available():
            if self._stockfish_service.start():
                chess_controller.set_ai_action_provider(
                    self._stockfish_service.get_best_move
                )
                _LOG.info(
                    f"Stockfish AI configured: difficulty={config.difficulty}, "
                    f"skill={config.stockfish.skill_level}, "
                    f"depth={config.stockfish.depth}, "
                    f"time={config.stockfish.time_limit_ms}ms"
                )
                return f"Stockfish ({config.difficulty.capitalize()})"
            else:
                _LOG.warning("Failed to start Stockfish, falling back to random AI")
                self._status_bar.showMessage(
                    "Stockfish failed to start. Using random AI.",
                    5000
                )
        else:
            _LOG.warning("Stockfish not available, falling back to random AI")
            self._status_bar.showMessage(
                "Stockfish not installed. Using random AI. "
                "Install with: sudo apt install stockfish",
                8000
            )

        # Fall through to random if Stockfish failed
        self._stockfish_service = None
        chess_controller.set_ai_action_provider(None)
        return "Random AI"

    def on_ai_config_changed(
        self,
        opponent_type: str,
        difficulty: str,
        chess_controller: Optional["ChessGameController"],
        get_ai_config: Callable[[], "HumanVsAgentConfig"],
    ) -> Optional[str]:
        """Handle AI opponent selection change.

        If a game is currently running, update the AI provider.

        Args:
            opponent_type: Type of AI opponent ("random", "stockfish", "custom").
            difficulty: Difficulty level for engines like Stockfish.
            chess_controller: The chess game controller (if active).
            get_ai_config: Callable to get the full config from the tab.

        Returns:
            AI display name if updated, None if no game active.
        """
        self._current_ai_opponent = opponent_type
        self._current_ai_difficulty = difficulty

        # If a chess game is active, update the AI provider
        if chess_controller is not None and chess_controller.is_game_active():
            ai_config = get_ai_config()
            ai_name = self.setup_ai_provider(ai_config, chess_controller)
            _LOG.info(f"AI opponent updated: {ai_name}")
            return ai_name

        return None

    def cleanup(self) -> None:
        """Clean up AI resources (stop Stockfish, etc.)."""
        if self._stockfish_service is not None:
            self._stockfish_service.stop()
            self._stockfish_service = None


__all__ = ["HumanVsAgentHandler"]
