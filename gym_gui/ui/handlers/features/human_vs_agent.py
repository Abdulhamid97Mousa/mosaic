"""Human vs Agent game handlers.

This module provides handler classes for Human vs Agent gameplay mode,
including AI opponent setup, game state management, and move handling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.controllers.chess_controller import ChessGameController
    from gym_gui.core.adapters.chess_adapter import ChessState
    from gym_gui.ui.widgets.human_vs_agent_board import InteractiveChessBoard
    from gym_gui.ui.widgets.human_vs_agent_config_form import HumanVsAgentConfig
    from gym_gui.ui.widgets.multi_agent_tab import HumanVsAgentTab

from gym_gui.services.chess_ai import StockfishService
from gym_gui.services.chess_ai.stockfish_service import StockfishConfig

_LOG = logging.getLogger(__name__)


class HumanVsAgentHandler:
    """Handles Human vs Agent game setup, state management, and AI provider.

    This handler manages:
    - AI opponent initialization (Stockfish, random, custom)
    - AI provider callbacks for ChessGameController
    - Game state updates (board position, highlights, turn indicator)
    - Game lifecycle events (started, over)
    - Move handling from interactive board
    - Cleanup of AI resources

    Args:
        status_bar: The status bar for showing feedback messages.
    """

    def __init__(self, status_bar: "QStatusBar") -> None:
        self._status_bar = status_bar
        self._stockfish_service: Optional[StockfishService] = None
        self._current_ai_opponent: str = "stockfish"
        self._current_ai_difficulty: str = "medium"

        # Game components (bound after game creation)
        self._chess_board: Optional["InteractiveChessBoard"] = None
        self._chess_controller: Optional["ChessGameController"] = None
        self._human_vs_agent_tab: Optional["HumanVsAgentTab"] = None

    def bind_game_components(
        self,
        chess_board: "InteractiveChessBoard",
        chess_controller: "ChessGameController",
        human_vs_agent_tab: "HumanVsAgentTab",
    ) -> None:
        """Bind game components after game creation.

        Args:
            chess_board: The interactive chess board widget.
            chess_controller: The chess game controller.
            human_vs_agent_tab: The Human vs Agent tab for UI updates.
        """
        self._chess_board = chess_board
        self._chess_controller = chess_controller
        self._human_vs_agent_tab = human_vs_agent_tab

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
    ) -> tuple[str, bool]:
        """Set up the AI action provider for chess from configuration.

        Args:
            config: Full HumanVsAgentConfig object with all settings.
            chess_controller: The chess game controller to configure.

        Returns:
            Tuple of (display_name, is_fallback) where is_fallback indicates
            if we fell back to Random AI because Stockfish wasn't available.
        """
        # Store current settings
        self._current_ai_opponent = config.opponent_type
        self._current_ai_difficulty = config.difficulty

        # Clean up existing Stockfish service
        self.cleanup()

        if config.opponent_type == "stockfish":
            ai_name, is_fallback = self._setup_stockfish(config, chess_controller)
            # Update the UI to show which AI is actually active
            if self._human_vs_agent_tab is not None:
                self._human_vs_agent_tab.set_active_ai(ai_name, is_fallback)
            return ai_name, is_fallback

        # Default: random AI (no action provider = uses default random)
        chess_controller.set_ai_action_provider(None)
        if self._human_vs_agent_tab is not None:
            self._human_vs_agent_tab.set_active_ai("Random AI", is_fallback=False)
        return "Random AI", False

    def _setup_stockfish(
        self,
        config: "HumanVsAgentConfig",
        chess_controller: "ChessGameController",
    ) -> tuple[str, bool]:
        """Set up Stockfish as the AI provider.

        Args:
            config: Configuration with Stockfish settings.
            chess_controller: The chess game controller.

        Returns:
            Tuple of (display_name, is_fallback) where is_fallback is True
            if we fell back to Random AI because Stockfish wasn't available.
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
                return f"Stockfish ({config.difficulty.capitalize()})", False
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
        return "Random AI (Stockfish unavailable)", True

    def on_ai_config_changed(
        self,
        opponent_type: str,
        difficulty: str,
        chess_controller: Optional["ChessGameController"],
        get_ai_config: Callable[[], "HumanVsAgentConfig"],
    ) -> Optional[tuple[str, bool]]:
        """Handle AI opponent selection change.

        If a game is currently running, update the AI provider.

        Args:
            opponent_type: Type of AI opponent ("random", "stockfish", "custom").
            difficulty: Difficulty level for engines like Stockfish.
            chess_controller: The chess game controller (if active).
            get_ai_config: Callable to get the full config from the tab.

        Returns:
            Tuple of (ai_name, is_fallback) if updated, None if no game active.
        """
        self._current_ai_opponent = opponent_type
        self._current_ai_difficulty = difficulty

        # If a chess game is active, update the AI provider
        if chess_controller is not None and chess_controller.is_game_active():
            ai_config = get_ai_config()
            ai_name, is_fallback = self.setup_ai_provider(ai_config, chess_controller)
            _LOG.info(f"AI opponent updated: {ai_name} (fallback={is_fallback})")
            return ai_name, is_fallback

        return None

    def cleanup(self) -> None:
        """Clean up AI resources (stop Stockfish, etc.)."""
        if self._stockfish_service is not None:
            self._stockfish_service.stop()
            self._stockfish_service = None

    # -------------------------------------------------------------------------
    # Game State Handlers
    # -------------------------------------------------------------------------

    def on_chess_state_changed(self, state: "ChessState") -> None:
        """Handle chess state update from controller.

        Updates the board display and control panel with current game state.

        Args:
            state: New chess game state from controller.
        """
        _LOG.debug(f"Chess state changed: player={state.current_player}, fen={state.fen[:30]}...")

        if self._chess_board is None:
            _LOG.warning("Chess board is None in state_changed handler")
            return

        # Update board position
        self._chess_board.set_position(state.fen)
        self._chess_board.set_legal_moves(state.legal_moves)
        self._chess_board.set_current_player(state.current_player)

        # Update highlights
        if state.last_move:
            from_sq = state.last_move[:2]
            to_sq = state.last_move[2:4]
            self._chess_board.set_last_move(from_sq, to_sq)
        else:
            self._chess_board.set_last_move(None, None)

        # Update check status
        if state.is_check:
            king_sq = self._find_king_square(state.fen, state.current_player)
            self._chess_board.set_check(True, king_sq)
        else:
            self._chess_board.set_check(False, None)

        # Enable/disable board based on turn
        if self._chess_controller is not None:
            is_human_turn = self._chess_controller.is_human_turn()
            self._chess_board.set_enabled(is_human_turn and not state.is_game_over)

        # Update game status in control panel
        if self._human_vs_agent_tab is not None:
            turn_text = f"{state.current_player.capitalize()}'s turn"
            if state.is_check:
                turn_text += " (CHECK!)"

            score_text = f"Move {state.move_count}"

            result = None
            if state.is_game_over:
                if state.winner == "draw":
                    result = "Draw"
                elif state.winner:
                    result = f"{state.winner.capitalize()} wins!"

            self._human_vs_agent_tab.update_game_status(
                current_turn=turn_text,
                score=score_text,
                result=result,
            )

    def on_chess_game_started(self) -> None:
        """Handle chess game start event."""
        _LOG.info("Chess game started")

        if self._human_vs_agent_tab is not None:
            self._human_vs_agent_tab._start_btn.setEnabled(True)
            self._human_vs_agent_tab._reset_btn.setEnabled(True)

    def on_chess_game_over(self, winner: str) -> None:
        """Handle chess game end event.

        Args:
            winner: "white", "black", or "draw"
        """
        _LOG.info(f"Chess game over: winner={winner}")

        if self._chess_board is not None:
            self._chess_board.set_enabled(False)

    def on_chess_move_made(self, from_sq: str, to_sq: str) -> None:
        """Handle move from the interactive chess board.

        Submits the human move to the chess controller.

        Args:
            from_sq: Source square (e.g., "e2")
            to_sq: Destination square (e.g., "e4")
        """
        if self._chess_controller is not None:
            self._chess_controller.submit_human_move(from_sq, to_sq)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _find_king_square(fen: str, player: str) -> Optional[str]:
        """Find the king's square for a given player from FEN.

        Args:
            fen: FEN position string
            player: "white" or "black"

        Returns:
            King's square in algebraic notation, or None if not found
        """
        king_char = "K" if player == "white" else "k"
        position = fen.split()[0]

        row = 7
        col = 0

        for char in position:
            if char == "/":
                row -= 1
                col = 0
            elif char.isdigit():
                col += int(char)
            else:
                if char == king_char:
                    file_char = chr(ord("a") + col)
                    rank_char = str(row + 1)
                    return f"{file_char}{rank_char}"
                col += 1

        return None


__all__ = ["HumanVsAgentHandler"]
