"""Test Chess integration with PettingZoo Classic adapter.

Tests that the Chess adapter:
1. Can be created and loaded
2. Provides complete render payloads with FEN and legal moves
3. Handles UCI move execution
4. Tracks game state correctly
5. Works with the ChessHandler for Human Control Mode
"""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from gym_gui.core.adapters.pettingzoo_classic import ChessEnvironmentAdapter


class TestChessAdapterLoads:
    """Test that the Chess adapter can be created and loaded."""

    @pytest.fixture
    def chess_adapter(self) -> "ChessEnvironmentAdapter":
        """Create a fresh Chess adapter for each test."""
        from gym_gui.core.adapters.pettingzoo_classic import ChessEnvironmentAdapter

        adapter = ChessEnvironmentAdapter()
        adapter.load()
        return adapter

    def test_chess_adapter_loads(self, chess_adapter: "ChessEnvironmentAdapter") -> None:
        """ChessEnvironmentAdapter loads successfully."""
        assert chess_adapter._aec_env is not None
        assert chess_adapter._board is None  # Board available after reset

    def test_chess_adapter_reset(self, chess_adapter: "ChessEnvironmentAdapter") -> None:
        """ChessEnvironmentAdapter reset returns valid step."""
        result = chess_adapter.reset(seed=42)

        assert result.observation is not None
        assert result.terminated is False
        assert result.truncated is False
        assert chess_adapter._board is not None

    def test_chess_adapter_has_id(self, chess_adapter: "ChessEnvironmentAdapter") -> None:
        """ChessEnvironmentAdapter has correct id."""
        assert chess_adapter.id == "chess_v6"


class TestChessRenderPayload:
    """Test that the Chess adapter provides complete render payloads."""

    @pytest.fixture
    def chess_adapter(self) -> "ChessEnvironmentAdapter":
        """Create and reset a Chess adapter."""
        from gym_gui.core.adapters.pettingzoo_classic import ChessEnvironmentAdapter

        adapter = ChessEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_render_payload_has_fen(self, chess_adapter: "ChessEnvironmentAdapter") -> None:
        """Render payload includes FEN position string."""
        step = chess_adapter.reset(seed=42)
        payload = step.render_payload

        assert payload is not None
        assert "fen" in payload
        # Starting position FEN
        assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" in payload["fen"]

    def test_render_payload_has_legal_moves(
        self, chess_adapter: "ChessEnvironmentAdapter"
    ) -> None:
        """Render payload includes legal moves list."""
        step = chess_adapter.reset(seed=42)
        payload = step.render_payload

        assert payload is not None
        assert "legal_moves" in payload
        assert isinstance(payload["legal_moves"], list)
        # Initial position has 20 legal moves
        assert len(payload["legal_moves"]) == 20

    def test_render_payload_has_current_player(
        self, chess_adapter: "ChessEnvironmentAdapter"
    ) -> None:
        """Render payload includes current player."""
        step = chess_adapter.reset(seed=42)
        payload = step.render_payload

        assert payload is not None
        assert "current_player" in payload
        assert payload["current_player"] == "white"

    def test_render_payload_has_game_state(
        self, chess_adapter: "ChessEnvironmentAdapter"
    ) -> None:
        """Render payload includes game state flags."""
        step = chess_adapter.reset(seed=42)
        payload = step.render_payload

        assert payload is not None
        assert "is_check" in payload
        assert "is_checkmate" in payload
        assert "is_stalemate" in payload
        assert "is_game_over" in payload
        # Initial position is not in check
        assert payload["is_check"] is False
        assert payload["is_game_over"] is False


class TestChessMovesUCI:
    """Test UCI move execution."""

    @pytest.fixture
    def chess_adapter(self) -> "ChessEnvironmentAdapter":
        """Create and reset a Chess adapter."""
        from gym_gui.core.adapters.pettingzoo_classic import ChessEnvironmentAdapter

        adapter = ChessEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_valid_uci_move(self, chess_adapter: "ChessEnvironmentAdapter") -> None:
        """Valid UCI move is executed correctly."""
        step = chess_adapter.step_uci("e2e4")

        assert step.terminated is False
        assert step.render_payload is not None
        # After e2e4, it's black's turn
        assert step.render_payload["current_player"] == "black"
        # e4 pawn should be in FEN
        assert "/4P3/" in step.render_payload["fen"] or "4P" in step.render_payload["fen"]

    def test_invalid_uci_format_raises(
        self, chess_adapter: "ChessEnvironmentAdapter"
    ) -> None:
        """Invalid UCI format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid UCI move format"):
            chess_adapter.step_uci("xyz")

    def test_illegal_move_raises(self, chess_adapter: "ChessEnvironmentAdapter") -> None:
        """Illegal move raises ValueError."""
        with pytest.raises(ValueError, match="Illegal move"):
            chess_adapter.step_uci("e2e5")  # Pawn can't move 3 squares

    def test_is_move_legal_valid(self, chess_adapter: "ChessEnvironmentAdapter") -> None:
        """is_move_legal returns True for legal move."""
        assert chess_adapter.is_move_legal("e2e4") is True
        assert chess_adapter.is_move_legal("d2d4") is True
        assert chess_adapter.is_move_legal("g1f3") is True  # Knight

    def test_is_move_legal_invalid(
        self, chess_adapter: "ChessEnvironmentAdapter"
    ) -> None:
        """is_move_legal returns False for illegal move."""
        assert chess_adapter.is_move_legal("e2e5") is False  # Too far
        assert chess_adapter.is_move_legal("e1e2") is False  # King blocked
        assert chess_adapter.is_move_legal("h1h3") is False  # Rook blocked


class TestChessGameState:
    """Test game state tracking."""

    @pytest.fixture
    def chess_adapter(self) -> "ChessEnvironmentAdapter":
        """Create and reset a Chess adapter."""
        from gym_gui.core.adapters.pettingzoo_classic import ChessEnvironmentAdapter

        adapter = ChessEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_move_count_increments(
        self, chess_adapter: "ChessEnvironmentAdapter"
    ) -> None:
        """Move count increments after each move."""
        assert chess_adapter._move_count == 0

        chess_adapter.step_uci("e2e4")
        assert chess_adapter._move_count == 1

        chess_adapter.step_uci("e7e5")
        assert chess_adapter._move_count == 2

    def test_current_agent_alternates(
        self, chess_adapter: "ChessEnvironmentAdapter"
    ) -> None:
        """Current agent alternates between white and black."""
        assert chess_adapter.current_agent() == "player_0"  # White

        chess_adapter.step_uci("e2e4")
        assert chess_adapter.current_agent() == "player_1"  # Black

        chess_adapter.step_uci("e7e5")
        assert chess_adapter.current_agent() == "player_0"  # White

    def test_get_legal_moves_from_square(
        self, chess_adapter: "ChessEnvironmentAdapter"
    ) -> None:
        """get_legal_moves_from_square returns valid destinations."""
        # Pawn on e2 can move to e3 or e4
        moves = chess_adapter.get_legal_moves_from_square("e2")
        assert "e3" in moves
        assert "e4" in moves
        assert len(moves) == 2

        # Knight on g1 can move to f3 or h3
        moves = chess_adapter.get_legal_moves_from_square("g1")
        assert "f3" in moves
        assert "h3" in moves
        assert len(moves) == 2

    def test_get_chess_state(self, chess_adapter: "ChessEnvironmentAdapter") -> None:
        """get_chess_state returns ChessRenderPayload."""
        from gym_gui.core.adapters.pettingzoo_classic import ChessRenderPayload

        state = chess_adapter.get_chess_state()

        assert isinstance(state, ChessRenderPayload)
        assert state.current_player == "white"
        assert len(state.legal_moves) == 20
        assert state.is_check is False
        assert state.is_game_over is False


class TestChessHandlerIntegration:
    """Test ChessHandler integration with adapter."""

    def test_handler_get_piece_from_fen(self) -> None:
        """ChessHandler._get_piece_from_fen parses FEN correctly."""
        from gym_gui.ui.handlers.chess_handlers import ChessHandler

        # Starting position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        # White pieces (row 1 and 2)
        assert ChessHandler._get_piece_from_fen(fen, "e1") == "K"  # King
        assert ChessHandler._get_piece_from_fen(fen, "d1") == "Q"  # Queen
        assert ChessHandler._get_piece_from_fen(fen, "e2") == "P"  # Pawn

        # Black pieces (row 7 and 8)
        assert ChessHandler._get_piece_from_fen(fen, "e8") == "k"  # King
        assert ChessHandler._get_piece_from_fen(fen, "d8") == "q"  # Queen
        assert ChessHandler._get_piece_from_fen(fen, "e7") == "p"  # Pawn

        # Empty squares
        assert ChessHandler._get_piece_from_fen(fen, "e4") is None
        assert ChessHandler._get_piece_from_fen(fen, "d5") is None

    def test_handler_is_chess_payload(self) -> None:
        """ChessHandler.is_chess_payload identifies chess payloads."""
        from gym_gui.ui.handlers.chess_handlers import ChessHandler

        chess_payload = {"fen": "...", "legal_moves": ["e2e4"]}
        assert ChessHandler.is_chess_payload(chess_payload) is True

        non_chess_payload = {"mode": "grid", "grid": []}
        assert ChessHandler.is_chess_payload(non_chess_payload) is False


class TestChessAdapterClose:
    """Test adapter cleanup."""

    def test_close_cleans_up(self) -> None:
        """close() cleans up resources."""
        from gym_gui.core.adapters.pettingzoo_classic import ChessEnvironmentAdapter

        adapter = ChessEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)

        assert adapter._aec_env is not None
        assert adapter._board is not None

        adapter.close()

        assert adapter._aec_env is None
        assert adapter._board is None

    def test_close_multiple_times_safe(self) -> None:
        """close() can be called multiple times safely."""
        from gym_gui.core.adapters.pettingzoo_classic import ChessEnvironmentAdapter

        adapter = ChessEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)

        adapter.close()
        adapter.close()  # Should not raise


class TestChessFactoryRegistration:
    """Test that Chess adapter is registered in factory."""

    def test_chess_in_factory_registry(self) -> None:
        """Chess adapter is registered in adapter factory."""
        from gym_gui.core.enums import GameId
        from gym_gui.core.factories.adapters import create_adapter

        adapter = create_adapter(GameId.CHESS)
        assert adapter is not None
        assert adapter.id == "chess_v6"

    def test_chess_game_id_exists(self) -> None:
        """GameId.CHESS exists in enum."""
        from gym_gui.core.enums import GameId

        assert hasattr(GameId, "CHESS")
        assert GameId.CHESS.value == "chess_v6"

    def test_chess_in_pettingzoo_classic_family(self) -> None:
        """Chess is in PETTINGZOO_CLASSIC environment family."""
        from gym_gui.core.enums import (
            EnvironmentFamily,
            GameId,
            ENVIRONMENT_FAMILY_BY_GAME,
        )

        assert GameId.CHESS in ENVIRONMENT_FAMILY_BY_GAME
        assert ENVIRONMENT_FAMILY_BY_GAME[GameId.CHESS] == EnvironmentFamily.PETTINGZOO_CLASSIC


class TestScholarsMate:
    """Test a complete game sequence (Scholar's Mate)."""

    def test_scholars_mate(self) -> None:
        """Play Scholar's Mate to test checkmate detection."""
        from gym_gui.core.adapters.pettingzoo_classic import ChessEnvironmentAdapter

        adapter = ChessEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)

        # Scholar's Mate: 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6 4.Qxf7#
        moves = [
            "e2e4",  # White
            "e7e5",  # Black
            "d1h5",  # White - Queen to h5
            "b8c6",  # Black - Knight to c6
            "f1c4",  # White - Bishop to c4
            "g8f6",  # Black - Knight to f6 (tries to attack queen)
            "h5f7",  # White - Qxf7# (checkmate)
        ]

        for i, move in enumerate(moves):
            step = adapter.step_uci(move)

            # Last move should be checkmate
            if i == len(moves) - 1:
                state = adapter.get_chess_state()
                assert state.is_checkmate is True
                assert state.is_game_over is True
                assert state.winner == "white"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
