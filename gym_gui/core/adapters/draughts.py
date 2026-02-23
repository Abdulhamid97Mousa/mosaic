"""Draughts/Checkers variants adapter with proper rule implementations.

Implements three variants:
1. American Checkers (8x8) - Standard rules, no backward captures, no flying kings
2. Russian Checkers (8x8) - Backward captures allowed, flying kings
3. International Draughts (10x10) - Backward captures, flying kings, 20 pieces per side

Each variant follows its official rules precisely.

References:
- American Checkers: https://en.wikipedia.org/wiki/English_draughts
- Russian Checkers: https://en.wikipedia.org/wiki/Russian_draughts
- International Draughts: https://en.wikipedia.org/wiki/International_draughts
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode

_LOG = logging.getLogger(__name__)


class Piece(IntEnum):
    """Piece types on the board."""
    EMPTY = 0
    BLACK_MAN = 1      # Player 0 regular piece
    BLACK_KING = 2     # Player 0 king
    WHITE_MAN = 3      # Player 1 regular piece
    WHITE_KING = 4     # Player 1 king


@dataclass(slots=True)
class Move:
    """Represents a move in draughts."""
    from_pos: Tuple[int, int]  # (row, col)
    to_pos: Tuple[int, int]    # (row, col)
    captures: List[Tuple[int, int]] = field(default_factory=list)  # Positions of captured pieces
    is_promotion: bool = False

    def to_string(self, board_size: int = 8) -> str:
        """Convert move to algebraic notation."""
        from_sq = f"{chr(ord('a') + self.from_pos[1])}{board_size - self.from_pos[0]}"
        to_sq = f"{chr(ord('a') + self.to_pos[1])}{board_size - self.to_pos[0]}"
        separator = "x" if self.captures else "-"
        return f"{from_sq}{separator}{to_sq}"


@dataclass(slots=True)
class DraughtsState:
    """Game state for draughts variants."""
    board: List[List[int]]
    current_player: int  # 0 = Black, 1 = White
    legal_moves: List[Move]
    last_move: Optional[Move] = None
    is_game_over: bool = False
    winner: Optional[int] = None  # 0, 1, or None for draw
    move_count: int = 0
    moves_without_capture: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for UI rendering."""
        return {
            "game_type": "draughts",
            "board": self.board,
            "current_player": f"player_{self.current_player}",
            "current_agent": f"player_{self.current_player}",
            "legal_moves": [m.to_string(len(self.board)) for m in self.legal_moves],
            "last_move": self.last_move.to_string(len(self.board)) if self.last_move else None,
            "is_game_over": self.is_game_over,
            "winner": f"player_{self.winner}" if self.winner is not None else None,
            "move_count": self.move_count,
        }


class DraughtsGame(ABC):
    """Abstract base class for draughts variants."""

    # Override these in subclasses
    BOARD_SIZE: int = 8
    PIECES_PER_PLAYER: int = 12
    MEN_CAN_CAPTURE_BACKWARD: bool = False
    FLYING_KINGS: bool = False
    MUST_CAPTURE_MAXIMUM: bool = False  # Must take path with most captures
    PROMOTION_ENDS_TURN: bool = True    # In American, promotion ends multi-jump

    def __init__(self) -> None:
        self._board: List[List[int]] = []
        self._current_player: int = 0  # 0 = Black, 1 = White
        self._move_count: int = 0
        self._moves_without_capture: int = 0
        self._last_move: Optional[Move] = None
        self._game_over: bool = False
        self._winner: Optional[int] = None

    def reset(self, seed: Optional[int] = None) -> DraughtsState:
        """Reset the game to initial state."""
        self._board = self._create_initial_board()
        self._current_player = 0  # Black moves first
        self._move_count = 0
        self._moves_without_capture = 0
        self._last_move = None
        self._game_over = False
        self._winner = None
        return self.get_state()

    def _create_initial_board(self) -> List[List[int]]:
        """Create the initial board setup."""
        board = [[Piece.EMPTY] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]

        # Calculate rows for pieces
        piece_rows = (self.PIECES_PER_PLAYER * 2) // self.BOARD_SIZE

        # Place black pieces (top of board, rows 0 to piece_rows-1)
        for row in range(piece_rows):
            for col in range(self.BOARD_SIZE):
                if self._is_playable_square(row, col):
                    board[row][col] = Piece.BLACK_MAN

        # Place white pieces (bottom of board)
        for row in range(self.BOARD_SIZE - piece_rows, self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self._is_playable_square(row, col):
                    board[row][col] = Piece.WHITE_MAN

        return board

    def _is_playable_square(self, row: int, col: int) -> bool:
        """Check if a square is playable (dark square)."""
        return (row + col) % 2 == 1

    def _is_valid_pos(self, row: int, col: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE

    def _get_piece_owner(self, piece: int) -> Optional[int]:
        """Get the owner of a piece (0=Black, 1=White, None=empty)."""
        if piece in (Piece.BLACK_MAN, Piece.BLACK_KING):
            return 0
        elif piece in (Piece.WHITE_MAN, Piece.WHITE_KING):
            return 1
        return None

    def _is_king(self, piece: int) -> bool:
        """Check if piece is a king."""
        return piece in (Piece.BLACK_KING, Piece.WHITE_KING)

    def _is_man(self, piece: int) -> bool:
        """Check if piece is a man (not king)."""
        return piece in (Piece.BLACK_MAN, Piece.WHITE_MAN)

    def _get_forward_directions(self, player: int) -> List[int]:
        """Get forward row directions for a player."""
        # Black (player 0) moves down (increasing row), White moves up
        return [1] if player == 0 else [-1]

    def _get_promotion_row(self, player: int) -> int:
        """Get the promotion row for a player."""
        # Black promotes at bottom row, White at top row
        return self.BOARD_SIZE - 1 if player == 0 else 0

    def _should_promote(self, player: int, row: int) -> bool:
        """Check if piece should be promoted."""
        return row == self._get_promotion_row(player)

    def _promote_piece(self, piece: int) -> int:
        """Promote a man to king."""
        if piece == Piece.BLACK_MAN:
            return Piece.BLACK_KING
        elif piece == Piece.WHITE_MAN:
            return Piece.WHITE_KING
        return piece

    def get_legal_moves(self) -> List[Move]:
        """Get all legal moves for current player."""
        if self._game_over:
            return []

        # First, check for captures (mandatory)
        captures = self._get_all_captures(self._current_player)
        if captures:
            if self.MUST_CAPTURE_MAXIMUM:
                # Filter to only maximum length captures
                max_len = max(len(m.captures) for m in captures)
                captures = [m for m in captures if len(m.captures) == max_len]
            return captures

        # No captures available, get regular moves
        return self._get_all_regular_moves(self._current_player)

    def _get_all_captures(self, player: int) -> List[Move]:
        """Get all capture moves for a player."""
        captures = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = self._board[row][col]
                if self._get_piece_owner(piece) == player:
                    piece_captures = self._get_captures_from(row, col, piece)
                    captures.extend(piece_captures)
        return captures

    def _get_captures_from(self, row: int, col: int, piece: int) -> List[Move]:
        """Get all capture sequences starting from a position."""
        player = self._get_piece_owner(piece)
        is_king = self._is_king(piece)

        # Determine capture directions
        if is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            forward_dirs = self._get_forward_directions(player)
            if self.MEN_CAN_CAPTURE_BACKWARD:
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            else:
                directions = [(d, -1) for d in forward_dirs] + [(d, 1) for d in forward_dirs]

        captures = []

        if self.FLYING_KINGS and is_king:
            # Flying king captures - can jump from distance
            captures = self._get_flying_captures(row, col, piece, directions, [])
        else:
            # Regular captures - jump over adjacent piece
            captures = self._get_regular_captures(row, col, piece, directions, [])

        return captures

    def _get_regular_captures(
        self,
        row: int,
        col: int,
        piece: int,
        directions: List[Tuple[int, int]],
        already_captured: List[Tuple[int, int]]
    ) -> List[Move]:
        """Get regular (non-flying) capture moves."""
        player = self._get_piece_owner(piece)
        captures = []

        for dr, dc in directions:
            # Position of piece to capture
            cap_row, cap_col = row + dr, col + dc
            # Landing position
            land_row, land_col = row + 2*dr, col + 2*dc

            if not self._is_valid_pos(land_row, land_col):
                continue

            cap_piece = self._board[cap_row][cap_col]
            land_piece = self._board[land_row][land_col]

            # Check if can capture
            cap_owner = self._get_piece_owner(cap_piece)
            if cap_owner is None or cap_owner == player:
                continue
            if (cap_row, cap_col) in already_captured:
                continue
            if land_piece != Piece.EMPTY:
                continue

            # Valid capture found
            new_captured = already_captured + [(cap_row, cap_col)]

            # Check for promotion
            will_promote = self._is_man(piece) and self._should_promote(player, land_row)

            # Look for multi-jumps (unless promotion ends turn)
            if will_promote and self.PROMOTION_ENDS_TURN:
                # Promotion ends the turn
                captures.append(Move(
                    from_pos=(row, col),
                    to_pos=(land_row, land_col),
                    captures=new_captured,
                    is_promotion=True
                ))
            else:
                # Check for continuation jumps
                # Temporarily update piece type if promoted
                next_piece = self._promote_piece(piece) if will_promote else piece

                continuations = self._get_regular_captures(
                    land_row, land_col, next_piece, directions, new_captured
                )

                if continuations:
                    for cont in continuations:
                        # Update the from_pos to original position
                        captures.append(Move(
                            from_pos=(row, col),
                            to_pos=cont.to_pos,
                            captures=cont.captures,
                            is_promotion=cont.is_promotion or will_promote
                        ))
                else:
                    # No more captures, this is a final move
                    captures.append(Move(
                        from_pos=(row, col),
                        to_pos=(land_row, land_col),
                        captures=new_captured,
                        is_promotion=will_promote
                    ))

        return captures

    def _get_flying_captures(
        self,
        row: int,
        col: int,
        piece: int,
        directions: List[Tuple[int, int]],
        already_captured: List[Tuple[int, int]]
    ) -> List[Move]:
        """Get flying king capture moves."""
        player = self._get_piece_owner(piece)
        captures = []

        for dr, dc in directions:
            # Scan along diagonal until we hit something
            dist = 1
            found_enemy = None

            while True:
                check_row = row + dist * dr
                check_col = col + dist * dc

                if not self._is_valid_pos(check_row, check_col):
                    break

                check_piece = self._board[check_row][check_col]

                if check_piece == Piece.EMPTY:
                    if found_enemy:
                        # Can land here after capturing
                        new_captured = already_captured + [found_enemy]

                        # Check for continuation captures from each landing spot
                        continuations = self._get_flying_captures(
                            check_row, check_col, piece, directions, new_captured
                        )

                        if continuations:
                            for cont in continuations:
                                captures.append(Move(
                                    from_pos=(row, col),
                                    to_pos=cont.to_pos,
                                    captures=cont.captures,
                                    is_promotion=False
                                ))
                        else:
                            captures.append(Move(
                                from_pos=(row, col),
                                to_pos=(check_row, check_col),
                                captures=new_captured,
                                is_promotion=False
                            ))
                    dist += 1
                else:
                    # Found a piece
                    piece_owner = self._get_piece_owner(check_piece)

                    if found_enemy:
                        # Already found an enemy, can't jump two pieces
                        break
                    elif piece_owner == player:
                        # Own piece blocks
                        break
                    elif (check_row, check_col) in already_captured:
                        # Already captured this piece
                        break
                    else:
                        # Found enemy to capture
                        found_enemy = (check_row, check_col)
                        dist += 1

        return captures

    def _get_all_regular_moves(self, player: int) -> List[Move]:
        """Get all non-capture moves for a player."""
        moves = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = self._board[row][col]
                if self._get_piece_owner(piece) == player:
                    piece_moves = self._get_regular_moves_from(row, col, piece)
                    moves.extend(piece_moves)
        return moves

    def _get_regular_moves_from(self, row: int, col: int, piece: int) -> List[Move]:
        """Get regular (non-capture) moves from a position."""
        player = self._get_piece_owner(piece)
        is_king = self._is_king(piece)
        moves = []

        # Determine move directions
        if is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            forward_dirs = self._get_forward_directions(player)
            directions = [(d, -1) for d in forward_dirs] + [(d, 1) for d in forward_dirs]

        if self.FLYING_KINGS and is_king:
            # Flying king can move multiple squares
            for dr, dc in directions:
                dist = 1
                while True:
                    new_row = row + dist * dr
                    new_col = col + dist * dc

                    if not self._is_valid_pos(new_row, new_col):
                        break
                    if self._board[new_row][new_col] != Piece.EMPTY:
                        break

                    moves.append(Move(
                        from_pos=(row, col),
                        to_pos=(new_row, new_col),
                        captures=[],
                        is_promotion=False
                    ))
                    dist += 1
        else:
            # Regular move - one square
            for dr, dc in directions:
                new_row = row + dr
                new_col = col + dc

                if not self._is_valid_pos(new_row, new_col):
                    continue
                if self._board[new_row][new_col] != Piece.EMPTY:
                    continue

                will_promote = self._is_man(piece) and self._should_promote(player, new_row)
                moves.append(Move(
                    from_pos=(row, col),
                    to_pos=(new_row, new_col),
                    captures=[],
                    is_promotion=will_promote
                ))

        return moves

    def make_move(self, move: Move) -> DraughtsState:
        """Execute a move and return new state."""
        if self._game_over:
            raise RuntimeError("Game is over")

        # Validate move
        legal_moves = self.get_legal_moves()
        if not any(self._moves_equal(move, m) for m in legal_moves):
            raise ValueError(f"Illegal move: {move.to_string(self.BOARD_SIZE)}")

        # Execute move
        piece = self._board[move.from_pos[0]][move.from_pos[1]]

        # Clear source square
        self._board[move.from_pos[0]][move.from_pos[1]] = Piece.EMPTY

        # Remove captured pieces
        for cap_pos in move.captures:
            self._board[cap_pos[0]][cap_pos[1]] = Piece.EMPTY

        # Place piece at destination (with promotion if applicable)
        if move.is_promotion:
            piece = self._promote_piece(piece)
        self._board[move.to_pos[0]][move.to_pos[1]] = piece

        # Update game state
        self._last_move = move
        self._move_count += 1

        if move.captures:
            self._moves_without_capture = 0
        else:
            self._moves_without_capture += 1

        # Switch player
        self._current_player = 1 - self._current_player

        # Check for game over
        self._check_game_over()

        return self.get_state()

    def make_move_by_index(self, move_index: int) -> DraughtsState:
        """Execute a move by its index in legal_moves list."""
        legal_moves = self.get_legal_moves()
        if move_index < 0 or move_index >= len(legal_moves):
            raise ValueError(f"Invalid move index: {move_index}")
        return self.make_move(legal_moves[move_index])

    def _moves_equal(self, m1: Move, m2: Move) -> bool:
        """Check if two moves are equal."""
        return (m1.from_pos == m2.from_pos and
                m1.to_pos == m2.to_pos and
                set(m1.captures) == set(m2.captures))

    def _check_game_over(self) -> None:
        """Check if game is over and determine winner."""
        # Check for no legal moves (current player loses)
        if not self.get_legal_moves():
            self._game_over = True
            self._winner = 1 - self._current_player
            return

        # Check for draw (40 moves without capture for American, 25 for others)
        draw_threshold = 40 if not self.MEN_CAN_CAPTURE_BACKWARD else 25
        if self._moves_without_capture >= draw_threshold:
            self._game_over = True
            self._winner = None  # Draw
            return

        # Check if one side has no pieces
        black_pieces = sum(
            1 for row in self._board for p in row
            if p in (Piece.BLACK_MAN, Piece.BLACK_KING)
        )
        white_pieces = sum(
            1 for row in self._board for p in row
            if p in (Piece.WHITE_MAN, Piece.WHITE_KING)
        )

        if black_pieces == 0:
            self._game_over = True
            self._winner = 1
        elif white_pieces == 0:
            self._game_over = True
            self._winner = 0

    def get_state(self) -> DraughtsState:
        """Get current game state."""
        return DraughtsState(
            board=[row[:] for row in self._board],
            current_player=self._current_player,
            legal_moves=self.get_legal_moves(),
            last_move=self._last_move,
            is_game_over=self._game_over,
            winner=self._winner,
            move_count=self._move_count,
            moves_without_capture=self._moves_without_capture
        )

    def find_move(self, from_sq: str, to_sq: str) -> Optional[Move]:
        """Find a move by source and destination squares (algebraic notation)."""
        from_pos = self._algebraic_to_pos(from_sq)
        to_pos = self._algebraic_to_pos(to_sq)

        for move in self.get_legal_moves():
            if move.from_pos == from_pos and move.to_pos == to_pos:
                return move
        return None

    def get_moves_from_square(self, square: str) -> List[str]:
        """Get all legal destination squares for moves from a given square."""
        from_pos = self._algebraic_to_pos(square)
        destinations = []

        for move in self.get_legal_moves():
            if move.from_pos == from_pos:
                destinations.append(self._pos_to_algebraic(move.to_pos))

        return destinations

    def _algebraic_to_pos(self, square: str) -> Tuple[int, int]:
        """Convert algebraic notation to (row, col)."""
        col = ord(square[0].lower()) - ord('a')
        row = self.BOARD_SIZE - int(square[1:])
        return (row, col)

    def _pos_to_algebraic(self, pos: Tuple[int, int]) -> str:
        """Convert (row, col) to algebraic notation."""
        row, col = pos
        return f"{chr(ord('a') + col)}{self.BOARD_SIZE - row}"


class AmericanCheckers(DraughtsGame):
    """American Checkers (English Draughts) - 8x8 board.

    Rules:
    - Men move diagonally forward only
    - Men capture diagonally forward only
    - Kings move one square in any diagonal direction
    - Kings capture one square in any diagonal direction
    - Captures are mandatory
    - Multiple jumps are required if available
    - Promotion ends the turn (can't continue jumping as king)
    """
    BOARD_SIZE = 8
    PIECES_PER_PLAYER = 12
    MEN_CAN_CAPTURE_BACKWARD = False
    FLYING_KINGS = False
    MUST_CAPTURE_MAXIMUM = False
    PROMOTION_ENDS_TURN = True


class RussianCheckers(DraughtsGame):
    """Russian Checkers - 8x8 board.

    Rules:
    - Men move diagonally forward only
    - Men can capture both forward AND backward
    - Flying kings - can move multiple squares diagonally
    - Flying kings can capture from distance
    - Captures are mandatory
    - Must take maximum captures if multiple options
    - Promotion during multi-jump allows continuing as king
    """
    BOARD_SIZE = 8
    PIECES_PER_PLAYER = 12
    MEN_CAN_CAPTURE_BACKWARD = True
    FLYING_KINGS = True
    MUST_CAPTURE_MAXIMUM = True
    PROMOTION_ENDS_TURN = False


class InternationalDraughts(DraughtsGame):
    """International Draughts - 10x10 board.

    Rules:
    - 20 pieces per player
    - Men move diagonally forward only
    - Men can capture both forward AND backward
    - Flying kings - can move multiple squares diagonally
    - Flying kings can capture from distance
    - Captures are mandatory
    - Must take maximum captures if multiple options
    - Promotion during multi-jump allows continuing as king
    """
    BOARD_SIZE = 10
    PIECES_PER_PLAYER = 20
    MEN_CAN_CAPTURE_BACKWARD = True
    FLYING_KINGS = True
    MUST_CAPTURE_MAXIMUM = True
    PROMOTION_ENDS_TURN = False


# =============================================================================
# Adapter Classes for GUI Integration
# =============================================================================

@dataclass(slots=True)
class DraughtsRenderPayload:
    """Render payload for draughts games."""
    game_type: str
    variant: str  # "american", "russian", "international"
    board: List[List[int]]
    board_size: int
    current_player: str
    legal_moves: List[str]
    last_move: Optional[str]
    is_game_over: bool
    winner: Optional[str]
    move_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_type": self.game_type,
            "variant": self.variant,
            "board": self.board,
            "board_size": self.board_size,
            "current_player": self.current_player,
            "current_agent": self.current_player,
            "legal_moves": self.legal_moves,
            "last_move": self.last_move,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "move_count": self.move_count,
        }


class BaseDraughtsAdapter(EnvironmentAdapter[Dict[str, Any], int]):
    """Base adapter for draughts variants."""

    # Override in subclasses
    GAME_CLASS: type = DraughtsGame
    VARIANT_NAME: str = "draughts"

    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    PLAYER_0 = "player_0"  # Black
    PLAYER_1 = "player_1"  # White

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context)
        self._game: Optional[DraughtsGame] = None
        self._move_history: List[Move] = []

    def load(self) -> None:
        """Load the draughts game."""
        self._game = self.GAME_CLASS()
        self._game.reset()
        self._move_history = []
        self._episode_step = 0
        self._episode_return = 0.0

    def reset(self, seed: int | None = None) -> AdapterStep[Dict[str, Any]]:
        """Reset the game."""
        if self._game is None:
            self.load()

        self._game.reset(seed)
        self._move_history = []
        self._episode_step = 0
        self._episode_return = 0.0

        state = self._game.get_state()
        return self._package_step(state, 0.0, False, False, {})

    def step(self, action: int) -> AdapterStep[Dict[str, Any]]:
        """Execute a move by index."""
        if self._game is None:
            raise RuntimeError("Game not loaded. Call load() first.")

        state_before = self._game.get_state()
        state = self._game.make_move_by_index(action)

        self._episode_step += 1
        self._move_history.append(state.last_move)

        # Calculate reward
        reward = 0.0
        if state.is_game_over:
            if state.winner == 0:  # Black wins
                reward = 1.0 if state_before.current_player == 0 else -1.0
            elif state.winner == 1:  # White wins
                reward = 1.0 if state_before.current_player == 1 else -1.0
            # Draw = 0 reward

        self._episode_return += reward

        return self._package_step(
            state, reward, state.is_game_over, False, {}
        )

    def close(self) -> None:
        """Clean up resources."""
        self._game = None
        self._move_history = []

    def render(self) -> np.ndarray | None:
        """Generate RGB image of board."""
        if self._game is None:
            return None
        return self._generate_board_image()

    def _generate_board_image(self) -> np.ndarray:
        """Generate a simple RGB image of the board."""
        if self._game is None:
            return np.zeros((480, 480, 3), dtype=np.uint8)

        board_size = self._game.BOARD_SIZE
        square_size = 480 // board_size
        img_size = square_size * board_size

        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        for row in range(board_size):
            for col in range(board_size):
                y = row * square_size
                x = col * square_size

                # Board color
                if (row + col) % 2 == 0:
                    color = (240, 217, 181)  # Light
                else:
                    color = (181, 136, 99)  # Dark

                img[y:y+square_size, x:x+square_size] = color

                # Draw pieces
                piece = self._game._board[row][col]
                if piece > 0:
                    center_y = y + square_size // 2
                    center_x = x + square_size // 2
                    radius = square_size // 2 - 4

                    # Piece color
                    if piece in (Piece.BLACK_MAN, Piece.BLACK_KING):
                        piece_color = (50, 50, 50)
                    else:
                        piece_color = (255, 255, 255)

                    # Draw circle
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dx*dx + dy*dy <= radius*radius:
                                py = center_y + dy
                                px = center_x + dx
                                if 0 <= py < img_size and 0 <= px < img_size:
                                    img[py, px] = piece_color

                    # King indicator
                    if piece in (Piece.BLACK_KING, Piece.WHITE_KING):
                        king_radius = radius // 3
                        king_color = (255, 215, 0)  # Gold
                        for dy in range(-king_radius, king_radius + 1):
                            for dx in range(-king_radius, king_radius + 1):
                                if dx*dx + dy*dy <= king_radius*king_radius:
                                    py = center_y + dy
                                    px = center_x + dx
                                    if 0 <= py < img_size and 0 <= px < img_size:
                                        img[py, px] = king_color

        return img

    def get_draughts_state(self) -> DraughtsRenderPayload:
        """Get structured state for Qt rendering."""
        if self._game is None:
            raise RuntimeError("Game not loaded.")

        state = self._game.get_state()

        return DraughtsRenderPayload(
            game_type="draughts",
            variant=self.VARIANT_NAME,
            board=[row[:] for row in state.board],
            board_size=self._game.BOARD_SIZE,
            current_player=f"player_{state.current_player}",
            legal_moves=[m.to_string(self._game.BOARD_SIZE) for m in state.legal_moves],
            last_move=state.last_move.to_string(self._game.BOARD_SIZE) if state.last_move else None,
            is_game_over=state.is_game_over,
            winner=f"player_{state.winner}" if state.winner is not None else None,
            move_count=state.move_count,
        )

    def get_legal_moves(self) -> List[int]:
        """Get legal move indices."""
        if self._game is None:
            return []
        return list(range(len(self._game.get_legal_moves())))

    def get_legal_move_strings(self) -> List[str]:
        """Get legal moves as strings."""
        if self._game is None:
            return []
        return [m.to_string(self._game.BOARD_SIZE) for m in self._game.get_legal_moves()]

    def find_action_for_move(self, from_sq: str, to_sq: str) -> Optional[int]:
        """Find action index for a move."""
        if self._game is None:
            return None

        move = self._game.find_move(from_sq, to_sq)
        if move is None:
            return None

        # Find index
        for i, m in enumerate(self._game.get_legal_moves()):
            if self._game._moves_equal(move, m):
                return i
        return None

    def get_moves_from_square(self, square: str) -> List[str]:
        """Get destinations for moves from a square."""
        if self._game is None:
            return []
        return self._game.get_moves_from_square(square)

    def cell_to_algebraic(self, row: int, col: int) -> str:
        """Convert cell to algebraic notation."""
        if self._game is None:
            return ""
        return self._game._pos_to_algebraic((row, col))

    def algebraic_to_cell(self, square: str) -> Tuple[int, int]:
        """Convert algebraic to cell coordinates."""
        if self._game is None:
            return (0, 0)
        return self._game._algebraic_to_pos(square)

    def current_agent(self) -> str:
        """Get current player."""
        if self._game is None:
            return self.PLAYER_0
        return f"player_{self._game._current_player}"

    @property
    def num_agents(self) -> int:
        """Number of agents."""
        return 2

    def _package_step(
        self,
        state: DraughtsState,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> AdapterStep[Dict[str, Any]]:
        """Package step result."""
        render_payload = self.get_draughts_state().to_dict() if self._game else None

        # Determine active agent based on current player
        current_player = self._game._current_player if self._game else 0
        active_agent = "player_0" if current_player == 0 else "player_1"

        step_state = StepState(
            active_agent=active_agent,
            agents=[
                AgentSnapshot(name="player_0", role="black"),
                AgentSnapshot(name="player_1", role="white"),
            ],
            metrics={
                "step_index": self._episode_step,
                "terminated": terminated,
                "truncated": truncated,
            },
        )

        return AdapterStep(
            observation=state.to_dict(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            render_payload=render_payload,
            state=step_state,
        )


class AmericanCheckersAdapter(BaseDraughtsAdapter):
    """Adapter for American Checkers (8x8)."""

    id = GameId.AMERICAN_CHECKERS.value if hasattr(GameId, 'AMERICAN_CHECKERS') else "american_checkers"
    GAME_CLASS = AmericanCheckers
    VARIANT_NAME = "american"


class RussianCheckersAdapter(BaseDraughtsAdapter):
    """Adapter for Russian Checkers (8x8)."""

    id = GameId.RUSSIAN_CHECKERS.value if hasattr(GameId, 'RUSSIAN_CHECKERS') else "russian_checkers"
    GAME_CLASS = RussianCheckers
    VARIANT_NAME = "russian"


class InternationalDraughtsAdapter(BaseDraughtsAdapter):
    """Adapter for International Draughts (10x10)."""

    id = GameId.INTERNATIONAL_DRAUGHTS.value if hasattr(GameId, 'INTERNATIONAL_DRAUGHTS') else "international_draughts"
    GAME_CLASS = InternationalDraughts
    VARIANT_NAME = "international"


# Registry mapping GameId to adapter class
DRAUGHTS_ADAPTERS: dict[GameId, type[BaseDraughtsAdapter]] = {
    GameId.AMERICAN_CHECKERS: AmericanCheckersAdapter,
    GameId.RUSSIAN_CHECKERS: RussianCheckersAdapter,
    GameId.INTERNATIONAL_DRAUGHTS: InternationalDraughtsAdapter,
}

__all__ = [
    "Piece",
    "Move",
    "DraughtsState",
    "DraughtsGame",
    "AmericanCheckers",
    "RussianCheckers",
    "InternationalDraughts",
    "DraughtsRenderPayload",
    "BaseDraughtsAdapter",
    "AmericanCheckersAdapter",
    "RussianCheckersAdapter",
    "InternationalDraughtsAdapter",
    "DRAUGHTS_ADAPTERS",
]
