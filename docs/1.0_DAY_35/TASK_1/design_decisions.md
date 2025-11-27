# PettingZoo Integration - Critical Design Decisions

**Date:** 2025-11-28
**Status:** Architecture Review

---

## Summary of Contrarian Insights

Three key architectural decisions that simplify implementation:

| Decision | Naive Approach | Better Approach |
|----------|---------------|-----------------|
| **API Handling** | Support both AEC and Parallel | Convert AEC→Parallel immediately |
| **Rendering** | Use `env.render()` for pixel display | Send game state, let Qt draw the board |
| **Development** | Build generic adapter first | Build ChessAdapter first, then generalize |

---

## 1. The Interaction Trap (The Biggest Risk)

### Problem

If we rely on `env.render()` which returns a flat numpy RGB array, the UI is "blind":
- It can display an image of a chessboard
- But it has **no idea where the pieces are**
- Mouse clicks on the image cannot be mapped to logical game squares

### Consequence

Without state-based rendering, users would have to type moves like "e2e4" in a text box because the UI cannot:
- Know which square was clicked
- Highlight valid moves
- Implement drag-and-drop

### Solution: State-Based Interface

```python
# Instead of this:
frame = env.render()  # Flat RGB array - UI is blind
render_view.display(frame)

# Do this:
state = adapter.get_game_state()
# state = {
#     "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
#     "current_player": "black",
#     "legal_moves": ["e7e5", "e7e6", "d7d5", ...],
#     "last_move": "e2e4",
# }
chess_board_widget.update_state(state)  # Qt draws the board
```

### Benefits

| env.render() | State-Based |
|--------------|-------------|
| Flat pixels | Structured data |
| UI is blind | UI knows piece positions |
| No click-to-move | Full mouse interaction |
| No move highlighting | Can highlight legal moves |
| No drag-and-drop | Drag-and-drop supported |

### Implementation for Chess

PettingZoo Chess exposes the board state via the observation:

```python
from pettingzoo.classic import chess_v6

env = chess_v6.env()
env.reset()

# Get observation for current player
obs = env.observe(env.agent_selection)

# obs["observation"] is a (8,8,111) array encoding piece positions
# But we can get FEN string from the underlying chess board:
fen = env.unwrapped.board.fen()
legal_moves = [str(m) for m in env.unwrapped.board.legal_moves]
```

### PyQt6 Mouse Event Integration

The RenderView widget (currently using QLabel) needs to be extended for interactive games:

```python
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSignal

class InteractiveChessBoard(QtWidgets.QWidget):
    """Chess board widget with mouse interaction."""

    # Signals
    square_clicked = pyqtSignal(str)  # e.g., "e4"
    move_made = pyqtSignal(str, str)  # from_square, to_square (e.g., "e2", "e4")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._board_state = None  # FEN or piece positions
        self._selected_square = None
        self._legal_moves = []
        self._square_size = 60

        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse click on board."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            square = self._pixel_to_square(event.position())
            if square:
                self._handle_square_click(square)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse movement for hover highlighting."""
        square = self._pixel_to_square(event.position())
        self._update_hover(square)
        super().mouseMoveEvent(event)

    def _pixel_to_square(self, pos: QtCore.QPointF) -> str | None:
        """Convert pixel coordinates to chess square notation."""
        x, y = int(pos.x()), int(pos.y())
        col = x // self._square_size
        row = 7 - (y // self._square_size)  # Chess ranks are bottom-up

        if 0 <= col < 8 and 0 <= row < 8:
            file = chr(ord('a') + col)
            rank = str(row + 1)
            return f"{file}{rank}"
        return None

    def _handle_square_click(self, square: str) -> None:
        """Handle click on a square."""
        if self._selected_square is None:
            # First click - select piece
            if self._has_piece_at(square):
                self._selected_square = square
                self.square_clicked.emit(square)
                self.update()  # Redraw with selection highlight
        else:
            # Second click - attempt move
            move = f"{self._selected_square}{square}"
            if move in self._legal_moves or self._is_legal_move(self._selected_square, square):
                self.move_made.emit(self._selected_square, square)
            self._selected_square = None
            self.update()

    def set_game_state(self, fen: str, legal_moves: list[str]) -> None:
        """Update board state from FEN string."""
        self._board_state = fen
        self._legal_moves = legal_moves
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Draw the chess board with pieces."""
        painter = QtGui.QPainter(self)
        self._draw_squares(painter)
        self._draw_pieces(painter)
        self._draw_highlights(painter)
        painter.end()
```

### Use Cases

| Use Case | env.render() | State-Based |
|----------|-------------|-------------|
| Spectator mode (watch AI play) | ✅ Sufficient | ✅ Works |
| Human vs AI (keyboard input) | ⚠️ Requires text input | ✅ Click to move |
| Human vs AI (mouse input) | ❌ Impossible | ✅ Full interaction |
| Debugging/development | ✅ Quick visual | ✅ + detailed state |

### Recommendation

Use **hybrid approach**:
1. For Classic games (Chess, Go, Tic-Tac-Toe): **State-based Qt rendering**
2. For visual games (Atari, Butterfly): **env.render() pixel display**

---

## 2. The "AEC is Unnecessary" Argument

### Problem

Supporting both AEC (turn-based) and Parallel (simultaneous) APIs means:
- Different `step()` signatures
- Agent selection loop management in SessionController
- Two code paths for everything

### Solution: Convert AEC to Parallel Immediately

PettingZoo provides a built-in wrapper:

```python
from pettingzoo.classic import chess_v6
from pettingzoo.utils import aec_to_parallel

# Original AEC environment
aec_env = chess_v6.env()

# Convert to Parallel API
parallel_env = aec_to_parallel(aec_env)

# Now always use Parallel API
observations, infos = parallel_env.reset()
# observations = {"player_0": obs0, "player_1": obs1}

# Step with action dict (inactive players get None or no-op)
actions = {
    "player_0": my_action,  # Active player's action
    "player_1": None,       # Inactive player (wrapper handles this)
}
observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
```

### Benefits

| Aspect | Supporting Both APIs | Parallel-Only |
|--------|---------------------|---------------|
| Controller complexity | High (two code paths) | Low (one path) |
| Step signature | Varies | Always `step(actions_dict)` |
| Agent loop management | In controller | In wrapper |
| Testing surface | Large | Small |

### Implementation

```python
class PettingZooAdapter:
    """Always uses Parallel API internally."""

    def load(self, env_id: str) -> None:
        # Import the appropriate module
        env = self._create_env(env_id)

        # If it's AEC, convert to Parallel
        if hasattr(env, 'agent_iter'):  # AEC signature
            from pettingzoo.utils import aec_to_parallel
            env = aec_to_parallel(env)

        self._env = env
        self._api_type = "parallel"  # Always parallel after conversion

    def step(self, actions: dict[str, Any]) -> MultiAgentStep:
        """Always takes action dict, regardless of original API."""
        observations, rewards, terminations, truncations, infos = self._env.step(actions)
        return MultiAgentStep(...)
```

### Caveat

For Human vs Agent in turn-based games, we still need to know whose turn it is. The wrapper handles this via the `infos` dict:

```python
# After step, check who should move next
current_player = None
for agent, info in infos.items():
    if info.get("action_mask") is not None:  # This agent can act
        current_player = agent
        break
```

---

## 3. The YAGNI Approach: ChessAdapter First

### Problem

Building a generic `PettingZooAdapter` requires solving:
- Environment import/instantiation for all families
- API conversion for all environments
- State extraction for all game types
- Async input blocking for human play

This is a lot to get right before seeing anything work.

### Solution: Hardcode ChessAdapter First

Build a specific adapter just for Chess to solve the hardest problem: **async input blocking**.

```python
class ChessAdapter:
    """Specific adapter for Chess - solves the async input problem first."""

    def __init__(self):
        from pettingzoo.classic import chess_v6
        from pettingzoo.utils import aec_to_parallel

        self._aec_env = chess_v6.env(render_mode="rgb_array")
        self._env = aec_to_parallel(self._aec_env)
        self._board = None  # Reference to underlying chess.Board

    def load(self) -> None:
        self._env.reset()
        self._board = self._aec_env.unwrapped.board

    def get_state(self) -> dict:
        """Return structured game state for Qt rendering."""
        return {
            "fen": self._board.fen(),
            "current_player": "white" if self._board.turn else "black",
            "legal_moves": [str(m) for m in self._board.legal_moves],
            "is_check": self._board.is_check(),
            "is_checkmate": self._board.is_checkmate(),
            "is_stalemate": self._board.is_stalemate(),
        }

    def make_move(self, uci_move: str) -> dict:
        """Execute a move in UCI notation (e.g., 'e2e4')."""
        import chess
        move = chess.Move.from_uci(uci_move)

        # Convert to action index for PettingZoo
        action = self._move_to_action(move)

        # Determine which player is moving
        current_agent = "player_0" if self._board.turn else "player_1"
        other_agent = "player_1" if self._board.turn else "player_0"

        # Step with Parallel API
        actions = {
            current_agent: action,
            other_agent: None,  # Inactive
        }
        observations, rewards, terminations, truncations, infos = self._env.step(actions)

        return self.get_state()

    def get_render_frame(self) -> np.ndarray | None:
        """Get pixel render (for debug/spectator mode)."""
        return self._env.render()
```

### The Async Input Problem

The core challenge for Human vs Agent:

```
┌─────────────────────────────────────────────────────────────────┐
│  Python Backend                      Qt UI Thread              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  def game_loop():                    [Chess Board Widget]       │
│      while not done:                       │                   │
│          if current_player == human:       │                   │
│              action = ???  ◄───────────────┤ Wait for click    │
│              # BLOCKING!                   │                   │
│          else:                             │                   │
│              action = ai.get_action()      │                   │
│          env.step(action)                  │                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Solutions to Async Input

**Option A: Qt Event Loop Integration (Recommended)**

```python
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QEventLoop

class ChessGameController(QObject):
    """Controls chess game flow with Qt event loop integration."""

    state_changed = pyqtSignal(dict)  # Emit new game state
    awaiting_human_input = pyqtSignal()
    game_over = pyqtSignal(str)  # winner

    def __init__(self, adapter: ChessAdapter, human_player: str = "white"):
        super().__init__()
        self._adapter = adapter
        self._human_player = human_player
        self._pending_move = None
        self._waiting_for_input = False

    def start_game(self) -> None:
        """Initialize and emit first state."""
        self._adapter.load()
        state = self._adapter.get_state()
        self.state_changed.emit(state)
        self._check_next_turn(state)

    def _check_next_turn(self, state: dict) -> None:
        """Determine what to do based on current player."""
        if state.get("is_checkmate") or state.get("is_stalemate"):
            self.game_over.emit(self._get_winner(state))
            return

        if state["current_player"] == self._human_player:
            # Human's turn - wait for input
            self._waiting_for_input = True
            self.awaiting_human_input.emit()
        else:
            # AI's turn - compute and execute move
            self._execute_ai_move(state)

    @pyqtSlot(str)
    def on_human_move(self, uci_move: str) -> None:
        """Called when human makes a move via UI."""
        if not self._waiting_for_input:
            return

        self._waiting_for_input = False
        state = self._adapter.make_move(uci_move)
        self.state_changed.emit(state)
        self._check_next_turn(state)

    def _execute_ai_move(self, state: dict) -> None:
        """Get AI move and execute it."""
        # For now, random legal move (replace with actual policy)
        import random
        ai_move = random.choice(state["legal_moves"])

        state = self._adapter.make_move(ai_move)
        self.state_changed.emit(state)

        # Use QTimer to allow UI to update before next turn check
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self._check_next_turn(state))
```

**Option B: Threading with Queue**

```python
import queue
import threading

class ChessGameThread(threading.Thread):
    """Game loop in separate thread, communicates via queues."""

    def __init__(self, adapter, input_queue, output_queue):
        super().__init__(daemon=True)
        self._adapter = adapter
        self._input_queue = input_queue  # Receives human moves
        self._output_queue = output_queue  # Sends state updates

    def run(self):
        self._adapter.load()
        state = self._adapter.get_state()
        self._output_queue.put(("state", state))

        while not self._is_game_over(state):
            if state["current_player"] == "white":  # Human
                self._output_queue.put(("awaiting_input", None))
                move = self._input_queue.get()  # Blocks until UI sends move
            else:  # AI
                move = self._get_ai_move(state)

            state = self._adapter.make_move(move)
            self._output_queue.put(("state", state))
```

### Development Order

1. **ChessAdapter** - Hardcode chess-specific logic
2. **InteractiveChessBoard** - Qt widget with mouse events
3. **ChessGameController** - Async input handling
4. **Integration** - Wire it up in MainWindow
5. **Generalize** - Extract patterns for other games

---

## 4. Updated Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Human vs Agent Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Human Vs Agent Tab]                                           │
│       │                                                         │
│       │ load_environment_requested("chess_v6", seed=1)          │
│       ▼                                                         │
│  [MainWindow._on_multi_agent_load_requested()]                  │
│       │                                                         │
│       │ Create ChessAdapter                                     │
│       │ Create ChessGameController                              │
│       │ Create InteractiveChessBoard                            │
│       ▼                                                         │
│  ┌──────────────────┐    state_changed    ┌──────────────────┐ │
│  │ ChessAdapter     │◄──────────────────► │ChessGameController│ │
│  │ (PettingZoo)     │                     │ (Async Logic)    │ │
│  └──────────────────┘                     └──────────────────┘ │
│       │                                           │             │
│       │ get_state()                               │             │
│       ▼                                           ▼             │
│  ┌──────────────────┐    move_made        ┌──────────────────┐ │
│  │ InteractiveChess │◄────────────────────│   (signals)      │ │
│  │ Board (Qt)       │                     │                  │ │
│  │                  │─────────────────────►                  │ │
│  │ - Mouse events   │    on_human_move()                     │ │
│  │ - Draw pieces    │                                         │ │
│  │ - Highlight moves│                                         │ │
│  └──────────────────┘                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Next Steps (Revised Priority)

### Phase 3A: Chess-Specific Implementation

1. **Create ChessAdapter** (`gym_gui/core/adapters/chess_adapter.py`)
   - Hardcode chess environment loading
   - Implement `get_state()` returning FEN + legal moves
   - Implement `make_move(uci)` for executing moves
   - Use `aec_to_parallel` wrapper internally

2. **Create InteractiveChessBoard** (`gym_gui/ui/widgets/chess_board.py`)
   - Qt widget that draws chess board from FEN
   - Mouse event handling (click to select, click to move)
   - Legal move highlighting
   - Last move highlighting

3. **Create ChessGameController** (`gym_gui/controllers/chess_game.py`)
   - Qt event loop integration for async human input
   - Turn management (human vs AI)
   - Signal emission for state changes

4. **Wire Up in MainWindow**
   - Replace placeholder `_on_multi_agent_load_requested()`
   - Add chess board to Render View area
   - Connect signals

### Phase 3B: Generalize

5. **Extract PettingZooAdapter base class**
6. **Create TicTacToeAdapter** (simpler than chess, good for testing)
7. **Create generic game board widget framework**

---

## 6. File Structure (Revised)

```
gym_gui/
├── core/
│   └── adapters/
│       ├── pettingzoo_base.py       # Base class (later)
│       └── chess_adapter.py         # Chess-specific (first)
├── ui/
│   ├── widgets/
│   │   ├── chess_board.py           # Interactive chess widget
│   │   └── game_boards/             # Other game boards (later)
│   └── environments/
│       └── multi_agent_env/
│           └── pettingzoo/
│               └── config_panel.py   # Already exists
└── controllers/
    └── chess_game.py                 # Game flow controller
```

---

## Summary

| Original Plan | Revised Plan |
|--------------|--------------|
| Generic PettingZooAdapter | ChessAdapter first |
| Support AEC + Parallel | Convert AEC→Parallel always |
| env.render() for display | State-based Qt rendering |
| Handle async "somehow" | Qt event loop with signals |

The key insight: **Solve the specific hard problem (Chess with mouse input) before generalizing.**
