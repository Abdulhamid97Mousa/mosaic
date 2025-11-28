# PettingZoo Integration - Critical Design Decisions

**Date:** 2025-11-28
**Status:** ✅ IMPLEMENTED

---

## Summary of Design Decisions

| Decision | Chosen Approach | Status |
|----------|-----------------|--------|
| **Rendering** | State-based Qt rendering in Grid tab | ✅ Implemented |
| **File Organization** | Consolidated `board_game.py` strategy | ✅ Implemented |
| **Tab Structure** | Dynamic Grid tab title, no separate PettingZoo tab | ✅ Implemented |
| **API Handling** | AEC→Parallel conversion with adapters | ✅ Working |
| **Development** | Built Chess/Go/C4 together, not iteratively | ✅ Complete |

---

## 1. State-Based Rendering (Implemented)

### The Problem

If we rely on `env.render()` which returns a flat numpy RGB array, the UI is "blind":
- It can display an image of a chessboard
- But it has **no idea where the pieces are**
- Mouse clicks on the image cannot be mapped to logical game squares

### The Solution

Send structured game state to Qt widgets for rendering:

```python
# Adapters return structured data
payload = {
    "game_type": "chess",  # or "connect_four", "go"
    "fen": "rnbqkbnr/...",
    "legal_moves": ["e2e4", ...],
    "current_player": "white",
}

# Strategy renders interactive board
strategy = BoardGameRendererStrategy()
strategy.render(payload)
```

### Benefits

| env.render() | State-Based |
|--------------|-------------|
| Flat pixels | Structured data |
| UI is blind | UI knows piece positions |
| No click-to-move | Full mouse interaction |
| No move highlighting | Can highlight legal moves |
| No drag-and-drop | Drag-and-drop supported |

---

## 2. Consolidated File Organization (Implemented)

### The Problem

Original approach created separate files for each game:
- `gym_gui/ui/widgets/chess_board.py`
- `gym_gui/ui/widgets/go_board.py`
- `gym_gui/ui/widgets/connect_four_board.py`
- `gym_gui/ui/environments/multi_agent_env/pettingzoo/pettingzoo_tab.py`

This doesn't scale as PettingZoo has 50+ environments.

### The Solution

Single consolidated strategy file with internal renderers:

```python
# gym_gui/rendering/strategies/board_game.py

class BoardGameRendererStrategy(RendererStrategy):
    """Main entry point for board game rendering."""
    SUPPORTED_GAMES = frozenset({GameId.CHESS, GameId.CONNECT_FOUR, GameId.GO})

class _BoardGameWidget(QStackedWidget):
    """Manages game-specific renderers."""

class _ChessBoardRenderer(QWidget):
    """Chess-specific rendering and interaction."""

class _ConnectFourBoardRenderer(QWidget):
    """Connect Four-specific rendering and interaction."""

class _GoBoardRenderer(QWidget):
    """Go-specific rendering and interaction."""
```

### Benefits

1. **Reduced File Count**: 1 file instead of 5
2. **Easier Maintenance**: All board game logic in one place
3. **Scalability**: Easy to add new games as internal classes
4. **Consistency**: Shared patterns across games

---

## 3. Grid Tab Integration (Implemented)

### The Problem

User feedback: "Why do we have to create a tab called PettingZoo? It could have been part of Grid."

Creating separate tabs for each environment family doesn't scale.

### The Solution

Board games render in the Grid tab with dynamic title:

```
Grid Tab (when showing Chess)  → "Grid - Chess"
Grid Tab (when showing Go)     → "Grid - Go"
Grid Tab (when showing C4)     → "Grid - Connect Four"
```

### Implementation

```python
# render_tabs.py
def _display_board_game_payload(self, payload, game_id):
    # Initialize strategy on first use
    if self._board_game_strategy is None:
        self._board_game_strategy = BoardGameRendererStrategy(self)
        # Replace Grid tab content with strategy widget
        ...

    # Update tab title based on current game
    game_names = {
        GameId.CHESS: "Chess",
        GameId.CONNECT_FOUR: "Connect Four",
        GameId.GO: "Go",
    }
    self.setTabText(self._grid_tab_index, f"Grid - {game_names[game_id]}")
```

---

## 4. Game Detection Logic (Implemented)

### The Problem

Adapters return payloads in different formats:
- Chess: `{"fen": "...", "legal_moves": [...]}` (flat)
- Connect Four: `{"game_type": "connect_four", "board": [...]}` (with game_type)
- Go: `{"game_type": "go", "board": [...]}` (with game_type)

### The Solution

Detect from multiple sources:

```python
def get_game_from_payload(payload):
    # 1. Check content keys (Chess uses 'fen')
    if "chess" in payload or "fen" in payload:
        return GameId.CHESS
    if "connect_four" in payload:
        return GameId.CONNECT_FOUR
    if "go" in payload:
        return GameId.GO

    # 2. Check game_type field (from adapter to_dict())
    game_type = payload.get("game_type")
    if game_type == "chess":
        return GameId.CHESS
    if game_type == "connect_four":
        return GameId.CONNECT_FOUR
    if game_type == "go":
        return GameId.GO

    return None
```

---

## 5. Signal Architecture (Implemented)

### Original Plan

Use `PettingZooClassicHandler` as intermediary:

```
BoardWidget → PettingZooClassicHandler → SessionController
```

### Final Implementation

Direct connections to game-specific handlers:

```python
# main_window.py
self._render_tabs.chess_move_made.connect(self._chess_handler.on_chess_move)
self._render_tabs.connect_four_column_clicked.connect(
    self._connect_four_handler.on_column_clicked
)
self._render_tabs.go_intersection_clicked.connect(
    self._go_handler.on_intersection_clicked
)
```

### Benefits

1. Simpler signal chain
2. Less indirection
3. Easier to debug
4. Game handlers remain reusable

---

## Files Changed

### Created
- `gym_gui/rendering/strategies/board_game.py` - Consolidated board game strategy

### Updated
- `gym_gui/ui/widgets/render_tabs.py` - Uses BoardGameRendererStrategy
- `gym_gui/ui/main_window.py` - Direct handler connections
- `gym_gui/ui/handlers/__init__.py` - Removed PettingZooClassicHandler

### Removed
- `gym_gui/ui/widgets/chess_board.py`
- `gym_gui/ui/widgets/go_board.py`
- `gym_gui/ui/widgets/connect_four_board.py`
- `gym_gui/ui/environments/multi_agent_env/pettingzoo/pettingzoo_tab.py`
- `gym_gui/ui/handlers/pettingzoo_classic_handler.py`

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Board Game Rendering                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Adapter]                                                      │
│       │                                                         │
│       │ payload = {"game_type": "chess", "fen": "...", ...}    │
│       ▼                                                         │
│  [RenderTabs._display_board_game_payload()]                    │
│       │                                                         │
│       │ Detects game type, initializes strategy                │
│       ▼                                                         │
│  [BoardGameRendererStrategy]                                    │
│       │                                                         │
│       │ render(payload)                                         │
│       ▼                                                         │
│  [_BoardGameWidget (QStackedWidget)]                           │
│       │                                                         │
│       ├── _ChessBoardRenderer                                   │
│       ├── _ConnectFourBoardRenderer                            │
│       └── _GoBoardRenderer                                      │
│                │                                                │
│                │ Signals: move_made, column_clicked, etc.      │
│                ▼                                                │
│  [MainWindow Handlers]                                          │
│       │                                                         │
│       ├── ChessHandler.on_chess_move()                         │
│       ├── ConnectFourHandler.on_column_clicked()               │
│       └── GoHandler.on_intersection_clicked()                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Success Criteria Met

| Criteria | Status |
|----------|--------|
| State-based rendering with mouse interaction | ✅ |
| Consolidated file structure | ✅ |
| Grid tab integration with dynamic titles | ✅ |
| Game detection from multiple payload formats | ✅ |
| Direct handler connections | ✅ |
| Games switch correctly when loading new environment | ✅ |
