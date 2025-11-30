# Files In Use - Architecture Update (Refactored)

**Date:** 2025-11-28 (Updated: 2025-11-29)
**Status:** ✅ COMPLETE

## New Architecture: Consolidated Board Game Rendering

Board games (Chess, Go, Connect Four) are now rendered through a unified strategy pattern
integrated into the Grid tab, eliminating the need for separate standalone widget files.

## Key Files

### `gym_gui/rendering/strategies/board_game.py` (NEW)
**Purpose:** Consolidated board game renderer strategy with pluggable game-specific renderers
**Contains:**
- `BoardGameRendererStrategy` - Main strategy class implementing RendererStrategy protocol
- `_BoardGameWidget` - QStackedWidget that manages game-specific renderers
- `_ChessBoardRenderer` - Chess board rendering and interaction
- `_ConnectFourBoardRenderer` - Connect Four board rendering and interaction
- `_GoBoardRenderer` - Go board rendering and interaction

**Key Features:**
- Detects game type from payload via `game_type` field or content keys (`fen`, etc.)
- Lazy-loads game renderers on demand
- Forwards signals (move_made, column_clicked, etc.) to MainWindow handlers
- Handles both wrapped payloads (`{"chess": {...}}`) and flat payloads from adapters

**Used by:** `gym_gui/ui/widgets/render_tabs.py`

### `gym_gui/ui/widgets/render_tabs.py` (UPDATED)
**Changes:**
- Removed imports for standalone board widgets
- Removed `PettingZooTab` import
- Added `BoardGameRendererStrategy` for handling board games in the Grid tab
- Board games now display with dynamic tab title: "Grid - Chess", "Grid - Go", "Grid - Connect Four"
- Lazy initialization of board game strategy on first board game payload

### `gym_gui/ui/main_window.py` (UPDATED)
**Changes:**
- Removed `PettingZooClassicHandler` import
- Board game signals now connect directly to legacy handlers:
  - `chess_move_made` → `ChessHandler.on_chess_move`
  - `connect_four_column_clicked` → `ConnectFourHandler.on_column_clicked`
  - `go_intersection_clicked` → `GoHandler.on_intersection_clicked`
  - `go_pass_requested` → `GoHandler.on_pass_requested`

## Handlers (KEEP)

These handlers process user input from the board game strategy:

### `gym_gui/ui/handlers/chess_handlers.py`
**Purpose:** Chess-specific move handling for Human Control Mode
**Used by:** `gym_gui/ui/main_window.py`

### `gym_gui/ui/handlers/go_handlers.py`
**Purpose:** Go-specific move handling for Human Control Mode
**Used by:** `gym_gui/ui/main_window.py`

### `gym_gui/ui/handlers/connect_four_handlers.py`
**Purpose:** Connect Four-specific move handling for Human Control Mode
**Used by:** `gym_gui/ui/main_window.py`

## Removed Files

The following files have been removed as part of the consolidation:

| File | Reason |
|------|--------|
| `gym_gui/ui/widgets/chess_board.py` | Merged into board_game.py |
| `gym_gui/ui/widgets/go_board.py` | Merged into board_game.py |
| `gym_gui/ui/widgets/connect_four_board.py` | Merged into board_game.py |
| `gym_gui/ui/environments/multi_agent_env/pettingzoo/pettingzoo_tab.py` | Replaced by Grid tab integration |
| `gym_gui/ui/handlers/pettingzoo_classic_handler.py` | Replaced by direct handler connections |

## Game Detection Logic

The `BoardGameRendererStrategy.get_game_from_payload()` detects games via:

1. **Content keys:** `"fen"` → Chess, `"chess"` key → Chess
2. **game_type field:** `"game_type": "connect_four"` → Connect Four
3. **game_type field:** `"game_type": "go"` → Go

This supports both wrapped payloads and flat payloads from adapter `to_dict()` methods.

## Benefits of New Architecture

1. **Reduced File Count:** Single file instead of 5 separate files
2. **Scalability:** Easy to add new board games by adding new renderer subclasses
3. **Consistency:** All board games share common rendering infrastructure
4. **Integration:** Board games render in Grid tab like other environments
5. **Maintainability:** Centralized code is easier to update and test
6. **Dynamic Tab Naming:** Tab shows "Grid - Chess" etc. based on current game

## Controllers (KEEP)

### `gym_gui/controllers/chess_controller.py`
**Purpose:** Chess game controller for Human vs Agent gameplay
**Status:** Independent of UI widgets, works with any UI via signals

---

## Human vs Agent Files (Added 2025-11-29)

### `gym_gui/services/chess_ai/__init__.py`
**Purpose:** Package init for chess AI services
**Exports:** `StockfishService`

### `gym_gui/services/chess_ai/stockfish_service.py`
**Purpose:** Stockfish chess engine wrapper with difficulty presets
**Contains:**
- `StockfishConfig` - Configuration dataclass for Stockfish settings
- `StockfishService` - Wrapper for Stockfish UCI protocol
- `DIFFICULTY_PRESETS` - Named presets (beginner, easy, medium, hard, expert)
**Used by:** `gym_gui/ui/handlers/human_vs_agent_handlers.py`

### `gym_gui/ui/widgets/human_vs_agent_config_form.py`
**Purpose:** Configuration dialog for Human vs Agent gameplay settings
**Contains:**
- `StockfishConfig` - Skill level, depth, time limit, threads, hash size
- `HumanVsAgentConfig` - Opponent type, difficulty, custom policy path
- `HumanVsAgentConfigForm` - QDialog with radio buttons, spinboxes, tooltips
- `DIFFICULTY_PRESETS` - Preset configurations
- `DIFFICULTY_DESCRIPTIONS` - User-friendly descriptions
**Used by:** `gym_gui/ui/widgets/multi_agent_tab.py`

### `gym_gui/ui/widgets/human_vs_agent_board.py`
**Purpose:** Interactive chess board widget for Human vs Agent mode
**Contains:**
- `InteractiveChessBoard` - QWidget with chess board rendering
  - Parses FEN strings
  - Shows legal moves, last move, check highlights
  - Emits `move_made(from_sq, to_sq)` signal on user moves
**Used by:** `gym_gui/ui/main_window.py`

### `gym_gui/ui/handlers/human_vs_agent_handlers.py`
**Purpose:** Handler for Human vs Agent AI opponent management
**Contains:**
- `HumanVsAgentHandler` - Manages AI providers (Stockfish, random)
  - `setup_ai_provider()` - Initializes AI based on config
  - `on_ai_config_changed()` - Updates AI when settings change
  - `cleanup()` - Stops Stockfish process
**Used by:** `gym_gui/ui/main_window.py`

## Updated Files (2025-11-29)

### `gym_gui/ui/widgets/multi_agent_tab.py`
**Changes:**
- Added `HumanVsAgentConfig` dataclass import
- Replaced inline AI opponent dropdowns with "Configure AI Opponent..." button
- Added `_config_summary` label showing current configuration
- Added `_on_configure_clicked()` - Opens config dialog
- Added `_on_config_accepted()` - Handles dialog result
- Added `_update_config_summary()` - Updates summary display
- Added `get_ai_config()` - Returns full config object

### `gym_gui/ui/main_window.py`
**Changes:**
- Fixed env_id comparison: `"chess"` → `"chess_v6"` (line 665)
- Added `HumanVsAgentHandler` import and instance variable
- Removed `_stockfish_service`, `_current_ai_opponent`, `_current_ai_difficulty`
- Removed `_setup_chess_ai_provider_from_config()` method (moved to handler)
- Updated `_load_chess_game()` to use handler
- Updated `_on_ai_opponent_changed()` to delegate to handler

### `gym_gui/ui/handlers/__init__.py`
**Changes:**
- Added `HumanVsAgentHandler` to exports

### `requirements/base.txt`
**Changes:**
- Added `stockfish>=3.28.0` for Stockfish Python bindings
