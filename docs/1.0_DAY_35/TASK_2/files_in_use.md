# Files In Use - Architecture Update (Refactored)

## New Architecture: Consolidated Board Game Rendering

Board games (Chess, Go, Connect Four) are now rendered through a unified strategy pattern
integrated into the Grid tab, eliminating the need for separate standalone widget files.

### `gym_gui/rendering/strategies/board_game.py` (NEW)
**Purpose:** Consolidated board game renderer strategy with pluggable game-specific renderers
**Contains:**
- `BoardGameRendererStrategy` - Main strategy class implementing RendererStrategy protocol
- `_BoardGameWidget` - QStackedWidget that manages game-specific renderers
- `_ChessBoardRenderer` - Chess board rendering and interaction
- `_ConnectFourBoardRenderer` - Connect Four board rendering and interaction
- `_GoBoardRenderer` - Go board rendering and interaction

**Used by:** `gym_gui/ui/widgets/render_tabs.py` - Lazy-loaded when PettingZoo Classic games are displayed

### `gym_gui/ui/widgets/render_tabs.py` (UPDATED)
**Changes:**
- Removed imports for `InteractiveChessBoard`, `InteractiveGoBoard`, `InteractiveConnectFourBoard`
- Removed `PettingZooTab` import
- Added `BoardGameRendererStrategy` for handling board games in the Grid tab
- Board games now display with tab title "Grid - Chess", "Grid - Go", etc.

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

- `gym_gui/ui/widgets/chess_board.py` - Merged into board_game.py
- `gym_gui/ui/widgets/go_board.py` - Merged into board_game.py
- `gym_gui/ui/widgets/connect_four_board.py` - Merged into board_game.py
- `gym_gui/ui/environments/multi_agent_env/pettingzoo/pettingzoo_tab.py` - No longer needed
- `gym_gui/ui/handlers/pettingzoo_classic_handler.py` - Replaced by direct handler connections

## Benefits of New Architecture

1. **Reduced File Count:** Single file instead of 5 separate files
2. **Scalability:** Easy to add new board games by adding new renderer subclasses
3. **Consistency:** All board games share common rendering infrastructure
4. **Integration:** Board games render in Grid tab like other environments
5. **Maintainability:** Centralized code is easier to update and test

## Controllers (KEEP)

### `gym_gui/controllers/chess_controller.py`
**Purpose:** Chess game controller for Human vs Agent gameplay
**Status:** Independent of UI widgets, works with any UI via signals
