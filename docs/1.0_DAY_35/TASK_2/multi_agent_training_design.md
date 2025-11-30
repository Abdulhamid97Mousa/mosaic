# Multi-Agent Mode: Human vs Agent Training Support

**Date:** 2025-11-28 (Updated: 2025-11-29)
**Status:** Board Game Rendering ✅ COMPLETE | Stockfish Integration ✅ COMPLETE | Training Support ⏳ PENDING

---

## ✅ Completed: Board Game Rendering Refactoring

### What Was Done

Board games (Chess, Go, Connect Four) now render through a **unified strategy pattern** in the Grid tab:

```
Grid Tab (Dynamic Title: "Grid - Chess", "Grid - Go", "Grid - Connect Four")
└── BoardGameRendererStrategy (gym_gui/rendering/strategies/board_game.py)
    ├── _BoardGameWidget (QStackedWidget)
    ├── _ChessBoardRenderer
    ├── _ConnectFourBoardRenderer
    └── _GoBoardRenderer
```

### Files Removed (Consolidated)
| File | Reason |
|------|--------|
| `gym_gui/ui/widgets/chess_board.py` | Merged into board_game.py |
| `gym_gui/ui/widgets/go_board.py` | Merged into board_game.py |
| `gym_gui/ui/widgets/connect_four_board.py` | Merged into board_game.py |
| `gym_gui/ui/environments/multi_agent_env/pettingzoo/pettingzoo_tab.py` | Grid tab integration |
| `gym_gui/ui/handlers/pettingzoo_classic_handler.py` | Direct handler connections |

### Key Implementation Details

1. **Game Detection**: Uses `game_type` field from adapter payloads or content keys (`fen`, etc.)
2. **Lazy Loading**: Renderers created on-demand when first payload displayed
3. **Signal Forwarding**: Board signals connected directly to handlers in MainWindow
4. **Dynamic Tab Title**: Tab shows current game name ("Grid - Chess", etc.)

---

## ✅ Completed: Stockfish Integration & Human vs Agent Chess (2025-11-29)

### What Was Done

Implemented a working Human vs Agent Chess mode using Stockfish as the AI opponent:

```
Multi-Agent Mode → Human vs Agent Tab
├── Environment selection (Chess)
├── Environment Configuration dialog
│   ├── AI Opponent type (Random, Stockfish, Custom Policy)
│   ├── Difficulty presets (Beginner, Easy, Medium, Hard, Expert)
│   └── Advanced settings (Skill level, Search depth, Time limit, Threads, Hash)
├── Player assignment (Human as White or Black)
└── Game controls (Load Environment, Start Game, Reset)
```

### Files Created/Modified

| Action | File | Description |
|--------|------|-------------|
| ✅ Created | `gym_gui/services/chess_ai/__init__.py` | Chess AI service package |
| ✅ Created | `gym_gui/services/chess_ai/stockfish_service.py` | Stockfish engine wrapper with difficulty presets |
| ✅ Created | `gym_gui/ui/widgets/human_vs_agent_config_form.py` | Configuration dialog for AI opponent settings |
| ✅ Created | `gym_gui/ui/widgets/human_vs_agent_board.py` | Interactive chess board widget for gameplay |
| ✅ Created | `gym_gui/ui/handlers/human_vs_agent_handlers.py` | Handler for AI provider management |
| ✅ Updated | `gym_gui/ui/widgets/multi_agent_tab.py` | Added Configure button, config summary display |
| ✅ Updated | `gym_gui/ui/main_window.py` | Integrated handler, fixed env_id comparison |
| ✅ Updated | `gym_gui/ui/handlers/__init__.py` | Added HumanVsAgentHandler export |
| ✅ Updated | `requirements/base.txt` | Added `stockfish>=3.28.0` dependency |

### Bug Fixed

**Root Cause:** The env_id comparison in `main_window.py` was checking `if env_id == "chess"` but the PettingZoo enum returns `"chess_v6"`. This caused the "Human vs Agent - Chess" tab to never be created.

### Architecture

```
HumanVsAgentTab (Control Panel)
│
├── "Configure AI Opponent..." button
│   └── HumanVsAgentConfigForm dialog
│       ├── StockfishConfig dataclass
│       ├── HumanVsAgentConfig dataclass
│       └── DIFFICULTY_PRESETS dict
│
├── "Load Environment" button
│   └── MainWindow._on_multi_agent_load_requested()
│       └── _load_chess_game()
│           ├── Creates InteractiveChessBoard widget
│           ├── Creates ChessGameController
│           ├── Creates HumanVsAgentHandler
│           │   └── setup_ai_provider() → StockfishService
│           └── Adds "Human vs Agent - Chess" tab to RenderTabs
│
└── Game flow
    ├── Human moves via InteractiveChessBoard
    ├── AI moves via StockfishService.get_best_move()
    └── State updates via ChessGameController
```

### Requirements

- **System:** `sudo apt install stockfish`
- **Python:** `pip install stockfish>=3.28.0` (included in requirements/base.txt)

---

## ⏳ Pending: Training Support

### Problem Statement

The Multi-Agent Mode sidebar lacks the ability to:
1. Select a training worker (CleanRL, etc.)
2. Configure and train agents for board games (Chess, Go, Connect Four)
3. Load trained policies for Human vs Agent gameplay

### Current Architecture Analysis

#### Single-Agent Mode Flow (Working)
```
Control Panel (Single-Agent Tab)
├── Active Actor Group
├── Worker Integration Group
│   └── Worker dropdown (cleanrl_worker, spade_bdi_worker, pettingzoo_worker)
└── Headless Training Group
    ├── Configure Agent button → Opens cleanrl_train_form.py dialog
    ├── Train Agent button → Submits training job
    └── Load Trained Policy button → Opens cleanrl_policy_form.py dialog
```

#### Multi-Agent Mode Flow (Current - Incomplete)
```
MultiAgentTab
├── HumanVsAgentTab (NEEDS WORK)
│   ├── Environment selection (Chess, Go, Connect Four)
│   ├── Load Policy button (no training option)
│   └── Game controls
├── CooperationTab (has worker selection, train/eval buttons)
└── CompetitionTab (has worker selection, train/eval buttons)
```

### CleanRL Algorithms Available for PettingZoo

From `3rd_party/cleanrl_worker/cleanrl/cleanrl/`:
- `ppo_pettingzoo_ma_atari.py` - Multi-agent PPO for PettingZoo Atari environments

**NOTE:** CleanRL does not have a dedicated algorithm for PettingZoo Classic board games (Chess, Go, Connect Four). We need to either:
1. Adapt existing algorithms for AEC environments
2. Use/create a self-play PPO implementation
3. Integrate with another library (e.g., RLlib, TianShou)

---

## Proposed Solution

### Phase 1: Add Training Support to HumanVsAgentTab

#### 1.1 UI Changes

Add a "Training" group box:

```
HumanVsAgentTab (Updated)
├── Environment Group (existing)
├── **NEW** Training Group
│   ├── Worker dropdown (CleanRL Worker, etc.)
│   ├── Configure Training button → Opens training form
│   └── Train Agent button → Starts headless training
├── AI Policy Group (existing)
└── Game Controls (existing)
```

#### 1.2 New Signals

```python
worker_changed = pyqtSignal(str)  # worker_id
configure_training_requested = pyqtSignal(str, str)  # worker_id, env_id
train_agent_requested = pyqtSignal(str, str)  # worker_id, env_id
```

### Phase 2: Create PettingZoo Classic Training Form

New file: `gym_gui/ui/widgets/pettingzoo_classic_train_form.py`

```python
class PettingZooClassicTrainForm(QtWidgets.QDialog):
    """Training configuration for PettingZoo Classic board games.

    Features:
    - Self-play PPO configuration
    - Board game specific parameters (board size for Go)
    - Opponent policy selection (random, previous checkpoint)
    """
```

### Phase 3: Implement Self-Play Training Algorithm

New file: `3rd_party/cleanrl_worker/cleanrl/cleanrl/ppo_pettingzoo_classic_selfplay.py`

Key components:
1. AEC environment wrapper
2. Self-play opponent management
3. Action mask handling
4. Policy checkpointing for opponent pool

### Phase 4: Policy Loading for Human vs Agent

```python
class PettingZooClassicPolicyForm(QtWidgets.QDialog):
    """Load trained policy for Human vs Agent gameplay.

    Features:
    - Discover policies trained for specific game
    - Filter by algorithm (PPO, etc.)
    - Preview policy metadata
    """
```

---

## Current File Structure

```
gym_gui/
├── rendering/
│   └── strategies/
│       └── board_game.py              # ✅ COMPLETE: Consolidated rendering
├── ui/
│   ├── widgets/
│   │   ├── multi_agent_tab.py         # ⏳ UPDATE: Add training UI
│   │   ├── pettingzoo_classic_train_form.py  # ⏳ NEW
│   │   └── pettingzoo_classic_policy_form.py # ⏳ NEW
│   └── handlers/
│       ├── chess_handlers.py          # ✅ KEEP: Human Control Mode
│       ├── go_handlers.py             # ✅ KEEP: Human Control Mode
│       └── connect_four_handlers.py   # ✅ KEEP: Human Control Mode

3rd_party/cleanrl_worker/cleanrl/cleanrl/
└── ppo_pettingzoo_classic_selfplay.py  # ⏳ NEW
```

---

## Questions to Resolve

1. **Algorithm Choice**: Start with Self-Play PPO for simplicity
2. **Opponent Selection**: Start with latest checkpoint, add pool later
3. **Training Duration**: Chess ~1M, Connect Four ~500K, Go ~2M timesteps
4. **Policy Format**: Use existing CleanRL checkpoint format (.pt files)

---

## Success Criteria

| Criteria | Status |
|----------|--------|
| Board games render in Grid tab with proper switching | ✅ Complete |
| Human vs Agent Chess tab appears when Load Environment clicked | ✅ Complete |
| Stockfish AI opponent works with adjustable difficulty | ✅ Complete |
| Environment Configuration dialog with detailed settings | ✅ Complete |
| User can select CleanRL worker in Human vs Agent tab | ⏳ Pending |
| User can configure and start training | ⏳ Pending |
| Training runs in background with telemetry | ⏳ Pending |
| User can load custom trained policy | ⏳ Pending |
