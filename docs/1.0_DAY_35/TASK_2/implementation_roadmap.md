# Implementation Roadmap: Multi-Agent Training for Human vs Agent Mode

**Date:** 2025-11-28 (Updated: 2025-11-29)
**Status:** Phase 1 (Board Game Rendering) ✅ COMPLETE | Phase 1.5 (Stockfish Integration) ✅ COMPLETE

---

## Overview

This roadmap outlines the step-by-step implementation to add training support for Multi-Agent Mode (Human vs Agent) board games.

---

## ✅ COMPLETED: Board Game Rendering Consolidation

### What Was Accomplished

Board games (Chess, Go, Connect Four) now render through a unified `BoardGameRendererStrategy` in the Grid tab, eliminating separate widget files.

### Architecture
```
RenderTabs
└── Grid Tab (dynamically renamed: "Grid - Chess", "Grid - Go", etc.)
    └── BoardGameRendererStrategy
        └── _BoardGameWidget (QStackedWidget)
            ├── _ChessBoardRenderer
            ├── _ConnectFourBoardRenderer
            └── _GoBoardRenderer
```

### Files Changed

| Action | File | Description |
|--------|------|-------------|
| ✅ Created | `gym_gui/rendering/strategies/board_game.py` | Consolidated strategy with all board renderers |
| ✅ Updated | `gym_gui/ui/widgets/render_tabs.py` | Uses BoardGameRendererStrategy |
| ✅ Updated | `gym_gui/ui/main_window.py` | Direct handler connections |
| ✅ Updated | `gym_gui/ui/handlers/__init__.py` | Removed PettingZooClassicHandler |
| ✅ Updated | `gym_gui/ui/environments/multi_agent_env/pettingzoo/__init__.py` | Removed PettingZooTab |
| ✅ Updated | `gym_gui/ui/environments/multi_agent_env/__init__.py` | Removed PettingZooTab |
| ❌ Deleted | `gym_gui/ui/widgets/chess_board.py` | Merged into board_game.py |
| ❌ Deleted | `gym_gui/ui/widgets/go_board.py` | Merged into board_game.py |
| ❌ Deleted | `gym_gui/ui/widgets/connect_four_board.py` | Merged into board_game.py |
| ❌ Deleted | `gym_gui/ui/environments/multi_agent_env/pettingzoo/pettingzoo_tab.py` | Replaced by Grid tab |
| ❌ Deleted | `gym_gui/ui/handlers/pettingzoo_classic_handler.py` | Direct connections |

### Key Features
- **Game Detection**: Detects game type from `game_type` field or content keys (`fen`, etc.)
- **Lazy Loading**: Renderers created on-demand
- **Flat Payload Support**: Handles both wrapped and flat payloads from adapters
- **Dynamic Tab Title**: Shows "Grid - Chess", "Grid - Connect Four", "Grid - Go"

---

## ✅ COMPLETED: Phase 1.5 - Stockfish Integration (2025-11-29)

### What Was Accomplished

Human vs Agent Chess mode now works with Stockfish as the AI opponent:
- Configuration dialog for AI opponent settings
- 5 difficulty presets with detailed explanations
- Interactive chess board with move highlighting
- Handler-based architecture for clean code separation

### Files Changed

| Action | File | Description |
|--------|------|-------------|
| ✅ Created | `gym_gui/services/chess_ai/__init__.py` | Chess AI service package |
| ✅ Created | `gym_gui/services/chess_ai/stockfish_service.py` | Stockfish wrapper with presets |
| ✅ Created | `gym_gui/ui/widgets/human_vs_agent_config_form.py` | Config dialog with tooltips |
| ✅ Created | `gym_gui/ui/widgets/human_vs_agent_board.py` | Interactive chess board |
| ✅ Created | `gym_gui/ui/handlers/human_vs_agent_handlers.py` | AI provider handler |
| ✅ Updated | `gym_gui/ui/widgets/multi_agent_tab.py` | Configure button + summary |
| ✅ Updated | `gym_gui/ui/main_window.py` | Handler integration, bug fix |
| ✅ Updated | `requirements/base.txt` | Added stockfish dependency |

### Bug Fixed

**Root Cause:** `main_window.py` compared `env_id == "chess"` but enum returns `"chess_v6"`, so tab never appeared.

**Solution:** Changed to `env_id == "chess_v6"` at line 665.

---

## ⏳ PENDING: Training Support Implementation

### Current State

- ✅ Single-Agent Mode has full training pipeline (Configure → Train → Load Policy)
- ✅ Board widgets render and accept user input via BoardGameRendererStrategy
- ✅ Human Control Mode allows playing against environment
- ✅ Human vs Agent Chess works with Stockfish AI opponent
- ✅ Environment Configuration dialog with detailed settings
- ❌ HumanVsAgentTab has no worker selection for custom training
- ❌ HumanVsAgentTab has no training configuration
- ❌ No self-play algorithm for PettingZoo Classic games
- ❌ "Custom Policy" option exists but loading not yet implemented

---

## Implementation Phases

### Phase 2: UI Enhancement

**Goal:** Add training UI components to HumanVsAgentTab

#### Task 2.1: Add Worker Integration Group
File: `gym_gui/ui/widgets/multi_agent_tab.py`

```python
# In HumanVsAgentTab._build_ui(), add after env_group:

# Worker Integration group
worker_group = QtWidgets.QGroupBox("Training", self)
worker_layout = QtWidgets.QFormLayout(worker_group)

self._worker_combo = QtWidgets.QComboBox(worker_group)
worker_layout.addRow("Worker:", self._worker_combo)

# Training buttons
train_btn_layout = QtWidgets.QHBoxLayout()
self._configure_btn = QtWidgets.QPushButton("Configure Training", worker_group)
self._train_btn = QtWidgets.QPushButton("Train Agent", worker_group)
self._train_btn.setEnabled(False)
train_btn_layout.addWidget(self._configure_btn)
train_btn_layout.addWidget(self._train_btn)
worker_layout.addRow("", train_btn_layout)
```

#### Task 2.2: Add New Signals
```python
worker_changed = pyqtSignal(str)
configure_training_requested = pyqtSignal(str, str)  # worker_id, env_id
train_agent_requested = pyqtSignal(str, str)  # worker_id, env_id
```

### Phase 3: Training Form

**Goal:** Create PettingZoo Classic training configuration dialog

File: `gym_gui/ui/widgets/pettingzoo_classic_train_form.py`

```python
class PettingZooClassicTrainForm(QtWidgets.QDialog):
    """Training configuration for PettingZoo Classic games.

    Parameters:
    - Algorithm: Self-Play PPO (default)
    - Total timesteps
    - Learning rate
    - Batch size
    - Self-play settings (opponent update frequency)
    - Game-specific settings (board size for Go)
    """

    def get_config(self) -> Dict[str, Any]:
        return {
            "algorithm": "ppo_selfplay",
            "env_id": self._env_id,
            "total_timesteps": self._timesteps_spin.value(),
            # ...
        }
```

### Phase 4: Self-Play Algorithm

**Goal:** Implement self-play PPO for PettingZoo AEC environments

File: `3rd_party/cleanrl_worker/cleanrl/cleanrl/ppo_pettingzoo_classic_selfplay.py`

Key components:
1. AEC environment wrapper with action masking
2. Self-play opponent management
3. Policy network (shared for both players)
4. Training loop adapted for turn-based games

### Phase 5: Policy Loading

**Goal:** Enable loading trained policies for Human vs Agent gameplay

File: `gym_gui/ui/widgets/pettingzoo_classic_policy_form.py`

Features:
- Discover policies in `VAR_TRAINER_DIR/runs/` for specific game
- Filter by game type (Chess, Go, Connect Four)
- Show metadata (algorithm, training steps, date)

### Phase 6: Integration

**Goal:** Connect all components in MainWindow

```python
def _connect_multi_agent_signals(self) -> None:
    hva = self._multi_agent_tab.human_vs_agent

    # Training signals
    hva.configure_training_requested.connect(self._on_configure_pettingzoo_training)
    hva.train_agent_requested.connect(self._on_train_pettingzoo_agent)

    # Policy loading
    hva.load_policy_requested.connect(self._on_load_pettingzoo_policy)
```

---

## File Checklist

### Phase 1 (COMPLETE)
- [x] `gym_gui/rendering/strategies/board_game.py` - Created
- [x] `gym_gui/ui/widgets/render_tabs.py` - Updated
- [x] `gym_gui/ui/main_window.py` - Updated
- [x] Old board widget files - Deleted

### Phase 1.5 (COMPLETE - Stockfish Integration)
- [x] `gym_gui/services/chess_ai/__init__.py` - Created
- [x] `gym_gui/services/chess_ai/stockfish_service.py` - Created
- [x] `gym_gui/ui/widgets/human_vs_agent_config_form.py` - Created
- [x] `gym_gui/ui/widgets/human_vs_agent_board.py` - Created
- [x] `gym_gui/ui/handlers/human_vs_agent_handlers.py` - Created
- [x] `gym_gui/ui/widgets/multi_agent_tab.py` - Updated with Configure button
- [x] `gym_gui/ui/main_window.py` - Fixed env_id bug, integrated handler

### Phase 2-6 (PENDING)
- [ ] `gym_gui/ui/widgets/multi_agent_tab.py` - Add worker selection for training
- [ ] `gym_gui/ui/widgets/pettingzoo_classic_train_form.py` - Create
- [ ] `gym_gui/ui/widgets/pettingzoo_classic_policy_form.py` - Create
- [ ] `3rd_party/cleanrl_worker/cleanrl/cleanrl/ppo_pettingzoo_classic_selfplay.py` - Create
- [ ] `3rd_party/cleanrl_worker/cleanrl_worker/cli.py` - Register algorithm

### Files to KEEP (In Use)
- `gym_gui/ui/handlers/chess_handlers.py` - Chess-specific logic (Human Control Mode)
- `gym_gui/ui/handlers/go_handlers.py` - Go-specific logic (Human Control Mode)
- `gym_gui/ui/handlers/connect_four_handlers.py` - Connect Four logic (Human Control Mode)
- `gym_gui/ui/handlers/human_vs_agent_handlers.py` - AI opponent management
- `gym_gui/controllers/chess_controller.py` - Game flow controller
- `gym_gui/services/chess_ai/stockfish_service.py` - Stockfish engine wrapper

---

## Success Metrics

| Metric | Status |
|--------|--------|
| Board games render in Grid tab correctly | ✅ Complete |
| Games switch properly when loading different environments | ✅ Complete |
| Human vs Agent Chess tab appears on Load Environment | ✅ Complete |
| Stockfish AI opponent with difficulty presets | ✅ Complete |
| Environment Configuration dialog functional | ✅ Complete |
| Human can play against Stockfish AI | ✅ Complete |
| Training pipeline works (configure → train → load) | ⏳ Pending |
| Custom policies discovered and loadable | ⏳ Pending |

---

## Dependencies

- PyTorch (for neural network)
- PettingZoo Classic (`pip install 'pettingzoo[classic]'`)
- python-chess (for Chess environment)
- Existing CleanRL worker infrastructure
- **Stockfish** (system): `sudo apt install stockfish`
- **stockfish** (Python): `pip install stockfish>=3.28.0` (in requirements/base.txt)
