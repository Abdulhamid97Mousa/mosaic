# Implementation Roadmap: Multi-Agent Training for Human vs Agent Mode

**Date:** 2025-11-28
**Status:** Phase 1 (Board Game Rendering) ✅ COMPLETE

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

## ⏳ PENDING: Training Support Implementation

### Current State

- ✅ Single-Agent Mode has full training pipeline (Configure → Train → Load Policy)
- ✅ Board widgets render and accept user input via BoardGameRendererStrategy
- ✅ Human Control Mode allows playing against environment
- ❌ HumanVsAgentTab has no worker selection
- ❌ HumanVsAgentTab has no training configuration
- ❌ No self-play algorithm for PettingZoo Classic games
- ❌ "Load Trained Policy" button exists but no policies to load

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

### Phase 2-6 (PENDING)
- [ ] `gym_gui/ui/widgets/multi_agent_tab.py` - Add training UI
- [ ] `gym_gui/ui/widgets/pettingzoo_classic_train_form.py` - Create
- [ ] `gym_gui/ui/widgets/pettingzoo_classic_policy_form.py` - Create
- [ ] `3rd_party/cleanrl_worker/cleanrl/cleanrl/ppo_pettingzoo_classic_selfplay.py` - Create
- [ ] `3rd_party/cleanrl_worker/cleanrl_worker/cli.py` - Register algorithm

### Files to KEEP (In Use)
- `gym_gui/ui/handlers/chess_handlers.py` - Chess-specific logic
- `gym_gui/ui/handlers/go_handlers.py` - Go-specific logic
- `gym_gui/ui/handlers/connect_four_handlers.py` - Connect Four logic
- `gym_gui/controllers/chess_controller.py` - Game flow controller

---

## Success Metrics

| Metric | Status |
|--------|--------|
| Board games render in Grid tab correctly | ✅ Complete |
| Games switch properly when loading different environments | ✅ Complete |
| Training pipeline works (configure → train → load) | ⏳ Pending |
| Policies discovered and loadable | ⏳ Pending |
| Human vs AI gameplay functional | ⏳ Pending |

---

## Dependencies

- PyTorch (for neural network)
- PettingZoo Classic (`pip install 'pettingzoo[classic]'`)
- python-chess (for Chess environment)
- Existing CleanRL worker infrastructure
