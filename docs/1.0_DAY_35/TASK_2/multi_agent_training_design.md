# Multi-Agent Mode: Human vs Agent Training Support

## Problem Statement

The Multi-Agent Mode sidebar lacks the ability to:
1. Select a training worker (CleanRL, etc.)
2. Configure and train agents for board games (Chess, Go, Connect Four)
3. Load trained policies for Human vs Agent gameplay

Currently, the **HumanVsAgentTab** only has:
- Environment selection (Family, Game, Seed)
- "Load Trained Policy..." button (but no way to train a policy first)
- Player assignment and game controls

**What's Missing:**
- No Worker Selection widget in Human vs Agent tab
- No "Configure Agent" / "Train Agent" buttons
- No algorithm selection for training
- No integration with the existing CleanRL training forms

## Current Architecture Analysis

### Single-Agent Mode Flow (Working)
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

### Multi-Agent Mode Flow (Current - Incomplete)
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

## Proposed Solution

### Phase 1: Add Training Support to HumanVsAgentTab

#### 1.1 UI Changes to HumanVsAgentTab

Add a "Training" group box between Environment and AI Policy groups:

```
HumanVsAgentTab (Updated)
├── Environment Group (existing)
│   ├── Family dropdown
│   ├── Game dropdown
│   ├── Seed spinner
│   └── Load Environment button
├── **NEW** Training Group
│   ├── Worker dropdown (CleanRL Worker, etc.)
│   ├── Worker description label
│   ├── Configure Training button → Opens training form
│   └── Train Agent button → Starts headless training
├── AI Policy Group (existing)
│   ├── Load Trained Policy button
│   └── Policy status label
└── Game Controls (existing)
```

#### 1.2 New Signals for HumanVsAgentTab

```python
# Add to HumanVsAgentTab
worker_changed = pyqtSignal(str)  # worker_id
configure_training_requested = pyqtSignal(str, str)  # worker_id, env_id
train_agent_requested = pyqtSignal(str, str)  # worker_id, env_id
```

### Phase 2: Create PettingZoo Classic Training Form

Since CleanRL doesn't have a dedicated Chess/Go/Connect Four algorithm, we need:

#### 2.1 New File: `gym_gui/ui/widgets/pettingzoo_classic_train_form.py`

A specialized training configuration dialog for AEC board games:

```python
class PettingZooClassicTrainForm(QtWidgets.QDialog):
    """Training configuration for PettingZoo Classic board games.

    Features:
    - Self-play PPO configuration
    - Board game specific parameters (board size for Go)
    - Opponent policy selection (random, previous checkpoint)
    - Training duration (episodes, timesteps)
    """
```

#### 2.2 Algorithm Options

For board games, common approaches are:
1. **Self-Play PPO** - Agent plays against itself
2. **Population-Based Training** - Train pool of agents
3. **AlphaZero-style MCTS + Neural Network** (future)

Initial implementation will use **Self-Play PPO** since it's:
- Simpler to implement
- Works with existing CleanRL infrastructure
- Effective for two-player games

### Phase 3: Implement Self-Play Training Algorithm

#### 3.1 New File: `3rd_party/cleanrl_worker/cleanrl/cleanrl/ppo_pettingzoo_classic_selfplay.py`

```python
"""Self-Play PPO for PettingZoo Classic AEC environments.

Based on CleanRL's PPO implementation, adapted for:
- AEC (turn-based) environments
- Self-play training
- Action masking for legal moves
"""

# Key components:
# 1. AEC environment wrapper
# 2. Self-play opponent management
# 3. Action mask handling
# 4. Policy checkpointing for opponent pool
```

### Phase 4: Policy Loading for Human vs Agent

#### 4.1 Update Policy Form for Multi-Agent

The existing `cleanrl_policy_form.py` needs to be extended or a new form created:

```python
class PettingZooClassicPolicyForm(QtWidgets.QDialog):
    """Load trained policy for Human vs Agent gameplay.

    Features:
    - Discover policies trained for specific game (Chess, Go, Connect Four)
    - Filter by algorithm (PPO, etc.)
    - Preview policy metadata (training steps, win rate)
    - Select which player the AI controls
    """
```

### Phase 5: Connect Everything in MainWindow

#### 5.1 Signal Connections

```python
# In MainWindow.__init__ or _connect_multi_agent_signals()

# HumanVsAgentTab training signals
self._multi_agent_tab.human_vs_agent.configure_training_requested.connect(
    self._on_configure_pettingzoo_training
)
self._multi_agent_tab.human_vs_agent.train_agent_requested.connect(
    self._on_train_pettingzoo_agent
)
```

#### 5.2 Handler Methods

```python
def _on_configure_pettingzoo_training(self, worker_id: str, env_id: str) -> None:
    """Open training configuration for PettingZoo Classic game."""
    form = PettingZooClassicTrainForm(env_id=env_id, parent=self)
    if form.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        self._pending_training_config = form.get_config()

def _on_train_pettingzoo_agent(self, worker_id: str, env_id: str) -> None:
    """Submit PettingZoo Classic training job."""
    if self._pending_training_config:
        self._submit_training_job(self._pending_training_config)
```

## Implementation Plan

### Step 1: Update HumanVsAgentTab UI
- Add Worker Integration group box
- Add worker dropdown populated from `get_worker_catalog()`
- Add "Configure Training" and "Train Agent" buttons
- Add new signals

**Files to modify:**
- `gym_gui/ui/widgets/multi_agent_tab.py`

### Step 2: Create PettingZoo Classic Training Form
- Implement training configuration dialog
- Support algorithm selection (Self-Play PPO)
- Add game-specific parameters
- Generate training config compatible with CleanRL worker

**Files to create:**
- `gym_gui/ui/widgets/pettingzoo_classic_train_form.py`

### Step 3: Implement Self-Play PPO Algorithm
- Create CleanRL-compatible self-play algorithm
- Handle AEC environment step loop
- Implement action masking
- Add opponent policy checkpointing

**Files to create:**
- `3rd_party/cleanrl_worker/cleanrl/cleanrl/ppo_pettingzoo_classic_selfplay.py`

### Step 4: Create/Update Policy Loading Form
- Implement policy discovery for PettingZoo Classic games
- Add policy metadata display
- Support loading policy for Human vs Agent mode

**Files to create/modify:**
- `gym_gui/ui/widgets/pettingzoo_classic_policy_form.py` (new)
- OR extend `gym_gui/ui/widgets/cleanrl_policy_form.py`

### Step 5: Connect to MainWindow
- Add signal handlers for new training/policy signals
- Integrate with trainer daemon client
- Handle policy loading for gameplay

**Files to modify:**
- `gym_gui/ui/main_window.py`

### Step 6: Update CleanRL Worker CLI
- Register new algorithm
- Add PettingZoo Classic environment support

**Files to modify:**
- `3rd_party/cleanrl_worker/cleanrl_worker/cli.py`
- `3rd_party/cleanrl_worker/cleanrl_worker/config.py`

## Board Widgets Status

**CONFIRMED IN USE - DO NOT REMOVE:**
- `gym_gui/ui/widgets/chess_board.py` - Used by PettingZooTab
- `gym_gui/ui/widgets/go_board.py` - Used by PettingZooTab
- `gym_gui/ui/widgets/connect_four_board.py` - Used by PettingZooTab

These provide the interactive UI for Human vs Agent gameplay.

## File Structure Summary

```
gym_gui/ui/widgets/
├── multi_agent_tab.py           # UPDATE: Add training UI to HumanVsAgentTab
├── pettingzoo_classic_train_form.py  # NEW: Training config dialog
├── pettingzoo_classic_policy_form.py # NEW: Policy loading dialog
├── chess_board.py               # KEEP: Used for Chess gameplay
├── go_board.py                  # KEEP: Used for Go gameplay
└── connect_four_board.py        # KEEP: Used for Connect Four gameplay

3rd_party/cleanrl_worker/cleanrl/cleanrl/
├── ppo_pettingzoo_classic_selfplay.py  # NEW: Self-play training algorithm
└── ... (existing algorithms)

gym_gui/ui/
└── main_window.py               # UPDATE: Connect new signals
```

## Questions to Resolve

1. **Algorithm Choice**: Should we start with Self-Play PPO or implement something more sophisticated (MCTS+NN)?
   - Recommendation: Start with Self-Play PPO for simplicity

2. **Opponent Selection**: How should the AI select opponents during self-play?
   - Options: Latest checkpoint, random from pool, curriculum-based
   - Recommendation: Start with latest checkpoint, add pool later

3. **Training Duration**: What defaults for board games?
   - Chess: ~1M timesteps
   - Connect Four: ~500K timesteps
   - Go: ~2M timesteps (game is more complex)

4. **Policy Format**: How to store/load policies?
   - Recommendation: Use existing CleanRL checkpoint format (.pt files)

## Success Criteria

1. User can select CleanRL worker in Human vs Agent tab
2. User can configure and start training for Chess/Go/Connect Four
3. Training runs in background with telemetry
4. User can load trained policy
5. Human can play against trained AI agent
