# Implementation Roadmap: Multi-Agent Training for Human vs Agent Mode

## Overview

This roadmap outlines the step-by-step implementation to add training support for Multi-Agent Mode (Human vs Agent) board games.

## Current State

### What Works
- Single-Agent Mode has full training pipeline (Configure → Train → Load Policy)
- Board widgets (Chess, Go, Connect Four) render and accept user input
- Human Control Mode allows playing against environment (not trained agent)
- CleanRL worker is integrated and functional

### What's Missing
- HumanVsAgentTab has no worker selection
- HumanVsAgentTab has no training configuration
- No self-play algorithm for PettingZoo Classic games
- "Load Trained Policy" button exists but no policies to load

## Implementation Phases

### Phase 1: UI Enhancement (Day 1-2)

**Goal:** Add training UI components to HumanVsAgentTab

#### Task 1.1: Add Worker Integration Group
File: `gym_gui/ui/widgets/multi_agent_tab.py`

```python
# In HumanVsAgentTab._build_ui(), add after env_group:

# Worker Integration group
worker_group = QtWidgets.QGroupBox("Training", self)
worker_layout = QtWidgets.QFormLayout(worker_group)

self._worker_combo = QtWidgets.QComboBox(worker_group)
worker_layout.addRow("Worker:", self._worker_combo)

self._worker_info = QtWidgets.QLabel("", worker_group)
self._worker_info.setWordWrap(True)
self._worker_info.setStyleSheet("color: #666; font-size: 11px;")
worker_layout.addRow("", self._worker_info)

# Training buttons
train_btn_layout = QtWidgets.QHBoxLayout()
self._configure_btn = QtWidgets.QPushButton("Configure Training", worker_group)
self._train_btn = QtWidgets.QPushButton("Train Agent", worker_group)
self._train_btn.setEnabled(False)
train_btn_layout.addWidget(self._configure_btn)
train_btn_layout.addWidget(self._train_btn)
worker_layout.addRow("", train_btn_layout)

layout.addWidget(worker_group)
```

#### Task 1.2: Add New Signals
```python
# New signals in HumanVsAgentTab
worker_changed = pyqtSignal(str)
configure_training_requested = pyqtSignal(str, str)  # worker_id, env_id
train_agent_requested = pyqtSignal(str, str)  # worker_id, env_id
```

#### Task 1.3: Populate Workers
```python
def _populate_workers(self) -> None:
    """Populate worker dropdown from catalog."""
    self._worker_combo.clear()
    catalog = get_worker_catalog()
    for worker in catalog:
        if worker.supports_training:
            self._worker_combo.addItem(worker.display_name, worker.worker_id)
```

### Phase 2: Training Form (Day 3-4)

**Goal:** Create PettingZoo Classic training configuration dialog

#### Task 2.1: Create Training Form
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

    def __init__(
        self,
        env_id: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._env_id = env_id
        self._build_ui()

    def get_config(self) -> Dict[str, Any]:
        """Return training configuration."""
        return {
            "algorithm": "ppo_selfplay",
            "env_id": self._env_id,
            "total_timesteps": self._timesteps_spin.value(),
            "learning_rate": self._lr_spin.value(),
            "batch_size": self._batch_spin.value(),
            "opponent_update_freq": self._opponent_freq_spin.value(),
            # ... other parameters
        }
```

#### Task 2.2: Form UI Components
- Algorithm selection (dropdown)
- Training hyperparameters (collapsible group)
- Self-play settings (opponent pool size, update frequency)
- Analytics options (TensorBoard, WandB)
- Validation and preview

### Phase 3: Self-Play Algorithm (Day 5-7)

**Goal:** Implement self-play PPO for PettingZoo AEC environments

#### Task 3.1: Create Algorithm File
File: `3rd_party/cleanrl_worker/cleanrl/cleanrl/ppo_pettingzoo_classic_selfplay.py`

Key components:
1. AEC environment wrapper with action masking
2. Self-play opponent management
3. Policy network (shared for both players)
4. Training loop adapted for turn-based games

```python
"""Self-Play PPO for PettingZoo Classic AEC games."""

import torch
from pettingzoo.classic import chess_v6, connect_four_v3, go_v5

class SelfPlayPPO:
    """PPO with self-play for two-player board games."""

    def __init__(self, env_fn, **kwargs):
        self.env_fn = env_fn
        self.policy = Agent(...)  # Shared policy network
        self.opponent_policy = None  # Copy of policy for opponent

    def train(self, total_timesteps: int):
        """Run self-play training."""
        env = self.env_fn()

        for step in range(total_timesteps):
            # Self-play episode
            obs, _ = env.reset()

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    action = None
                else:
                    # Use current policy or opponent policy
                    policy = self.policy if self._is_learning_agent(agent) else self.opponent_policy
                    action = policy.get_action(observation)

                env.step(action)

            # Update policy periodically
            if step % self.opponent_update_freq == 0:
                self._update_opponent()
```

#### Task 3.2: Register Algorithm in CleanRL Worker
File: `3rd_party/cleanrl_worker/cleanrl_worker/cli.py`

```python
ALGORITHMS = {
    "ppo": "cleanrl.ppo",
    "dqn": "cleanrl.dqn",
    # ... existing algorithms
    "ppo_pettingzoo_classic_selfplay": "cleanrl.ppo_pettingzoo_classic_selfplay",
}
```

### Phase 4: Policy Loading (Day 8-9)

**Goal:** Enable loading trained policies for Human vs Agent gameplay

#### Task 4.1: Create/Update Policy Form
File: `gym_gui/ui/widgets/pettingzoo_classic_policy_form.py`

Features:
- Discover policies in `VAR_TRAINER_DIR/runs/` for specific game
- Filter by game type (Chess, Go, Connect Four)
- Show metadata (algorithm, training steps, date)
- Preview performance stats if available

```python
class PettingZooClassicPolicyForm(QtWidgets.QDialog):
    """Policy selection dialog for PettingZoo Classic games."""

    def __init__(
        self,
        env_id: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._env_id = env_id
        self._selected_policy_path: Optional[Path] = None
        self._build_ui()
        self._discover_policies()
```

#### Task 4.2: Policy Discovery
```python
def _discover_policies(self) -> List[PolicyInfo]:
    """Find trained policies for the selected game."""
    runs_dir = VAR_TRAINER_DIR / "runs"
    policies = []

    for run_dir in runs_dir.iterdir():
        # Check if this run is for our game
        config_file = run_dir / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            if config.get("env_id") == self._env_id:
                # Find checkpoint files
                for ckpt in run_dir.glob("*.pt"):
                    policies.append(PolicyInfo(
                        path=ckpt,
                        env_id=config["env_id"],
                        algorithm=config.get("algorithm"),
                        timesteps=config.get("total_timesteps"),
                        date=run_dir.stat().st_mtime,
                    ))
    return policies
```

### Phase 5: Integration (Day 10-11)

**Goal:** Connect all components in MainWindow

#### Task 5.1: Signal Connections
File: `gym_gui/ui/main_window.py`

```python
def _connect_multi_agent_signals(self) -> None:
    """Connect Multi-Agent Mode signals."""
    hva = self._multi_agent_tab.human_vs_agent

    # Training signals
    hva.configure_training_requested.connect(self._on_configure_pettingzoo_training)
    hva.train_agent_requested.connect(self._on_train_pettingzoo_agent)

    # Policy loading (existing)
    hva.load_policy_requested.connect(self._on_load_pettingzoo_policy)
```

#### Task 5.2: Handler Methods
```python
def _on_configure_pettingzoo_training(self, worker_id: str, env_id: str) -> None:
    """Open training configuration dialog."""
    form = PettingZooClassicTrainForm(env_id=env_id, parent=self)
    if form.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        self._pending_pettingzoo_config = form.get_config()
        # Enable Train button
        self._multi_agent_tab.human_vs_agent._train_btn.setEnabled(True)

def _on_train_pettingzoo_agent(self, worker_id: str, env_id: str) -> None:
    """Submit training job to daemon."""
    if self._pending_pettingzoo_config:
        config = self._build_training_config(self._pending_pettingzoo_config)
        self._trainer_client.submit_job(config)

def _on_load_pettingzoo_policy(self, env_id: str) -> None:
    """Open policy selection dialog."""
    form = PettingZooClassicPolicyForm(env_id=env_id, parent=self)
    if form.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        policy_path = form.get_selected_policy()
        self._load_policy_for_gameplay(policy_path, env_id)
```

### Phase 6: Testing & Polish (Day 12-14)

**Goal:** End-to-end testing and bug fixes

#### Task 6.1: Manual Testing
1. Select Chess in Human vs Agent tab
2. Select CleanRL worker
3. Click Configure Training → Verify form opens
4. Configure and start training → Verify job submits
5. Wait for training completion
6. Click Load Policy → Verify policy discovered
7. Load policy and start game → Verify AI plays

#### Task 6.2: Add Pyright Type Checking
Run pyright on all new files to ensure type safety.

#### Task 6.3: Documentation
- Update user guide with new workflow
- Add tooltips to UI components

## File Checklist

### Files to Create
- [ ] `gym_gui/ui/widgets/pettingzoo_classic_train_form.py`
- [ ] `gym_gui/ui/widgets/pettingzoo_classic_policy_form.py`
- [ ] `3rd_party/cleanrl_worker/cleanrl/cleanrl/ppo_pettingzoo_classic_selfplay.py`

### Files to Modify
- [ ] `gym_gui/ui/widgets/multi_agent_tab.py` - Add training UI
- [ ] `gym_gui/ui/main_window.py` - Connect signals
- [ ] `3rd_party/cleanrl_worker/cleanrl_worker/cli.py` - Register algorithm

### Files to KEEP (In Use)
- `gym_gui/ui/widgets/chess_board.py` - Used by PettingZooTab, main_window, render_tabs
- `gym_gui/ui/widgets/go_board.py` - Used by PettingZooTab, render_tabs
- `gym_gui/ui/widgets/connect_four_board.py` - Used by PettingZooTab, render_tabs
- `gym_gui/ui/handlers/pettingzoo_classic_handler.py` - Game move handling
- `gym_gui/ui/handlers/chess_handlers.py` - Chess-specific logic
- `gym_gui/ui/handlers/go_handlers.py` - Go-specific logic
- `gym_gui/ui/handlers/connect_four_handlers.py` - Connect Four logic

## Success Metrics

1. **Training Pipeline Works**
   - User can configure training from UI
   - Training jobs run successfully
   - Checkpoints are saved

2. **Policy Loading Works**
   - Policies are discovered correctly
   - Policy metadata is displayed
   - Selected policy loads without errors

3. **Human vs Agent Works**
   - Human can play moves via board widget
   - AI responds with trained policy
   - Game state updates correctly
   - Win/loss detection works

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 2 days | UI updates to HumanVsAgentTab |
| Phase 2 | 2 days | Training configuration form |
| Phase 3 | 3 days | Self-play PPO algorithm |
| Phase 4 | 2 days | Policy loading form |
| Phase 5 | 2 days | MainWindow integration |
| Phase 6 | 3 days | Testing and polish |
| **Total** | **14 days** | Complete training pipeline |

## Dependencies

- PyTorch (for neural network)
- PettingZoo Classic (`pip install 'pettingzoo[classic]'`)
- python-chess (for Chess environment)
- Existing CleanRL worker infrastructure
