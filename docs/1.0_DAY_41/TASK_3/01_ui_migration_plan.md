# UI Migration Plan: Thin Wrappers + Advanced Tab

## Overview

This document outlines the step-by-step plan to:
1. Refactor existing tabs to use `PolicyMappingService` internally (thin wrappers)
2. Add new "Advanced" tab for full per-agent configuration flexibility

**No breaking changes for users** - existing tabs continue to work, just refactored internally.

---

## Phase 1: Advanced Tab (New Widget)

### 1.1 Create Core Widgets

```
gym_gui/ui/widgets/advanced_config/
├── __init__.py
├── environment_selector.py    # Step 1: Environment + paradigm detection
├── agent_config_table.py      # Step 2: Per-agent policy bindings
├── worker_config_panel.py     # Step 3: Dynamic worker configuration
├── run_mode_selector.py       # Step 4: Interactive/Headless/Eval
└── advanced_config_tab.py     # Main container widget
```

### 1.2 EnvironmentSelector Widget

```python
class EnvironmentSelector(QWidget):
    """Step 1: Environment selection with paradigm auto-detection."""

    # Signals
    environment_changed = Signal(str)  # env_id
    paradigm_detected = Signal(SteppingParadigm)
    agents_detected = Signal(list)  # List of agent IDs

    # UI Elements:
    # - Family dropdown (Gymnasium, PettingZoo, ViZDoom, etc.)
    # - Environment dropdown (filtered by family)
    # - Seed input
    # - Info panel showing: paradigm, agent count, action spaces
```

### 1.3 AgentConfigTable Widget

```python
class AgentConfigTable(QWidget):
    """Step 2: Per-agent policy/worker binding configuration."""

    # Signals
    bindings_changed = Signal(dict)  # Dict[agent_id, AgentPolicyBinding]

    # UI Elements:
    # ┌──────────────────────────────────────────────────────────────┐
    # │  Agent        Actor           Worker           Mode          │
    # ├──────────────────────────────────────────────────────────────┤
    # │  player_0     [Human      ▼]  [—           ▼]  [Play     ▼] │
    # │  player_1     [CleanRL    ▼]  [cleanrl    ▼]  [Train    ▼] │
    # └──────────────────────────────────────────────────────────────┘
    #
    # For many agents: "Apply to All" checkbox + bulk configuration

    def set_agents(self, agent_ids: List[str]) -> None:
        """Populate table with agent rows."""

    def get_bindings(self) -> Dict[str, AgentPolicyBinding]:
        """Get all configured bindings."""
```

### 1.4 WorkerConfigPanel Widget

```python
class WorkerConfigPanel(QWidget):
    """Step 3: Dynamic worker-specific configuration."""

    # Dynamically generates form based on selected workers
    # Each worker provides a config schema

    # UI Elements:
    # ┌──────────────────────────────────────────────────────────────┐
    # │  CleanRL Worker (player_1)                                   │
    # │  ├── Algorithm:       [PPO ▼]                                │
    # │  ├── Learning Rate:   [0.0003  ]                             │
    # │  ├── Total Timesteps: [100000  ]                             │
    # │  └── [Advanced Settings...]                                  │
    # └──────────────────────────────────────────────────────────────┘
```

### 1.5 RunModeSelector Widget

```python
class RunModeSelector(QWidget):
    """Step 4: Run mode selection."""

    # Signals
    mode_changed = Signal(str)  # "interactive", "headless", "evaluation"

    # UI Elements:
    # ○ Interactive (with rendering)
    # ○ Headless Training (no rendering)
    # ○ Evaluation (load trained policy)
```

### 1.6 AdvancedConfigTab (Container)

```python
class AdvancedConfigTab(QWidget):
    """Main Advanced tab combining all steps."""

    # Signals
    launch_requested = Signal(LaunchConfig)

    def __init__(self):
        # Layout:
        # ┌─────────────────────────────────────┐
        # │ Step 1: Environment Selection       │
        # │ EnvironmentSelector widget          │
        # ├─────────────────────────────────────┤
        # │ Step 2: Agent Configuration         │
        # │ AgentConfigTable widget             │
        # ├─────────────────────────────────────┤
        # │ Step 3: Worker Configuration        │
        # │ WorkerConfigPanel widget            │
        # ├─────────────────────────────────────┤
        # │ Step 4: Run Mode                    │
        # │ RunModeSelector widget              │
        # ├─────────────────────────────────────┤
        # │ [ Launch ]                          │
        # └─────────────────────────────────────┘

    def _on_launch(self):
        """Generate PolicyMappingService config and emit."""
        policy_mapping = get_service_locator().get(PolicyMappingService)

        # Configure from UI
        policy_mapping.set_paradigm(self._env_selector.paradigm)
        policy_mapping.set_agents(self._env_selector.agents)

        for agent_id, binding in self._agent_table.get_bindings().items():
            policy_mapping.bind_agent_policy(
                agent_id,
                binding.policy_id,
                worker_id=binding.worker_id,
                config=binding.config,
            )

        # Emit launch signal
        self.launch_requested.emit(LaunchConfig(...))
```

---

## Phase 2: Integrate Advanced Tab into ControlPanelWidget

### 2.1 Add Tab to control_panel.py

```python
# In ControlPanelWidget._build_ui()

# ... existing tabs ...

# Add Advanced tab (NEW)
from gym_gui.ui.widgets.advanced_config import AdvancedConfigTab

self._advanced_tab = AdvancedConfigTab(self)
self._tab_widget.addTab(self._advanced_tab, "Advanced")

# Connect signals
self._advanced_tab.launch_requested.connect(self._on_advanced_launch)
```

### 2.2 Handle Advanced Launch

```python
def _on_advanced_launch(self, config: LaunchConfig):
    """Handle launch from Advanced tab."""
    # PolicyMappingService is already configured by the tab
    # Just need to trigger the appropriate action based on run mode

    if config.run_mode == "interactive":
        self.start_game_requested.emit()
    elif config.run_mode == "headless":
        self.train_agent_requested.emit(config.primary_worker_id)
    elif config.run_mode == "evaluation":
        self.trained_agent_requested.emit(config.primary_worker_id)
```

---

## Phase 3: Refactor Existing Tabs as Thin Wrappers

### 3.1 HumanVsAgentTab Refactor

**Before (hardcoded):**
```python
def _on_start_game(self):
    human_agent = self._human_player_combo.currentData()
    seed = self._seed_spin.value()
    self.start_game_requested.emit(self._selected_env.value, human_agent, seed)
```

**After (thin wrapper):**
```python
def _on_start_game(self):
    # Get UI selections
    human_agent = self._human_player_combo.currentData()
    ai_agent = "player_1" if human_agent == "player_0" else "player_0"
    ai_policy = self._get_ai_policy_id()  # "stockfish", "cleanrl", "random"

    # Configure PolicyMappingService
    policy_mapping = get_service_locator().get(PolicyMappingService)
    policy_mapping.set_paradigm(SteppingParadigm.SEQUENTIAL)
    policy_mapping.set_agents([human_agent, ai_agent])
    policy_mapping.bind_agent_policy(human_agent, "human_keyboard")
    policy_mapping.bind_agent_policy(ai_agent, ai_policy)

    # Emit signal (same as before - no breaking change)
    seed = self._seed_spin.value()
    self.start_game_requested.emit(self._selected_env.value, human_agent, seed)
```

### 3.2 SingleAgentTab Refactor

**Before:**
```python
def _emit_train_agent_requested(self):
    worker_id = self._current_worker_id
    self.train_agent_requested.emit(worker_id)
```

**After:**
```python
def _emit_train_agent_requested(self):
    worker_id = self._current_worker_id

    # Configure PolicyMappingService for single-agent
    policy_mapping = get_service_locator().get(PolicyMappingService)
    policy_mapping.set_paradigm(SteppingParadigm.SINGLE_AGENT)
    policy_mapping.set_agents(["agent_0"])
    policy_mapping.bind_agent_policy("agent_0", worker_id, worker_id=worker_id)

    # Emit signal (same as before)
    self.train_agent_requested.emit(worker_id)
```

### 3.3 CooperationTab / CompetitionTab Refactor

Similar pattern - configure PolicyMappingService before emitting signals.

---

## Phase 4: Update SessionController

### 4.1 Use PolicyMappingService for Action Selection

```python
# In SessionController

def _select_agent_action(self, agent_id: str) -> Optional[int]:
    """Select action using PolicyMappingService."""
    policy_mapping = get_service_locator().get(PolicyMappingService)
    return policy_mapping.select_action(agent_id, self._current_snapshot)
```

### 4.2 PettingZoo AEC Loop Integration

```python
def _run_pettingzoo_step(self):
    """Run one step in PettingZoo AEC environment."""
    for agent_id in self._adapter.agent_iter():
        obs, reward, terminated, truncated, info = self._adapter.last()

        if terminated or truncated:
            action = None
        else:
            action = self._select_agent_action(agent_id)

        self._adapter.step(action)
        self._notify_step(agent_id)
```

---

## Implementation Order

| Phase | Task | Depends On | Priority |
|-------|------|------------|----------|
| 1.1 | Create `EnvironmentSelector` widget | — | High |
| 1.2 | Create `AgentConfigTable` widget | PolicyMappingService (done) | High |
| 1.3 | Create `WorkerConfigPanel` widget | Worker config schemas | Medium |
| 1.4 | Create `RunModeSelector` widget | — | High |
| 1.5 | Create `AdvancedConfigTab` container | 1.1-1.4 | High |
| 2.1 | Add Advanced tab to ControlPanelWidget | 1.5 | High |
| 3.1 | Refactor HumanVsAgentTab | PolicyMappingService | Medium |
| 3.2 | Refactor SingleAgentTab | PolicyMappingService | Medium |
| 3.3 | Refactor Cooperation/Competition tabs | PolicyMappingService | Low |
| 4.1 | Update SessionController | 3.x | Medium |

---

## Success Criteria

1. **Advanced tab works**: Can configure any agent→policy→worker combination
2. **Existing tabs unchanged for users**: Same UI, same behavior
3. **Existing tabs use PolicyMappingService internally**: No duplicated logic
4. **New scenarios expressible**: Self-play, mixed workers, 3+ agents, LLM+CleanRL

---

## Files to Create/Modify

### New Files
```
gym_gui/ui/widgets/advanced_config/__init__.py
gym_gui/ui/widgets/advanced_config/environment_selector.py
gym_gui/ui/widgets/advanced_config/agent_config_table.py
gym_gui/ui/widgets/advanced_config/worker_config_panel.py
gym_gui/ui/widgets/advanced_config/run_mode_selector.py
gym_gui/ui/widgets/advanced_config/advanced_config_tab.py
```

### Modified Files
```
gym_gui/ui/widgets/control_panel.py          # Add Advanced tab
gym_gui/ui/widgets/multi_agent_tab.py        # Refactor to use PolicyMappingService
gym_gui/controllers/session.py               # Use PolicyMappingService for action selection
```

---

## Related Documents

- [TASK_1: PolicyMappingService](../TASK_1/03_policy_mapping_service_plan.md) - Backend service (Phase 2.1 ✅)
- [UI Architecture Analysis](./00_ui_architecture_analysis.md) - Problem analysis
