# PettingZoo Integration Plan

**Date:** 2025-11-27 (Updated: 2025-11-28)
**Status:** Phase 1-2 Complete, Phase 3 In Progress
**Goal:** Integrate PettingZoo as the Multi-Agent environment backend for gym_gui

---

## Current Status Summary

### Completed (Day 35)
- [x] Created `gym_gui/core/pettingzoo_enums.py` with environment IDs, families, API types
- [x] Created `gym_gui/ui/widgets/multi_agent_tab.py` with three subtabs:
  - Human vs Agent tab
  - Cooperation tab
  - Competition tab
- [x] Updated `control_panel.py` to use the new MultiAgentTab
- [x] Created PettingZoo game documentation in `gym_gui/game_docs/PettingZoo/`
- [x] Extended ALE documentation with multi-player Atari games

### Completed (Day 35 - Session 2)
- [x] Renamed "Competitive" tab to "Competition"
- [x] Added environment loading UI to Human vs Agent tab (seed, checkbox, Load Environment button)
- [x] Created `gym_gui/ui/environments/multi_agent_env/` directory structure
- [x] Created `gym_gui/ui/environments/multi_agent_env/pettingzoo/config_panel.py`
- [x] Added PETTINGZOO to EnvironmentFamily enum
- [x] Added Multi-Agent log codes (`LOG750`-`LOG759`)
- [x] Connected signal chain: `MultiAgentTab` → `ControlPanelWidget` → `MainWindow`

### In Progress (Revised Approach)
- [ ] **Create ChessAdapter first** (`gym_gui/core/adapters/chess_adapter.py`) - YAGNI approach
- [ ] **Create InteractiveChessBoard widget** (`gym_gui/ui/widgets/chess_board.py`) - State-based rendering

### Pending
- [ ] Create ChessGameController for async input handling
- [ ] Wire up chess in MainWindow
- [ ] Generalize to PettingZooAdapter base class
- [ ] Multi-agent training integration

### Key Design Decisions (See `design_decisions.md`)
1. **AEC→Parallel Conversion**: Always convert AEC environments to Parallel API using `aec_to_parallel` wrapper
2. **State-Based Rendering**: Send game state (FEN) to Qt, not just pixels - enables mouse interaction
3. **YAGNI**: Build ChessAdapter first, then generalize

---

## 1. Overview

### Why PettingZoo?

| Feature | PettingZoo |
|---------|------------|
| **Installation** | `pip install pettingzoo` |
| **Maintained by** | Farama Foundation (Gymnasium team) |
| **API Compatibility** | Gymnasium-like interface |
| **Python Native** | Yes |
| **Multi-Agent Types** | AEC (turn-based) + Parallel |
| **Environment Count** | 50+ built-in |

### PettingZoo Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PettingZoo APIs                            │
├─────────────────────────────────────────────────────────────────┤
│  AEC API (Agent Environment Cycle)                              │
│  ─────────────────────────────────                              │
│  • Sequential turn-based environments                           │
│  • env.agent_iter() loops through agents                        │
│  • Chess, Go, Card games, etc.                                  │
│                                                                 │
│  Parallel API                                                   │
│  ────────────                                                   │
│  • Simultaneous actions from all agents                         │
│  • env.step(actions_dict) for all agents at once                │
│  • MPE, SISL, cooperative games                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Problem: Why Render View Is Empty

### Signal Flow Analysis

When you click "Load Environment" in the Multi-Agent Mode (Human vs Agent) tab:

```
┌─────────────────────┐     ┌────────────────────┐     ┌─────────────┐
│  HumanVsAgentTab    │────►│ ControlPanelWidget │────►│ MainWindow  │
│  (Load Env Button)  │     │ (Signal Forward)   │     │ (Handler)   │
└─────────────────────┘     └────────────────────┘     └─────────────┘
         │                           │                        │
         ▼                           ▼                        ▼
   Signal: load_environment_requested(env_id, seed)           │
                                                              │
                                           ┌──────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │  _on_multi_agent_load_  │
                              │       requested()       │
                              │                         │
                              │  ❌ NO IMPLEMENTATION   │
                              │                         │
                              │  Just shows status msg: │
                              │  "Not yet implemented"  │
                              └─────────────────────────┘
```

### What's Missing

The `_on_multi_agent_load_requested` handler in `main_window.py` (line 603) currently does:

```python
def _on_multi_agent_load_requested(self, env_id: str, seed: int) -> None:
    # Just logs and shows status message
    self._status_bar.showMessage(
        f"Multi-agent environment loading not yet implemented: {env_id} (seed={seed})",
        5000
    )
```

**What it SHOULD do:**
1. Create a PettingZoo environment instance
2. Reset the environment with the provided seed
3. Get the initial observation and render frame
4. Pass the render frame to the Render View widget
5. Set up the game loop for turn management (AEC) or simultaneous play (Parallel)

### Why This Is Non-Trivial

The existing single-agent flow uses this architecture:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ControlPanel   │────►│  MainWindow     │────►│ SessionController│
│  load_requested │     │ _on_load_       │     │ load_environment │
│                 │     │  requested()    │     │                  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  AdapterFactory │
                                               │  create_adapter │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ GymnasiumAdapter│
                                               │   (single-agent)│
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   gym.make()    │
                                               │  env.reset()    │
                                               │  env.render()   │
                                               └─────────────────┘
```

For PettingZoo, we need a **completely different adapter** because:

1. **Different API**:
   - Gymnasium: `env.step(action)` → returns `(obs, reward, done, truncated, info)`
   - PettingZoo AEC: `env.step(action)` for **current agent only**, must iterate through `env.agent_iter()`
   - PettingZoo Parallel: `env.step(actions_dict)` → returns dicts of `{agent: value}`

2. **Multiple Agents**:
   - Gymnasium: Single observation, single action, single reward
   - PettingZoo: Multiple observations, multiple actions, multiple rewards (per agent)

3. **Turn Management (AEC)**:
   - Chess: Player 1 moves, then Player 2 moves (alternating)
   - Need to track whose turn it is
   - Human needs to wait for AI, or vice versa

4. **Observation Structure**:
   ```python
   # Gymnasium (single-agent)
   obs = env.reset()  # Just one observation

   # PettingZoo AEC
   env.reset()
   obs = env.observe(agent)  # Must specify which agent

   # PettingZoo Parallel
   observations, infos = env.reset()  # Dict of all observations
   # observations = {"player_0": obs0, "player_1": obs1, ...}
   ```

---

## 3. Architecture Solution

### 3.1 New PettingZoo Adapter

File: `gym_gui/core/adapters/pettingzoo.py`

```python
"""PettingZoo multi-agent environment adapter.

This adapter wraps PettingZoo environments to work with the gym_gui
session controller and rendering pipeline.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    StepState,
)
from gym_gui.core.enums import ControlMode, RenderMode
from gym_gui.core.pettingzoo_enums import PettingZooAPIType, PettingZooEnvId

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class MultiAgentStep:
    """Step result for multi-agent environments."""

    observations: Dict[str, Any]  # {agent_id: observation}
    rewards: Dict[str, float]     # {agent_id: reward}
    terminations: Dict[str, bool] # {agent_id: terminated}
    truncations: Dict[str, bool]  # {agent_id: truncated}
    infos: Dict[str, Any]         # {agent_id: info}
    render_payload: Any | None = None
    current_agent: str | None = None  # For AEC environments
    active_agents: List[str] = field(default_factory=list)


class PettingZooAdapter:
    """Base adapter for PettingZoo multi-agent environments."""

    supported_control_modes = (
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )
    default_render_mode = RenderMode.RGB_ARRAY

    def __init__(
        self,
        env_id: PettingZooEnvId,
        *,
        context: AdapterContext | None = None,
    ) -> None:
        self._env_id = env_id
        self._context = context
        self._env: Any = None  # AECEnv or ParallelEnv
        self._api_type: PettingZooAPIType | None = None
        self._agents: List[str] = []
        self._current_agent: str | None = None

    @property
    def api_type(self) -> PettingZooAPIType | None:
        return self._api_type

    @property
    def agents(self) -> List[str]:
        return self._agents

    @property
    def current_agent(self) -> str | None:
        return self._current_agent

    def load(self, render_mode: str = "rgb_array") -> None:
        """Create the PettingZoo environment."""
        # Import and create based on env_id
        # This is where the actual environment is instantiated
        pass

    def reset(self, *, seed: int | None = None) -> MultiAgentStep:
        """Reset the environment and return initial state."""
        pass

    def step(self, action: Any) -> MultiAgentStep:
        """Execute one step (AEC: single agent, Parallel: all agents)."""
        pass

    def render(self) -> np.ndarray | None:
        """Get current render frame."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass


class AECAdapter(PettingZooAdapter):
    """Adapter for AEC (turn-based) PettingZoo environments.

    AEC environments work with sequential turns:
    1. Get current agent from env.agent_selection
    2. Get observation for that agent
    3. Execute action for that agent
    4. Move to next agent
    """

    def __init__(self, env_id: PettingZooEnvId, **kwargs) -> None:
        super().__init__(env_id, **kwargs)
        self._api_type = PettingZooAPIType.AEC

    def step(self, action: Any) -> MultiAgentStep:
        """Step the current agent only."""
        # In AEC, env.step(action) advances to the next agent
        self._env.step(action)

        # Get next agent
        self._current_agent = self._env.agent_selection

        # Get observation for the new current agent
        obs, reward, term, trunc, info = self._env.last()

        return MultiAgentStep(
            observations={self._current_agent: obs},
            rewards={self._current_agent: reward},
            terminations={self._current_agent: term},
            truncations={self._current_agent: trunc},
            infos={self._current_agent: info},
            render_payload=self.render(),
            current_agent=self._current_agent,
            active_agents=list(self._env.agents),
        )


class ParallelAdapter(PettingZooAdapter):
    """Adapter for Parallel PettingZoo environments.

    Parallel environments take all actions simultaneously:
    1. Collect actions from all agents
    2. Execute env.step(actions_dict)
    3. All agents receive observations/rewards at once
    """

    def __init__(self, env_id: PettingZooEnvId, **kwargs) -> None:
        super().__init__(env_id, **kwargs)
        self._api_type = PettingZooAPIType.PARALLEL

    def step(self, actions: Dict[str, Any]) -> MultiAgentStep:
        """Step all agents simultaneously."""
        observations, rewards, terminations, truncations, infos = self._env.step(actions)

        return MultiAgentStep(
            observations=observations,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
            render_payload=self.render(),
            current_agent=None,  # No "current" agent in parallel
            active_agents=list(self._env.agents),
        )
```

### 3.2 Session Controller Changes

The `SessionController` needs to handle multi-agent environments differently:

```python
# Current (single-agent):
def step(self, action):
    result = self._adapter.step(action)
    # Process single observation, reward, etc.

# Needed (multi-agent AEC):
def step_aec(self, action):
    result = self._adapter.step(action)  # Steps current agent only
    if result.current_agent == "human":
        # Wait for human input
        self._awaiting_human.emit()
    else:
        # Get AI action and continue
        ai_action = self._policy.get_action(result.observations[result.current_agent])
        self.step_aec(ai_action)  # Recursive until human's turn
```

### 3.3 Render Integration

The render pipeline can mostly stay the same since PettingZoo also returns `rgb_array`:

```python
# Both work the same way
frame = env.render()  # Returns numpy array (H, W, 3)
```

The difference is in **when** to render:
- Single-agent: Render after every step
- AEC: Render after every agent's turn (or just after human's turn)
- Parallel: Render after every step (all agents moved)

---

## 4. Implementation Phases

### Phase 1: Core Infrastructure ✅ COMPLETE
- [x] Create `gym_gui/core/pettingzoo_enums.py` with environment enums
- [x] Add `PETTINGZOO` to `EnvironmentFamily` enum
- [x] Add `pettingzoo` to `requirements.txt`

### Phase 2: GUI Tab Widget ✅ COMPLETE
- [x] Create `gym_gui/ui/widgets/multi_agent_tab.py` with subtabs
- [x] Replace placeholder in `control_panel.py`
- [x] Connect signals through to `MainWindow`
- [x] Add environment loading UI (seed, checkbox, Load Environment button)

### Phase 3: PettingZoo Adapter 🚧 IN PROGRESS
- [ ] Create `gym_gui/core/adapters/pettingzoo.py`
- [ ] Implement `AECAdapter` for turn-based games
- [ ] Implement `ParallelAdapter` for simultaneous games
- [ ] Test basic environment creation and stepping

### Phase 4: Session Integration ⏳ PENDING
- [ ] Add `load_pettingzoo_environment()` method to session controller
- [ ] Handle multi-agent observation/reward/done dicts
- [ ] Implement turn management for AEC environments

### Phase 5: Rendering Integration ⏳ PENDING
- [ ] Pass render frames from PettingZoo to Render View
- [ ] Handle multi-agent frame display
- [ ] Support both AEC and Parallel rendering

### Phase 6: Human Control ⏳ PENDING
- [ ] Implement human control for AEC environments (turn-based)
- [ ] Map keyboard actions to PettingZoo action spaces
- [ ] Add agent-specific action input UI

### Phase 7: Training Integration ⏳ PENDING
- [ ] Create `pettingzoo_worker` for MARL training
- [ ] Integrate with existing CleanRL patterns
- [ ] Add multi-agent telemetry

---

## 5. Files Status

### Created Files ✅

| File | Status | Purpose |
|------|--------|---------|
| `gym_gui/core/pettingzoo_enums.py` | ✅ Complete | Environment and API type enums |
| `gym_gui/ui/widgets/multi_agent_tab.py` | ✅ Complete | Multi-Agent tab with subtabs |
| `gym_gui/ui/environments/multi_agent_env/__init__.py` | ✅ Complete | Package init |
| `gym_gui/ui/environments/multi_agent_env/pettingzoo/__init__.py` | ✅ Complete | PettingZoo helpers |
| `gym_gui/ui/environments/multi_agent_env/pettingzoo/config_panel.py` | ✅ Complete | Family-specific config panels |
| `gym_gui/game_docs/PettingZoo/__init__.py` | ✅ Complete | Game documentation |

### Modified Files ✅

| File | Status | Change |
|------|--------|--------|
| `gym_gui/ui/widgets/control_panel.py` | ✅ Complete | Added MultiAgentTab, signals |
| `gym_gui/core/enums.py` | ✅ Complete | Added PETTINGZOO to EnvironmentFamily |
| `gym_gui/ui/main_window.py` | ✅ Complete | Added multi_agent_load handler (placeholder) |
| `gym_gui/logging_config/log_constants.py` | ✅ Complete | Added LOG750-LOG759 for multi-agent |
| `gym_gui/game_docs/ALE/__init__.py` | ✅ Complete | Extended with multi-player Atari |

### Pending Files ⏳

| File | Status | Purpose |
|------|--------|---------|
| `gym_gui/core/adapters/pettingzoo.py` | ⏳ Not started | Multi-agent environment adapter |

---

## 6. Log Codes Added

| Code | Level | Description |
|------|-------|-------------|
| `LOG750` | INFO | Multi-agent environment load requested |
| `LOG751` | INFO | Multi-agent environment loaded successfully |
| `LOG752` | ERROR | Multi-agent environment load failed |
| `LOG753` | INFO | Multi-agent policy load requested |
| `LOG754` | INFO | Multi-agent game start requested |
| `LOG755` | INFO | Multi-agent reset requested |
| `LOG756` | DEBUG | Multi-agent action submitted |
| `LOG757` | INFO | Multi-agent training requested |
| `LOG758` | INFO | Multi-agent evaluation requested |
| `LOG759` | WARNING | Action attempted without environment |

---

## 7. Next Steps (Priority Order)

1. **Create PettingZoo Adapter** (`gym_gui/core/adapters/pettingzoo.py`)
   - Start with a simple Classic game (Chess or Tic-Tac-Toe)
   - Implement AECAdapter first (turn-based is easier to debug)
   - Test environment creation and basic stepping

2. **Implement MainWindow Handler**
   - Replace placeholder in `_on_multi_agent_load_requested()`
   - Create adapter instance
   - Reset environment and get initial frame
   - Pass frame to Render View

3. **Add Render View Display**
   - Ensure RGB frames from PettingZoo display correctly
   - Handle the initial observation display

4. **Human Input for AEC**
   - For Chess: map clicks to move selection
   - For simpler games (Tic-Tac-Toe): map number keys to positions

---

## References

- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [PettingZoo GitHub](https://github.com/Farama-Foundation/PettingZoo)
- [PettingZoo Paper (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/7ed2d3454c5eea71148b11d0c25104ff-Paper.pdf)
- [CleanRL + PettingZoo Tutorial](https://pettingzoo.farama.org/tutorials/cleanrl/)
