# Why the Render View Is Empty When Loading PettingZoo Environments

**Date:** 2025-11-28

---

## Quick Answer

When you click "Load Environment" in the Multi-Agent Mode (Human vs Agent) tab, the button's signal reaches `main_window.py`, but **the handler just shows a status message** - it doesn't actually create the environment or render anything.

---

## The Signal Flow (What Currently Happens)

```
┌─────────────────────┐
│  Human vs Agent Tab │
│  [Load Environment] │──────┐
└─────────────────────┘      │
                             │ Signal: load_environment_requested("chess_v6", 1)
                             ▼
┌─────────────────────┐
│  ControlPanelWidget │
│  _on_multi_agent_   │──────┐
│  load_requested()   │      │
└─────────────────────┘      │ Signal: multi_agent_load_requested("chess_v6", 1)
                             ▼
┌─────────────────────┐
│     MainWindow      │
│  _on_multi_agent_   │
│  load_requested()   │
└─────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  def _on_multi_agent_load_requested(self, env_id, seed):   │
│      # THIS IS ALL IT DOES:                                 │
│      self._status_bar.showMessage(                          │
│          f"Multi-agent environment loading not yet          │
│           implemented: {env_id} (seed={seed})",             │
│          5000                                               │
│      )                                                      │
│      # NO environment creation                              │
│      # NO render frame generation                           │
│      # NO passing frame to Render View                      │
└─────────────────────────────────────────────────────────────┘
```

---

## What SHOULD Happen (But Doesn't Yet)

```
┌─────────────────────┐
│  Human vs Agent Tab │
│  [Load Environment] │
└─────────────────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│  _on_multi_agent_   │────►│  PettingZooAdapter  │
│  load_requested()   │     │  (NOT YET CREATED)  │
└─────────────────────┘     └─────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │  pettingzoo.classic │
                            │  .chess_v6.env()    │
                            │  env.reset(seed=1)  │
                            │  frame = env.render()│
                            └─────────────────────┘
                                      │
                                      ▼ frame (numpy RGB array)
                            ┌─────────────────────┐
                            │     Render View     │
                            │    (QLabel/Image)   │
                            │   Shows chess board │
                            └─────────────────────┘
```

---

## Why Can't We Just Use the Existing Adapter?

The existing `GymnasiumAdapter` in `gym_gui/core/adapters/base.py` is designed for **single-agent** environments. PettingZoo environments are fundamentally different:

### Single-Agent (Gymnasium) - What We Have

```python
# One agent, one action, one observation, one reward
env = gym.make("FrozenLake-v1")
obs, info = env.reset(seed=42)
obs, reward, done, truncated, info = env.step(action)
frame = env.render()  # One frame
```

### Multi-Agent AEC (PettingZoo) - What Chess Needs

```python
# Multiple agents, turn-based
from pettingzoo.classic import chess_v6

env = chess_v6.env(render_mode="rgb_array")
env.reset(seed=42)

# Must iterate through agents in sequence
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None  # Agent is done
    else:
        # Which agent is it?
        if agent == "player_0":  # White
            action = get_human_move()  # Wait for human input
        else:  # Black
            action = ai_policy.get_action(observation)  # AI moves

    env.step(action)  # Only advances THIS agent
    frame = env.render()
```

### Multi-Agent Parallel (PettingZoo) - What Cooperative Games Need

```python
# Multiple agents, simultaneous actions
from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.parallel_env(render_mode="rgb_array")
observations, infos = env.reset(seed=42)
# observations = {"agent_0": obs0, "agent_1": obs1, "agent_2": obs2}

# All agents act at once
actions = {
    "agent_0": policy.get_action(observations["agent_0"]),
    "agent_1": policy.get_action(observations["agent_1"]),
    "agent_2": policy.get_action(observations["agent_2"]),
}

# Single step advances ALL agents
observations, rewards, terminations, truncations, infos = env.step(actions)
# rewards = {"agent_0": 0.5, "agent_1": -0.2, "agent_2": 0.3}
```

---

## The Key Differences

| Aspect | Gymnasium (Single-Agent) | PettingZoo AEC | PettingZoo Parallel |
|--------|--------------------------|----------------|---------------------|
| `reset()` returns | `(obs, info)` | Nothing (use `env.last()`) | `(obs_dict, info_dict)` |
| `step()` takes | `action` | `action` (current agent only) | `{agent: action}` dict |
| `step()` returns | `(obs, reward, done, trunc, info)` | Nothing (use `env.last()`) | All as dicts |
| Current agent | Always the same (1 agent) | `env.agent_selection` | N/A (all at once) |
| Observation | Single value | Per agent | Dict of all |
| Reward | Single float | Per agent | Dict of floats |
| When to render | After every step | After each agent's turn | After every step |

---

## What Needs to Be Built

### 1. PettingZoo Adapter (`gym_gui/core/adapters/pettingzoo.py`)

```python
class PettingZooAdapter:
    """Wraps PettingZoo environments to work with gym_gui."""

    def load(self, env_id: str, render_mode: str = "rgb_array"):
        """Create the environment based on env_id."""
        # e.g., "chess_v6" → pettingzoo.classic.chess_v6.env()

    def reset(self, seed: int) -> dict:
        """Reset and return initial observations for all agents."""

    def step(self, action) -> dict:
        """Execute action (AEC: single agent, Parallel: all agents)."""

    def render(self) -> np.ndarray:
        """Return current RGB frame."""
```

### 2. MainWindow Handler Update

```python
def _on_multi_agent_load_requested(self, env_id: str, seed: int) -> None:
    # 1. Create adapter
    self._pettingzoo_adapter = PettingZooAdapter()
    self._pettingzoo_adapter.load(env_id, render_mode="rgb_array")

    # 2. Reset environment
    initial_state = self._pettingzoo_adapter.reset(seed)

    # 3. Get render frame
    frame = self._pettingzoo_adapter.render()

    # 4. Display in Render View
    self._render_tabs.display_frame(frame)

    # 5. Set up game loop for human/AI turns
    if self._pettingzoo_adapter.api_type == "aec":
        self._setup_turn_based_game()
```

### 3. Session Controller Integration

The `SessionController` needs new methods to handle multi-agent game loops, especially for AEC environments where we need to:
- Track whose turn it is
- Wait for human input when it's the human's turn
- Auto-step AI agents
- Update the render after each turn

---

## Summary

**The Render View is empty because:**

1. The signal chain works correctly (Load Environment button → MainWindow)
2. But the handler in MainWindow is just a placeholder that shows a status message
3. No PettingZoo adapter exists to actually create and manage the environment
4. No code passes render frames to the Render View widget

**To fix this, we need to:**

1. Create `gym_gui/core/adapters/pettingzoo.py` with AECAdapter and ParallelAdapter classes
2. Update `_on_multi_agent_load_requested()` to use the adapter
3. Integrate with the session controller for game loop management
4. Handle human input for turn-based games (Chess, Tic-Tac-Toe, etc.)

This is Phase 3 in the integration plan.
