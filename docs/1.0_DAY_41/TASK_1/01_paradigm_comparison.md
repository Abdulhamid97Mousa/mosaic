# Paradigm Comparison: POSG vs AEC vs Others

## Related Documents

| Document | Description |
|----------|-------------|
| [00_multi_paradigm_orchestrator_plan.md](./00_multi_paradigm_orchestrator_plan.md) | Main architecture plan with implementation phases |
| [02_naming_resolution_plan.md](./02_naming_resolution_plan.md) | Directory naming conventions (✅ completed) |

---

## 1. Mathematical Models

### 1.1 POSG (Partially Observable Stochastic Game)

**Used by:** RLlib, PettingZoo Parallel API, Gymnasium (multi-agent), XuanCe

**Definition:**
A POSG is a tuple (N, S, {A_i}, {O_i}, T, {R_i}, {Obs_i}, γ) where:
- N = set of agents
- S = state space
- A_i = action space for agent i
- O_i = observation space for agent i
- T: S × A → Δ(S) = transition function (joint action to next state)
- R_i: S × A → ℝ = reward function for agent i
- Obs_i: S → O_i = observation function for agent i
- γ = discount factor

**Key Property:** All agents act **simultaneously** in each time step.

```
         ┌─────────────────────────────────────────────────────┐
         │                    Time Step t                       │
         │                                                      │
         │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
         │  │ Agent 1  │  │ Agent 2  │  │ Agent N  │           │
         │  │ obs → a₁ │  │ obs → a₂ │  │ obs → aₙ │           │
         │  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
         │       │             │             │                  │
         │       └─────────────┼─────────────┘                  │
         │                     │                                │
         │                     ▼                                │
         │             ┌──────────────┐                         │
         │             │ Environment  │                         │
         │             │ step(a₁..aₙ) │                         │
         │             └──────┬───────┘                         │
         │                    │                                 │
         │       ┌────────────┼────────────┐                    │
         │       ▼            ▼            ▼                    │
         │   (obs₁, r₁)   (obs₂, r₂)   (obsₙ, rₙ)              │
         │                                                      │
         └─────────────────────────────────────────────────────┘
```

---

### 1.2 AEC (Agent Environment Cycle)

**Used by:** PettingZoo AEC API, OpenSpiel (via Shimmy)

**Definition:**
An AEC game is a tuple (N, S, {A_i}, {O_i}, T, {R_i}, π, ε) where:
- N, S, A_i, O_i, T, R_i = same as POSG
- π: H → N ∪ {ε} = agent selection function (history → next agent)
- ε = environment agent (handles stochastic transitions)

**Key Property:** Agents act **sequentially**, one at a time.

```
         ┌─────────────────────────────────────────────────────┐
         │                    Agent Cycle                       │
         │                                                      │
         │  ┌──────────────────────────────────────────────┐   │
         │  │                  Agent 1                      │   │
         │  │  last() → (obs, r, done, trunc, info)        │   │
         │  │  step(action)                                 │   │
         │  └─────────────────────┬────────────────────────┘   │
         │                        │                             │
         │                        ▼                             │
         │  ┌──────────────────────────────────────────────┐   │
         │  │               Environment Step 1              │   │
         │  │  (internal state update)                      │   │
         │  └─────────────────────┬────────────────────────┘   │
         │                        │                             │
         │                        ▼                             │
         │  ┌──────────────────────────────────────────────┐   │
         │  │                  Agent 2                      │   │
         │  │  last() → (obs, r, done, trunc, info)        │   │
         │  │  step(action)                                 │   │
         │  └─────────────────────┬────────────────────────┘   │
         │                        │                             │
         │                        ▼                             │
         │  ┌──────────────────────────────────────────────┐   │
         │  │               Environment Step 2              │   │
         │  │  (internal state update)                      │   │
         │  └─────────────────────┬────────────────────────┘   │
         │                        │                             │
         │                        ▼                             │
         │                   (cycle repeats)                    │
         │                                                      │
         └─────────────────────────────────────────────────────┘
```

---

### 1.3 EFG (Extensive Form Game)

**Used by:** OpenSpiel

**Definition:**
An EFG is a tree structure where:
- Nodes = game states
- Edges = actions
- Leaves = terminal states with payoffs
- Nature player = handles stochastic events

**Key Property:** Explicit tree representation of all possible game paths.

```
                              Root
                               │
                  ┌────────────┼────────────┐
                  │            │            │
                  ▼            ▼            ▼
              Player 1     Player 1     Nature
                 │            │            │
              ┌──┴──┐      ┌──┴──┐      ┌──┴──┐
              ▼     ▼      ▼     ▼      ▼     ▼
           Player 2  ...  ...   ...   ...   ...
```

---

## 2. API Comparison

### 2.1 Code Examples

**Gymnasium (Single-Agent):**
```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()

for _ in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

**RLlib (POSG/Simultaneous):**
```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv

env = MultiAgentEnv()
observations = env.reset()  # Dict[AgentID, Obs]

for _ in range(1000):
    actions = {
        agent_id: policy(obs)
        for agent_id, obs in observations.items()
    }
    observations, rewards, dones, infos = env.step(actions)
    if dones["__all__"]:
        observations = env.reset()
```

**PettingZoo Parallel API (POSG/Simultaneous):**
```python
from pettingzoo.butterfly import pistonball_v6

# parallel_env() returns POSG-style environment
env = pistonball_v6.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # All agents act simultaneously
    actions = {agent: policy(obs) for agent, obs in observations.items()}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

**PettingZoo AEC API (Sequential):**
```python
from pettingzoo.classic import chess_v6

# env() returns AEC-style environment
env = chess_v6.env()
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = policy(agent, observation)

    env.step(action)
```

**OpenSpiel (EFG):**
```python
import pyspiel

game = pyspiel.load_game("chess")
state = game.new_initial_state()

while not state.is_terminal():
    if state.is_chance_node():
        action = np.random.choice(state.legal_actions())
    else:
        current_player = state.current_player()
        action = policy(current_player, state.observation_tensor())
    state.apply_action(action)
```

---

## 3. Equivalence Theorems

### 3.1 AEC ↔ POSG Equivalence (PettingZoo Paper)

**Theorem (from PettingZoo paper, Appendix D):**

> For every AEC game, there exists an equivalent POSG, and for every POSG, there exists an equivalent AEC game.

**AEC → POSG Conversion:**
- Expand state space to include "current agent" indicator
- All agents observe (but don't act) during others' turns
- Use no-op actions for non-acting agents

**POSG → AEC Conversion:**
- Order agents arbitrarily (e.g., by ID)
- Each agent acts in sequence within one "logical step"
- Combine all actions before environment transition

### 3.2 Practical Implications

| Aspect | POSG Native | AEC Native |
|--------|-------------|------------|
| Simultaneous games (MPE) | Natural | Requires queuing |
| Turn-based games (Chess) | Requires no-ops | Natural |
| Agent death/creation | Awkward | Natural |
| Race condition risk | High | None |
| Reward attribution | Summed | Per-agent |
| Implementation complexity | Lower | Higher |

---

## 4. Why Mosaic Must Support Both

### 4.1 RL Worker Paradigm Matrix

| Worker | Primary Paradigm | Secondary | Notes |
|--------|------------------|-----------|-------|
| CleanRL | SINGLE_AGENT | - | Gymnasium (incl. MuJoCo envs) |
| RLlib | SIMULTANEOUS | SINGLE_AGENT | POSG native |
| PettingZoo | SEQUENTIAL + SIMULTANEOUS | - | **Both AEC and Parallel APIs** |
| Jason/BDI | SEQUENTIAL | - | Goal-driven + RL |
| OpenSpiel (Shimmy) | SEQUENTIAL | SIMULTANEOUS | Via PettingZoo wrappers |

### 4.1.1 Non-RL Controllers (Separate from RL Paradigms)

| Controller | Type | Approach |
|------------|------|----------|
| MuJoCo MPC | Model Predictive Control | Optimization (iLQG, Cross Entropy) - NOT RL |

> **Note:** MuJoCo MPC is NOT an RL system. It uses planning/optimization algorithms,
> not policy gradients or value functions. It has its own controller in
> `gym_gui/services/mujoco_mpc_controller/` and should not be mixed with RL paradigms.

### 4.2 Environment Paradigm Matrix (RL Environments)

| Environment | Native Paradigm | Can Convert To | API |
|-------------|-----------------|----------------|-----|
| Gymnasium Atari | SINGLE_AGENT | - | `gym.make()` |
| Gymnasium MuJoCo (RL) | SINGLE_AGENT | - | `gym.make("HalfCheetah-v4")` |
| PettingZoo Classic (Chess) | SEQUENTIAL | SIMULTANEOUS | `env()` / `parallel_env()` |
| PettingZoo Butterfly | SIMULTANEOUS | SEQUENTIAL | `parallel_env()` / `env()` |
| PettingZoo MPE | SIMULTANEOUS | SEQUENTIAL | `parallel_env()` / `env()` |
| RLlib MultiAgent | SIMULTANEOUS | SEQUENTIAL | Custom conversion |
| OpenSpiel (Shimmy) | SEQUENTIAL | - | Via Shimmy wrapper |
| MeltingPot (Shimmy) | SIMULTANEOUS | - | Via Shimmy wrapper |

> **MuJoCo Clarification:**
> - **Gymnasium MuJoCo** (HalfCheetah, Hopper, Walker2d) → RL training with CleanRL/RLlib
> - **MuJoCo MPC** (Cartpole, Humanoid Track) → Optimal control, NOT RL

### 4.3 PettingZoo API Conversion

```python
from pettingzoo.utils import aec_to_parallel, parallel_to_aec

# AEC → Parallel (with restrictions: must update only at cycle end)
parallel_env = aec_to_parallel(aec_env)

# Parallel → AEC (always possible)
aec_env = parallel_to_aec(parallel_env)
```

**Ecosystem Support:**
- **SuperSuit**: Frame stacking, observation normalization, action clipping
- **Shimmy**: OpenSpiel, DeepMind Control Soccer, MeltingPot compatibility

---

## 5. Mosaic's Unified Abstraction

### 5.1 The ParadigmAdapter Solution

```python
class ParadigmAdapter(ABC):
    """Bridge between Mosaic and paradigm-specific workers."""

    @abstractmethod
    def get_stepping_mode(self) -> SteppingParadigm:
        """Which paradigm this adapter implements."""
        pass

    @abstractmethod
    def get_agents_to_act(self) -> List[str]:
        """Return agents that need actions NOW.

        SINGLE_AGENT: ["agent_0"]
        SIMULTANEOUS: [all agents]
        SEQUENTIAL: [current_agent]
        """
        pass

    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> StepResult:
        """Execute paradigm-appropriate step."""
        pass
```

### 5.2 Usage Pattern

```python
# Mosaic GUI doesn't care about paradigm
adapter = get_adapter_for_environment(env_id)
policy_service = PolicyMappingService()

# Game loop is paradigm-agnostic
while not adapter.is_done():
    agents = adapter.get_agents_to_act()
    observations = adapter.get_observations(agents)

    # PolicyMappingService handles both modes
    if adapter.get_stepping_mode() == SteppingParadigm.SIMULTANEOUS:
        actions = policy_service.select_actions(observations)
    else:
        actions = {
            agent: policy_service.select_action(agent, obs)
            for agent, obs in observations.items()
        }

    result = adapter.step(actions)
```

---

## 6. References

1. Terry, J.K. et al. "PettingZoo: Gym for Multi-Agent RL" NeurIPS 2021
2. Liang, E. et al. "RLlib: Abstractions for Distributed RL" ICML 2018
3. Lanctot, M. et al. "OpenSpiel: A Framework for RL in Games" 2019
4. Shapley, L. "Stochastic Games" 1953
5. Bernstein, D. et al. "The Complexity of Decentralized Control" 2002
