# Ray/RLlib Architecture Analysis

## Overview

This document analyzes the Ray and RLlib architecture to understand the concepts of **Actors**, **Workers**, **Policies**, and **Agents** as used in distributed reinforcement learning systems.

---

## 1. Ray Core Concepts

### 1.1 Ray Actor Model

**Source:** `ray/python/ray/actor.py`

Ray's Actor model is based on the **Actor Model** from distributed computing (Carl Hewitt, 1973). In Ray, an Actor is a **stateful distributed process**.

```python
# From ray/python/ray/actor.py (lines 1188-1196)
@PublicAPI
class ActorClass(Generic[T]):
    """An actor class.

    This is a decorated class. It can be used to create actors.

    Attributes:
        __ray_metadata__: Contains metadata for the actor.
    """
```

**Key characteristics:**
- Created with `@ray.remote` decorator
- Runs in its own process
- Maintains state across method calls
- Communicates via message passing (`.remote()` calls)

### 1.2 Ray Tasks vs Actors

| Concept | Ray Task | Ray Actor |
|---------|----------|-----------|
| State | Stateless | Stateful |
| Execution | Single function call | Long-lived process |
| Use case | Parallel function execution | Stateful services |

---

## 2. RLlib Architecture

### 2.1 Core Components

**Source:** `ray/rllib/` directory structure

```
rllib/
├── algorithms/           # Training algorithms (PPO, DQN, SAC, etc.)
├── core/
│   ├── rl_module/       # Neural network modules
│   │   ├── rl_module.py        # Single RLModule
│   │   └── multi_rl_module.py  # Container for multiple RLModules
│   └── learner/         # Training logic
├── env/
│   ├── env_runner.py           # Base class for rollout workers
│   ├── single_agent_env_runner.py
│   ├── multi_agent_env_runner.py
│   ├── multi_agent_env.py      # Multi-agent environment interface
│   ├── external_env.py         # For external control (human, etc.)
│   └── policy_client.py        # Client for policy serving
└── evaluation/          # Legacy rollout workers
```

### 2.2 EnvRunner (Rollout Worker)

**Source:** `ray/rllib/env/env_runner.py` (lines 36-51)

```python
@PublicAPI(stability="alpha")
class EnvRunner(FaultAwareApply, metaclass=abc.ABCMeta):
    """Base class for distributed RL-style data collection from an environment.

    The EnvRunner API's core functionalities can be summarized as:
    - Gets configured via passing a AlgorithmConfig object to the constructor.
    Normally, subclasses of EnvRunner then construct their own environment (possibly
    vectorized) copies and RLModules/Policies and use the latter to step through the
    environment in order to collect training data.
    - Clients of EnvRunner can use the `sample()` method to collect data for training
    from the environment(s).
    - EnvRunner offers parallelism via creating n remote Ray Actors based on this class.
    """
```

**Key insight:** `EnvRunner` is a **Ray Actor** that:
1. Owns a copy of the environment
2. Owns a copy of the policy (RLModule)
3. Collects training data via `sample()`

### 2.3 Multi-Agent Architecture

**Source:** `ray/rllib/env/multi_agent_env.py` (lines 29-54)

```python
@PublicAPI(stability="beta")
class MultiAgentEnv(gym.Env):
    """An environment that hosts multiple independent agents.

    Agents are identified by AgentIDs (string).
    """

    # Optional mappings from AgentID to individual agents' spaces.
    observation_spaces: Optional[Dict[AgentID, gym.Space]] = None
    action_spaces: Optional[Dict[AgentID, gym.Space]] = None

    # All agents currently active in the environment.
    agents: List[AgentID] = []
    # All agents that may appear in the environment, ever.
    possible_agents: List[AgentID] = []
```

### 2.4 Policy Mapping

**Source:** `ray/rllib/algorithms/algorithm_config.py` (lines 3434-3448)

```python
def multi_agent(
    self,
    *,
    policies: Optional[
        Union[MultiAgentPolicyConfigDict, Collection[PolicyID]]
    ] = NotProvided,
    policy_mapping_fn: Optional[
        Callable[[AgentID, "EpisodeType"], PolicyID]
    ] = NotProvided,
    policies_to_train: Optional[
        Union[Collection[PolicyID], Callable[[PolicyID, SampleBatchType], bool]]
    ] = NotProvided,
    ...
)
```

**Example from `ray/rllib/tuned_examples/ppo/multi_agent_cartpole_ppo.py`:**

```python
config = (
    PPOConfig()
    .multi_agent(
        # Maps agent_id → policy_id
        policy_mapping_fn=lambda aid, *arg, **kw: f"p{aid}",
        # Multiple policies, each can be different
        policies={f"p{i}" for i in range(num_agents)},
    )
)
```

---

## 3. MultiRLModule (Multiple Brains)

**Source:** `ray/rllib/core/rl_module/multi_rl_module.py` (lines 47-71)

```python
@PublicAPI(stability="alpha")
class MultiRLModule(RLModule):
    """Base class for an RLModule that contains n sub-RLModules.

    This class holds a mapping from ModuleID to underlying RLModules. It provides
    a convenient way of accessing each individual module, as well as accessing all of
    them with only one API call.

    The extension of this class can include any arbitrary neural networks as part of
    the MultiRLModule. For example, a MultiRLModule can include a shared encoder network
    that is used by all the individual (single-agent) RLModules.
    """
```

**This is the key evidence for "multiple brains":**
- Each `ModuleID` maps to a separate neural network
- Modules can share components (e.g., shared encoder)
- Each agent in the environment can use a different module

---

## 4. External Control (Human-in-the-Loop)

### 4.1 ExternalEnv

**Source:** `ray/rllib/env/external_env.py` (lines 24-52)

```python
@OldAPIStack
class ExternalEnv(threading.Thread):
    """An environment that interfaces with external agents.

    Unlike simulator envs, control is inverted: The environment queries the
    policy to obtain actions and in return logs observations and rewards for
    training. This is in contrast to gym.Env, where the algorithm drives the
    simulation through env.step() calls.

    You can use ExternalEnv as the backend for policy serving (by serving HTTP
    requests in the run loop), for ingesting offline logs data (by reading
    offline transitions in the run loop), or other custom use cases not easily
    expressed through gym.Env.

    ExternalEnv supports both on-policy actions (through self.get_action()),
    and off-policy actions (through self.log_action()).
    """
```

### 4.2 PolicyClient

**Source:** `ray/rllib/env/policy_client.py` (lines 35-92)

```python
@OldAPIStack
class PolicyClient:
    """REST client to interact with an RLlib policy server."""

    def get_action(
        self, episode_id: str, observation: Union[EnvObsType, MultiAgentDict]
    ) -> Union[EnvActionType, MultiAgentDict]:
        """Get action from policy for given observation."""

    def log_action(
        self,
        episode_id: str,
        observation: Union[EnvObsType, MultiAgentDict],
        action: Union[EnvActionType, MultiAgentDict],
    ) -> None:
        """Log an action taken by an external agent (e.g., human)."""
```

**This enables:**
- Human players to provide actions via HTTP
- Logging human demonstrations for imitation learning
- Mixed human-AI gameplay

---

## 5. Terminology Mapping

| RLlib Term | Meaning | Your Project Equivalent |
|------------|---------|------------------------|
| **Ray Actor** | Distributed stateful process | Worker process |
| **EnvRunner** | Ray Actor that samples from env | Worker (cleanrl_worker) |
| **Policy/RLModule** | Neural network (brain) | Actor (decision-maker) |
| **Agent** | Entity in environment (player) | Agent (player_0, player_1) |
| **PolicyMapping** | agent_id → policy_id | Actor assignment per agent |
| **ExternalEnv** | Env controlled externally | Human control mode |

---

## 6. Key Architectural Patterns

### Pattern 1: Multiple Policies per Environment

```
Environment                    Policy Mapping              Policies
───────────────────────────────────────────────────────────────────
agent_0 (red team)    ────►    "aggressive"     ────►    PPO Net A
agent_1 (red team)    ────►    "aggressive"     ────►    PPO Net A (shared)
agent_2 (blue team)   ────►    "defensive"      ────►    PPO Net B
agent_3 (human)       ────►    "human"          ────►    External input
```

### Pattern 2: Self-Play

```
agent_0  ────►  "current"      ────►  Latest policy (training)
agent_1  ────►  "opponent_v1"  ────►  Frozen snapshot from 1000 steps ago
agent_2  ────►  "opponent_v2"  ────►  Frozen snapshot from 5000 steps ago
```

### Pattern 3: Human-AI Collaboration

```
agent_0 (human)  ────►  PolicyClient.log_action()  ────►  Human keyboard
agent_1 (AI)     ────►  RLModule.forward()         ────►  Trained policy
```

---

## 7. Implications for gym_gui

Based on this analysis, your `ActorService` aligns with RLlib's **policy_mapping** concept:

```python
# RLlib approach
policy_mapping_fn = lambda agent_id, episode: {
    "player_0": "human",      # Human plays white
    "player_1": "stockfish",  # AI plays black
}[agent_id]

# Your approach (conceptually similar)
actor_service.set_actor_for_agent("player_0", HumanKeyboardActor())
actor_service.set_actor_for_agent("player_1", StockfishActor())
```

**Recommendation:** Rename `ActorService` to `PolicyMappingService` or `AgentControllerService` to better reflect its purpose and align with industry terminology.
