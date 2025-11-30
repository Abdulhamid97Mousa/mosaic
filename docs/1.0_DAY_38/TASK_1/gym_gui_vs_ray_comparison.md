# gym_gui vs Ray/RLlib Architecture Comparison

## Overview

This document compares your project's architecture with Ray/RLlib to identify alignments and potential improvements.

---

## 1. Side-by-Side Comparison

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Ray / RLlib                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Algorithm (PPO, DQN, etc.)                                                │
│        │                                                                     │
│        ├── Learner (training logic)                                         │
│        │       └── RLModule (neural network)                                │
│        │                                                                     │
│        └── EnvRunnerGroup                                                   │
│                └── EnvRunner[0..N] (Ray Actors)                             │
│                        ├── Environment (vectorized)                         │
│                        └── RLModule (inference copy)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              gym_gui                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   MainWindow (Qt UI)                                                        │
│        │                                                                     │
│        ├── SessionController (game loop)                                    │
│        │       └── Adapter (GymAdapter, PettingZooAdapter)                  │
│        │                                                                     │
│        ├── ActorService (who controls)                                      │
│        │       └── Actor (HumanKeyboardActor, CleanRLWorkerActor)          │
│        │                                                                     │
│        └── TrainerClient ──────► Worker Process (cleanrl_worker)            │
│                                        ├── Training loop                    │
│                                        └── Policy (neural network)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Mapping

| Ray/RLlib | gym_gui | Notes |
|-----------|---------|-------|
| `Algorithm` | `TrainerClient` + Worker | Training orchestration |
| `EnvRunner` | Worker process | Runs env.step() loop |
| `RLModule` | Policy in worker | Neural network |
| `MultiRLModule` | — | Not implemented |
| `policy_mapping_fn` | `ActorService` | Maps agent → controller |
| `ExternalEnv` | `SessionController` | GUI-controlled env |
| `PolicyClient` | `HumanInputController` | External action source |

---

## 2. Key Differences

### 2.1 Who Owns the Environment?

**Ray/RLlib:**
```python
# EnvRunner owns the environment
class EnvRunner:
    def __init__(self, config):
        self.env = gym.make(config.env_id)  # EnvRunner owns env
        self.module = RLModule()             # EnvRunner owns policy

    def sample(self):
        obs = self.env.reset()
        while not done:
            action = self.module.forward(obs)  # Policy decides
            obs, reward, done, _ = self.env.step(action)
```

**gym_gui:**
```python
# GUI owns the environment (via SessionController)
class SessionController:
    def __init__(self):
        self._adapter = GymAdapter()  # GUI owns env

    def step(self, action):  # Action comes from outside (UI or worker)
        return self._adapter.step(action)

# Worker owns a SEPARATE environment copy
class CleanRLWorker:
    def run(self):
        env = gym.make(env_id)  # Worker owns its own env
        policy = load_policy()
        while training:
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
```

**Key difference:** In gym_gui, there are TWO environment instances:
1. GUI's `SessionController._adapter` (for visualization)
2. Worker's internal env (for training)

### 2.2 Policy Location

**Ray/RLlib:**
```
┌─────────────────────────────────────────────┐
│              Learner (main process)          │
│  ┌───────────────────────────────────────┐  │
│  │  RLModule (authoritative weights)     │  │
│  └───────────────────────────────────────┘  │
│                    │                         │
│             weight sync                      │
│                    ▼                         │
│  ┌───────────────────────────────────────┐  │
│  │  EnvRunner[0] ─► RLModule (copy)      │  │
│  │  EnvRunner[1] ─► RLModule (copy)      │  │
│  │  EnvRunner[N] ─► RLModule (copy)      │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**gym_gui:**
```
┌─────────────────────────────────────────────┐
│              GUI Process                     │
│  ┌───────────────────────────────────────┐  │
│  │  ActorService                         │  │
│  │    └── CleanRLWorkerActor (stub)      │  │
│  │            returns None               │  │
│  └───────────────────────────────────────┘  │
│                    │                         │
│               gRPC call                      │
│                    ▼                         │
│  ┌───────────────────────────────────────┐  │
│  │  Worker Process (separate)            │  │
│  │    └── Policy (actual neural network) │  │
│  │            returns action             │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## 3. Multi-Agent Support

### 3.1 RLlib Multi-Agent

**Source:** `ray/rllib/algorithms/algorithm_config.py`

```python
config.multi_agent(
    policies={"p0": PolicySpec(), "p1": PolicySpec()},
    policy_mapping_fn=lambda agent_id, episode, **kw: {
        "player_0": "p0",
        "player_1": "p1",
    }[agent_id],
)
```

**Result:**
- `player_0` uses policy `p0` (neural network A)
- `player_1` uses policy `p1` (neural network B)
- Both can train simultaneously

### 3.2 gym_gui Multi-Agent

**Current implementation:**

```python
# gym_gui/services/actor.py
class ActorService:
    def __init__(self):
        self._actors: Dict[str, Actor] = {}
        self._active_actor_id: Optional[str] = None  # Only ONE active

    def select_action(self, snapshot: StepSnapshot) -> Optional[int]:
        actor = self.get_active_actor()  # Single active actor
        return actor.select_action(snapshot)
```

**Limitation:** Only ONE active actor at a time, not per-agent mapping.

### 3.3 Proposed Enhancement

```python
# Proposed: gym_gui/services/policy_mapping.py
class PolicyMappingService:
    def __init__(self):
        self._agent_to_policy: Dict[AgentID, PolicyController] = {}

    def set_policy_for_agent(self, agent_id: str, policy: PolicyController):
        self._agent_to_policy[agent_id] = policy

    def select_action(self, agent_id: str, obs) -> int:
        policy = self._agent_to_policy.get(agent_id)
        return policy.select_action(obs)

# Usage
service = PolicyMappingService()
service.set_policy_for_agent("player_0", HumanPolicyController())
service.set_policy_for_agent("player_1", StockfishPolicyController())

# In game loop
for agent_id in env.agents:
    obs = observations[agent_id]
    action = service.select_action(agent_id, obs)
    actions[agent_id] = action
```

---

## 4. External Control Patterns

### 4.1 RLlib ExternalEnv

**Source:** `ray/rllib/env/external_env.py`

```python
class ExternalEnv(threading.Thread):
    """Control is inverted: Environment queries policy."""

    def get_action(self, episode_id, observation):
        """Get action from RLlib policy."""
        pass

    def log_action(self, episode_id, observation, action):
        """Log action from external source (human)."""
        pass
```

### 4.2 gym_gui HumanInputController

**Source:** `gym_gui/controllers/human_input.py`

```python
class HumanInputController:
    """Captures keyboard input and maps to actions."""

    def get_pending_action(self) -> Optional[int]:
        """Return buffered keyboard action."""
        pass
```

**Similarity:** Both allow external (non-policy) action sources.

---

## 5. Recommendations

### 5.1 Terminology Alignment

| Current | Recommended | Ray Equivalent |
|---------|-------------|----------------|
| `Actor` | `PolicyController` | `Policy` / `RLModule` |
| `ActorService` | `PolicyMappingService` | `policy_mapping_fn` |
| `Worker` | `EnvRunner` | `EnvRunner` |
| `SessionController` | `LocalEnvRunner` | N/A (unique to GUI) |

### 5.2 Architecture Enhancement

```
Proposed gym_gui Architecture
─────────────────────────────

┌─────────────────────────────────────────────────────────────┐
│                     MainWindow                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PolicyMappingService                                        │
│    ├── agent_0 → HumanPolicyController                      │
│    ├── agent_1 → StockfishPolicyController                  │
│    └── agent_2 → CleanRLPolicyController ──► Worker         │
│                                                              │
│  LocalEnvRunner (SessionController)                          │
│    └── Adapter (owns env for visualization)                 │
│                                                              │
│  TrainerClient                                               │
│    └── gRPC connection to worker processes                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 File Structure Proposal

```
gym_gui/
├── core/
│   ├── adapters/           # Environment adapters (keep)
│   └── policy_controllers/ # Renamed from actor concepts
│       ├── base.py         # PolicyController protocol
│       ├── human.py        # HumanPolicyController
│       ├── stockfish.py    # StockfishPolicyController
│       └── external.py     # ExternalPolicyController (for workers)
├── services/
│   ├── policy_mapping.py   # PolicyMappingService (renamed ActorService)
│   └── ...
└── ...
```

---

## 6. Conclusion

Your architecture is conceptually similar to RLlib but with different terminology. The key differences are:

1. **Terminology**: Your "Actor" = RLlib's "Policy"
2. **Multi-agent**: RLlib has per-agent policy mapping; yours has single active actor
3. **Env ownership**: Your GUI owns an env copy; workers own separate copies

**Recommended next steps:**
1. Rename `ActorService` → `PolicyMappingService`
2. Add per-agent policy mapping (not just single active)
3. Consider aligning with RLlib terminology for easier onboarding
