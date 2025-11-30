# Common Ground and Differences: gym_gui vs Ray

## Executive Summary

| Aspect | Ray/RLlib | gym_gui | Status |
|--------|-----------|---------|--------|
| **Distributed Workers** | EnvRunner (Ray Actor) | 3rd_party workers | ✅ Similar |
| **Stateless Tasks** | @ray.remote functions | — | ❌ Missing |
| **Stateful Actors** | @ray.remote classes | — | ❌ Missing (ActorService is different) |
| **Object Store** | Plasma shared memory | gRPC | ✅ Different but functional |
| **Policy Mapping** | policy_mapping_fn | ActorService | 🔸 Partial |
| **External Control** | ExternalEnv, PolicyClient | HumanInputController | ✅ Similar |
| **Env Ownership** | EnvRunner owns env | GUI owns env + workers have copies | ✅ Different (intentional) |

---

## 1. Common Ground

### 1.1 Independent Worker Processes ✅

**Ray:**
```python
# EnvRunner is a Ray Actor (separate process)
@ray.remote
class EnvRunner:
    def __init__(self, config):
        self.env = gym.make(config.env_id)
        self.module = RLModule(config)

    def sample(self):
        # Collect trajectories
        pass
```

**gym_gui:**
```python
# 3rd_party/cleanrl_worker - Independent process
class CleanRLRuntime:
    def __init__(self, config):
        self.env = gym.make(config.env_id)
        self.policy = load_policy(config)

    def run(self):
        # Training loop
        pass
```

**Verdict:** ✅ Your workers are correctly independent. They own their own environments and policies, just like Ray's EnvRunners.

---

### 1.2 External Control / Human-in-the-Loop ✅

**Ray:**
```python
# ExternalEnv - Environment controlled by external source
class ExternalEnv:
    def get_action(self, episode_id, obs):
        """Query policy for action."""

    def log_action(self, episode_id, obs, action):
        """Log externally-provided action (e.g., human)."""

# PolicyClient - REST client for external control
class PolicyClient:
    def get_action(self, episode_id, obs):
        return requests.post(server, {"obs": obs})
```

**gym_gui:**
```python
# HumanInputController - Keyboard capture
class HumanInputController:
    def get_pending_action(self) -> Optional[int]:
        return self._action_buffer.get()

# SessionController - GUI owns environment
class SessionController:
    def step(self, action):
        return self._adapter.step(action)
```

**Verdict:** ✅ Both support external control. Your approach is more integrated (Qt keyboard capture), while Ray uses HTTP/REST.

---

### 1.3 gRPC Communication ✅

**Ray:** Uses custom serialization + Plasma object store, but also supports gRPC for external services.

**gym_gui:** Uses gRPC for trainer-worker communication.

```protobuf
// gym_gui/services/trainer/proto/trainer.proto
service TrainerService {
    rpc SubmitRun(RunConfig) returns (RunStatus);
    rpc StreamTelemetry(RunId) returns (stream TelemetryEvent);
}
```

**Verdict:** ✅ gRPC is a valid alternative to Ray's object store for inter-process communication.

---

## 2. Key Differences

### 2.1 Ray Tasks (Stateless) ❌ Missing

**Ray has:**
```python
@ray.remote
def process_frame(frame):
    """Stateless function - can run anywhere."""
    return preprocess(frame)

# Launch 1000 parallel tasks
futures = [process_frame.remote(f) for f in frames]
results = ray.get(futures)
```

**gym_gui doesn't have:**
- No concept of distributing stateless work
- Everything is either GUI-bound or worker-bound

**Do you need it?**
- **Maybe not** for typical RL training
- **Yes** if you want parallel preprocessing, hyperparameter search, etc.

---

### 2.2 Ray Actors (Stateful Distributed) ❌ Missing

**Ray has:**
```python
@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

# Create actor on remote node
counter = Counter.remote()
# Call methods remotely
ray.get(counter.increment.remote())  # Returns 1
```

**gym_gui has (but different):**
```python
# ActorService is NOT a Ray Actor
# It's an in-process registry of decision-makers
class ActorService:
    def __init__(self):
        self._actors = {}  # In-process, not distributed

    def select_action(self, snapshot):
        return self._actors[self._active_id].select_action(snapshot)
```

**Critical difference:**
| Aspect | Ray Actor | gym_gui ActorService |
|--------|-----------|---------------------|
| Location | Remote process | Same process as GUI |
| State | Distributed | Local |
| Scaling | Can have 1000s | Single instance |
| Purpose | Distributed computation | Action routing |

---

### 2.3 Environment Ownership ✅ Different (Intentional)

**Ray:**
```
┌─────────────────────────────────────────────────────────┐
│  Learner (central)                                       │
│     │                                                    │
│     └── EnvRunner[0] ──► owns Env ──► collects data     │
│     └── EnvRunner[1] ──► owns Env ──► collects data     │
│     └── EnvRunner[N] ──► owns Env ──► collects data     │
│                                                          │
│  No central env - each worker has its own copy          │
└─────────────────────────────────────────────────────────┘
```

**gym_gui:**
```
┌─────────────────────────────────────────────────────────┐
│  GUI Process                                             │
│     │                                                    │
│     └── SessionController ──► owns Env ──► VISUALIZATION│
│                                                          │
│  Worker Process                                          │
│     │                                                    │
│     └── CleanRLRuntime ──► owns Env ──► TRAINING        │
│                                                          │
│  Two separate envs - one for display, one for training  │
└─────────────────────────────────────────────────────────┘
```

**Why different?** Your GUI needs to visualize the environment. Ray is headless.

**This is correct for your use case.**

---

### 2.4 Policy Mapping 🔸 Partial Implementation

**Ray (full multi-agent):**
```python
# Each agent can have different policy
config.multi_agent(
    policies={
        "human": PolicySpec(policy_class=ExternalPolicy),
        "ai_aggressive": PolicySpec(config={"lr": 0.01}),
        "ai_defensive": PolicySpec(config={"lr": 0.001}),
    },
    policy_mapping_fn=lambda agent_id, episode, **kw: {
        "player_0": "human",
        "player_1": "ai_aggressive",
        "player_2": "ai_defensive",
    }[agent_id],
)
```

**gym_gui (single active):**
```python
class ActorService:
    def __init__(self):
        self._active_actor_id: Optional[str] = None  # Only ONE

    def select_action(self, snapshot):
        # Same actor for ALL agents
        return self._actors[self._active_actor_id].select_action(snapshot)
```

**Gap:** You can't have `player_0 = human` and `player_1 = AI` simultaneously.

---

## 3. What's Missing vs What's Not Needed

### 3.1 Definitely Missing (Should Add)

| Feature | Why Needed | Complexity |
|---------|------------|------------|
| **Per-agent policy mapping** | Human vs AI in same game | Low |
| **Policy controller abstraction** | Clean interface for different brains | Low |

### 3.2 Maybe Missing (Depends on Use Case)

| Feature | Why Maybe Needed | Complexity |
|---------|------------------|------------|
| **Distributed tasks** | Parallel preprocessing, hyperparameter search | Medium |
| **Checkpointing** | Resume training, model versioning | Medium |
| **Policy pool** | Self-play with historical opponents | High |

### 3.3 Not Needed (Different Architecture)

| Feature | Why Not Needed |
|---------|----------------|
| **Ray object store** | gRPC works fine for your scale |
| **Distributed actors** | GUI is single-node by design |
| **EnvRunner as separate process** | GUI needs env for visualization |

---

## 4. Proposed Architecture Improvements

### 4.1 Rename for Clarity

```
Current                          Proposed
───────────────────────────────────────────────────
ActorService                 →   PolicyMappingService
Actor (protocol)             →   PolicyController (protocol)
HumanKeyboardActor           →   HumanPolicyController
CleanRLWorkerActor           →   ExternalPolicyController
```

### 4.2 Add Per-Agent Policy Mapping

```python
# Proposed: gym_gui/services/policy_mapping.py

from typing import Dict, Protocol, Optional
from gym_gui.core.enums import AgentID

class PolicyController(Protocol):
    """Protocol for any action-selecting brain."""

    def select_action(self, agent_id: AgentID, observation: object) -> Optional[int]:
        """Return action for given agent and observation."""
        ...

    def on_step_result(self, agent_id: AgentID, reward: float, done: bool) -> None:
        """Receive feedback after step."""
        ...


class PolicyMappingService:
    """Maps agents to their controlling policies."""

    def __init__(self):
        self._mapping: Dict[AgentID, PolicyController] = {}
        self._default_policy: Optional[PolicyController] = None

    def set_policy(self, agent_id: AgentID, policy: PolicyController) -> None:
        """Assign a policy to control a specific agent."""
        self._mapping[agent_id] = policy

    def set_default_policy(self, policy: PolicyController) -> None:
        """Set fallback policy for unmapped agents."""
        self._default_policy = policy

    def get_policy(self, agent_id: AgentID) -> Optional[PolicyController]:
        """Get the policy controlling an agent."""
        return self._mapping.get(agent_id, self._default_policy)

    def select_action(self, agent_id: AgentID, observation: object) -> Optional[int]:
        """Select action for an agent."""
        policy = self.get_policy(agent_id)
        if policy is None:
            return None
        return policy.select_action(agent_id, observation)
```

### 4.3 Implement Concrete Policy Controllers

```python
# gym_gui/services/policy_controllers/human.py
class HumanPolicyController:
    """Routes actions from keyboard input."""

    def __init__(self, input_controller: HumanInputController):
        self._input = input_controller

    def select_action(self, agent_id: AgentID, observation: object) -> Optional[int]:
        return self._input.get_pending_action()


# gym_gui/services/policy_controllers/stockfish.py
class StockfishPolicyController:
    """Uses Stockfish engine for chess moves."""

    def __init__(self, stockfish_service: StockfishService):
        self._engine = stockfish_service

    def select_action(self, agent_id: AgentID, observation: object) -> Optional[int]:
        # Convert observation to board state, query engine
        return self._engine.get_best_move(observation)


# gym_gui/services/policy_controllers/external.py
class ExternalPolicyController:
    """Delegates to external worker process."""

    def __init__(self, trainer_client: TrainerClient, run_id: str):
        self._client = trainer_client
        self._run_id = run_id

    def select_action(self, agent_id: AgentID, observation: object) -> Optional[int]:
        # Worker owns the env and policy - this is just a stub
        # Action comes via telemetry stream, not direct query
        return None
```

### 4.4 Usage in Multi-Agent Game

```python
# In MainWindow or game setup

# Create policy mapping service
policy_service = PolicyMappingService()

# Human vs Stockfish chess
policy_service.set_policy("player_0", HumanPolicyController(self._human_input))
policy_service.set_policy("player_1", StockfishPolicyController(self._stockfish))

# In game loop (PettingZoo style)
for agent_id in env.agent_iter():
    obs, reward, done, truncated, info = env.last()

    if done:
        action = None
    else:
        action = policy_service.select_action(agent_id, obs)

    env.step(action)
```

---

## 5. Summary: Common vs Different

### What's Common (Keep)

```
✅ Independent worker processes (3rd_party/*)
✅ gRPC for communication
✅ External control support (HumanInputController)
✅ GUI owns visualization env, workers own training env
✅ Telemetry streaming architecture
```

### What's Different (Intentional)

```
🔸 No distributed Ray Actors (GUI is single-node)
🔸 No Ray object store (gRPC is sufficient)
🔸 GUI-centric design vs headless training
```

### What's Missing (Should Add)

```
❌ Per-agent policy mapping (currently single active actor)
❌ PolicyController abstraction (clean protocol)
❌ Clear terminology (Actor → PolicyController)
```

### What's NOT Missing (Don't Need)

```
⬜ Ray Tasks (stateless distributed) - not needed for GUI app
⬜ Ray Actors (stateful distributed) - GUI is single-process
⬜ Policy pools for self-play - future enhancement, not critical
```
