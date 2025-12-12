# Multi-Paradigm Orchestrator Architecture Plan

## Executive Summary

This document addresses a critical architectural problem: **Mosaic's current implementation is tightly coupled to PettingZoo's AEC (Agent Environment Cycle) paradigm**, forcing all workers and actors to conform to a single stepping model. This is fundamentally wrong for a distributed system that must support multiple RL frameworks with incompatible stepping semantics.

---

## Progress Tracking

### Prerequisites ✅

- [x] **Naming Resolution**: Rename `ui/environments/` → `ui/config_panels/` (see [02_naming_resolution_plan.md](./02_naming_resolution_plan.md))
- [x] **Import Updates**: All imports updated to new paths
- [x] **Pyright Verification**: No import-related errors
- [x] **Pytest Verification**: 78 tests passed (minigrid integration)

### Phase 1: Abstraction Layer ✅

- [x] Define SteppingParadigm enum (`gym_gui/core/enums.py`)
- [x] Define WorkerCapabilities dataclass (`gym_gui/core/adapters/base.py`)
- [x] Define PolicyController protocol (`gym_gui/services/actor.py`)
- [x] Create ParadigmAdapter ABC (`gym_gui/core/adapters/paradigm.py`)
- [x] Add paradigm field to existing adapters (`EnvironmentAdapter`, `PettingZooAdapter`)

### Phase 2: PolicyMappingService 🔄

#### Phase 2.1: Core Service ✅ (LOGIC ONLY)

- [x] Create PolicyMappingService class (`gym_gui/services/policy_mapping.py`)
- [x] Define AgentPolicyBinding dataclass
- [x] Add per-agent policy mapping (bindings dict)
- [x] Add paradigm-aware action selection (`select_action`, `select_actions`)
- [x] Bootstrap registration (`gym_gui/services/bootstrap.py`)

#### Phase 2.2: SessionController Integration 📋 (LOGIC)

- [ ] Update `_select_agent_action()` to use PolicyMappingService
- [ ] Update `_record_step()` for per-agent notification
- [ ] Update `_finalize_episode()` for per-agent cleanup

#### Phase 2.3: UI Components 📋 (UI)

- [ ] Create PolicyMappingPanel widget
- [ ] Update ControlPanelContainer
- [ ] Add preset save/load functionality
- [ ] Deprecate single-actor dropdown for multi-agent envs

### Phase 3: Worker Registry 📋

- [ ] Create WorkerTypeRegistry
- [ ] Define WorkerTypeDefinition schema
- [ ] Register existing workers (CleanRL, Jason, MuJoCo MPC)
- [ ] Add capability discovery
- [ ] Update TrainerService to use registry

### Phase 4: Paradigm Adapters 📋

- [ ] Implement GymParadigmAdapter (SINGLE_AGENT)
- [ ] Implement PettingZooParadigmAdapter (SEQUENTIAL)
- [ ] Implement RLlibParadigmAdapter (SIMULTANEOUS)
- [ ] Add adapter selection logic
- [ ] Integration testing across paradigms

### Phase 5: WorkerOrchestrator 📋

- [ ] Create WorkerOrchestrator class
- [ ] Implement paradigm inference
- [ ] Implement worker selection
- [ ] Migrate from TrainerDispatcher
- [ ] End-to-end testing

---

## 1. The Core Problem

### 1.1 Current State: PettingZoo-Centric Design

The current architecture was rushed to support PettingZoo, creating implicit assumptions throughout:

```
Current Architecture (PettingZoo-Biased)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│                              GUI (Mosaic)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ActorService                                                           ││
│  │    └── _active_actor_id: Optional[str]  ← SINGLE ACTOR FOR ALL AGENTS! ││
│  │    └── select_action(snapshot) → int    ← ASSUMES SEQUENTIAL STEPPING  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                              gRPC (Trainer)                                  │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  TrainerService                                                         ││
│  │    └── CleanRL Worker ONLY                                              ││
│  │    └── Assumes Gym/PettingZoo environments                              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘

PROBLEMS:
├── ActorService assumes ONE active actor (not per-agent mapping)
├── StepSnapshot designed for single-agent (no agent_id field)
├── Workers tightly coupled to CleanRL implementation
├── No abstraction for different stepping paradigms
└── Adding RLlib/XuanCe would require hacking around PettingZoo assumptions
```

### 1.2 The Two Stepping Paradigms

#### POSG Model (Simultaneous) - RLlib, PettingZoo Parallel API

```python
# Simultaneous stepping - ALL agents act at once
# Used by: RLlib, PettingZoo parallel_env(), Gymnasium Multi-Agent

# PettingZoo Parallel API example:
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.parallel_env()
observations, infos = env.reset()

while env.agents:
    actions = {agent: policy(obs) for agent, obs in observations.items()}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

#### AEC Model (Sequential) - PettingZoo AEC API, OpenSpiel

```python
# Sequential stepping - agents act ONE AT A TIME
# Used by: PettingZoo env(), OpenSpiel (via Shimmy)

# PettingZoo AEC API example:
from pettingzoo.classic import chess_v6
env = chess_v6.env()
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy(agent, observation) if not termination else None
    env.step(action)
```

### 1.3 PettingZoo Supports BOTH Paradigms

**Important Clarification:** PettingZoo is NOT AEC-only. It provides:

| API | Creation | Paradigm | Use Case |
|-----|----------|----------|----------|
| `env()` | `chess_v6.env()` | AEC (Sequential) | Turn-based games |
| `parallel_env()` | `pistonball_v6.parallel_env()` | POSG (Simultaneous) | Continuous games |

**Conversion Wrappers:**

```python
from pettingzoo.utils import aec_to_parallel, parallel_to_aec

# Convert AEC to Parallel (with restrictions)
parallel_env = aec_to_parallel(aec_env)

# Convert Parallel to AEC
aec_env = parallel_to_aec(parallel_env)
```

**Additional Ecosystem:**

- **SuperSuit** - Pre-processing wrappers (frame stacking, observation normalization)
- **Shimmy** - Compatibility wrappers (OpenSpiel, DeepMind Control, Melting Pot)

### 1.4 API Comparison Table

| Aspect | POSG (Parallel) | AEC (Sequential) |
|--------|-----------------|------------------|
| Step semantics | `env.step(Dict[Agent, Action])` | `env.step(single_action)` |
| Action format | `Dict[AgentID, Action]` | Single `Action` |
| Agent iteration | All at once | One at a time via `agent_iter()` |
| Observation access | `env.reset()` returns dict | `env.last()` for current agent |
| Reward timing | All returned together | Per-agent, per-step |
| Race conditions | Possible (tie-breaking needed) | Impossible |
| Best for | Continuous control, MPE | Turn-based, board games |

**Mathematical equivalence is proven** (PettingZoo paper, Appendix D), but **API patterns differ significantly**.

---

## 2. Current 3rd Party Workers

```
3rd_party/
│
│ ─── RL TRAINING WORKERS ───────────────────────────────────────────────────
├── cleanrl_worker/       # Single-agent, Gymnasium (incl. MuJoCo envs) (WORKING)
├── pettingzoo_worker/    # Multi-agent, AEC + Parallel (IN PROGRESS)
│   └── PettingZoo/       # Submodule with both APIs
├── ray_worker/           # Multi-agent, POSG (PLANNED)
├── jason_worker/         # BDI + RL, sequential (WORKING)
├── vizdoom_worker/       # FPS environment, single-agent (WORKING)
├── airsim_worker/        # Drone simulation (PLANNED)
├── llm_worker/           # LLM-based agent (PLANNED)
├── vlm_worker/           # Vision-language (PLANNED)
├── supersuit_worker/     # SuperSuit wrappers
│   └── SuperSuit/        # Submodule
├── shimmy_worker/        # Shimmy compatibility (OpenSpiel, MeltingPot)
│   └── Shimmy/           # Submodule
│
│ ─── NON-RL CONTROLLERS ────────────────────────────────────────────────────
├── mujoco_mpc_worker/    # Model Predictive Control (NOT RL!)
│   └── mujoco_mpc/       # DeepMind's MJPC (optimization-based planning)
└── unreal_map_worker/    # Unreal Engine environments (PLANNED)
```

> **Important Distinction:**
>
> - **MuJoCo environments** (HalfCheetah, Hopper, etc.) use CleanRL for RL training
> - **MuJoCo MPC** is optimal control/planning (iLQG, Cross Entropy) - NOT RL
> - MuJoCo MPC is managed by `gym_gui/services/mujoco_mpc_controller/`, not the RL orchestrator

### Worker Paradigm Support Matrix (RL Workers)

| Worker | Single-Agent | Parallel (POSG) | Sequential (AEC) | Notes |
|--------|--------------|-----------------|------------------|-------|
| cleanrl | ✅ | ❌ | ❌ | Gymnasium (incl. MuJoCo envs) |
| pettingzoo | ❌ | ✅ | ✅ | Both APIs supported |
| ray/rllib | ✅ | ✅ | ❌ | POSG native |
| jason/bdi | ❌ | ❌ | ✅ | Goal-driven + RL |
| vizdoom | ✅ | ❌ | ❌ | FPS environment |
| shimmy | ❌ | ✅ | ✅ | OpenSpiel, MeltingPot |

### Non-RL Controllers (Separate Management)

| Controller | Type | Notes |
|------------|------|-------|
| mujoco_mpc | Model Predictive Control | Optimization-based planning, NOT RL |
| (future) | Classical Control | PID, LQR, etc. |

Each worker has different:

- Environment interface (Gym, PettingZoo AEC, PettingZoo Parallel, custom)
- Stepping paradigm (sequential, simultaneous, continuous)
- Action space (discrete, continuous, hierarchical)
- Observation format (vector, image, structured)

---

## 3. Proposed Architecture: Paradigm-Agnostic Orchestrator

### 3.1 High-Level Design

```
Proposed Architecture (Paradigm-Agnostic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│                              GUI (Mosaic)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  PolicyMappingService (replaces ActorService)                           ││
│  │    └── _agent_policies: Dict[AgentID, PolicyController]                 ││
│  │    └── select_action(agent_id, obs) → Action                            ││
│  │    └── select_actions(observations) → Dict[AgentID, Action]             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                    ┌───────────────┼───────────────┐                        │
│                    │               │               │                        │
│                    ▼               ▼               ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  WorkerOrchestrator (replaces TrainerService)                           ││
│  │    └── ParadigmAdapter (interface)                                      ││
│  │           ├── GymAdapter (single-agent, POSG)                           ││
│  │           ├── PettingZooAdapter (multi-agent, AEC)                      ││
│  │           ├── RLlibAdapter (multi-agent, POSG)                          ││
│  │           └── CustomAdapter (extensible)                                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                    ┌───────────────┼───────────────┐                        │
│                    │               │               │                        │
│                    ▼               ▼               ▼                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │ CleanRL      │  │ RLlib        │  │ Jason/BDI    │                       │
│  │ Worker       │  │ Worker       │  │ Worker       │                       │
│  │ (POSG)       │  │ (POSG)       │  │ (Sequential) │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Abstractions

#### 3.2.1 SteppingParadigm Enum

```python
from enum import Enum, auto

class SteppingParadigm(Enum):
    """Defines how RL agents interact with the environment.

    NOTE: This enum is for RL training paradigms ONLY.
    Non-RL systems like MuJoCo MPC (optimal control) are managed separately.
    """

    SINGLE_AGENT = auto()      # Gymnasium: one agent, one step
    SIMULTANEOUS = auto()       # RLlib/POSG/PettingZoo Parallel: all agents step together
    SEQUENTIAL = auto()         # PettingZoo AEC/OpenSpiel: agents step one at a time
    HIERARCHICAL = auto()       # BDI: high-level goals → low-level RL actions
```

> **Note on MuJoCo vs MuJoCo MPC:**
>
> - **MuJoCo** (physics engine) is used with CleanRL/RLlib for RL training → `SINGLE_AGENT`
> - **MuJoCo MPC** is Model Predictive Control (optimization-based planning), NOT RL
> - MuJoCo MPC has its own controller (`gym_gui/services/mujoco_mpc_controller/`)
> - Do NOT include MuJoCo MPC in RL paradigms - it uses planning, not learning

#### 3.2.2 WorkerCapabilities

```python
@dataclass
class WorkerCapabilities:
    """Declares what a worker can do."""

    worker_type: str                    # "cleanrl", "rllib", "jason", etc.
    paradigm: SteppingParadigm          # Primary stepping model
    supported_paradigms: List[SteppingParadigm]  # All supported

    # Environment compatibility
    env_types: List[str]                # ["gymnasium", "pettingzoo", "custom"]
    action_spaces: List[str]            # ["discrete", "continuous", "multi_discrete"]
    observation_spaces: List[str]       # ["box", "dict", "image"]

    # Multi-agent support
    max_agents: int                     # 1 for single-agent, N for multi-agent
    supports_self_play: bool
    supports_population: bool

    # Resource requirements
    requires_gpu: bool
    gpu_memory_mb: Optional[int]
    cpu_cores: int
```

#### 3.2.3 ParadigmAdapter Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ParadigmAdapter(ABC):
    """Adapts different stepping paradigms to a common interface."""

    @property
    @abstractmethod
    def paradigm(self) -> SteppingParadigm:
        """Return the paradigm this adapter handles."""
        ...

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset and return initial observations.

        Returns:
            For SINGLE_AGENT: {"agent_0": obs}
            For SIMULTANEOUS: {agent_id: obs for all agents}
            For SEQUENTIAL: {current_agent: obs}
        """
        ...

    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> StepResult:
        """Execute actions according to paradigm.

        Args:
            actions: For SINGLE_AGENT: {"agent_0": action}
                     For SIMULTANEOUS: {agent_id: action for all}
                     For SEQUENTIAL: {current_agent: action}

        Returns:
            StepResult with observations, rewards, dones, infos
        """
        ...

    @abstractmethod
    def get_current_agents(self) -> List[str]:
        """Return agents that need to act NOW.

        For SINGLE_AGENT: ["agent_0"]
        For SIMULTANEOUS: [all agent_ids]
        For SEQUENTIAL: [current_agent_id]
        """
        ...

    @abstractmethod
    def is_episode_done(self) -> bool:
        """Check if episode has ended for ALL agents."""
        ...
```

#### 3.2.4 PolicyController Protocol

```python
from typing import Protocol, Optional, Any

class PolicyController(Protocol):
    """Unified interface for all policy types."""

    @property
    def id(self) -> str:
        """Unique identifier for this policy."""
        ...

    @property
    def paradigm(self) -> SteppingParadigm:
        """Paradigm this policy expects."""
        ...

    def select_action(
        self,
        agent_id: str,
        observation: Any,
        info: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Select action for a specific agent."""
        ...

    def select_actions(
        self,
        observations: Dict[str, Any],
        infos: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Select actions for multiple agents (POSG mode)."""
        ...

    def on_step_result(
        self,
        agent_id: str,
        reward: float,
        done: bool,
        info: Dict[str, Any]
    ) -> None:
        """Receive feedback after step."""
        ...
```

### 3.3 PolicyMappingService (Replaces ActorService)

```python
class PolicyMappingService:
    """Maps agents to policies, supporting both AEC and POSG paradigms."""

    def __init__(self):
        self._agent_to_policy: Dict[str, PolicyController] = {}
        self._default_policy: Optional[PolicyController] = None
        self._paradigm_hint: Optional[SteppingParadigm] = None

    # ──────────────────────────────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────────────────────────────
    def set_policy(self, agent_id: str, policy: PolicyController) -> None:
        """Assign a policy to control a specific agent."""
        self._agent_to_policy[agent_id] = policy

    def set_default_policy(self, policy: PolicyController) -> None:
        """Fallback policy for unmapped agents."""
        self._default_policy = policy

    def set_paradigm_hint(self, paradigm: SteppingParadigm) -> None:
        """Hint for which stepping mode to use."""
        self._paradigm_hint = paradigm

    # ──────────────────────────────────────────────────────────────────────
    # Sequential (AEC) Mode
    # ──────────────────────────────────────────────────────────────────────
    def select_action(
        self,
        agent_id: str,
        observation: Any,
        info: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Get action for ONE agent (AEC/sequential mode)."""
        policy = self._agent_to_policy.get(agent_id, self._default_policy)
        if policy is None:
            return None
        return policy.select_action(agent_id, observation, info)

    # ──────────────────────────────────────────────────────────────────────
    # Simultaneous (POSG) Mode
    # ──────────────────────────────────────────────────────────────────────
    def select_actions(
        self,
        observations: Dict[str, Any],
        infos: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Get actions for ALL agents at once (POSG/simultaneous mode)."""
        actions = {}
        for agent_id, obs in observations.items():
            info = infos.get(agent_id) if infos else None
            action = self.select_action(agent_id, obs, info)
            if action is not None:
                actions[agent_id] = action
        return actions

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────
    def get_policy(self, agent_id: str) -> Optional[PolicyController]:
        return self._agent_to_policy.get(agent_id, self._default_policy)

    def list_agents(self) -> List[str]:
        return list(self._agent_to_policy.keys())

    def clear(self) -> None:
        self._agent_to_policy.clear()
        self._default_policy = None
```

---

## 4. Worker Registry and Discovery

### 4.1 Worker Type Registry

```python
@dataclass
class WorkerTypeDefinition:
    """Static definition of a worker type."""

    type_id: str                        # "cleanrl", "rllib", "jason"
    display_name: str
    description: str
    capabilities: WorkerCapabilities

    # Launch configuration
    module_path: str                    # e.g., "3rd_party.cleanrl_worker"
    entry_point: str                    # e.g., "main:run"
    config_schema: Optional[Type]       # Pydantic model for config validation

    # Requirements
    pip_dependencies: List[str]
    system_dependencies: List[str]


class WorkerTypeRegistry:
    """Registry of all available worker types."""

    _types: Dict[str, WorkerTypeDefinition] = {}

    @classmethod
    def register(cls, definition: WorkerTypeDefinition) -> None:
        cls._types[definition.type_id] = definition

    @classmethod
    def get(cls, type_id: str) -> Optional[WorkerTypeDefinition]:
        return cls._types.get(type_id)

    @classmethod
    def find_compatible(
        cls,
        paradigm: SteppingParadigm,
        env_type: str,
        action_space: str,
    ) -> List[WorkerTypeDefinition]:
        """Find workers compatible with given requirements."""
        return [
            defn for defn in cls._types.values()
            if paradigm in defn.capabilities.supported_paradigms
            and env_type in defn.capabilities.env_types
            and action_space in defn.capabilities.action_spaces
        ]
```

### 4.2 Registering Built-in Workers

```python
# In gym_gui/workers/registry.py

WorkerTypeRegistry.register(WorkerTypeDefinition(
    type_id="cleanrl",
    display_name="CleanRL",
    description="Single-file RL implementations (PPO, DQN, SAC)",
    capabilities=WorkerCapabilities(
        worker_type="cleanrl",
        paradigm=SteppingParadigm.SINGLE_AGENT,
        supported_paradigms=[SteppingParadigm.SINGLE_AGENT],
        env_types=["gymnasium"],
        action_spaces=["discrete", "continuous"],
        observation_spaces=["box"],
        max_agents=1,
        supports_self_play=False,
        supports_population=False,
        requires_gpu=True,
        gpu_memory_mb=2048,
        cpu_cores=4,
    ),
    module_path="3rd_party.cleanrl_worker",
    entry_point="cleanrl_worker.main:run",
    config_schema=CleanRLConfig,
    pip_dependencies=["torch", "gymnasium", "wandb"],
    system_dependencies=[],
))

WorkerTypeRegistry.register(WorkerTypeDefinition(
    type_id="rllib",
    display_name="RLlib",
    description="Scalable distributed RL (Ray-based)",
    capabilities=WorkerCapabilities(
        worker_type="rllib",
        paradigm=SteppingParadigm.SIMULTANEOUS,
        supported_paradigms=[SteppingParadigm.SINGLE_AGENT, SteppingParadigm.SIMULTANEOUS],
        env_types=["gymnasium", "rllib_multi_agent"],
        action_spaces=["discrete", "continuous", "multi_discrete"],
        observation_spaces=["box", "dict"],
        max_agents=100,
        supports_self_play=True,
        supports_population=True,
        requires_gpu=True,
        gpu_memory_mb=4096,
        cpu_cores=8,
    ),
    module_path="3rd_party.ray_worker",
    entry_point="ray_worker.main:run",
    config_schema=RLlibConfig,
    pip_dependencies=["ray[rllib]", "torch"],
    system_dependencies=[],
))

WorkerTypeRegistry.register(WorkerTypeDefinition(
    type_id="pettingzoo",
    display_name="PettingZoo",
    description="Multi-agent AEC environments",
    capabilities=WorkerCapabilities(
        worker_type="pettingzoo",
        paradigm=SteppingParadigm.SEQUENTIAL,
        supported_paradigms=[SteppingParadigm.SEQUENTIAL],
        env_types=["pettingzoo"],
        action_spaces=["discrete"],
        observation_spaces=["box", "dict"],
        max_agents=20,
        supports_self_play=True,
        supports_population=False,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=2,
    ),
    module_path="3rd_party.pettingzoo_worker",
    entry_point="pettingzoo_worker.main:run",
    config_schema=PettingZooConfig,
    pip_dependencies=["pettingzoo", "torch"],
    system_dependencies=[],
))
```

---

## 5. WorkerOrchestrator (Replaces TrainerService)

### 5.1 Unified Orchestrator

```python
class WorkerOrchestrator:
    """Manages workers across different paradigms."""

    def __init__(
        self,
        registry: RunRegistry,
        dispatcher: WorkerDispatcher,
        policy_service: PolicyMappingService,
    ):
        self._registry = registry
        self._dispatcher = dispatcher
        self._policy_service = policy_service
        self._active_adapters: Dict[str, ParadigmAdapter] = {}

    async def submit_run(
        self,
        config: UnifiedRunConfig,
    ) -> str:
        """Submit a training run, auto-selecting paradigm adapter."""

        # 1. Determine paradigm from config
        paradigm = self._infer_paradigm(config)

        # 2. Find compatible worker
        worker_type = self._select_worker(config, paradigm)

        # 3. Create paradigm adapter
        adapter = self._create_adapter(paradigm, config)

        # 4. Launch worker process
        run_id = await self._dispatcher.launch(
            worker_type=worker_type,
            config=config,
            adapter=adapter,
        )

        self._active_adapters[run_id] = adapter
        return run_id

    def _infer_paradigm(self, config: UnifiedRunConfig) -> SteppingParadigm:
        """Infer paradigm from environment type and config."""

        if config.env_type == "gymnasium":
            return SteppingParadigm.SINGLE_AGENT
        elif config.env_type == "pettingzoo":
            return SteppingParadigm.SEQUENTIAL
        elif config.env_type == "rllib_multi_agent":
            return SteppingParadigm.SIMULTANEOUS
        elif config.env_type == "mujoco_mpc":
            return SteppingParadigm.CONTINUOUS
        else:
            raise ValueError(f"Unknown env_type: {config.env_type}")

    def _select_worker(
        self,
        config: UnifiedRunConfig,
        paradigm: SteppingParadigm,
    ) -> WorkerTypeDefinition:
        """Select best worker for the job."""

        # User override
        if config.worker_type:
            return WorkerTypeRegistry.get(config.worker_type)

        # Auto-select based on requirements
        compatible = WorkerTypeRegistry.find_compatible(
            paradigm=paradigm,
            env_type=config.env_type,
            action_space=config.action_space,
        )

        if not compatible:
            raise ValueError(f"No compatible worker for {paradigm}")

        # Prefer GPU workers if available
        return sorted(compatible, key=lambda w: w.capabilities.requires_gpu)[-1]
```

---

## 6. Migration Plan

### Phase 1: Abstraction Layer ✅

```
Tasks:
├── [x] Define SteppingParadigm enum
├── [x] Define WorkerCapabilities dataclass
├── [x] Define PolicyController protocol
├── [x] Create ParadigmAdapter ABC
└── [x] Add paradigm field to existing adapters
```

### Phase 2: PolicyMappingService 🔄

```
Tasks (Phase 2.1 - LOGIC):
├── [x] Create PolicyMappingService class
├── [x] Define AgentPolicyBinding dataclass
├── [x] Add per-agent policy mapping
└── [x] Bootstrap registration

Tasks (Phase 2.2 - LOGIC):
├── [ ] Update SessionController to use new service
└── [ ] Add paradigm detection from adapter

Tasks (Phase 2.3 - UI):
├── [ ] Create PolicyMappingPanel widget
├── [ ] Update ControlPanelContainer
└── [ ] Deprecate single-actor dropdown
```

### Phase 3: Worker Registry (Week 5-6)

```
Tasks:
├── [ ] Create WorkerTypeRegistry
├── [ ] Define WorkerTypeDefinition schema
├── [ ] Register existing workers (CleanRL, Jason, MuJoCo MPC)
├── [ ] Add capability discovery
└── [ ] Update TrainerService to use registry
```

### Phase 4: Paradigm Adapters (Week 7-8)

```
Tasks:
├── [ ] Implement GymParadigmAdapter (SINGLE_AGENT)
├── [ ] Implement PettingZooParadigmAdapter (SEQUENTIAL)
├── [ ] Implement RLlibParadigmAdapter (SIMULTANEOUS)
├── [ ] Add adapter selection logic
└── [ ] Integration testing across paradigms
```

### Phase 5: WorkerOrchestrator (Week 9-10)

```
Tasks:
├── [ ] Create WorkerOrchestrator class
├── [ ] Implement paradigm inference
├── [ ] Implement worker selection
├── [ ] Migrate from TrainerDispatcher
└── [ ] End-to-end testing
```

---

## 7. File Structure Changes

> **Note:** Directory naming has been resolved - see [02_naming_resolution_plan.md](./02_naming_resolution_plan.md)
>
> - `ui/environments/` → `ui/config_panels/` ✅ DONE

```
gym_gui/
├── core/
│   ├── adapters/                    # Environment adapters (KEEP)
│   └── paradigms/                   # NEW: Stepping paradigm abstractions
│       ├── __init__.py
│       ├── base.py                  # SteppingParadigm, ParadigmAdapter
│       ├── single_agent.py          # GymParadigmAdapter
│       ├── sequential.py            # PettingZooParadigmAdapter
│       └── simultaneous.py          # RLlibParadigmAdapter
│
├── ui/
│   └── config_panels/               # UI configuration panels (RENAMED from environments/)
│       ├── single_agent/            # Single-agent env configs
│       └── multi_agent/             # Multi-agent env configs
│
├── services/
│   ├── actor.py                     # DEPRECATED → policy_mapping.py
│   ├── policy_mapping.py            # NEW: PolicyMappingService
│   ├── worker_registry.py           # NEW: WorkerTypeRegistry
│   ├── worker_orchestrator.py       # NEW: WorkerOrchestrator
│   └── trainer/                     # REFACTOR: Use orchestrator
│
└── workers/
    ├── __init__.py
    ├── registry.py                  # Worker type definitions
    ├── capabilities.py              # WorkerCapabilities
    └── protocols.py                 # PolicyController protocol
```

---

## 8. Backwards Compatibility

### 8.1 Deprecation Strategy

```python
# In gym_gui/services/actor.py

import warnings

class ActorService:
    """DEPRECATED: Use PolicyMappingService instead."""

    def __init__(self):
        warnings.warn(
            "ActorService is deprecated. Use PolicyMappingService instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._delegate = PolicyMappingService()

    def select_action(self, snapshot: StepSnapshot) -> Optional[int]:
        # Adapt old interface to new
        return self._delegate.select_action(
            agent_id="agent_0",  # Legacy single-agent assumption
            observation=snapshot.observation,
        )
```

### 8.2 Migration Examples

**Before (PettingZoo-coupled):**

```python
actor_service = ActorService()
actor_service.register_actor(HumanKeyboardActor())
action = actor_service.select_action(snapshot)
```

**After (Paradigm-agnostic):**

```python
policy_service = PolicyMappingService()
policy_service.set_policy("player_0", HumanPolicyController())
policy_service.set_policy("player_1", CleanRLPolicyController())

# Sequential (PettingZoo)
action = policy_service.select_action("player_0", observation)

# Simultaneous (RLlib)
actions = policy_service.select_actions({"player_0": obs0, "player_1": obs1})
```

---

## 9. Success Criteria

### 9.1 Technical Metrics

- [ ] All existing workers continue to function
- [ ] New RLlib worker can be added without modifying core code
- [ ] Policy mapping works for both AEC and POSG paradigms
- [ ] Pyright passes with 0 errors on new code
- [ ] Unit test coverage > 80% for new modules

### 9.2 Architectural Goals

- [ ] No paradigm-specific code in GUI layer
- [ ] Workers are interchangeable via registry
- [ ] Adding new paradigm requires only new adapter
- [ ] Clear separation of concerns

---

## 10. References

### Internal Documents

1. [02_naming_resolution_plan.md](./02_naming_resolution_plan.md) - Naming conventions and directory structure
2. [01_paradigm_comparison.md](./01_paradigm_comparison.md) - POSG vs AEC comparison
3. [DAY_38 Analysis](../1.0_DAY_38/TASK_1/) - Architecture comparison
4. [DAY_40 Vision](../1.0_DAY_40/TASK_1/) - Publication roadmap

### External Papers

5. [PettingZoo Paper](https://arxiv.org/abs/2009.14471) - AEC formalism
6. [RLlib Paper](https://arxiv.org/abs/1712.09381) - POSG/distributed RL
