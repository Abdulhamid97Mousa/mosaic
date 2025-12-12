# Phase 2: PolicyMappingService Architecture

## Related Documents

| Document | Description |
|----------|-------------|
| [00_multi_paradigm_orchestrator_plan.md](./00_multi_paradigm_orchestrator_plan.md) | Overall architecture |
| [01_paradigm_comparison.md](./01_paradigm_comparison.md) | POSG vs AEC paradigms |
| [TASK_2: Cognitive Orchestration](../TASK_2/README.md) | LLM/VLM integration (separate) |

---

## 1. System Architecture

### 1.1 Full Worker Ecosystem

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MOSAIC PLATFORM                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         GUI LAYER (PyQt6)                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │ │
│  │  │ MainWindow   │  │ ControlPanel │  │ PolicyMapping│  │ RenderTabs     │  │ │
│  │  │              │  │ Container    │  │ Panel (NEW)  │  │                │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                        │
│                                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       SERVICE LAYER                                         │ │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌────────────┐  │ │
│  │  │ ActorService │  │PolicyMappingServ │  │ TrainerServ  │  │ SeedMgr    │  │ │
│  │  │ (Legacy)     │◄─┤    (NEW)         │  │              │  │            │  │ │
│  │  └──────────────┘  └──────────────────┘  └──────────────┘  └────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                        │
│                                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                      CONTROLLER LAYER                                       │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      SessionController                                │  │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │ │
│  │  │  │_select_action() │  │ _record_step()  │  │_finalize_episode()  │   │  │ │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                        │
│                                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       ADAPTER LAYER                                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      ParadigmAdapter                                  │  │ │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │  │ │
│  │  │  │SingleAgent   │  │Sequential    │  │Simultaneous  │  │Hierarch. │  │  │ │
│  │  │  │Adapter       │  │Adapter (AEC) │  │Adapter (POSG)│  │Adapter   │  │  │ │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘  │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                        │
│                              gRPC / IPC │                                        │
│                                         ▼                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           3RD_PARTY WORKERS                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    REINFORCEMENT LEARNING WORKERS                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │    │
│  │  │ cleanrl_    │  │ xuance_     │  │ ray_        │  │ pettingzoo_     │ │    │
│  │  │ worker      │  │ worker      │  │ worker      │  │ worker          │ │    │
│  │  │ ──────────  │  │ ──────────  │  │ ──────────  │  │ ──────────────  │ │    │
│  │  │ PPO, DQN    │  │ MAPPO, QMIX │  │ RLlib       │  │ AEC/Parallel    │ │    │
│  │  │ SAC, TD3    │  │ MADDPG      │  │ Multi-Agent │  │ Wrappers        │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    COGNITIVE / BDI WORKERS                               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │    │
│  │  │ jason_      │  │ spade_bdi_  │  │ llm_        │  │ vlm_            │ │    │
│  │  │ worker      │  │ worker      │  │ worker      │  │ worker          │ │    │
│  │  │ ──────────  │  │ ──────────  │  │ ──────────  │  │ ──────────────  │ │    │
│  │  │ AgentSpeak  │  │ SPADE BDI   │  │ Ollama      │  │ CLIP/LLaVA      │ │    │
│  │  │ Java Bridge │  │ Python      │  │ LangChain   │  │ Vision Models   │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    SIMULATION / CONTROL WORKERS                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │    │
│  │  │ mujoco_mpc_ │  │ vizdoom_    │  │ airsim_     │  │ unreal_map_     │ │    │
│  │  │ worker      │  │ worker      │  │ worker      │  │ worker          │ │    │
│  │  │ ──────────  │  │ ──────────  │  │ ──────────  │  │ ──────────────  │ │    │
│  │  │ MPC Control │  │ FPS Games   │  │ Drones/Cars │  │ Unreal Engine   │ │    │
│  │  │ iLQG, CEM   │  │ Doom Envs   │  │ Simulation  │  │ Maps            │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    UTILITY WORKERS                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │    │
│  │  │ shimmy_     │  │ supersuit_  │  │ test_       │                      │    │
│  │  │ worker      │  │ worker      │  │ worker      │                      │    │
│  │  │ ──────────  │  │ ──────────  │  │ ──────────  │                      │    │
│  │  │ API Compat  │  │ Wrappers    │  │ Testing     │                      │    │
│  │  │ OpenSpiel   │  │ Frame Stack │  │ Mocks       │                      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Worker Capabilities Matrix

| Worker | Paradigm | Multi-Agent | Action Space | Env Types |
|--------|----------|-------------|--------------|-----------|
| `cleanrl_worker` | SINGLE_AGENT | No | Discrete, Continuous | Gymnasium |
| `xuance_worker` | SINGLE_AGENT, SIMULTANEOUS | Yes (MARL) | Discrete, Continuous | Gymnasium, PettingZoo |
| `ray_worker` | SIMULTANEOUS | Yes (RLlib) | All | Gymnasium, Multi-Agent |
| `pettingzoo_worker` | SEQUENTIAL, SIMULTANEOUS | Yes | Discrete | PettingZoo |
| `jason_worker` | HIERARCHICAL | Yes (BDI) | Discrete | Custom |
| `spade_bdi_worker` | HIERARCHICAL | Yes (BDI) | Discrete | Custom |
| `llm_worker` | HIERARCHICAL | Yes | Discrete, Text | PettingZoo, Custom |
| `vlm_worker` | HIERARCHICAL | Yes | Discrete | Vision-based |
| `mujoco_mpc_worker` | N/A (MPC) | No | Continuous | MuJoCo |
| `vizdoom_worker` | SINGLE_AGENT | No | Multi-Binary | ViZDoom |
| `airsim_worker` | SINGLE_AGENT | No | Continuous | AirSim |

---

## 2. PolicyMappingService Design

### 2.1 Core Interface

```python
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from gym_gui.core.enums import SteppingParadigm
from gym_gui.services.actor import Actor, ActorService, StepSnapshot, EpisodeSummary


@dataclass
class AgentPolicyBinding:
    """Binding between an agent and its policy controller."""

    agent_id: str
    policy_id: str  # References an Actor in ActorService
    worker_id: Optional[str] = None  # e.g., "cleanrl_worker", "llm_worker"
    config: Dict[str, Any] = field(default_factory=dict)


class PolicyMappingService:
    """Per-agent policy mapping for multi-agent environments.

    Extends ActorService to support:
    1. Multiple active policies (one per agent)
    2. Paradigm-aware action selection
    3. Worker-specific routing

    For single-agent environments, delegates to ActorService.
    For multi-agent, maintains agent_id → policy_id mapping.
    """

    def __init__(self, actor_service: ActorService) -> None:
        self._actor_service = actor_service
        self._bindings: Dict[str, AgentPolicyBinding] = {}
        self._paradigm: SteppingParadigm = SteppingParadigm.SINGLE_AGENT
        self._agent_ids: List[str] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_paradigm(self, paradigm: SteppingParadigm) -> None:
        """Set the stepping paradigm for this session."""
        self._paradigm = paradigm

    def set_agents(self, agent_ids: List[str]) -> None:
        """Configure the list of agents in the environment."""
        self._agent_ids = list(agent_ids)
        # Auto-bind to default policy if not already bound
        default_policy = self._actor_service.get_active_actor_id()
        for agent_id in agent_ids:
            if agent_id not in self._bindings and default_policy:
                self._bindings[agent_id] = AgentPolicyBinding(
                    agent_id=agent_id,
                    policy_id=default_policy,
                )

    def bind_agent_policy(
        self,
        agent_id: str,
        policy_id: str,
        *,
        worker_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Bind an agent to a specific policy."""
        if policy_id not in list(self._actor_service.available_actor_ids()):
            raise KeyError(f"Unknown policy '{policy_id}'")

        self._bindings[agent_id] = AgentPolicyBinding(
            agent_id=agent_id,
            policy_id=policy_id,
            worker_id=worker_id,
            config=config or {},
        )

    def get_binding(self, agent_id: str) -> Optional[AgentPolicyBinding]:
        """Get the policy binding for an agent."""
        return self._bindings.get(agent_id)

    def get_all_bindings(self) -> Dict[str, AgentPolicyBinding]:
        """Get all agent-policy bindings."""
        return dict(self._bindings)

    # ------------------------------------------------------------------
    # Action Selection (Paradigm-Aware)
    # ------------------------------------------------------------------

    def select_action(
        self,
        agent_id: str,
        snapshot: StepSnapshot,
    ) -> Optional[int]:
        """Select action for a specific agent (Sequential/AEC mode)."""
        binding = self._bindings.get(agent_id)
        if binding is None:
            # Fallback to legacy ActorService
            return self._actor_service.select_action(snapshot)

        # Get the actor for this agent's policy
        actor = self._actor_service._actors.get(binding.policy_id)
        if actor is None:
            return None

        return actor.select_action(snapshot)

    def select_actions(
        self,
        observations: Dict[str, Any],
        snapshots: Dict[str, StepSnapshot],
    ) -> Dict[str, Optional[int]]:
        """Select actions for all agents (Simultaneous/POSG mode)."""
        actions: Dict[str, Optional[int]] = {}

        for agent_id, snapshot in snapshots.items():
            actions[agent_id] = self.select_action(agent_id, snapshot)

        return actions

    # ------------------------------------------------------------------
    # Step Notification
    # ------------------------------------------------------------------

    def notify_step(
        self,
        agent_id: str,
        snapshot: StepSnapshot,
    ) -> None:
        """Notify the appropriate policy of a step result."""
        binding = self._bindings.get(agent_id)
        if binding is None:
            self._actor_service.notify_step(snapshot)
            return

        actor = self._actor_service._actors.get(binding.policy_id)
        if actor is not None:
            actor.on_step(snapshot)

    def notify_episode_end(
        self,
        agent_id: str,
        summary: EpisodeSummary,
    ) -> None:
        """Notify the appropriate policy of episode end."""
        binding = self._bindings.get(agent_id)
        if binding is None:
            self._actor_service.notify_episode_end(summary)
            return

        actor = self._actor_service._actors.get(binding.policy_id)
        if actor is not None:
            actor.on_episode_end(summary)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_multi_agent(self) -> bool:
        """Check if we're in multi-agent mode."""
        return len(self._agent_ids) > 1

    @property
    def paradigm(self) -> SteppingParadigm:
        return self._paradigm

    @property
    def agent_ids(self) -> List[str]:
        return list(self._agent_ids)
```

### 2.2 Integration with SessionController

```python
# In SessionController._select_agent_action()

def _select_agent_action(self) -> Optional[int]:
    """Select action using PolicyMappingService or ActorService."""

    if self._last_step is None:
        return None

    # Extract agent_id from step
    agent_id = self._last_step.agent_id or getattr(
        self._last_step.state, "active_agent", None
    )

    snapshot = self._build_step_snapshot(self._last_step)

    # Multi-agent path: use PolicyMappingService
    if self._policy_mapping is not None and agent_id is not None:
        try:
            return self._policy_mapping.select_action(agent_id, snapshot)
        except Exception as exc:
            self._logger.error(f"Policy selection failed for {agent_id}: {exc}")
            return None

    # Single-agent fallback: use legacy ActorService
    if self._actor_service is not None:
        try:
            return self._actor_service.select_action(snapshot)
        except Exception as exc:
            self._logger.error(f"Actor selection failed: {exc}")
            return None

    return None
```

---

## 3. Requirements Structure

### 3.1 Current Layout

```
requirements/
├── base.txt                 # Core GUI + shared libs (grpcio, gymnasium, PyQt6)
├── cleanrl_worker.txt       # CleanRL: torch, tensorboard, wandb
├── xuance_worker.txt        # XuanCe: MARL algorithms (TO CREATE)
├── ray_worker.txt           # Ray/RLlib (TO CREATE)
├── pettingzoo.txt           # PettingZoo environments
├── jason_worker.txt         # Jason BDI: Java bridge
├── spade_bdi_worker.txt     # SPADE BDI: Python
├── llm_worker.txt           # LLM: ollama, langchain
├── vlm_worker.txt           # VLM: CLIP, transformers (TO CREATE)
├── mujoco_mpc_worker.txt    # MuJoCo MPC: mujoco, dm_control
├── vizdoom.txt              # ViZDoom environments
└── airsim_worker.txt        # AirSim simulation
```

### 3.2 Worker → Requirements Mapping

| Worker Directory | Requirements File | Key Dependencies |
|------------------|-------------------|------------------|
| `3rd_party/cleanrl_worker/` | `cleanrl_worker.txt` | torch, tensorboard, wandb |
| `3rd_party/xuance_worker/` | (needs creation) | xuance, torch |
| `3rd_party/ray_worker/` | (needs creation) | ray[rllib] |
| `3rd_party/pettingzoo_worker/` | `pettingzoo.txt` | pettingzoo, supersuit |
| `3rd_party/jason_worker/` | `jason_worker.txt` | grpcio (Java bridge) |
| `3rd_party/spade_bdi_worker/` | `spade_bdi_worker.txt` | spade-bdi |
| `3rd_party/llm_worker/` | `llm_worker.txt` | ollama, langchain |
| `3rd_party/vlm_worker/` | (needs creation) | transformers, CLIP |
| `3rd_party/mujoco_mpc_worker/` | `mujoco_mpc_worker.txt` | mujoco, dm_control |
| `3rd_party/vizdoom_worker/` | `vizdoom.txt` | vizdoom |
| `3rd_party/airsim_worker/` | `airsim_worker.txt` | airsim |
| `3rd_party/shimmy_worker/` | (needs creation) | shimmy |
| `3rd_party/supersuit_worker/` | (needs creation) | supersuit |

---

## 4. UI Changes

### 4.1 PolicyMappingPanel (New Widget)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Agent Policy Configuration                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Environment: chess_v6 (PettingZoo Classic)                                 │
│  Paradigm: SEQUENTIAL (AEC)                                                 │
│  Agents: 2                                                                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Agent              Policy                    Worker                   │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  player_0           [Human Keyboard    ▼]    [Local         ▼]        │ │
│  │  player_1           [CleanRL Worker    ▼]    [cleanrl_worker ▼]       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Available Policies:                                                   │ │
│  │    • Human (Keyboard) - Manual control via keyboard                   │ │
│  │    • CleanRL Worker - Trained RL policy                               │ │
│  │    • LLM Agent - Language model decision making                       │ │
│  │    • Random Policy - Uniform random actions                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  [ Apply Configuration ]  [ Reset to Defaults ]  [ Save Preset ]            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Signal Flow

```
PolicyMappingPanel
       │
       │ policy_binding_changed(agent_id: str, policy_id: str)
       ▼
MainWindow._on_policy_binding_changed()
       │
       │ policy_mapping_service.bind_agent_policy(agent_id, policy_id)
       ▼
PolicyMappingService._bindings[agent_id] = binding
       │
       ▼
SessionController (uses bindings during step)
```

---

## 5. Implementation Phases

### Phase 2.1: PolicyMappingService Core ✅

- [x] Create `gym_gui/services/policy_mapping.py`
- [x] Define `AgentPolicyBinding` dataclass
- [x] Implement `PolicyMappingService` class
- [x] Add to `bootstrap.py` registration
- [x] Update `services/__init__.py` exports

### Phase 2.2: SessionController Integration ✅

- [x] Update `_select_agent_action()` to use PolicyMappingService
- [x] Update `_record_step()` for per-agent notification
- [x] Update `_finalize_episode()` for per-agent cleanup
- [x] Add paradigm detection from adapter (via `_get_active_agent()` helper)

### Phase 2.3: UI Components ✅

- [x] Create `PolicyMappingPanel` widget → Implemented as `AgentConfigTable` in AdvancedConfigTab
- [x] Update `ControlPanelContainer` to include new panel → `AdvancedConfigTab` integrated
- [x] Update `MainWindow` signal handling → `_on_advanced_launch()` handles LaunchConfig
- [ ] Add preset save/load functionality (future enhancement)

### Phase 2.4: Worker Integration

- [ ] Create missing requirements files (xuance, ray, vlm, shimmy, supersuit)
- [ ] Document worker installation procedures
- [ ] Add worker capability registration

---

## 6. Backward Compatibility

### 6.1 Single-Agent Mode (Unchanged)

```python
# SessionController continues to work exactly as before for single-agent
if not self._policy_mapping.is_multi_agent():
    # Delegates to ActorService
    action = self._actor_service.select_action(snapshot)
```

### 6.2 Legacy ActorService (Preserved)

```python
# ActorService remains unchanged
# PolicyMappingService wraps it, doesn't replace it
class PolicyMappingService:
    def __init__(self, actor_service: ActorService):
        self._actor_service = actor_service  # Composition, not inheritance
```

### 6.3 UI Compatibility

- Single-agent environments: Show legacy actor dropdown
- Multi-agent environments: Show PolicyMappingPanel
- Detection: Based on adapter's `num_agents` or `agent_ids`

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# test_policy_mapping_service.py

def test_single_agent_fallback():
    """Single agent should delegate to ActorService."""
    actor_service = ActorService()
    actor_service.register_actor(MockActor(id="test"), activate=True)

    mapping = PolicyMappingService(actor_service)
    mapping.set_agents(["agent_0"])

    snapshot = StepSnapshot(step_index=0, observation=None, reward=0, ...)
    action = mapping.select_action("agent_0", snapshot)

    assert action is not None


def test_multi_agent_binding():
    """Multiple agents should use their bound policies."""
    actor_service = ActorService()
    actor_service.register_actor(MockActor(id="policy_a"), activate=True)
    actor_service.register_actor(MockActor(id="policy_b"))

    mapping = PolicyMappingService(actor_service)
    mapping.set_agents(["player_0", "player_1"])
    mapping.bind_agent_policy("player_0", "policy_a")
    mapping.bind_agent_policy("player_1", "policy_b")

    assert mapping.get_binding("player_0").policy_id == "policy_a"
    assert mapping.get_binding("player_1").policy_id == "policy_b"
```

### 7.2 Integration Tests

- Test with PettingZoo Chess (AEC/Sequential)
- Test with PettingZoo MPE (Parallel/Simultaneous)
- Test mixed human + AI agents
