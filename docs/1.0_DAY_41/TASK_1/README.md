# DAY 41 - TASK 1: Multi-Paradigm Orchestrator Architecture

## Problem Statement

Mosaic's current architecture is **tightly coupled to PettingZoo's AEC paradigm**, forcing all workers to conform to sequential stepping. This is fundamentally wrong for a distributed platform that must support:

- **CleanRL** (single-agent, Gymnasium)
- **RLlib** (multi-agent, POSG/simultaneous)
- **PettingZoo** (multi-agent, AEC/sequential + parallel)
- **Jason/BDI** (goal-driven, sequential)
- **Future workers** (LLM, VLM, Unreal, AirSim)

> **Note:** MuJoCo MPC is NOT an RL system - it uses optimization-based planning (iLQG, Cross Entropy).
> It is managed separately by `gym_gui/services/mujoco_mpc_controller/`.

## Documents

| # | Document | Description | Status |
|---|----------|-------------|--------|
| 00 | [Multi-Paradigm Orchestrator Plan](./00_multi_paradigm_orchestrator_plan.md) | Complete architectural plan with migration steps | ✅ Phase 1 Done |
| 01 | [Paradigm Comparison](./01_paradigm_comparison.md) | POSG vs AEC vs EFG comparison | ✅ Complete |
| 02 | [Naming Resolution Plan](./02_naming_resolution_plan.md) | Directory naming conventions | ✅ Complete |
| 03 | [PolicyMappingService Plan](./03_policy_mapping_service_plan.md) | Phase 2 architecture with worker ecosystem | ✅ Phase 2.3 Done |
| 04 | [Worker Requirements](./04_worker_requirements.md) | Worker dependency management and installation | ✅ Complete |

> **See Also:** [TASK_2: Cognitive Orchestration Layer](../TASK_2/README.md) for LLM/VLM integration (separate from stepping paradigms)

## Progress Tracking

### Prerequisites ✅

- [x] **Naming Resolution**: `ui/environments/` → `ui/config_panels/`
- [x] **Import Updates**: All imports updated to new paths
- [x] **Pyright Verification**: No import-related errors
- [x] **Pytest Verification**: 82 tests passed

### Phase 1: Abstraction Layer ✅

- [x] **SteppingParadigm enum**: `gym_gui/core/enums.py` (SINGLE_AGENT, SEQUENTIAL, SIMULTANEOUS, HIERARCHICAL)
- [x] **WorkerCapabilities dataclass**: `gym_gui/core/adapters/base.py`
- [x] **PolicyController protocol**: `gym_gui/services/actor.py`
- [x] **ParadigmAdapter ABC**: `gym_gui/core/adapters/paradigm.py` (new file)
- [x] **Paradigm field on adapters**: Added to EnvironmentAdapter and PettingZooAdapter

### Phase 2: PolicyMappingService 🔄

#### Phase 2.1: Core Service ✅
- [x] **AgentPolicyBinding dataclass**: `gym_gui/services/policy_mapping.py`
- [x] **PolicyMappingService class**: Per-agent policy mapping with paradigm awareness
- [x] **Bootstrap integration**: Registered in `bootstrap.py`
- [x] **Module exports**: Added to `services/__init__.py`
- [x] **Pyright verification**: 0 errors on new code
- [x] **Pytest verification**: 396 tests passed

#### Phase 2.2: SessionController Integration ✅
- [x] Update `_select_agent_action()` to use PolicyMappingService
- [x] Update `_record_step()` for per-agent notification
- [x] Update `_finalize_episode()` for per-agent cleanup
- [x] Add `_get_active_agent()` helper for paradigm detection

#### Phase 2.3: UI Components ✅
- [x] Create PolicyMappingPanel widget → `AgentConfigTable` in AdvancedConfigTab
- [x] Update ControlPanelContainer → `AdvancedConfigTab` integrated
- [x] MainWindow signal handling → `_on_advanced_launch()` handles LaunchConfig
- [ ] Add preset save/load functionality (future enhancement)

### Phase 2.4: Worker Requirements ✅

- [x] Create `requirements/ray_worker.txt` - Ray/RLlib dependencies
- [x] Create `requirements/xuance_worker.txt` - XuanCe MARL dependencies
- [x] Create `3rd_party/ray_worker/pyproject.toml` - Package configuration
- [x] Create `3rd_party/xuance_worker/pyproject.toml` - Package configuration
- [x] Create `ray_worker/__init__.py` - Module initialization
- [x] Create `xuance_worker/__init__.py` - Module initialization
- [x] Update `pyproject.toml` with `ray-rllib` and `xuance` optional dependencies

### Phase 3: Worker Registry 📋

- [ ] Worker capability matching
- [ ] Paradigm compatibility checks

## Key Findings

### The Two Stepping Paradigms

```
POSG/Parallel (Simultaneous)          AEC (Sequential)
════════════════════════════          ════════════════
env.step(Dict[Agent, Action])         env.step(single_action)
All agents act simultaneously         Agents act one at a time
Dict of rewards returned              Per-agent reward returned
Race conditions possible              Race conditions impossible

Used by:                              Used by:
- RLlib                               - PettingZoo env()
- PettingZoo parallel_env()           - OpenSpiel (via Shimmy)
- XuanCe                              - Turn-based games
```

**Important:** PettingZoo supports BOTH paradigms:
- `env()` → AEC (sequential)
- `parallel_env()` → POSG (simultaneous)
- Conversion wrappers: `aec_to_parallel()`, `parallel_to_aec()`

### Current Problems

1. `ActorService` has single `_active_actor_id` (not per-agent)
2. `StepSnapshot` designed for single-agent (no `agent_id`)
3. `TrainerService` coupled to CleanRL implementation
4. No abstraction for stepping paradigms

## Proposed Solution

### New Core Abstractions

```python
class SteppingParadigm(Enum):
    """RL stepping paradigms ONLY. Non-RL systems (MuJoCo MPC) managed separately."""
    SINGLE_AGENT = auto()    # Gymnasium
    SIMULTANEOUS = auto()     # RLlib/POSG/PettingZoo Parallel
    SEQUENTIAL = auto()       # PettingZoo AEC/OpenSpiel
    HIERARCHICAL = auto()     # BDI

class PolicyMappingService:
    """Replaces ActorService with per-agent policy mapping."""

    def select_action(self, agent_id, obs) -> Action:
        """Sequential mode (AEC)"""

    def select_actions(self, observations) -> Dict[Agent, Action]:
        """Simultaneous mode (POSG)"""

class WorkerOrchestrator:
    """Paradigm-agnostic worker management."""

    async def submit_run(self, config) -> run_id:
        paradigm = self._infer_paradigm(config)
        worker = self._select_worker(paradigm)
        adapter = self._create_adapter(paradigm)
        return await self._dispatcher.launch(worker, adapter)
```

## Migration Timeline

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Abstraction layer (SteppingParadigm, WorkerCapabilities, ParadigmAdapter) | ✅ Complete |
| 2.1 | PolicyMappingService Core (AgentPolicyBinding, PolicyMappingService) | ✅ Complete |
| 2.2 | SessionController Integration | ✅ Complete |
| 2.3 | UI Components (AgentConfigTable in AdvancedConfigTab) | ✅ Complete |
| 2.4 | Worker Requirements (ray_worker, xuance_worker) | ✅ Complete |
| 3 | Worker Registry | 📋 Planned |
| 4 | Concrete Paradigm Adapters | 📋 Planned |
| 5 | WorkerOrchestrator | 📋 Planned |

> **Note:** Cognitive Orchestration Layer (LLM/VLM) is tracked separately in [TASK_2](../TASK_2/README.md)

## Success Criteria

### Prerequisites (Completed)
- [x] Directory naming resolved (`ui/config_panels/`)
- [x] Pyright passes with 0 import-related errors
- [x] All tests pass (78 passed)

### Future Implementation
- [ ] All existing workers continue to function
- [ ] RLlib worker can be added without modifying core code
- [ ] Policy mapping works for both AEC and POSG
- [ ] No paradigm-specific code in GUI layer
- [ ] Pyright passes with 0 errors on new code

## Related Documents

- [DAY 38: Ray Architecture Analysis](../1.0_DAY_38/TASK_1/)
- [DAY 40: Publication Roadmap](../1.0_DAY_40/TASK_1/)
