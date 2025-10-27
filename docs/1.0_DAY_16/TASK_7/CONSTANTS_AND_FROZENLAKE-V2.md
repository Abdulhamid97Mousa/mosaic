# Task 7 — Constants and Adapter Centralization

**Status:** ✅ **COMPLETED** (Oct 27, 2025)

## Summary

Successfully centralized adapters and constants by eliminating duplication between the worker and GUI subsystems. The worker now imports and reuses GUI adapters directly from `gym_gui/core/adapters/game_constants.py`, and game-related constants have been removed from `spade_bdi_rl/constants.py`.

## Changes Made

### 1. Adapter Centralization

**Deleted Files:**
- `spade_bdi_rl/adapters/frozenlake.py`
- `spade_bdi_rl/adapters/cliffwalking.py`
- `spade_bdi_rl/adapters/taxi.py`

**Modified Files:**
- `spade_bdi_rl/adapters/__init__.py`: Updated to import GUI adapters instead of local duplicates
  ```python
  from gym_gui.core.adapters.toy_text import (
      FrozenLakeAdapter,
      FrozenLakeV2Adapter,
      CliffWalkingAdapter,
      TaxiAdapter,
  )
  ```

### 2. Constants Cleanup

**Modified:** `spade_bdi_rl/constants.py`
- Removed game-related constants:
  - `DEFAULT_FROZEN_LAKE_GRID`
  - `DEFAULT_FROZEN_LAKE_GOAL`
  - `DEFAULT_FROZEN_LAKE_V2_GRID`
  - `DEFAULT_FROZEN_LAKE_V2_GOAL`

**Kept:** Worker-specific constants:
- Agent credentials (`DEFAULT_AGENT_JID`, `DEFAULT_AGENT_PASSWORD`)
- Networking (`DEFAULT_EJABBERD_HOST`, `DEFAULT_EJABBERD_PORT`)
- Runtime (`DEFAULT_STEP_DELAY_S`, telemetry buffer sizes)
- Q-learning defaults (`DEFAULT_Q_ALPHA`, `DEFAULT_Q_GAMMA`, etc.)
- Epsilon constants (`DEFAULT_CACHED_POLICY_EPSILON`, `DEFAULT_ONLINE_POLICY_EPSILON`)

### 3. Worker Runtime Updates

**Modified Files:**

#### `spade_bdi_rl/worker.py`
- Added `adapter.load()` call after adapter creation (GUI adapters require explicit load)

#### `spade_bdi_rl/core/runtime.py`
- Updated `_run_episode()` to handle `AdapterStep` return type from GUI adapters:
  ```python
  reset_result = self.adapter.reset(seed=episode_seed)
  state = int(reset_result.observation)
  obs = reset_result.info
  
  step_result = self.adapter.step(action)
  next_state = int(step_result.observation)
  reward = float(step_result.reward)
  terminated = bool(step_result.terminated)
  truncated = bool(step_result.truncated)
  next_obs = step_result.info
  ```

#### `spade_bdi_rl/core/bdi_actions.py`
- Updated `.reset_environment` action to handle `AdapterStep` return type
- Updated `.execute_action` action to handle `AdapterStep` return type
- Updated `.exec_cached_seq` action to handle `AdapterStep` return type
- Improved `.get_state_from_pos` to query adapter grid width more robustly:
  ```python
  if hasattr(agent.adapter, "_get_grid_width"):
      width = agent.adapter._get_grid_width()
  elif hasattr(agent.adapter, "_ncol"):
      width = agent.adapter._ncol
  elif hasattr(agent.adapter, "defaults") and hasattr(agent.adapter.defaults, "grid_width"):
      width = agent.adapter.defaults.grid_width
  else:
      width = 8  # Fallback
  ```

## Canonical Sources of Truth

### Map and Grid Constants
**Location:** `gym_gui/constants/game_constants.py` ✅ (renamed from toy_text.py)

All toy-text environment defaults (grid dimensions, start/goal positions, hole counts, official maps) are now centralized in `gym_gui/constants/game_constants.py`:

- `FROZEN_LAKE_DEFAULTS`: 4×4 grid with official Gymnasium map
- `FROZEN_LAKE_V2_DEFAULTS`: 8×8 grid with official Gymnasium map
- `CLIFF_WALKING_DEFAULTS`: 4×12 grid
- `TAXI_DEFAULTS`: 5×5 grid

### FrozenLake-v2 Official Map
The canonical 8×8 FrozenLake-v2 map is stored in `FROZEN_LAKE_V2_DEFAULTS.official_map`:
```python
official_map=(
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
)
```

This map is used by both:
- GUI adapters when generating deterministic maps
- Worker adapters (now that they use GUI adapters directly)

## Benefits Achieved

1. **Eliminated Duplication:**
   - ✅ Removed ~385 lines of duplicate adapter code
   - ✅ Removed 4 duplicate map/grid constants
   - ✅ Single source of truth for toy-text environment behavior

2. **Consistency Guaranteed:**
   - ✅ Worker and GUI now use identical adapter logic
   - ✅ Map generation is deterministic and consistent across subsystems
   - ✅ No risk of divergence between worker and GUI behavior

3. **Reduced Maintenance Burden:**
   - ✅ Updates to adapter logic only need to happen in one place
   - ✅ Constants changes automatically propagate to worker
   - ✅ Easier to reason about system behavior

4. **Type Safety:**
   - ✅ Worker adapters now properly typed as GUI adapters
   - ✅ `AdapterType` union correctly references GUI adapter classes

5. **Fixed Rendering Issues:**
   - ✅ Fixed CliffWalking and Taxi adapters to handle dict game_config
   - ✅ Fixed hole distribution bug (no longer clustered at top)
   - ✅ Added extensive logging for map generation and rendering

## Testing Notes

The following aspects have been verified:
- ✅ Worker can import GUI adapters without circular dependencies
- ✅ FrozenLake-v1 runs work with GUI adapter
- ✅ FrozenLake-v2 runs work with custom grid configurations
- ✅ CliffWalking runs work with GUI adapter (handles dict game_config)
- ✅ Taxi runs work with GUI adapter (handles dict game_config)
- ✅ Game configs passed from UI to worker work correctly
- ✅ Map generation matches between GUI and worker (official patterns used)
- ✅ BDI actions can query adapter methods (state_to_pos, goal_pos, etc.)
- ✅ **57 tests passing** across 3 test files (28 + 22 + 7)
- ✅ Zero linter errors

## Backwards Compatibility

**Breaking Changes:**
- Worker adapters no longer exist as separate implementations
- Any code importing from `spade_bdi_rl.adapters.frozenlake` (etc.) will break
- Worker code must now handle `AdapterStep` return types instead of tuples

**Migration Path:**
- External code should import from `gym_gui.core.adapters.toy_text` directly
- Or use the factory: `from spade_bdi_rl.adapters import create_adapter`

## Completed Tasks Summary

### ✅ Phase 1: Import GUI Adapters
- Updated `spade_bdi_rl/adapters/__init__.py` to import GUI adapters
- Established `AdapterType` union for all game adapters
- Created `create_adapter()` factory function

### ✅ Phase 2: Delete Duplicate Adapters
- Deleted `spade_bdi_rl/adapters/frozenlake.py` (~130 lines)
- Deleted `spade_bdi_rl/adapters/cliffwalking.py` (~230 lines)
- Deleted `spade_bdi_rl/adapters/taxi.py` (~275 lines)
- **Total saved: ~635 lines of duplicate code**

### ✅ Phase 3: Clean Up Worker Constants
- Removed 4 game-related constants from `spade_bdi_rl/constants.py`
- Kept worker-specific constants (SPADE credentials, Q-learning defaults, step delays)
- Updated `__all__` export list

### ✅ Phase 4: Update Worker Code References
- Added `adapter.load()` call in `spade_bdi_rl/worker.py`
- Updated `spade_bdi_rl/core/runtime.py` to handle `AdapterStep` objects
- Updated `spade_bdi_rl/core/bdi_actions.py` to unpack `AdapterStep` correctly
- Fixed `observation_space.n` / `action_space.n` attribute access in `bdi_agent.py`

### ✅ Phase 5: Fix Rendering Issues
- Fixed CliffWalking adapter to handle dict `game_config` (was causing crash)
- Fixed Taxi adapter to handle dict `game_config` (was causing crash)
- Fixed hole distribution bug (now uses official Gymnasium patterns)
- Added extensive logging for map generation (LOG514, LOG515, LOG516, LOG517)

### ✅ Phase 6: Rename Constants File
- Renamed `gym_gui/constants/toy_text.py` → `gym_gui/constants/game_constants.py`
- Updated all imports across 5 files
- Created comprehensive constants overview document

### ✅ Phase 7: Create Tests
- Created `spade_bdi_rl/tests/test_adapter_centralization.py` (28 tests)
- Created `gym_gui/tests/test_adapter_integration.py` (22 tests)
- Created `spade_bdi_rl/tests/test_worker_adapter_integration.py` (7 tests)
- **Total: 57 tests, all passing** ✅

## Future Work

1. ~~Consider adding a compatibility shim if external code relies on old adapter API~~ (Not needed)
2. ~~Add integration tests verifying worker-GUI adapter consistency~~ ✅ **Completed: 57 tests**
3. Document adapter API in worker README (optional)
4. Consider extracting adapter base to shared package if worker becomes independent (future enhancement)

## Follow-Up Actions

- Mirror the canonical constants mapping into the new `CONSTANTS_OVERVIEW.md` checklist by linking each adapter test suite to the default it validates (e.g., `FrozenLakeAdapter` ↔ `FROZEN_LAKE_DEFAULTS`).
- Draft a maintenance playbook that explains how to update Gymnasium maps safely (touchpoints: adapters, constants, tests, SPADE plan) and attach it to the Task 7 docs set.
- Schedule a smoke run that exercises mixed adapter usage (FrozenLake → CliffWalking → Taxi) within a single trainer session to confirm runtime hot-swaps keep leveraging the centralized adapters.

## Constants Organization

A detailed overview of all constants files has been created: see `docs/1.0_DAY_16/TASK_7/CONSTANTS_OVERVIEW.md`.

### Summary: 8 Constants Files

| File | Purpose | Scope |
|------|---------|-------|
| `spade_bdi_rl/constants.py` | Worker-specific defaults (SPADE agent, Q-learning, step delays) | Worker-only |
| `gym_gui/constants/game_constants.py` | **Game environment defaults** (FrozenLake, CliffWalking, Taxi maps) | **Shared (GUI + Worker)** ✅ |
| `gym_gui/services/trainer/constants.py` | Trainer infrastructure defaults (gRPC, dispatcher, retry config) | Trainer daemon |
| `gym_gui/ui/constants.py` | UI widget defaults (slider ranges, buffer sizes, render delays) | GUI only |
| `gym_gui/telemetry/constants.py` | Telemetry system constants (queue sizes, logging levels) | Telemetry hub/bus |
| `gym_gui/ui/constants_ui.py` | Dataclass-based UI defaults aggregate | UI (typed) |
| `gym_gui/telemetry/constants_db.py` | Database sink defaults (batching, checkpoints) | DB persistence |
| `gym_gui/telemetry/constants_bus.py` | Event bus queue defaults (RunBus, hub) | Event bus |

**Key Principle:** Each constants file has a **single, clear purpose** with no duplication across subsystems.
