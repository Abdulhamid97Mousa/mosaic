# Constants Architecture Audit & Consolidation Plan

> **Update (Oct 29, 2025):** Consolidation completed. All constants now live in
> `gym_gui/constants/`. Notes below retained for historical context, with ✅
> markers indicating completed work.

## Current State (Post-Consolidation)

```
✓ gym_gui/constants/               # Canonical package for all constants
  - constants_core.py              # Episode counter + worker IDs
  - constants_ui.py                # UI defaults (simple + dataclass)
  - constants_telemetry.py         # Telemetry infrastructure buffers
  - constants_telemetry_bus.py     # RunBus / fan-out defaults
  - constants_telemetry_db.py      # Persistence defaults
  - constants_trainer.py           # Trainer client/daemon defaults
  - constants_game.py              # Environment defaults (pre-existing)
  - loader.py                      # Config loading utilities
```

Legacy modules such as `gym_gui/telemetry/constants.py` and
`gym_gui/services/trainer/constants.py` have been deleted after import
migration. The remainder of this document captures the original audit to show
why the consolidation was necessary.

## Historical Analysis

### Problem 1: Inconsistent File Naming

**Original constants layout (pre-refactor):**

### Problem 2: Unknown Overlapping Values

**Potential overlaps identified:**

| Constant | Current Location(s) | Scope | Purpose |
|----------|-------------------|-------|---------|
| Buffer sizes | `constants/constants_telemetry.py`, `constants/constants_ui.py` | ✅ Resolved | Separated by domain |
| Queue sizes | `constants/constants_telemetry.py` | ✅ Resolved | Telemetry | Event queuing |
| Defaults | `constants/constants_ui.py` | ✅ Resolved | UI | Form/widget defaults |
| Validation ranges | `ui/widgets/spade_bdi_train_form.py` (hardcoded) | 🟡 In progress | Training form | Episode/step/LR ranges |
| Episode counter config | `constants/constants_core.py` | ✅ Resolved | Episodes | Centralized |
| Worker ID config | `constants/constants_core.py` | ✅ Resolved | Episodes | Centralized (adoption underway) |

**Issue:** No master index of all constants → duplicate values, inconsistent naming, hard to maintain.

### Problem 3: Worker ID Not Implemented

**Progress as of Oct 29, 2025:**
- ✅ Train form captures `worker_id` and pushes it into metadata/environment
- ✅ `TrainRunConfig` keeps worker id; dispatcher writes it to `worker-{run_id}` configs
- ✅ Dispatcher/proxy append `--worker-id` and set `WORKER_ID` env vars
- ✅ Worker runtime (`RunConfig`, `TelemetryEmitter`) tags every step/episode payload
- ✅ Telemetry service + SQLite store persist `worker_id`; episode IDs now use `format_episode_id(run_id, ep_index, worker_id)`
- ✅ `RunCounterManager` + `SessionController` maintain worker-scoped counters
- 🟡 Live telemetry controllers still consolidate by `(run_id, agent_id)`; worker-aware buffers remain outstanding
- 🟡 BDI integration tests still assume single worker; follow-up tracked in TASK 3

---

## Solution Plan

### Phase 1: Consolidate Episode Counter Constants ✅

**Action:** Move `constants_episode_counter.py` → `constants/episode_counter.py`

```bash
# Before (inconsistent)
gym_gui/core/constants_episode_counter.py

# After (consistent with pattern)
gym_gui/constants/
├── game_constants.py
├── episode_counter.py  # NEW
└── loader.py
```

**Rationale:**
- Follows `gym_gui/constants/` package pattern
- Matches `game_constants.py` naming
- Can extend loader.py if needed

### Phase 2: Create Constants Master Index ✅

**Action:** Create a new doc listing all constants with:
- File location
- Purpose/scope
- Type (config, validation range, buffer size, etc.)
- Dependencies
- Usage count

```markdown
# All Project Constants

## Episode Counter (constants/episode_counter.py)
- DEFAULT_MAX_EPISODES_PER_RUN: 999,999
- EPISODE_COUNTER_WIDTH: 6
- format_episode_id(): Helper function
- parse_episode_id(): Helper function

## UI (ui/constants.py)
- RENDER_DELAY_MIN_MS: 10
- ... 20+ values for UI widgets

## Telemetry (telemetry/constants.py)
- STEP_BUFFER_SIZE: 64
- ... queue and buffer config

... etc
```

### Phase 3: Implement Worker ID Support 🟡

**Action 1:** Add worker_id to RunCounterManager properly

```python
# In gym_gui/constants/episode_counter.py
# Add worker configuration documentation
WORKER_ID_CONFIG = """
Worker IDs enable distributed/sharded training:
- Each worker has unique ULID (timestamp-based ordering)
- Episode index scoped to (run_id, worker_id) pair
- DB constraint: UNIQUE(run_id, worker_id, ep_index)
- Episode format: {run_id}-w{worker_id}-ep{ep_index:06d}
"""
```

**Action 2:** Update SessionController to support worker_id

```python
# In session.py
def set_run_context(self, run_id, max_episodes, db_conn, worker_id=None):
    self._run_counter_manager = RunCounterManager(
        db_conn, run_id, max_episodes, worker_id=worker_id
    )
```

**Action 3:** Update training form to optionally set worker_id

```python
# In spade_bdi_train_form.py
worker_id = self._get_worker_id()  # From form or generate ULID
```

**Action 4:** DB schema already supports it - no migration needed

### Phase 4: Update Imports ✅

**Files to update:**
- `gym_gui/core/run_counter_manager.py`: Change import path
- `gym_gui/controllers/session.py`: Change import path
- `gym_gui/ui/widgets/spade_bdi_train_form.py`: Change import path
- `gym_gui/core/tests/test_run_counter_manager.py`: Change import path

```python
# Before
from gym_gui.core.constants_episode_counter import ...

# After
from gym_gui.constants.episode_counter import ...
```

---

## Implementation Checklist

### Phase 1: File Organization
- [x] Create `gym_gui/constants/constants_core.py`
- [x] Copy contents from `gym_gui/core/constants_episode_counter.py`
- [x] Update docstrings to reference package location
- [x] Delete old `gym_gui/core/constants_episode_counter.py`
- [x] Update imports throughout codebase

### Phase 2: Documentation
- [x] Create/refresh constants README + quick reference
- [x] List all constants by module
- [x] Mark overlaps and potential consolidations
- [x] Update this file with resolution status

### Phase 3: Worker ID Implementation
- [x] Add worker_id parameter to SessionController.set_run_context()
- [x] Update RunCounterManager to use worker_id in format_episode_id()
- [x] Add optional worker_id field to training form UI
- [x] Update DB-related code to handle UNIQUE(run_id, worker_id, ep_index)
- [ ] Add tests for multi-worker scenarios

### Phase 4: Integration
- [x] Update imports to `gym_gui.constants`
- [x] Run full test suite (`pytest gym_gui/tests`)
- [x] Verify no regression in episode counter functionality

---

## Benefits

After consolidation:
1. **Consistency:** All constants in `gym_gui/constants/` package
2. **Clarity:** Master index shows ALL constants, no hidden duplicates
3. **Functionality:** Worker ID actually works end-to-end
4. **Maintainability:** Single source of truth for each config domain
5. **Scalability:** Easy to add new constant files with consistent naming

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Breaking imports | Update 4 files systematically, run tests |
| DB compatibility | Schema already exists, no migration needed |
| Incomplete worker_id | Phase 3 ensures full implementation |
| Missing constants in index | Systematic search and audit in Phase 2 |
