# Quick Start Action Plan — Day 18 Task 5

## ⚡ TL;DR — What to Do RIGHT NOW

### Step 1: Setup Environment (5 minutes)
```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL
source ./.venv/bin/activate
pip install -r requirements/cleanrl_worker.txt
```

### Step 2: Understand Current State (10 minutes)
- Read: `/docs/1.0_DAY_18/TASK_5/FAST_TRAINING_MODE_AND_CLEANRL_WORKER.md`
- Key insight: **Two-phase approach**
  1. **Phase 1 (THIS WEEK):** Test SPADE-BDI without telemetry + add toggle
  2. **Phase 2 (NEXT WEEK):** Create CleanRL worker with analytics tabs

### Step 3: Proof of Concept — Test SPADE-BDI Without Telemetry (Today)

#### 3a. Create a test script to verify the concept

**File:** `test_no_telemetry_poc.py` (in project root)

```python
"""Proof of concept: SPADE-BDI worker without telemetry."""
import json
import subprocess
import time
from pathlib import Path

# Config with telemetry disabled
config = {
    "env_id": "CartPole-v1",
    "episodes": 50,
    "seed": 42,
    "worker_id": "test-worker-1",
    "extra": {
        "telemetry_enabled": False,  # KEY: Disable telemetry
        "ui_rendering_enabled": False,  # KEY: Disable UI
    }
}

config_file = Path("test_config.json")
config_file.write_text(json.dumps(config, indent=2))

print("=" * 60)
print("Testing SPADE-BDI without telemetry...")
print("=" * 60)

start = time.time()
result = subprocess.run(
    ["python", "-m", "spade_bdi_worker", "--config", str(config_file)],
    capture_output=True,
    text=True,
)
elapsed = time.time() - start

print(f"\n✓ Training completed in {elapsed:.2f}s")
print(f"Episodes per second: {50 / elapsed:.0f}")

# Parse output for lifecycle events
if "run_completed" in result.stdout:
    print("✓ Lifecycle events detected")
else:
    print("⚠ No lifecycle events found (expected)")

print("\nOutput sample:")
print(result.stdout[:500] if result.stdout else "(no stdout)")

if result.returncode != 0:
    print(f"\n✗ Error:\n{result.stderr}")
else:
    print("\n✅ SUCCESS: Worker ran without telemetry!")
```

**Run it:**
```bash
python test_no_telemetry_poc.py
```

#### 3b. If the POC works, measure the speedup

Compare with telemetry enabled:
```bash
# Create config WITH telemetry
# Run training twice (one with, one without)
# Compare times
```

---

## 🎯 My Recommendation: DO NOT START WITH CPU/GPU CONTROLS

### Why?
- ✅ Fast training toggle is **simpler** and gives immediate value
- ✅ GPU controls are **orthogonal** — can be added in parallel/later
- ✅ You need to **prove the concept** first (no telemetry works)
- ✅ GPU allocation is a **trainer daemon concern**, not just UI

### When to add GPU/CPU controls?
- **After** Phase 1 works (fast training mode tested)
- In a **separate task** (Task 5b)
- Coordinate with trainer daemon resource registry

---

## 📋 Detailed Action Items (This Week)

### Item 1: Verify SPADE-BDI Worker Can Skip Telemetry
**Status:** Research-only
**Time:** 30 minutes
**Acceptance Criteria:**
- [ ] Can instantiate TelemetryEmitter with `disabled=True` flag
- [ ] Worker training completes successfully
- [ ] No telemetry appears on stdout
- [ ] Lifecycle events (run_started, run_completed) still emit

**Files to check:**
```
spade_bdi_worker/core/telemetry_worker.py
spade_bdi_worker/worker.py
spade_bdi_worker/core/runtime.py (or bdi_trainer.py)
```

---

### Item 2: Add `--no-telemetry` Flag to Worker
**Status:** Implementation
**Time:** 1 hour
**Acceptance Criteria:**
- [ ] `python -m spade_bdi_worker --help` shows `--no-telemetry`
- [ ] Flag is passed to RunConfig
- [ ] TelemetryEmitter respects the flag
- [ ] Test passes: `pytest spade_bdi_worker/tests/test_training_no_telemetry.py`

**Files to modify:**
```
spade_bdi_worker/worker.py                    (add --no-telemetry arg)
spade_bdi_worker/core/telemetry_worker.py     (add disabled parameter)
spade_bdi_worker/core/runtime.py              (or bdi_trainer.py) (use disabled flag)
spade_bdi_worker/tests/test_training_no_telemetry.py (new test)
```

---

### Item 3: Add "Fast Training Mode" Toggle to GUI
**Status:** Implementation
**Time:** 2 hours
**Acceptance Criteria:**
- [ ] Checkbox appears in spade_bdi_train_form.py
- [ ] Checkbox has tooltip explaining what it does
- [ ] When checked, disables live rendering options
- [ ] When checked, shows warning label
- [ ] Toggle value passed to TrainRequest

**Files to modify:**
```
gym_gui/ui/widgets/spade_bdi_train_form.py    (add checkbox + handler)
gym_gui/ui/presenters/workers/spade_bdi_worker_presenter.py (pass to TrainRequest)
```

---

### Item 4: Confirmation Dialog Before Fast Training
**Status:** Implementation
**Time:** 30 minutes
**Acceptance Criteria:**
- [ ] Dialog appears when user clicks "Train" with fast mode enabled
- [ ] Dialog clearly warns about lost telemetry/replay
- [ ] User can cancel before committing

**Files to modify:**
```
gym_gui/ui/presenters/workers/spade_bdi_worker_presenter.py (add dialog)
```

---

### Item 5: Test End-to-End
**Status:** Testing
**Time:** 30 minutes
**Acceptance Criteria:**
- [ ] Launch GUI: `python -m gym_gui.app`
- [ ] Open train form
- [ ] Enable "Fast Training Mode"
- [ ] Submit training
- [ ] See confirmation dialog
- [ ] Training completes
- [ ] No errors in logs

---

## 🔄 Week 2: Post-Training SQLite Population

Once fast training mode works, implement deferred persistence:

### Item 6: Create TensorBoard Importer Service
**Time:** 2-3 hours
**Files to create:**
```
gym_gui/services/tensorboard_importer.py
gym_gui/services/event_reader.py (helper)
```

### Item 7: Add `telemetry_summary` Table to SQLite
**Time:** 1 hour
**Files to modify:**
```
gym_gui/storage/telemetry_sqlite_store.py (add new table)
```

### Item 8: Hook Importer into Run Completion
**Time:** 1 hour
**Files to modify:**
```
gym_gui/controllers/session_controller.py (or appropriate service)
```

### Item 9: Create Analytics Tab Presenter
**Time:** 2 hours
**Files to create:**
```
gym_gui/ui/presenters/analytics_tab_presenter.py
gym_gui/ui/widgets/tensorboard_analytics_tab.py
gym_gui/ui/widgets/wandb_analytics_tab.py
```

---

## 🚀 Week 3: CleanRL Worker Scaffolding

After Items 1-5 are solid, start CleanRL worker:

### Item 10: Scaffold cleanrl_worker/
**Time:** 1 hour
**Create files:**
```
cleanrl_worker/__init__.py
cleanrl_worker/README.md
cleanrl_worker/worker.py
cleanrl_worker/constants.py
cleanrl_worker/core/__init__.py
cleanrl_worker/core/runtime.py
cleanrl_worker/core/config.py
cleanrl_worker/infrastructure/__init__.py
cleanrl_worker/infrastructure/tensorboard_tracker.py
cleanrl_worker/adapters/__init__.py (reuse from spade_bdi_worker)
cleanrl_worker/tests/__init__.py
```

### Item 11: Add Constants & Log Codes
**Time:** 30 minutes
**Files to modify:**
```
gym_gui/constants/constants_trainer.py       (add WorkerType.CLEANRL)
gym_gui/logging_config/log_constants.py      (add LOG_CLEANRL_* codes)
```

### Item 12: Implement CleanRL Runtime
**Time:** 3-4 hours
**Files to create:**
```
cleanrl_worker/core/runtime.py
cleanrl_worker/infrastructure/tensorboard_tracker.py
```

### Item 13: Update Presenter to Handle Analytics Path
**Time:** 2 hours
**Files to modify:**
```
gym_gui/ui/presenters/workers/cleanrl_worker_presenter.py (new)
gym_gui/ui/presenters/workers/__init__.py (register presenter)
```

---

## 📊 Risk Mitigation Checklist

- [ ] **Telemetry disabled = data loss:** Add BIG warning dialog + confirmation
- [ ] **GPU starvation:** Document in trainer daemon that only ONE GPU worker at a time
- [ ] **Backward compatibility:** Tag runs with `telemetry_mode` field
- [ ] **Constant duplication:** Review all imports from `gym_gui.constants`
- [ ] **TensorFlow import:** Make tensorboard optional (handle ImportError gracefully)

---

## 📝 Testing Commands (After Each Item)

```bash
# After Item 1 (verify concept)
python test_no_telemetry_poc.py

# After Item 2 (verify flag works)
python -m spade_bdi_worker --help | grep no-telemetry
pytest spade_bdi_worker/tests/test_training_no_telemetry.py -v

# After Item 3 (GUI toggle)
python -m pytest gym_gui/tests/ui/test_spade_bdi_train_form.py -v

# After Item 4 (end-to-end)
python -m gym_gui.app
# [Manually test: train dialog → fast mode → confirmation → submit]

# After Item 5 (imports)
pytest gym_gui/tests/services/test_tensorboard_importer.py -v

# After Item 6 (CleanRL basics)
pytest cleanrl_worker/tests/test_cleanrl_worker_basic.py -v
```

---

## 💡 Key Insights

### The Dual-Path Model
```
SPADE-BDI Worker:
  ├─ Normal mode: Telemetry path (real-time UI)
  └─ Fast mode: Analytics path (TensorBoard only) ← NEW

CleanRL Worker (next week):
  └─ Analytics-only path (TensorBoard/W&B tabs)
```

### Why This Order?
1. **Proof of concept first** (SPADE-BDI without telemetry)
2. **UI integration second** (toggle + confirmation)
3. **Persistence third** (TensorBoard importer)
4. **New worker fourth** (CleanRL with analytics tabs)

Each step builds on the previous, reducing risk of big design mistakes.

### The ID Hierarchy (Preserved Across Both Paths)
```
run_id: "run-abc123"
├─ worker_id: "worker-1"       (which process)
├─ agent_id: "agent-1"         (which policy)
└─ episode_id: "ep-0", "ep-1"  (which episode)
```

This structure works for:
- ✅ Real-time telemetry (SPADE-BDI normal mode)
- ✅ Analytics-only (SPADE-BDI fast mode + CleanRL)
- ✅ Multi-agent scenarios (future)
- ✅ Distributed training (future)

---

## Questions During Implementation?

Refer back to:
- `/docs/1.0_DAY_18/TASK_5/FAST_TRAINING_MODE_AND_CLEANRL_WORKER.md` — Full strategy
- `/docs/1.0_DAY_18/CLEANRL_WORKER_STRATEGY.md` — Architecture deep dive
- `/docs/1.0_DAY_14/TASK_1/TELEMETRY_ARCHITECTURE.md` — How telemetry works

---

## Next: Let's Start Implementation!

Would you like me to:
1. ✅ **Start with Item 1** — Check if SPADE-BDI can skip telemetry (analysis)
2. ✅ **Jump to Item 2** — Implement `--no-telemetry` flag in worker
3. ✅ **Start with Item 3** — Add toggle to GUI
4. ❌ **Something else?**

My recommendation: **Start with Item 2** (implement `--no-telemetry` flag first), then test with POC script, then add UI toggle in Item 3.


