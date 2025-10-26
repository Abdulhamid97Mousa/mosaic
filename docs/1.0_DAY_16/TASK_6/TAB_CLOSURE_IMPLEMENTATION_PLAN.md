# Tab Closure Implementation Plan — Day 16 Task 6

**Objective:** Implement pause-before-close workflow with keep/archive/delete options following the NOTIFICATIONS_AND_INDICATIONS.md design.

---

## 1. Current State (As-Is)

### File: `gym_gui/ui/widgets/render_tabs.py`

**Method:** `_close_dynamic_tab(run_id, tab_name, tab_index)`

**Current Flow:**
1. User clicks "×" close button on tab
2. Lambda directly calls `_close_dynamic_tab()`
3. Method immediately:
   - Removes widget from tracking
   - Removes tab at index
   - Calls `widget.cleanup()` if available
   - Calls `widget.deleteLater()`
   - **No user confirmation or data management**

**Problem:** Training continues after tab closes; user has no choice about data retention.

---

## 2. Desired State (To-Be)

### Workflow: Tab Closure Decision Tree

```
User clicks close button (×)
         ↓
[Is run still active?]
  ├─ YES → Show TabClosureDialog (modal)
  │         ├─ Pause Training (graceful signal)
  │         ├─ Then prompt: Keep / Archive / Delete
  │         ├─ [User selects action]
  │         ├─ Execute action (call TelemetryService API)
  │         └─ Close tab + cleanup
  │
  └─ NO → Check indicators
          ├─ Indicators present → Show warning + quick options
          └─ Clean state → Close immediately
```

---

## 3. Implementation Architecture

### 3.1 New Component: `TabClosureDialog`

**Location:** `gym_gui/ui/indicators/tab_closure_dialog.py`

**Purpose:** Reusable modal dialog for tab closure with keep/archive/delete options.

**Signals:**
- `action_selected(choice: TabClosureChoice)` — emitted when user makes choice
- `paused()` — emitted when training has been paused

**Methods:**
- `exec() -> TabClosureChoice` — blocking dialog, returns user's choice
- `set_run_summary(episodes, dropped_steps, etc.)` — populate dialog with stats

**Choices (Enum):**
```python
class TabClosureChoice(Enum):
    CANCEL = "cancel"           # User cancelled; don't close
    KEEP_AND_CLOSE = "keep"     # Keep data, close tab (training may continue)
    ARCHIVE = "archive"         # Freeze run snapshot, close tab
    DELETE = "delete"           # Purge run data, close tab
```

---

### 3.2 Enhanced: `RenderTabs._close_dynamic_tab()`

**Updated Flow:**

```python
def _close_dynamic_tab(self, run_id: str, tab_name: str, tab_index: int) -> None:
    """
    Intercept tab close and show closure workflow if run is active.
    """
    widget = self._agent_tabs.get(run_id, {}).get(tab_name)
    if not widget:
        return
    
    # Step 1: Check if run is still active
    if self._is_run_active(run_id):
        # Step 2: Show closure dialog with pause/keep/archive/delete options
        choice = self._show_tab_closure_dialog(run_id, tab_name, widget)
        
        if choice == TabClosureChoice.CANCEL:
            return  # Don't close
        
        # Step 3: Pause training if needed
        if choice != TabClosureChoice.KEEP_AND_CLOSE:
            self._pause_training_for_run(run_id)
        
        # Step 4: Execute data management action
        match choice:
            case TabClosureChoice.KEEP_AND_CLOSE:
                self._log_info(f"Keeping data for {run_id}, closing tab")
            case TabClosureChoice.ARCHIVE:
                self._archive_run(run_id)
            case TabClosureChoice.DELETE:
                self._delete_run(run_id)
    
    # Step 5: Clean up UI (always happens)
    self._perform_tab_cleanup(run_id, tab_name, tab_index, widget)
```

---

### 3.3 TelemetryService API Extensions

**Location:** `gym_gui/services/telemetry.py`

**New Methods:**

```python
def archive_run(self, run_id: str) -> bool:
    """
    Archive a run: mark it as archived in DB, freeze telemetry snapshot.
    Returns: True if successful, False otherwise.
    """

def delete_run(self, run_id: str) -> bool:
    """
    Delete a run: purge all steps/episodes from SQLite for this run_id.
    Returns: True if successful, False otherwise.
    """

def get_run_summary(self, run_id: str) -> RunSummary:
    """
    Get run statistics for dialog display.
    Returns: RunSummary(episodes_count, dropped_steps, dropped_episodes, etc.)
    """
```

---

### 3.4 New Dataclass: `RunSummary`

**Location:** `gym_gui/core/data_model/` (or `indicators/`)

```python
@dataclass
class RunSummary:
    run_id: str
    agent_id: str
    episodes_collected: int
    steps_collected: int
    dropped_episodes: int
    dropped_steps: int
    total_reward: float
    is_active: bool
    last_update_timestamp: str
```

---

### 3.5 Trainer/Worker Communication

**Pausing a Training Run:**

- Signal trainer daemon: "pause run X" (via gRPC or service locator)
- Trainer tells worker: stop accepting new steps, flush pending events
- Worker transitions to PAUSED state
- UI waits for confirmation, then proceeds with delete/archive

---

## 4. File Changes Summary

| File | Change | Scope |
|------|--------|-------|
| `gym_gui/ui/indicators/tab_closure_dialog.py` | **NEW** | Dialog + choice enum + signals |
| `gym_gui/ui/indicators/__init__.py` | MODIFY | Export `TabClosureDialog`, `TabClosureChoice` |
| `gym_gui/ui/widgets/render_tabs.py` | MODIFY | Integrate `_show_tab_closure_dialog()`, pause/archive/delete logic |
| `gym_gui/services/telemetry.py` | MODIFY | Add `archive_run()`, `delete_run()`, `get_run_summary()` |
| `gym_gui/core/data_model/` | MODIFY/NEW | Add `RunSummary` dataclass |
| `gym_gui/telemetry/sqlite_store.py` | MODIFY | Implement SQL for archive flag, delete rows |
| `gym_gui/services/trainer/service.py` | MODIFY | Add pause/resume methods |
| Tests | NEW | `test_tab_closure_workflow.py` |

---

## 5. Dialog UX Details

### TabClosureDialog Wireframe

```
┌─────────────────────────────────────────────┐
│  Close Live Training Tab                    │
├─────────────────────────────────────────────┤
│ Run "agent-A" collected:                    │
│  • 3 episodes (2 dropped)                   │
│  • 150 steps (24 dropped)                   │
│  • Total reward: 45.2                       │
│                                             │
│ Choose what happens to its data:            │
│                                             │
│ ◯ Keep data (default)                       │
│   Retain telemetry and keep run history     │
│   accessible. Training may continue.        │
│                                             │
│ ◯ Archive snapshot                          │
│   Seal the run for replay; move it to       │
│   archived run list for review.             │
│                                             │
│ ◯ Delete data (⚠ Warning)                   │
│   Remove telemetry from storage and         │
│   worker cache. CANNOT BE UNDONE.           │
│                                             │
│ ☐ Apply to other tabs from this run         │
│                                             │
│ [Cancel]  [Continue]                        │
└─────────────────────────────────────────────┘
```

### Dialog Content Extraction

- **Run summary stats** pulled from `TelemetryService.get_run_summary(run_id)`
- **Dropped metrics** from `LiveTelemetryTab._dropped_steps`, `_dropped_episodes`
- **Last update** from telemetry event timestamp

---

## 6. Implementation Phases (Incremental)

### Phase 1: Dialog & Enum (Foundation)
- [ ] Create `TabClosureDialog` component
- [ ] Define `TabClosureChoice`, `RunSummary`
- [ ] Add basic styling & layout
- **Test:** Dialog opens, user can select options

### Phase 2: RenderTabs Integration (UI Flow)
- [ ] Add `_is_run_active()` check
- [ ] Add `_show_tab_closure_dialog()` method
- [ ] Wire close button to new flow
- [ ] Add logging for user decisions
- **Test:** Close tab → dialog appears → selection works

### Phase 3: TelemetryService APIs (Data Ops)
- [ ] Implement `get_run_summary(run_id)`
- [ ] Implement `archive_run(run_id)` (update DB with archived flag)
- [ ] Implement `delete_run(run_id)` (SQL DELETE rows for run_id)
- **Test:** Archive/delete operations work correctly

### Phase 4: Trainer Pause Signal (Worker Integration)
- [ ] Add `pause_run(run_id)` to TrainerService
- [ ] Wire pause signal to worker
- [ ] Handle worker confirmation
- **Test:** Training pauses before delete/archive

### Phase 5: Testing & Polish
- [ ] Unit tests for TabClosureDialog
- [ ] Integration tests for close workflow
- [ ] UI tests for dialog UX
- [ ] Smoke tests for archive/delete data integrity

---

## 7. Database Schema Changes

### Archive Flag (Existing `episodes` Table)

```sql
ALTER TABLE episodes ADD COLUMN archived INTEGER DEFAULT 0;
-- archived = 0: active
-- archived = 1: archived
-- archived = -1: deleted (soft-delete marker)
```

### Delete Operation

**Option A: Soft Delete** (safer for concurrent readers)
```sql
UPDATE episodes SET archived = -1 WHERE run_id = ? AND agent_id = ?;
UPDATE steps SET archived = -1 WHERE run_id = ?;
```

**Option B: Hard Delete** (reclaims space immediately)
```sql
DELETE FROM steps WHERE run_id = ?;
DELETE FROM episodes WHERE run_id = ?;
```

**Recommendation:** Use soft-delete with periodic cleanup job to avoid WAL contention.

---

## 8. Error Handling

**Scenarios:**

1. **Archive fails** (DB locked): Show error dialog, offer retry or cancel
2. **Delete fails** (foreign key?): Rollback, explain constraint to user
3. **Pause times out**: Alert user, offer force-close or retry
4. **Network error** (trainer unreachable): Graceful fallback to local close

---

## 9. Logging & Observability

**Log Closure Decisions:**

```python
self.log_constant(
    LOG_UI_TAB_CLOSURE_DECISION,
    message=f"User selected: {choice.value} for run {run_id}",
    extra={
        "run_id": run_id,
        "agent_id": agent_id,
        "choice": choice.value,
        "episodes": summary.episodes_collected,
        "dropped_steps": summary.dropped_steps,
    }
)
```

---

## 10. Next Steps (Immediate)

1. **Create skeleton** of `TabClosureDialog` in `gym_gui/ui/indicators/`
2. **Define enums** and dataclasses
3. **Draft unit test** for dialog interaction
4. **Integrate into `RenderTabs._close_dynamic_tab()`** (start with phase 2)
5. **Implement TelemetryService APIs** (phase 3)
6. **Test end-to-end** with sample run

---

## 11. Design Compliance

✅ **Follows TASK_6 principles:**
- Persistent before modal ← Dialog shows only at close
- Run-centric signals ← Uses run_id + RunSummary
- Graduated severity ← Indicators inform decision
- Explicit retention choices ← Keep / Archive / Delete radio buttons
- Telemetry provenance ← Shows dropped metrics, episodes count

✅ **Aligns with constants work (Task 7):**
- Dialog thresholds pulled from `ui.constants` (not hard-coded)
- Archive/delete timeouts from `telemetry.constants`

