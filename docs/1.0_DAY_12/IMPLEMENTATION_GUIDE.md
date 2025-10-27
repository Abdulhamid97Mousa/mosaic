# Implementation Guide: Fixing Telemetry & Live Tab Issues

## Phase 1: Diagnostics (No Code Changes)

### Step 1: Run Diagnostic Checks

```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL

# Check 1: Episodes in database
echo "=== Episodes in database ==="
sqlite3 var/telemetry/telemetry.sqlite "SELECT COUNT(*) as episode_count FROM episodes;"

# Check 2: Credit logs
echo "=== Credit-related logs ==="
grep -i "credit\|no credit" var/logs/*.log 2>/dev/null | head -10

# Check 3: RunBus publishing
echo "=== RunBus publishing ==="
grep -i "published.*runbus\|step_appended\|episode_finalized" var/logs/*.log 2>/dev/null | head -10

# Check 4: Agent ID labels
echo "=== Agent ID labels ==="
grep -i "agent_id" var/logs/*.log 2>/dev/null | grep -E "proxy|controller|tab" | head -10

# Check 5: Overflow events
echo "=== Overflow events ==="
grep -i "overflow\|queue.*full" var/logs/*.log 2>/dev/null | head -10

# Check 6: Game ID in database
echo "=== Game IDs in database ==="
sqlite3 var/telemetry/telemetry.sqlite "SELECT DISTINCT game_id FROM steps LIMIT 5;" 2>/dev/null

# Check 7: Frame references
echo "=== Frame references ==="
sqlite3 var/telemetry/telemetry.sqlite "SELECT DISTINCT frame_ref FROM steps WHERE frame_ref IS NOT NULL LIMIT 5;" 2>/dev/null
```

### Step 2: Analyze Results

- **Episodes exist but tab is empty?** → Issue 1.1 or 1.2 (credit/RunBus)
- **No credit logs?** → Issue 1.1 (credits not initialized)
- **No RunBus logs?** → Issue 1.2 (publishing disabled)
- **Agent ID mismatches?** → Issue 2.2 (routing broken)
- **Overflow events?** → Issue 3.1 (buffer overflow)
- **game_id is NULL?** → Issue 3.2 (context missing)
- **frame_ref is NULL?** → Issue 3.3 (frame generation disabled)

---

## Phase 2: Priority 1 Fixes (Critical Deadlock)

### Fix 1.1: Pre-Tab Credit Grant

**File**: `gym_gui/controllers/live_telemetry.py`

**Changes**:
1. Add pending credits buffer in `__init__`
2. Grant credits immediately when first event arrives (before tab creation)
3. Apply buffered credits when tab is registered

**Code Pattern**:
```python
def __init__(self, ...):
    self._pending_credits: Dict[tuple[str, str], int] = {}  # (run_id, agent_id) -> credits

def _process_steps(self):
    # When first event arrives for new agent:
    if key not in self._tabs:
        # Grant credits BEFORE tab creation
        self._grant_credits_for_stream(evt.run_id, evt.agent_id, 200)
        self._pending_credits[key] = 200
        # Then request tab creation
        self.run_tab_requested.emit(...)

def register_tab(self, run_id, agent_id, tab):
    key = (run_id, agent_id)
    self._tabs[key] = tab
    # Apply any pending credits
    if key in self._pending_credits:
        credits = self._pending_credits.pop(key)
        # Tab can now use these credits
```

### Fix 1.2: Verify RunBus Publishing

**File**: `gym_gui/services/trainer/streams.py`

**Verification**:
1. Confirm `bus.publish()` calls exist in `_drain_loop()` (lines 535-590)
2. Verify publishing happens for both STEP_APPENDED and EPISODE_FINALIZED
3. Add logging before each publish

**Code Pattern**:
```python
# In _drain_loop(), after normalization:
try:
    bus = get_bus()
    evt = TelemetryEvent(
        topic=Topic.STEP_APPENDED,
        run_id=run_id,
        agent_id=agent_id,
        seq_id=step.seq_id,
        ts_iso=payload_dict.get("timestamp", ""),
        payload=payload_dict,
    )
    bus.publish(evt)
    _LOGGER.debug("Published STEP_APPENDED", extra={
        "run_id": run_id, "agent_id": agent_id, "seq_id": step.seq_id
    })
except Exception as e:
    _LOGGER.warning("Failed to publish", extra={"error": str(e)})
```

---

## Phase 3: Priority 2 Fixes (Data Model)

### Fix 2.1: Episode Identity Consistency

**File**: `gym_gui/services/trainer/service.py`

**Changes**:
1. Verify `_episode_from_proto()` correctly reconstructs episode_id
2. Add validation to ensure episode_id is consistent
3. Add logging for episode_id generation

**Code Pattern**:
```python
def _episode_from_proto(self, message):
    # Reconstruct episode_id from episode_index
    episode_suffix = int(message.episode_index)
    episode_id = f"{message.run_id}-ep{episode_suffix:04d}"
    
    _LOGGER.debug("Episode ID reconstructed", extra={
        "episode_index": episode_suffix,
        "episode_id": episode_id,
        "run_id": message.run_id
    })
    
    return EpisodeRollup(
        episode_id=episode_id,
        # ... other fields
    )
```

### Fix 2.2: Agent ID Standardization

**File**: `gym_gui/services/trainer/streams.py`

**Changes**:
1. Standardize agent_id format (no transformations)
2. Add validation to catch mismatches
3. Log agent_id at every routing decision

**Code Pattern**:
```python
def _normalize_payload(payload, run_id, default_agent_id="default"):
    # Extract agent_id
    agent_id = payload.get("agent_id", default_agent_id)
    
    # Validate format
    if not isinstance(agent_id, str) or not agent_id:
        agent_id = default_agent_id
    
    _LOGGER.debug("Agent ID normalized", extra={
        "original": payload.get("agent_id"),
        "normalized": agent_id,
        "run_id": run_id
    })
    
    return {
        "agent_id": agent_id,
        # ... other fields
    }
```

---

## Phase 4: Priority 3 Fixes (Display)

### Fix 3.1: game_id Propagation

**File**: `gym_gui/ui/main_window.py`

**Changes**:
1. Extract game_id from run metadata
2. Pass to tab constructors
3. Initialize before first render

**Code Pattern**:
```python
def _create_agent_tabs_for(self, run_id, agent_id, first_payload):
    # Extract game_id from run metadata (not first_payload)
    game_id = self._get_game_id_for_run(run_id)
    
    # Pass to grid tab
    grid = AgentOnlineGridTab(
        run_id, agent_id,
        game_id=game_id,  # NEW
        renderer_registry=renderer_registry,
        parent=self
    )
```

### Fix 3.2: Frame Reference Generation

**File**: `gym_gui/core/adapters/base.py`

**Changes**:
1. Implement `build_frame_reference()` in all adapters
2. Generate timestamped filenames
3. Return relative paths

**Code Pattern**:
```python
def build_frame_reference(self, render_payload, state):
    if render_payload is None:
        return None
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    step_index = getattr(state, "step_index", 0)
    frame_ref = f"frames/{timestamp}_{step_index:06d}.png"
    
    _LOGGER.debug("Frame reference generated", extra={
        "frame_ref": frame_ref,
        "step_index": step_index
    })
    
    return frame_ref
```

---

## Testing Checklist

- [ ] Run diagnostic checks (Phase 1)
- [ ] Implement Priority 1 fixes
- [ ] Verify RunBus publishing with logs
- [ ] Implement Priority 2 fixes
- [ ] Test episode identity end-to-end
- [ ] Test agent_id routing with multiple agents
- [ ] Implement Priority 3 fixes
- [ ] Test game_id propagation
- [ ] Test frame reference generation
- [ ] Run full integration test
- [ ] Verify no regressions

---

## Rollback Plan

If issues arise:
1. Revert changes to specific file
2. Re-run diagnostic checks
3. Identify which fix caused regression
4. Adjust fix and re-test

---

## Success Indicators

- [ ] Dynamic tabs appear when first event arrives
- [ ] Tabs display steps and episodes
- [ ] Multiple agents have separate tabs
- [ ] Correct game is rendered
- [ ] Frames display correctly
- [ ] No events are dropped
- [ ] No deadlocks occur

