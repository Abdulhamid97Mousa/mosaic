# Telemetry Counter and Reward Display Analysis
**Date**: October 22, 2025  
**Status**: Investigation Complete → Fixes Pending  
**Priority**: CRITICAL

---

## Executive Summary

Three interconnected bugs discovered in Live Training telemetry display:

| Issue | Symptom | Root Cause | Impact |
|-------|---------|-----------|--------|
| **Counter** | Shows "Step: 100 \| Episodes: 20" regardless of actual episode progression | Counter uses buffer length instead of episode/step indices | Data stale, user sees incorrect metrics |
| **Reward** | Displays "+0.000" instead of "+1.000" for goal achievement | Telemetry producer sending reward=0.0 (not UI formatting) | Training feedback incomplete |
| **Human Mode** | Counter doesn't update during manual play | Telemetry routing different for human input | Manual mode appears frozen |

**Severity**: HIGH - Violates core principle of accurate real-time telemetry visualization

---

## Issue #1: Counter Shows Buffer Size, Not Episode Metrics

### The Bug (with Contrarian Analysis)

**Current Code** (`gym_gui/ui/widgets/live_telemetry_tab.py:784`):
```python
def _update_stats(self) -> None:
    """Refresh step/episode counters."""
    self._stats_label.setText(
        f"Steps: {len(self._step_buffer)} | Episodes: {len(self._episode_buffer)}"
    )
```

**What's Wrong**:
- Uses `len(self._step_buffer)` which returns deque occupancy (0-100), not actual step count
- Deque defined as `deque(maxlen=100)` at line 44-45
- This confuses **buffer size** with **training metrics**

**Why This Is a Data Corruption Bug**:

Consider a scenario:
- Episode 1: 50 steps → counter correctly shows "Step: 50"
- Episode 2: 50 steps → buffer fills, counter shows "Step: 100"
- Episode 3: 1 step → counter STILL shows "Step: 100" (buffer hasn't rotated yet)
- User sees: "Episode 20, Step: 100" even though current episode is 3 steps in
- **Result: STALE DATA** - counter doesn't reset at episode boundaries

**Architectural Problem**:
The counter was never designed as a true metrics display. Someone confused "buffer occupancy" with "episode progress", creating a fundamental architectural flaw.

### Secondary Bug: AgentOnlineGridTab

**Location** (`gym_gui/ui/widgets/agent_online_grid_tab.py:18-20, 98-102`):
```python
# Initialization
self._steps = 0          # CUMULATIVE, never resets
self._episodes = 0       # Only tracks max, never used for increment

# In on_step() callback
self._steps += 1         # Increments forever
self._episodes = max(self._episodes, int(episode_index) + 1)

# Display (line 117)
self._steps_label.setText(str(self._steps))  # Shows cumulative!
```

**Problem**: If episode 1 has 100 steps, episode 2 shows step counter at 100+, 101, 102...

### Expected Behavior

```
Episode 1:
  Step: 0 → Step: 1 → Step: 2 → ... → Step: 98 → (terminated)
  
Episode 2:
  Step: 0 → Step: 1 → Step: 2 → ... → Step: 47 → (terminated)
  
Counter Should Display:
  Episode: 1 Step: 0
  Episode: 1 Step: 1
  Episode: 1 Step: 2
  ...
  Episode: 2 Step: 0  ← RESETS on episode boundary
  Episode: 2 Step: 1
  ...
```

### Why User Sees "Step: 100 | Episodes: 20"

**Data Flow**:
```
Telemetry Producer (worker/gRPC)
  ↓ (sends payloads with episode_index, step_index)
TelemetryBridge (deserializes)
  ↓ (adds to circular buffers)
LiveTelemetryTab._step_buffer (deque, maxlen=100)
  ↓ (buffer size used for display)
Counter Widget: len(buffer) = 100 → "Step: 100"
```

---

## Issue #2: Reward Shows "+0.000"

### The Bug

**Current Code** (`gym_gui/ui/widgets/live_telemetry_tab.py:428`):
```python
reward = _get_field(payload, "reward", default=0.0)  # line 369
# ... later ...
self._steps_table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{reward:+.3f}"))
```

**Display Format Is Correct**:
- `f"{1.0:+.3f}"` produces `"+1.000"` ✓
- `f"{0.0:+.3f}"` produces `"+0.000"` ✓
- Format code is NOT the bug

**The Real Problem**:
The telemetry PAYLOAD itself contains `reward=0.0`, not the UI formatter

**Evidence**:
- User reports: "Episode 17 has 8 steps in Telemetry Recent Episodes table (correct)"
- If telemetry producer was completely broken, database wouldn't have correct data
- BUT: live stream shows "+0.000" consistently
- This suggests: **Live telemetry payload has reward=0.0, but historical data may differ**

### Possible Causes

1. **FrozenLakeAdapter returning 0.0**
   - Check gym_gui/core/adapters/toy_text.py FrozenLakeAdapter.step()
   - Verify environment.step() returns correct reward

2. **TelemetryBridge serialization losing reward**
   - Check gym_gui/telemetry/bridge.py payload construction
   - Verify reward field name matches (camelCase in protobuf?)

3. **SessionController not capturing reward**
   - Check gym_gui/controllers/session.py _record_step()
   - Verify reward extracted from env.step() tuple

4. **_get_field() falling back to default**
   - Check if payload field name is wrong
   - Verify "reward" vs "reward_value" vs other naming

### Impact

Without real reward values, training feedback is incomplete:
- Agent can't distinguish goal achievement from normal movement
- User can't verify training progress visually
- Debugging becomes harder

---

## Issue #3: Human Mode Counter Not Updating

### The Bug

**Location** (`gym_gui/ui/widgets/agent_online_grid_tab.py`):
- Counters only update in `on_step()` callback
- Human mode may not trigger this callback

**Possible Root Causes**:

1. **Human steps don't emit telemetry**
   - SessionController.human_step() may not call telemetry producer
   - Result: on_step() never called for human input

2. **Human telemetry goes to different sink**
   - Human telemetry may use different signal/event path
   - Not connected to LiveTelemetryTab/AgentOnlineGridTab callbacks

3. **Human telemetry payload missing episode_index**
   - Counter tracking relies on episode_index field
   - If human mode doesn't set it, counter update fails silently

### Data Flow Comparison

**Agent Mode (Working)**:
```
Worker sends gRPC payload
  ↓
TelemetryAsyncHub receives
  ↓
Emits Qt signals (step_received, episode_finished)
  ↓
on_step() callback in LiveTelemetryTab
  ↓
Counter updated
```

**Human Mode (Broken?)**:
```
SessionController.human_step() called
  ↓
????? (unknown path)
  ↓
Counter NOT updated
```

---

## Data Structure Analysis

### LiveTelemetryTab Buffers (Line 44-45)

```python
self._step_buffer: Deque[Any] = deque(maxlen=buffer_size)
self._episode_buffer: Deque[Any] = deque(maxlen=episode_buffer_size)
```

**Purpose**: Circular buffers for rendering (keep last 100 items)

**Problem**: Counter display confuses buffer size with actual metrics

**Size Progression Example**:
- After 50 steps: buffer has 50 items, len()=50, display shows "Step: 50" ✓
- After 100 steps: buffer has 100 items, len()=100, display shows "Step: 100" ✓
- After 150 steps: buffer has 100 items (rotates), len()=100, display shows "Step: 100" ✗
- After 200 steps: buffer has 100 items, len()=100, display shows "Step: 100" ✗

**Result**: Counter stuck at 100 after buffer fills

---

## Payload Structure Analysis

### Expected Telemetry Payload Fields

From code examination:

```python
# Producer (worker/gRPC)
{
    "run_id": "3310c533dfff",
    "agent_id": "Agent-1",
    "episode_index": 17,      # 1-based
    "step_index": 8,          # 0-based or 1-based?
    "reward": 0.0,            # ← PROBLEM: should be 1.0 for goal
    "observation": [...],
    "action": 2,
    "terminated": False,
    "truncated": False,
    "game_id": "FrozenLake-v1"
}
```

### Field Extraction (Line 369)

```python
def _get_field(payload, key, default=None):
    """Recursively extract field from nested dict."""
    # ... implementation ...
    return default  # Falls back to 0.0 if field missing
```

**Issue**: If payload structure is different, _get_field() returns default

---

## Root Cause Verification

### Counter Issue: CONFIRMED

**Files Involved**:
- `gym_gui/ui/widgets/live_telemetry_tab.py:784` - Uses len(buffer)
- `gym_gui/ui/widgets/agent_online_grid_tab.py:98-102` - Uses cumulative counters

**Verification**: Tracing buffer construction at line 44-45 confirms maxlen=100

### Reward Issue: SUSPECTED (needs verification)

**Files to Check**:
- `gym_gui/core/adapters/toy_text.py` - FrozenLakeAdapter.step()
- `gym_gui/telemetry/bridge.py` - Payload serialization
- `gym_gui/controllers/session.py` - _record_step() reward capture

### Human Mode Issue: SUSPECTED (needs verification)

**Files to Check**:
- `gym_gui/controllers/session.py` - human_step() implementation
- Signal routing from SessionController → TelemetryBridge

---

## Fix Strategy (Summary)

### Fix #1: Counter Logic (HIGH PRIORITY)

**Approach**: Separate buffer from metrics

1. Add tracking fields to LiveTelemetryTab:
   ```python
   self._current_episode_index: int = 0
   self._current_step_in_episode: int = 0
   self._previous_episode_index: int = -1
   ```

2. Extract from each payload:
   ```python
   episode_index = payload.get('episode_index', 0)
   step_index = payload.get('step_index', 0)
   
   if episode_index != self._previous_episode_index:
       self._current_step_in_episode = 0
       self._previous_episode_index = episode_index
   
   self._current_episode_index = episode_index
   self._current_step_in_episode = step_index
   ```

3. Update display:
   ```python
   f"Episode: {self._current_episode_index} Step: {self._current_step_in_episode}"
   ```

**Result**: Counter resets per episode, shows actual indices

### Fix #2: Reward Value (HIGH PRIORITY)

**Approach**: Trace and fix source

1. Add logging in adapter.step()
2. Add logging in _record_step()
3. Verify FrozenLakeConfig (is_slippery, goal_position)
4. Check if reward field is being overwritten

### Fix #3: Human Mode Telemetry (MEDIUM PRIORITY)

**Approach**: Verify/fix signal routing

1. Trace SessionController.human_step() → telemetry producer
2. Ensure human payloads have episode_index, step_index
3. Verify on_step() callback receives human telemetry

---

## Testing Plan

### Unit Test: Counter Reset

```python
# Simulate episode boundary
payload1 = {"episode_index": 1, "step_index": 0, "reward": 0.0, ...}
payload2 = {"episode_index": 1, "step_index": 1, "reward": 0.0, ...}
payload3 = {"episode_index": 2, "step_index": 0, "reward": 1.0, ...}  # Goal!

# Assert counter resets
assert counter.episode_index == 2 and counter.step_index == 0
```

### Integration Test: Reward Display

```python
# Train FrozenLake-v1 to goal
# Verify counter shows correct episode/step
# Verify reward shows "+1.000" on goal achievement
```

### Manual Test: Human Mode

```
1. Start training in Agent Only mode
2. Switch to Human mode
3. Make manual moves
4. Assert counter updates with each move
5. Assert episode resets on goal/truncation
```

---

## References

- **Counter code**: gym_gui/ui/widgets/live_telemetry_tab.py:784
- **Alternative counter**: gym_gui/ui/widgets/agent_online_grid_tab.py:98-102
- **Reward display**: gym_gui/ui/widgets/live_telemetry_tab.py:428
- **Buffer definition**: gym_gui/ui/widgets/live_telemetry_tab.py:44-45
- **Telemetry producer**: gym_gui/services/trainer/streams.py (gRPC) / gym_gui/controllers/session.py (headless)
- **Signal routing**: gym_gui/ui/main_window.py:770-870
