# Comprehensive Telemetry & Live Tab System Analysis

## Executive Summary

This document provides a structured analysis of 7 critical issues preventing dynamic agent tabs from displaying data correctly. Issues are ranked by severity and include root cause analysis, diagnostic steps, and implementation guidance.

---

## Quick Diagnostic Checklist

Run these checks FIRST (no code changes):

```bash
# 1. Check if episodes exist in database (producer working?)
sqlite3 var/telemetry/telemetry.sqlite "SELECT COUNT(*) FROM episodes;"

# 2. Check for credit-related log messages
grep -i "credit\|no credit" var/logs/*.log | head -20

# 3. Check for RunBus publishing
grep -i "published.*runbus\|step_appended\|episode_finalized" var/logs/*.log | head -20

# 4. Check agent_id labels
grep -i "agent_id" var/logs/*.log | grep -E "proxy|controller|tab" | head -20

# 5. Check for overflow events
grep -i "overflow\|queue.*full" var/logs/*.log | head -20

# 6. Check game_id in payloads
sqlite3 var/telemetry/telemetry.sqlite "SELECT DISTINCT game_id FROM steps LIMIT 5;"

# 7. Check frame_ref values
sqlite3 var/telemetry/telemetry.sqlite "SELECT DISTINCT frame_ref FROM steps WHERE frame_ref IS NOT NULL LIMIT 5;"
```

---

## Priority 1: Critical Deadlock Issues

### Issue 1.1: Credit Backpressure Chicken-and-Egg Deadlock

**Problem**: Producer won't publish UI events until it receives credits, but the first credit grant is sent by the Live tab AFTER the tab is created. If the tab is only created after the first step arrives, this creates a deadlock.

**Flow**:
```
Worker sends step → No tab exists → No credits granted → Step dropped
                                                          ↓
                                                    No step → No tab
```

**Root Cause**:
- Credits initialized only when first event arrives (lazy initialization)
- Tab created only after first event received
- Credit grant sent only after tab created
- Race condition: credits needed before tab exists

**Diagnostic Steps**:
1. Check `CreditManager.initialize_stream()` - when is it called?
2. Check `LiveTelemetryTab._grant_initial_credits()` - when is it called?
3. Check logs for "Initialized credits" vs "Tab registered"
4. Verify credit grant is published to RunBus CONTROL topic

**Fix Strategy**:
- Pre-initialize credits when run starts (not when first event arrives)
- Grant initial credits in LiveTelemetryController BEFORE tab creation
- Store pending credits in buffer if tab not yet created
- Ensure credits flow from controller → producer, not tab → producer

---

### Issue 1.2: RunBus vs Qt-Signal Delivery Split

**Problem**: Two separate paths to UI (Qt signals and RunBus). If one got unplugged, UI routing stalls.

**Expected Flow**:
```
TelemetryAsyncHub._drain_loop()
  ├─ Publish to RunBus (STEP_APPENDED, EPISODE_FINALIZED)
  └─ Emit Qt signals (TelemetryBridge)
       ↓
LiveTelemetryController
  ├─ Subscribe to RunBus
  └─ Route to tabs
```

**Diagnostic Steps**:
1. Search `streams.py` for `bus.publish()` calls
2. Verify publishing happens in `_drain_loop()` (lines 535-590)
3. Check if publishing is conditional or filtered
4. Verify `LiveTelemetryController._run()` subscribes to RunBus
5. Check queue sizes (should be 64 for UI)

**Fix Strategy**:
- Verify both paths are active
- Add logging for every publish/subscribe
- Ensure no filtering or conditional logic blocks events
- Test end-to-end event delivery

---

## Priority 2: Data Model Mismatches

### Issue 2.1: Episode Identity Mismatch

**Problem**: Protobuf carries `episode_index` (uint64), Python uses `episode_id` (string). Mismatch causes episodes to "vanish".

**Mapping**:
```
Protobuf: episode_index (0, 1, 2, ...)
   ↓
Python: episode_id (e.g., "FrozenLake-v1-seed_1-ep0001-a1b2c3d4")
   ↓
Database: episode_id (PRIMARY KEY)
```

**Diagnostic Steps**:
1. Check `_episode_from_proto()` in `service.py` (line 759)
2. Verify episode_id reconstruction from episode_index
3. Query database: `SELECT episode_id FROM episodes LIMIT 5`
4. Check if episode_id format is consistent
5. Verify no episodes are lost in conversion

**Fix Strategy**:
- Carry both `episode_index` and `episode_id` end-to-end
- Add `episode_id` field to RunEpisode proto
- Ensure deterministic episode_id generation
- Add validation to catch mismatches

---

### Issue 2.2: Agent ID Filtering Mismatch

**Problem**: Proxy injects `agent_id` (e.g., "agent_1"), but tab key uses different format. Events silently dropped.

**Diagnostic Steps**:
1. Check proxy default: `--agent-id agent_1` (line 304 in trainer_telemetry_proxy.py)
2. Check tab key creation: `(run_id, agent_id)` in controller
3. Check for transformations (underscores, dashes, case)
4. Log agent_id at every step
5. Verify tab key matches event agent_id

**Fix Strategy**:
- Standardize agent_id format across all components
- Add validation to catch mismatches
- Log agent_id at every routing decision
- Test with multiple agents

---

## Priority 3: Buffer & Display Issues

### Issue 3.1: Buffer Overflow Masking

**Problem**: UI buffers overflow silently. Tab counter doesn't advance even though events are flowing.

**Diagnostic Steps**:
1. Check UI queue size: 64 (line 259 in live_telemetry.py)
2. Check writer queue size: 256-512
3. Check for overflow logs
4. Monitor queue depth during high-frequency training
5. Verify overflow feedback is visible

**Fix Strategy**:
- Align credits with buffer capacity
- Add overflow metrics and visibility
- Implement backpressure feedback
- Test with high-frequency events

---

### Issue 3.2: game_id Context Propagation

**Problem**: Dynamic tabs receive `game_id=None`, fall back to FROZEN_LAKE.

**Diagnostic Steps**:
1. Check `_create_agent_tabs_for()` (line 1144 in main_window.py)
2. Verify game_id extraction from first_payload
3. Check if game_id is available in run metadata
4. Verify RendererContext initialization
5. Check renderer fallback logic (line 53 in grid.py)

**Fix Strategy**:
- Extract game_id from run metadata (not first payload)
- Pass game_id to tab constructors
- Initialize self._game_id before first render
- Add assertions to catch missing game_id

---

### Issue 3.3: Frame Reference Generation

**Problem**: `frame_ref` is None because `build_frame_reference()` returns None by default.

**Diagnostic Steps**:
1. Check adapter implementations
2. Verify `build_frame_reference()` is called
3. Query database: `SELECT DISTINCT frame_ref FROM steps`
4. Check if frames are saved to disk
5. Verify frame_ref format

**Fix Strategy**:
- Implement `build_frame_reference()` in all adapters
- Generate timestamped filenames
- Persist frames before telemetry emission
- Implement frame storage service

---

## Data Flow Verification

Expected end-to-end flow:

```
Worker (JSONL)
  ↓
Proxy (gRPC)
  ↓
TelemetryAsyncHub._drain_loop()
  ├─ Normalize payload
  ├─ Publish to RunBus (STEP_APPENDED)
  ├─ Publish to RunBus (EPISODE_FINALIZED)
  └─ Emit Qt signals
       ↓
LiveTelemetryController._process_events()
  ├─ Receive from RunBus
  ├─ Extract agent_id
  ├─ Create tab if needed
  └─ Route to tab
       ↓
LiveTelemetryTab.add_step()
  ├─ Buffer step
  ├─ Update stats
  └─ Render (throttled)
```

---

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test full data flow
4. **Stress Tests**: Test with high-frequency events
5. **Multi-Agent Tests**: Test with multiple agents

---

## Success Criteria

- [ ] All 7 issues have root cause identified
- [ ] Diagnostic steps can be run without code changes
- [ ] Each issue has clear fix strategy
- [ ] Tests verify each fix
- [ ] No regressions in existing functionality
- [ ] Performance impact < 5%

