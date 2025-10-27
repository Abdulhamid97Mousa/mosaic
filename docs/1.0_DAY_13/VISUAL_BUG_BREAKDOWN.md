# Telemetry Display Issues: Visual Breakdown

## Issue #1: Counter Shows Buffer Size (DATA CORRUPTION)

### Current (Broken) Flow
```
Telemetry Payload
├─ episode_index: 17
├─ step_index: 8
└─ reward: 0.0
     │
     └─> LiveTelemetryTab
          ├─ Add to _step_buffer (deque, maxlen=100)
          └─ Display Counter: len(_step_buffer) = 100 ✗
               └─> Widget shows "Step: 100" (WRONG!)
```

### Why It's Broken

```
Training Progress:        Buffer State:            Counter Display:
─────────────────         ─────────────            ─────────────────
Step 0                    [empty]                  Step: 0
Step 50                   [50 items]               Step: 50   ✓ Correct
Step 100                  [100 items, full]        Step: 100  ✓ Correct
Step 101                  [100 items, rotated]     Step: 100  ✗ STALE
Step 150                  [100 items, rotated]     Step: 100  ✗ STALE
Step 200 (Episode 2)      [100 items, rotated]     Step: 100  ✗ STALE

Data Corruption:
┌─────────────────────────────────────────────────┐
│ User sees Step: 100 for multiple episodes       │
│ Can't tell if training is progressing           │
│ Can't see when episode boundaries occur         │
│ Can't match counter with telemetry table        │
└─────────────────────────────────────────────────┘
```

### Fixed Flow
```
Telemetry Payload
├─ episode_index: 17
├─ step_index: 8
└─ reward: 0.0
     │
     └─> LiveTelemetryTab
          ├─ Add to _step_buffer (for rendering)
          ├─ Track _current_episode_index = 17
          ├─ Track _current_step_in_episode = 8
          ├─ Detect episode boundary (index changes)
          ├─ Reset step counter on boundary
          └─ Display Counter: "Episode: 17 Step: 8" ✓
               └─> Widget shows correct metrics
```

### Before vs After
```
BEFORE (Broken):
┌────────────────────────────────────────────┐
│ Live Training Tab                          │
├────────────────────────────────────────────┤
│ Steps: 100 | Episodes: 20 [STALE!]        │ ← Always shows 100
│                                            │
│ Episode │ Steps │ Avg Reward              │
│───────────────────────────────────────────│
│   17    │   8   │   0.125                 │ ← Actual data correct
│   16    │  100  │   0.050                 │
│   15    │  100  │   0.045                 │
└────────────────────────────────────────────┘
   ↑ MISMATCH! Counter shows 100, table shows 8

AFTER (Fixed):
┌────────────────────────────────────────────┐
│ Live Training Tab                          │
├────────────────────────────────────────────┤
│ Episode: 17 Step: 8 [CORRECT]              │ ← Shows real metrics
│                                            │
│ Episode │ Steps │ Avg Reward              │
│───────────────────────────────────────────│
│   17    │   8   │   0.125                 │ ← Consistent!
│   16    │  100  │   0.050                 │
│   15    │  100  │   0.045                 │
└────────────────────────────────────────────┘
   ✓ MATCH! Counter and table synchronized
```

---

## Issue #2: Reward Shows "+0.000"

### Data Flow Analysis

```
FrozenLake Environment
│
├─ Normal step: env.step() → reward=0.0 ✓
└─ Goal step: env.step() → reward=1.0 ✓
     │
     └─> SessionController._record_step()
          │
          ├─ Extracts: reward = step_result[1]
          └─ Expected: reward = 1.0 ✓
               │
               └─> Telemetry Producer (gRPC)
                    │
                    ├─ Creates payload with reward
                    └─ Expected: {"reward": 1.0} ✓
                         │
                         └─> TelemetryBridge
                              │
                              └─> LiveTelemetryTab._on_telemetry_event()
                                   │
                                   ├─ Extract: reward = _get_field(payload, "reward", default=0.0)
                                   │              ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                                   │ If field missing → Falls back to 0.0
                                   │
                                   └─> Format: f"{reward:+.3f}"
                                        │
                                        └─> Display: "+0.000" (WRONG!)
                                             or: "+1.000" (CORRECT if source has 1.0)
```

### Problem Diagnosis Tree

```
                    Reward shows "+0.000"
                            │
                ┌───────────┴───────────┐
                │                       │
        Source has 0.0         Field name wrong
                │                       │
        ┌───────┴────────┐              │
        │                │              │
    Adapter    SessionController   _get_field() uses
    returns 0  extracts wrong      "reward" but payload
              value from env       has "rewardValue" etc
              
        → Check              → Check              → Check
        FrozenLakeAdapter    _record_step()      TelemetryBridge
        .step()             payload capture     field naming
```

### Expected vs Actual

```
EXPECTED DATA FLOW (goal step):
env.step()         SessionController     TelemetryBridge    UI
reward=1.0    →    reward=1.0       →    {reward:1.0}   →   "+1.000" ✓

ACTUAL DATA FLOW (broken):
env.step()         SessionController     TelemetryBridge    UI
reward=1.0    →    reward=???       →    {reward:0.0}   →   "+0.000" ✗
             ???: Either not captured correctly, or sent as 0.0
```

### Side-by-Side Comparison

```
┌──────────────────────────┬──────────────────────────┐
│     Normal Step          │      Goal Achievement    │
├──────────────────────────┼──────────────────────────┤
│ Env: reward=0.0          │ Env: reward=1.0          │
│ UI: +0.000       ✓       │ UI: +0.000       ✗       │
│ Reward: Invisible        │ Reward: Invisible        │
│ Feedback: Missing        │ Feedback: BROKEN!        │
└──────────────────────────┴──────────────────────────┘
```

---

## Issue #3: Human Mode Counter Frozen

### Signal Routing Comparison

```
AGENT MODE (Working):
┌──────────────────────────────────────┐
│ Worker (gRPC)                        │
│ Sends: {episode_index, step_index}   │
└──────────────┬───────────────────────┘
               │
               └─→ TelemetryAsyncHub
                   │
                   ├─→ Qt Signal: step_received(payload)
                   │
                   └─→ LiveTelemetryTab.on_step(payload)
                       └─→ Counter increments ✓

HUMAN MODE (Broken?):
┌──────────────────────────────────────┐
│ SessionController.human_step()        │
│ ???: Sends telemetry?                │
└──────────────┬───────────────────────┘
               │
               ├─ Path 1: Does NOT emit telemetry → on_step() never called ✗
               │
               ├─ Path 2: Emits to different sink → doesn't reach UI ✗
               │
               └─ Path 3: Payload missing episode_index → counter fails silently ✗
                   
                   → Counter appears frozen ✗
```

### State Machine Comparison

```
AGENT MODE:
Idle
  ↓ (Agent takes action)
  └─→ Worker sends telemetry
      └─→ TelemetryBridge emits signal
          └─→ UI updates counter ✓
              └─→ Episode 1 Step: 0,1,2,...50
                  └─→ Episode boundary detected
                      └─→ Counter resets
                          └─→ Episode 2 Step: 0,1,2,... ✓

HUMAN MODE (Broken):
Idle
  ↓ (User clicks move button)
  └─→ SessionController.human_step() called
      └─→ ???: Telemetry emitted? Unknown
          └─→ ???: Signal routed? Unknown
              └─→ UI counter: No update ✗
                  └─→ Episode 1 Step: 0,0,0,... (frozen)
                      └─→ User sees frozen mode ✗
```

### Verification Checklist

```
Human mode telemetry verification:

□ SessionController.human_step() calls telemetry producer?
  ├─ If YES → Check if payload has episode_index, step_index
  │          Are they correct values?
  └─ If NO  → Human telemetry never emitted (BROKEN)

□ Payload structure matches agent payload?
  ├─ If YES → Signal routing issue?
  │          └─ Check if on_step() callback receives it
  └─ If NO  → Fields missing or named differently

□ on_step() callback receives human payloads?
  ├─ If YES → Counter should update (but doesn't?)
  │          └─ Debug counter update logic
  └─ If NO  → Signal routing broken

□ Episode boundary detection works in human mode?
  └─ Does episode_index field match?
     Does terminated/truncated trigger boundary reset?
```

---

## Integration Architecture

```
Current (Broken) Architecture:
┌─────────────────────────────────────────────────────────┐
│                  Live Training Tab                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Counter Widget            Telemetry Tables             │
│  ─────────────────         ─────────────────            │
│  "Step: 100" ✗     ← len(buffer)                        │
│                    ← Uses buffer size as metric        │
│                                                         │
│  Episode │ Step │ Reward                                │
│  ─────────────────────────                              │
│  17      │  8   │ +0.000 ✗                              │
│  16      │ 100  │ +0.000 ✗                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
Data corruption: Counter and table show different values!

Fixed Architecture:
┌─────────────────────────────────────────────────────────┐
│                  Live Training Tab                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Counter Widget            Telemetry Tables             │
│  ─────────────────         ─────────────────            │
│  "Episode: 17 Step: 8" ✓  ← Tracked fields            │
│                ↑                                         │
│                └─ current_episode_index = 17            │
│                └─ current_step_in_episode = 8           │
│                └─ Reset on episode boundary             │
│                                                         │
│  Episode │ Step │ Reward                                │
│  ─────────────────────────                              │
│  17      │  8   │ +0.000 → +1.000 (after fix)          │
│  16      │ 100  │ +0.000 → mixed values (after fix)    │
│                                                         │
└─────────────────────────────────────────────────────────┘
Data synchronized: Counter and table consistent!
```

---

## Summary Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    3 Bugs, 3 Root Causes                      │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  BUG #1: Counter stuck at 100                                │
│  ─────────────────────────────                               │
│  ROOT: len(buffer) instead of episode/step tracking          │
│  FIX:  Add tracking fields, reset per episode                │
│  TYPE: Architectural (design flaw)                           │
│  RISK: Low (isolated change)                                 │
│                                                               │
│  BUG #2: Reward shows +0.000                                 │
│  ──────────────────────────                                  │
│  ROOT: Telemetry producer sending reward=0.0                 │
│  FIX:  Trace and fix source (adapter/controller)             │
│  TYPE: Data source (implementation issue)                    │
│  RISK: Medium (multiple possible sources)                    │
│                                                               │
│  BUG #3: Human mode counter frozen                           │
│  ──────────────────────────────                              │
│  ROOT: Human telemetry not reaching UI callbacks             │
│  FIX:  Verify/fix signal routing                             │
│  TYPE: Signal routing (pipeline issue)                       │
│  RISK: Medium (multiple paths to check)                      │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Code Fix Locations Map

```
gym_gui/
├── ui/
│   ├── widgets/
│   │   ├── live_telemetry_tab.py ← FIX #1 (line 784)
│   │   │                          ← FIX #2 (line 369, 428)
│   │   │
│   │   └── agent_online_grid_tab.py ← FIX #3 (lines 18-20, 98-102)
│   │
│   └── main_window.py ← Signal routing (for FIX #3 verification)
│
├── core/
│   └── adapters/
│       └── toy_text.py ← FIX #2 (trace reward in FrozenLakeAdapter)
│
├── controllers/
│   └── session.py ← FIX #2 (trace reward in _record_step)
│                  ← FIX #3 (human_step implementation)
│
└── telemetry/
    └── bridge.py ← FIX #2 (payload serialization)
                  ← FIX #3 (human payload routing)
```
