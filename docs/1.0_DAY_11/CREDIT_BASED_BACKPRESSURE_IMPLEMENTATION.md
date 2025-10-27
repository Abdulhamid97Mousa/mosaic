# Credit-Based Backpressure Implementation

## Overview

Implemented a sophisticated credit-based backpressure mechanism to solve telemetry queue overflow issues. This elegant solution prevents overflow at the source by controlling the producer rate, rather than just handling overflow after it occurs.

## Problem Statement

**Issue 1: Telemetry Queue Overflow Warnings**
- UI queue (64 events) was overflowing because the producer (worker) sends telemetry faster than the UI can consume
- Rendering throttle was applied AFTER events were received, not BEFORE
- Simply increasing queue sizes is a band-aid solution that doesn't address the root cause

**Issue 2: UI Rendering Throttle Slider Not Working**
- Slider value from Train Agent Form was not being passed to LiveTelemetryTab
- Rendering throttle interval was hardcoded to 20 steps
- Slider value was set in environment variables but never used by the GUI

## Solution Architecture

### 1. Credit Manager (`gym_gui/telemetry/credit_manager.py`)

New module that implements the credit system:
- **Initial Credits**: Each stream starts with 200 credits
- **Credit Consumption**: Producer consumes 1 credit per event published to UI
- **Credit Granting**: UI sends CREDIT_GRANT messages when queue drops below threshold
- **Overflow Tracking**: Tracks total events dropped due to no credits

**Key Methods:**
- `get_credits()` - Get current credits for a stream
- `consume_credit()` - Attempt to consume one credit (returns True if successful)
- `grant_credits()` - Grant credits to a stream (called when UI queue drops)
- `get_dropped_count()` - Get total events dropped due to no credits

### 2. Control Topic (`gym_gui/telemetry/events.py`)

Added `Topic.CONTROL` to the event enum for control plane messages:
- Used for credit grants and backpressure signals
- Separate from data plane (STEP_APPENDED, EPISODE_FINALIZED)
- Enables bidirectional communication between UI and producer

### 3. LiveTelemetryTab Credit Granting (`gym_gui/ui/widgets/live_telemetry_tab.py`)

Enhanced to send credit grants to the producer:
- **Initial Grant**: Sends 200 credits when tab is created
- **Threshold Monitoring**: Checks queue depth after each step
- **Dynamic Granting**: Grants 100 credits when queue drops below 50% threshold
- **Control Messages**: Publishes CREDIT_GRANT events via RunBus

**New Methods:**
- `_grant_initial_credits()` - Send initial credit grant on tab creation
- `_check_and_grant_credits()` - Monitor queue and grant credits as needed
- `set_render_throttle_interval()` - Set rendering throttle from slider value

### 4. TelemetryAsyncHub Credit Checking (`gym_gui/services/trainer/streams.py`)

Updated to respect credits before publishing to UI:
- **Credit Checking**: Checks available credits before queue.put_nowait()
- **Selective Publishing**: Skips UI publish when credits=0 (but DB writes continue)
- **Control Listener**: Listens for CREDIT_GRANT messages from UI
- **No Data Loss**: All events always written to database regardless of credits

**Key Changes:**
- `_listen_for_control_messages()` - Subscribe to CONTROL topic for a run
- `_process_control_messages()` - Process credit grants from UI
- Updated `_stream_steps()` and `_stream_episodes()` to check credits

### 5. LiveTelemetryController Throttle Wiring (`gym_gui/controllers/live_telemetry.py`)

New methods to wire slider value to tabs:
- `set_render_throttle_for_run()` - Store throttle value per run
- `register_tab()` - Apply throttle when tab is created
- Extracts throttle from environment variables and applies to tabs

### 6. MainWindow Integration (`gym_gui/ui/main_window.py`)

Updated to pass rendering throttle to controller:
- Extracts `TELEMETRY_SAMPLING_INTERVAL` from config
- Calls `set_render_throttle_for_run()` before subscribing
- Ensures slider value is applied to all tabs for the run

## Data Flow

### Credit Grant Flow
```
1. LiveTelemetryTab created
   ↓
2. _grant_initial_credits() sends CREDIT_GRANT(200)
   ↓
3. Event published to RunBus CONTROL topic
   ↓
4. TelemetryAsyncHub._process_control_messages() receives it
   ↓
5. CreditManager.grant_credits() updates credit counter
   ↓
6. Producer can now publish more events to UI
```

### Event Publishing Flow
```
1. Worker sends step/episode to TelemetryAsyncHub
   ↓
2. Extract agent_id from payload
   ↓
3. Check: has_credits = credit_manager.consume_credit(run_id, agent_id)
   ↓
4. If has_credits:
     - Publish to UI queue (queue.put_nowait)
   Else:
     - Skip UI publish (but continue DB writes)
     - Log debug message
   ↓
5. All events always written to database (no sampling)
```

### Rendering Throttle Flow
```
1. User sets slider to value N in Train Agent Form
   ↓
2. TELEMETRY_SAMPLING_INTERVAL = N in environment
   ↓
3. MainWindow._on_training_submitted() extracts value
   ↓
4. Calls live_controller.set_render_throttle_for_run(run_id, N)
   ↓
5. When tab is created, register_tab() applies throttle
   ↓
6. LiveTelemetryTab.set_render_throttle_interval(N) sets it
   ↓
7. add_step() uses throttle to render every Nth step
```

## Benefits

1. **Prevents Overflow at Source**: Producer respects credits before publishing
2. **No Data Loss**: All events written to database regardless of credits
3. **Self-Regulating**: System adapts to UI processing speed automatically
4. **Cleaner Logs**: No more repeated overflow warnings
5. **Responsive UI**: Rendering throttle slider now works correctly
6. **Elegant Design**: Separates control plane from data plane

## Files Modified

1. `gym_gui/telemetry/credit_manager.py` - NEW: Credit management system
2. `gym_gui/telemetry/events.py` - Added CONTROL topic
3. `gym_gui/services/trainer/streams.py` - Credit checking in producer
4. `gym_gui/ui/widgets/live_telemetry_tab.py` - Credit granting in UI
5. `gym_gui/controllers/live_telemetry.py` - Throttle wiring
6. `gym_gui/ui/main_window.py` - Throttle extraction and passing

## Testing Recommendations

1. **Overflow Prevention**: Monitor logs for "Skipped UI publish due to no credits"
2. **Credit Grants**: Verify CREDIT_GRANT messages in logs
3. **Rendering Throttle**: Set slider to different values and verify update frequency
4. **Data Integrity**: Verify all events in database despite UI skips
5. **Performance**: Monitor CPU and memory usage during high-frequency training

## Future Enhancements

1. **Adaptive Credits**: Adjust initial credit allocation based on system performance
2. **Per-Agent Throttling**: Different throttle rates for different agents
3. **Metrics Dashboard**: Display credit stats and overflow metrics in UI
4. **Backpressure Signals**: Send explicit backpressure signals to worker

