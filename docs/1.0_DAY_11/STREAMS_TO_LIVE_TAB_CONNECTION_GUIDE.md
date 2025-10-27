# Complete Connection Guide: streams.py → Live-{agent-id}-Training Tab

## Overview

The `streams.py` file is **NOT truncated** - it's a complete, refactored telemetry streaming system. This guide explains how telemetry flows from the worker process through `streams.py` to the Live-{agent-id}-Training tab in the GUI.

## Complete Telemetry Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ WORKER PROCESS (spadeBDI_RL_refactored/worker.py)                  │
│ - BDITrainer or HeadlessTrainer runs episodes                       │
│ - Emits JSONL telemetry to stdout via TelemetryEmitter              │
└────────────────────────┬────────────────────────────────────────────┘
                         │ JSONL Events (step, episode, run_started, etc.)
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ TELEMETRY PROXY (gym_gui/services/trainer/trainer_telemetry_proxy)  │
│ - Parses JSONL from worker stdout                                   │
│ - Converts to gRPC protobuf messages                                │
│ - Sends to Trainer Daemon via gRPC                                  │
└────────────────────────┬────────────────────────────────────────────┘
                         │ gRPC RunStep, RunEpisode messages
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ TRAINER DAEMON (gym_gui/services/trainer/service.py)                │
│ - Receives gRPC messages from proxy                                 │
│ - Stores in database                                                │
│ - Broadcasts to all connected GUI clients                           │
└────────────────────────┬────────────────────────────────────────────┘
                         │ gRPC stream_run_steps(), stream_run_episodes()
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ GUI CLIENT - TelemetryAsyncHub (streams.py)                         │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ _stream_steps() & _stream_episodes()                         │   │
│ │ - Subscribe to gRPC streams from daemon                      │   │
│ │ - Receive protobuf payloads                                  │   │
│ │ - Put into shared asyncio.Queue                              │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                         ↓                                            │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ _drain_loop()                                                │   │
│ │ - Dequeue from shared queue                                  │   │
│ │ - Convert protobuf to dict via _proto_to_dict()             │   │
│ │ - Normalize payload via _normalize_payload()                 │   │
│ │ - Create TelemetryStep/TelemetryEpisode objects              │   │
│ │ - Call bridge.emit_step() or bridge.emit_episode()           │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                         ↓                                            │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ TelemetryBridge (Qt signals)                                 │   │
│ │ - emit_step(TelemetryStep) → posts custom Qt event           │   │
│ │ - emit_episode(TelemetryEpisode) → posts custom Qt event     │   │
│ │ - Signals: step_received, episode_received, queue_overflow   │   │
│ └──────────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────┘
                         │ Qt signals on main thread
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LiveTelemetryController (gym_gui/controllers/live_telemetry.py)     │
│ - Subscribes to TelemetryBridge signals                             │
│ - Routes step/episode to appropriate LiveTelemetryTab               │
│ - Handles tab creation on first step (run_tab_requested signal)     │
│ - Manages per-run buffer sizes and render throttling                │
└────────────────────────┬────────────────────────────────────────────┘
                         │ Calls tab.add_step() or tab.add_episode()
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LiveTelemetryTab (gym_gui/ui/widgets/live_telemetry_tab.py)         │
│ - add_step(payload) → buffers step, updates UI tables               │
│ - add_episode(payload) → buffers episode, updates UI tables         │
│ - Renders live grid visualization (throttled)                       │
│ - Displays step/episode tables with details                         │
│ - Implements credit-based backpressure via RunBus                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components in streams.py

### 1. **_normalize_payload(payload, run_id)**
- Converts protobuf messages to dictionaries
- Ensures all payloads have required fields: `run_id`, `agent_id`, `episode_index`, `step_index`, `payload_version`
- Handles both dict and protobuf inputs

### 2. **TelemetryBridge (Qt signals)**
- `step_received` signal: emitted when step arrives
- `episode_received` signal: emitted when episode arrives
- `queue_overflow` signal: emitted when buffer is full
- `run_completed` signal: emitted when run finishes
- Uses custom `_TelemetryEvent` for thread-safe delivery

### 3. **TelemetryAsyncHub (main orchestrator)**
- `subscribe_run(run_id, client)`: Start streaming for a run
- `_stream_steps()`: Async task receiving step stream from daemon
- `_stream_episodes()`: Async task receiving episode stream from daemon
- `_drain_loop()`: Async task consuming from shared queue and emitting signals
- Handles reconnection with exponential backoff
- Manages per-run buffers via `RunStreamBuffer`

### 4. **RunStreamBuffer**
- Circular buffer for steps and episodes
- Tracks dropped events when buffer is full
- Per-run tracking to prevent cross-run data mixing

## Connection Points

### From streams.py to LiveTelemetryController

**File**: `gym_gui/controllers/live_telemetry.py`

```python
# Line 69-70: Subscribe to bridge signals
self._hub.bridge.queue_overflow.connect(self._on_queue_overflow)
self._hub.bridge.run_completed.connect(self._on_run_completed_from_bridge)

# Line 258-263: Subscribe to RunBus for independent event delivery
self._step_queue = self._bus.subscribe_with_size(Topic.STEP_APPENDED, "live-ui", 64)
self._episode_queue = self._bus.subscribe_with_size(Topic.EPISODE_FINALIZED, "live-ui", 64)

# Line 296-410: Process events from RunBus queues
async def _process_step_queue(self) -> None:
    # Route to tab or buffer until tab is registered
    tab = self._tabs.get((run_id, agent_id))
    if tab is not None:
        tab.add_step(evt.payload)
    else:
        # Request tab creation on first step
        self.run_tab_requested.emit(run_id, agent_id, tab_title)
        # Buffer until tab is registered
        self._step_buffer[key].append(evt.payload)
```

### From LiveTelemetryController to LiveTelemetryTab

**File**: `gym_gui/ui/widgets/live_telemetry_tab.py`

```python
# Line 247-293: Receive step from controller
def add_step(self, payload: Any) -> None:
    self._steps_received += 1
    self._step_buffer.append(payload)
    
    # Schedule UI updates on main thread
    QtCore.QMetaObject.invokeMethod(
        self, "_render_step_row_main_thread",
        QtCore.Qt.ConnectionType.QueuedConnection,
        QtCore.Q_ARG(object, payload)
    )
    
    # Check queue depth and grant credits (backpressure)
    self._check_and_grant_credits()

# Line 480-496: Receive episode from controller
def add_episode(self, payload: Any) -> None:
    self._episode_buffer.append(payload)
    QtCore.QMetaObject.invokeMethod(
        self, "_render_episode_row_main_thread",
        QtCore.Qt.ConnectionType.QueuedConnection,
        QtCore.Q_ARG(object, payload)
    )
```

## Data Flow Example: Single Step

1. **Worker emits**: `{"type": "step", "run_id": "run-123", "episode": 5, "step_index": 10, "reward": 1.0, ...}`
2. **Proxy converts**: Protobuf `RunStep` message
3. **Daemon broadcasts**: gRPC `stream_run_steps()` to all clients
4. **TelemetryAsyncHub._stream_steps()**: Receives protobuf, puts in queue
5. **TelemetryAsyncHub._drain_loop()**: Dequeues, normalizes, creates `TelemetryStep`
6. **TelemetryBridge.emit_step()**: Posts custom Qt event
7. **Qt Main Thread**: Receives event, emits `step_received` signal
8. **LiveTelemetryController._process_step_queue()**: Routes to tab
9. **LiveTelemetryTab.add_step()**: Buffers, schedules UI update
10. **LiveTelemetryTab._render_step_row_main_thread()**: Adds row to table, renders grid

## Critical Design Decisions

### Why Two Paths? (TelemetryBridge + RunBus)

- **TelemetryBridge**: Fast UI path with Qt signals (immediate rendering)
- **RunBus**: Durable path for database persistence (all events recorded)
- Both paths receive ALL events - no sampling at source
- UI rendering throttling happens at LiveTelemetryTab level (not in streams.py)

### Why Custom Qt Events?

- `_TelemetryEvent` ensures thread-safe delivery from async hub to main thread
- `QtCore.QCoreApplication.postEvent()` is the only safe way to cross threads
- Avoids race conditions and segmentation faults

### Why Reconnection Logic?

- gRPC streams can close due to network hiccups or episode boundaries
- `_stream_steps()` and `_stream_episodes()` reconnect automatically
- Exponential backoff prevents hammering the daemon
- `since_seq` parameter ensures no data loss on reconnect

## Testing the Connection

```bash
# 1. Start trainer daemon
python -m gym_gui.services.trainer.service

# 2. Run worker with gRPC telemetry
python -m spadeBDI_RL_refactored.worker --grpc --config config.json

# 3. Check logs for:
# - "TelemetryAsyncHub initialized"
# - "Step stream opened for run: run-123"
# - "Emitting bridge.step_received"
# - "Step event received from RunBus"
# - "Tab registered and ready for telemetry"
```

## Summary

The `streams.py` file is a complete, production-ready telemetry streaming system that:
- Receives gRPC streams from the trainer daemon
- Normalizes protobuf payloads to consistent dictionaries
- Delivers events to the UI via Qt signals (thread-safe)
- Publishes to RunBus for database persistence
- Handles reconnection and backpressure
- Routes to appropriate LiveTelemetryTab based on (run_id, agent_id)

No code is missing - the file is fully functional and ready to use!

