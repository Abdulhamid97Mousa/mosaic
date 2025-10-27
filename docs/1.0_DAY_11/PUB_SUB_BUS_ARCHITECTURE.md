# Pub/Sub Bus Architecture - Ray-Inspired Dual-Path Design

## Problem Statement

The current telemetry system has a single blocking path that serves two fundamentally different consumers:

1. **Live UI**: Needs ultra-low latency, best-effort delivery, per-subscriber buffers
2. **Durable Storage**: Needs lossless persistence, can tolerate latency, requires consistency

This causes:
- Head-of-line blocking: UI waits for DB I/O
- "First step only" symptom: Queue overflows when DB writes block
- Replay stays empty: UI can't distinguish between "no data yet" and "data in DB"

## Solution: Dual-Path Architecture

Inspired by Ray's design, we separate:

- **Live Path (Pub/Sub Bus)**: Fast, ephemeral, per-subscriber buffers
- **Durable Path (SQLite)**: Asynchronous, lossless, background persistence

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Worker (spadeBDI_RL_refactored/worker.py)                  │
│ Emits: step(run_id, episode, step_index, ...)              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ TelemetryProxy (gym_gui/services/trainer/streams.py)       │
│ Normalizes JSONL → TelemetryEvent                          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
        ┌────────────┴────────────┐
        ↓                         ↓
   ┌─────────────┐         ┌──────────────┐
   │ LIVE PATH   │         │ DURABLE PATH │
   │ (Pub/Sub)   │         │ (SQLite)     │
   └─────────────┘         └──────────────┘
        ↓                         ↓
   ┌─────────────┐         ┌──────────────┐
   │ RunBus      │         │ DB Sink Task │
   │ (in-memory) │         │ (async)      │
   └─────────────┘         └──────────────┘
        ↓                         ↓
   ┌─────────────┐         ┌──────────────┐
   │ UI Subs     │         │ SQLite Store │
   │ (Online)    │         │ (persistent) │
   └─────────────┘         └──────────────┘
```

## Key Components

### 1. Events & Topics (`gym_gui/telemetry/events.py`)

```python
class Topic(Enum):
    RUN_STARTED = auto()
    RUN_HEARTBEAT = auto()
    RUN_COMPLETED = auto()
    STEP_APPENDED = auto()
    EPISODE_FINALIZED = auto()
    OVERFLOW = auto()

@dataclass(slots=True)
class TelemetryEvent:
    topic: Topic
    run_id: str
    agent_id: Optional[str]
    seq_id: int              # Strictly increasing per stream
    ts_iso: str              # ISO-8601 timestamp
    payload: Dict[str, Any]  # Event-specific data
```

### 2. RunBus (`gym_gui/telemetry/run_bus.py`)

```python
class RunBus:
    def subscribe(topic: Topic, subscriber_id: str) -> asyncio.Queue
    def unsubscribe(topic: Topic, subscriber_id: str) -> None
    def publish(evt: TelemetryEvent) -> None  # Non-blocking
    def overflow_stats() -> Dict[str, int]
    def queue_sizes() -> Dict[str, int]
```

**Key properties**:
- Non-blocking publish: producers never wait
- Per-subscriber queues: independent buffers
- Drop policy: oldest events dropped if full
- Overflow tracking: metrics for monitoring

## Integration Points

### 1. Publish from TelemetryProxy

**File**: `gym_gui/services/trainer/streams.py`

After normalizing JSONL payload:

```python
from gym_gui.telemetry.events import TelemetryEvent, Topic
from gym_gui.telemetry.run_bus import get_bus

bus = get_bus()

# When it's a step
evt = TelemetryEvent(
    topic=Topic.STEP_APPENDED,
    run_id=payload["run_id"],
    agent_id=payload.get("agent_id"),
    seq_id=self._seq_counter,
    ts_iso=payload["ts"],
    payload=payload,
)
bus.publish(evt)

# When episode finalized
evt = TelemetryEvent(
    topic=Topic.EPISODE_FINALIZED,
    run_id=payload["run_id"],
    agent_id=payload.get("agent_id"),
    seq_id=self._seq_counter,
    ts_iso=payload["ts"],
    payload=payload,
)
bus.publish(evt)
```

### 2. Subscribe in UI (LiveTelemetryController)

**File**: `gym_gui/services/telemetry/live_controller.py`

```python
from gym_gui.telemetry.events import Topic
from gym_gui.telemetry.run_bus import get_bus

bus = get_bus()
queue = bus.subscribe(Topic.STEP_APPENDED, f"ui-{run_id}-{agent_id}")

# In async loop:
while True:
    evt = await queue.get()
    # Render immediately (no DB wait!)
    self._render_step(evt.payload)
```

### 3. Subscribe in DB Sink (Background Task)

**File**: `gym_gui/telemetry/db_sink.py` (NEW)

```python
from gym_gui.telemetry.events import Topic
from gym_gui.telemetry.run_bus import get_bus

async def db_sink_task():
    bus = get_bus()
    step_queue = bus.subscribe(Topic.STEP_APPENDED, "db-sink-steps")
    episode_queue = bus.subscribe(Topic.EPISODE_FINALIZED, "db-sink-episodes")
    
    pending_steps = []
    
    while True:
        # Batch steps
        try:
            evt = step_queue.get_nowait()
            pending_steps.append(evt.payload)
            if len(pending_steps) >= 32:
                store.flush_steps(pending_steps)
                pending_steps = []
        except asyncio.QueueEmpty:
            pass
        
        # Handle episodes
        try:
            evt = episode_queue.get_nowait()
            if pending_steps:
                store.flush_steps(pending_steps)
                pending_steps = []
            store.write_episode(evt.payload)
        except asyncio.QueueEmpty:
            pass
        
        await asyncio.sleep(0.1)
```

### 4. Subscribe in Replay Controller

**File**: `gym_gui/services/telemetry/replay_controller.py`

```python
from gym_gui.telemetry.events import Topic
from gym_gui.telemetry.run_bus import get_bus

bus = get_bus()
queue = bus.subscribe(Topic.RUN_COMPLETED, f"replay-{run_id}")

# In async loop:
evt = await queue.get()
if evt.payload["outcome"] == "success":
    # Load episodes from SQLite
    episodes = store.episodes_for_run(run_id)
    self._populate_replay_tab(episodes)
```

### 5. Subscribe in RunRegistry

**File**: `gym_gui/services/trainer/registry.py`

```python
from gym_gui.telemetry.events import Topic
from gym_gui.telemetry.run_bus import get_bus

bus = get_bus()
queue = bus.subscribe(Topic.RUN_COMPLETED, "registry")

# In async loop:
evt = await queue.get()
registry.update_run_outcome(
    run_id=evt.run_id,
    status=RunStatus.COMPLETED,
    outcome=evt.payload["outcome"],
)
```

## Tab Semantics (Clarified)

### Agent-{id}-Online (Primary Live View)

- Subscribes to `STEP_APPENDED` (live)
- Renders immediately (no DB wait)
- For ToyText: shows grid
- For visual: shows video
- No "Recent Steps/Episodes" panes

### Agent-{id}-Replay (Post-Training Browser)

- Subscribes to `RUN_COMPLETED`
- Loads from SQLite via `episodes_for_run(run_id)`
- Appears/activates only after training finishes
- No live subscriptions

### Agent-{id}-Online-Grid / Online-Video

- Optional specialized live views
- Or fold into Online with renderer selection

## Why This Fixes "First Step Only"

**Before** (blocking path):
```
step arrives → normalize → write to DB (blocks!) → queue fills → overflow → only first step visible
```

**After** (dual-path):
```
step arrives → normalize → publish to bus (non-blocking) → UI renders immediately
                                                        → DB sink batches & persists asynchronously
```

UI never waits for DB I/O. DB can lag without affecting live view.

## Safety Rails & Diagnostics

### Overflow Monitoring

```python
stats = bus.overflow_stats()
# {"STEP_APPENDED:ui-run1-agent6": 42, ...}
# If non-zero under normal load → investigate
```

### Queue Sizes

```python
sizes = bus.queue_sizes()
# {"STEP_APPENDED:ui-run1-agent6": 128, ...}
# Monitor for sustained high values
```

### Cursor-Based Backfill

If a subscriber restarts:
```python
last_seq_id = subscriber.last_seq_id
# Query SQLite for seq_id > last_seq_id
# Backfill from DB
```

### Health Heartbeat

Emit `RUN_HEARTBEAT` every N seconds. If UI misses 3 heartbeats:
- Show "stream stalled" banner
- Keep accepting DB writes
- Attempt reconnect

## Implementation Checklist

- [x] Create `events.py` with Topic enum and TelemetryEvent
- [x] Create `run_bus.py` with RunBus class
- [ ] Update `streams.py` to publish events
- [ ] Create `db_sink.py` for background persistence
- [ ] Update `live_controller.py` to subscribe to STEP_APPENDED
- [ ] Update `replay_controller.py` to subscribe to RUN_COMPLETED
- [ ] Update `registry.py` to subscribe to RUN_COMPLETED
- [ ] Update UI tabs to use new semantics
- [ ] Add overflow monitoring to debug pane
- [ ] Test end-to-end with training run

## Expected Outcomes

✅ Online tab updates every step without DB wait  
✅ Replay tab stays empty until RUN_COMPLETED  
✅ ToyText shows only Online (Grid) by default  
✅ Episodes rows contain agent_id (no NULL)  
✅ RunRegistry reflects state changes  
✅ overflow_stats() remains zero under normal load  

## Next Steps

1. Implement `db_sink.py` for background persistence
2. Update `streams.py` to publish events
3. Update controllers to subscribe
4. Test with diagnostic
5. Run full system test


