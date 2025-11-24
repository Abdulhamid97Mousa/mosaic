# Ray Architecture Deep Dive: Lessons for GUI_BDI_RL

## Ray's Core Insight: Decouple Fast & Slow Paths

Ray handles **billions of events/second** across distributed systems. Its solution:

**Two independent paths for every event:**
1. **Live Path** (Pub/Sub) - Fast, ephemeral, for UI/dashboards
2. **Durable Path** (KV Store) - Slow, persistent, for recovery/replay

**Key**: These paths are **completely independent**. Slow path never blocks fast path.

➡️ For how this philosophy materialized in our Qt GUI (fast lane vs durable lane, RunBus queue sizing, and shared buffers), see the implementation brief in `docs/1.0_DAY_28/TASK_1/README.md`.

---

## Ray's Protobuf Architecture

### 1. GCS (Global Control Store) - gcs.proto

**Pattern**: Table + Pub/Sub decoupling

```protobuf
enum TablePrefix {
  TASK = 2;           // Durable storage
  ACTOR = 6;          // Durable storage
  // ... 15+ more tables
}

enum TablePubsub {
  NO_PUBLISH = 1;           // No pub/sub
  TASK_PUBSUB = 2;          // Publish task events
  ACTOR_PUBSUB = 6;         // Publish actor events
  // Each has independent subscribers!
}
```

**How it works**:
- Write to `TASK` table (durable, slow)
- Simultaneously publish to `TASK_PUBSUB` (fast, ephemeral)
- Subscribers to `TASK_PUBSUB` get events immediately
- Subscribers to `TASK` table get events after persistence

**Result**: UI gets events in <1ms; DB writes happen asynchronously.

### 2. Core Worker - core_worker.proto

**Pattern**: Independent sequence numbers per consumer

```protobuf
message PushTaskRequest {
  int64 sequence_number = 3;      // Task sequence for ordering
  int64 max_sequence_number = 4;  // Last processed by client
  // Each client maintains its own cursor!
}
```

**How it works**:
- Client A processes up to seq 100
- Client B processes up to seq 95
- Server knows each client's position independently
- No blocking between clients

**Result**: Slow client doesn't block fast client.

---

## Ray's Consumer Cursor Pattern

### The Problem Ray Solves

```
Shared Queue (blocking):
  Producer → [Event1, Event2, Event3] → Single Consumer
                                           ↓
                                      Slow DB write
                                           ↓
                                      Queue fills up
                                           ↓
                                      Events dropped
```

### Ray's Solution: Independent Cursors

```
Pub/Sub (non-blocking):
  Producer → [Event1, Event2, Event3]
                ↓           ↓           ↓
            UI Consumer  DB Consumer  Metrics Consumer
            (cursor=3)   (cursor=1)   (cursor=2)
            
Each consumer:
- Reads at its own pace
- Maintains its own cursor
- Doesn't block others
```

### Implementation Pattern

```python
# Ray's approach (pseudocode from core_worker.proto)
class ConsumerCursor:
    def __init__(self, consumer_id: str):
        self.consumer_id = consumer_id
        self.last_seq_id = 0  # Persisted in DB
    
    def process_event(self, event: Event) -> None:
        # Process event
        handle_event(event)
        # Update cursor
        self.last_seq_id = event.seq_id
        self.persist_cursor()  # Save to DB
    
    def on_restart(self) -> None:
        # Resume from last cursor
        self.last_seq_id = load_cursor_from_db()
        # Request events since last_seq_id
        request_events(since_seq=self.last_seq_id)
```

---

## Ray's Heartbeat Pattern

Ray emits **heartbeat events** for monitoring:

```protobuf
message Heartbeat {
  string component_id = 1;      // "ui", "db", "metrics"
  int64 last_seq_id = 2;        // Last processed event
  int64 total_events = 3;       // Total processed
  int64 dropped_events = 4;     // Overflow count
  double lag_ms = 5;            // Lag in milliseconds
}
```

**Benefits**:
- Detect slow consumers
- Monitor queue health
- Alert on data loss
- Measure end-to-end latency

---

## Applying Ray's Patterns to GUI_BDI_RL

### Current Architecture (Blocking)

```
Worker → TelemetryAsyncHub._stream_steps()
           ↓
         Single Queue (1024 items)
           ↓
         _drain_loop() (single consumer)
           ↓
         TelemetryBridge.emit_step()
           ↓
         Qt Main Thread
           ↓
         LiveTelemetryController
           ↓
         AgentOnlineGridTab.on_step()  ← UI (fast)
         TelemetryService.record_step() ← DB (slow, BLOCKS!)
```

### Ray-Inspired Architecture (Non-Blocking)

```
Worker → RunBus (in-process pub/sub)
           ├→ UI Subscriber (fast)
           │   └→ cursor: seq_id=100
           ├→ DB Subscriber (slow)
           │   └→ cursor: seq_id=95
           └→ Metrics Subscriber
               └→ cursor: seq_id=98

Each subscriber:
- Reads at own pace
- Maintains own cursor
- Doesn't block others
```

### Implementation Roadmap

**Phase 1 (Quick Fixes)**:
- Increase queue size to 8192
- Make DB writes async
- Add heartbeat events

**Phase 2 (Ray-Inspired)**:
- Create RunBus class (pub/sub)
- Add consumer_offsets table
- Implement per-consumer cursors
- Emit EPISODE_FINALIZED events

**Phase 3 (Production)**:
- Multi-agent coordination
- Horizontal scaling
- Full replay capability
- Health monitoring

---

## Key Takeaways from Ray

1. **Never block fast paths for slow paths**
   - UI must never wait for DB
   - Metrics must never wait for logging

2. **Independent cursors per consumer**
   - Each consumer tracks its own position
   - Restart doesn't lose progress
   - Slow consumer doesn't block fast consumer

3. **Explicit event boundaries**
   - EPISODE_FINALIZED events (not inferred)
   - HEARTBEAT events (for monitoring)
   - Clear state transitions

4. **Bounded queues with overflow tracking**
   - Know when events are dropped
   - Alert on backpressure
   - Measure lag

5. **Pub/Sub for live paths**
   - Ephemeral, low-latency
   - Multiple independent subscribers
   - No persistence needed

---

## Ray Code References

- **GCS Pub/Sub**: `src/ray/protobuf/gcs.proto` (lines 48-64)
- **Consumer Cursors**: `src/ray/protobuf/core_worker.proto` (lines 84-96)
- **Task Sequencing**: `src/ray/protobuf/core_worker.proto` (lines 79-120)
- **C++ Implementation**: `src/ray/core_worker/core_worker.h`
- **Python Bindings**: `python/ray/includes/libcoreworker.pxd`

---

## Next Steps

1. ✅ Understand Ray's dual-path architecture
2. ✅ Identify blocking points in current code
3. ⏭️ Implement Phase 1 fixes (queue size, async DB)
4. ⏭️ Implement Phase 2 (RunBus with cursors)
5. ⏭️ Add monitoring and health checks

