# This is comparison between Intel Coach and gym_gui (MOSAIC) is one of the most instructive architectural contrasts in the repository

Let’s break the story down with what the current code actually does, so the trade-offs are grounded in the implementation.

🧩 1. Why Coach Doesn’t Need SQLite or Telemetry

Intel Coach predates the async/multi-process world. It runs a monolithic training loop, writes metrics directly from `Logger.dump_*`, and renders frames in the same Python interpreter that executes the environment. There is no gRPC, no sidecar, and no out-of-process consumer to feed.

| Function      | Coach implementation                                      | Why it works                                  |
| ------------- | --------------------------------------------------------- | --------------------------------------------- |
| Metrics       | Direct CSV / TensorBoard writes from the active TF1 graph | Single-threaded loop keeps I/O cheap          |
| Dashboard     | Bokeh server tails CSV files (tail -f semantics)          | Append-only text logs are trivial to stream   |
| Visualization | PyGame blits the ndarray produced by Gym                  | Rendering and training share memory + clock   |
| IPC           | None — training, logging, rendering live together         | No marshaling layer to maintain               |
| State         | GraphManager / NetworkWrapper objects in memory           | Restartability handled by TF checkpoints only |

✅ Result: low-latency visuals because nothing ever leaves the process.

❌ Trade-off: no separation of concerns. Headless trainers, SPADE-BDI workers, or remote inference are off the table.

🧮 2. What MOSAIC Is Doing (Validated Against Code)

MOSAIC is deliberately split into cooperating processes:

- Trainer daemon (`gym_gui/services/trainer/service.py`) hosts the `TrainerService` gRPC API and orchestrates runs.
- Workers (CleanRL, SPADE-BDI, etc.) send telemetry over `PublishRunSteps` / `PublishRunEpisodes` streams. JSONL workers go through the proxy in `gym_gui/services/trainer/trainer_telemetry_proxy.py` which converts lines into protobufs.
- Logic-oriented supervisors written in JASON (`3rd_party/jason_worker/`) connect through the same trainer service, letting BDI plans supervise or veto actions while still emitting standard telemetry.
- The GUI resolves a `TelemetryAsyncHub` instance (registered in `gym_gui/services/bootstrap.py`) and a `TrainerClient`. The hub keeps an asyncio loop, bridges to Qt, and feeds live tabs.
- Durable telemetry is persisted by `TelemetryDBSink` (`gym_gui/telemetry/db_sink.py`) into `TelemetrySQLiteStore` (`gym_gui/telemetry/sqlite_store.py`) using WAL and batched writes.
- Fan-out to live consumers happens through the in-process `RunBus` (`gym_gui/telemetry/run_bus.py`) so the UI, persistence layer, and any analytics consumer run independently.

✅ Result: the GUI can replay runs, tolerate trainer restarts, and schedule heterogeneous workers because the proto contract is language-agnostic.

❌ Cost: every useful datum must cross process boundaries, so latency depends on serialization, gRPC backpressure, and SQLite batching.

## 3. End-to-End Telemetry Path (With Code References)

1. Worker emits telemetry.

   - Direct gRPC workers push `trainer_pb2.RunStep` messages straight to `TrainerService.PublishRunSteps`.
   - JSONL workers go through `trainer_telemetry_proxy.py` (`run_proxy` → `_make_runstep` / `_make_runepisode`).

2. Trainer daemon normalizes and fans out.

   - `RunTelemetryBroadcaster` in `service.py` stores per-run history and services `StreamRunSteps/StreamRunEpisodes` subscribers with sequence IDs and automatic replay.
   - `TrainerService.PublishRunSteps` / `PublishRunEpisodes` also emit `TelemetryEvent`s onto the `RunBus` and enqueue durable writes through `TelemetrySQLiteStore.record_step` / `.record_episode` (Section `service.py:702-827`).

3. GUI consumes live data.

   - `TelemetryAsyncHub` (`services/trainer/streams.py`) keeps a bounded queue, pulls from `TrainerClient.stream_run_*`, and hands payloads to the Qt thread via `TelemetryBridge`.
   - The bridge calls `_normalize_payload`, producing dicts that tabs can render without touching protobuf APIs.

4. UI and storage decouple.

   - `LiveTelemetryController` (`controllers/live_telemetry_controllers.py`) subscribes to the `RunBus` with UI-sized queues, applies render throttling and credit accounting, and drives the Qt tabs.
   - `TelemetryDBSink` subscribes to the same bus with a larger queue, batches into SQLite on a background thread, and keeps WAL checkpoints under control.

The architecture is the “dual path” described in the inline comments: a live in-memory fan-out plus a durable SQLite pipeline.

## 4. Why Coach Appears “Faster”

| Layer            | Coach                               | MOSAIC (per current code)                                        |
| ---------------- | ----------------------------------- | ---------------------------------------------------------------- |
| Frame generation | Gym loop remains in the game thread | Workers run elsewhere, often through the telemetry proxy         |
| Frame transport  | Direct ndarray → PyGame             | RunStep protobuf across gRPC (optionally JSONL → protobuf first) |
| Queueing         | None — immediate draw               | `TelemetryAsyncHub` queue → `RunBus` queue → Qt event dispatch   |
| Persistence      | Optional CSV append                 | `TelemetryDBSink` + `TelemetrySQLiteStore` batched commits       |
| Rendering        | Same interpreter                    | Qt widgets repaint after payload normalization                   |

None of these steps are individually expensive (microseconds to low milliseconds), but they stack. With default settings you pay for:

- Protobuf serialization in the worker or proxy.
- gRPC flow control (per-message ack in `PublishRunSteps`).
- Async hub buffering (`TELEMETRY_HUB_MAX_QUEUE`) and normalization.
- RunBus fan-out, plus SQLite batching every 32 records by default.
- Qt render throttling in `LiveTelemetryTab`.

The cumulative delay is why Coach’s on-thread PyGame window “feels” more immediate.

## 5. What MOSAIC Gains That Coach Cannot Deliver

| Capability                    | Coach             | MOSAIC today                                                                                                                 |
| ----------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Remote or multi-agent workers | ❌                | ✅ (`RegisterWorker` session gating; RunBus fan-out)                                                                         |
| Crash-tolerant telemetry      | ❌                | ✅ (`TelemetrySQLiteStore`, WAL, run archiving)                                                                              |
| Replay / resume UI            | ❌                | ✅ (tabs pull history via `StreamRun*` replay + SQLite queries)                                                              |
| Cross-language orchestration  | Limited to Python | ✅ (CleanRL, SPADE-BDI, JASON BDI supervisors, future Go/C++ via protos)                                                    |
| Logic-oriented supervision    | ❌                | ✅ (JASON worker in `3rd_party/jason_worker/` bridges AgentSpeak plans into the trainer contract)                             |
| Credit / backpressure hooks   | N/A               | Partially ✅ — `CreditManager` exists but `_drain_loop` still needs `consume_credit()` wiring (tracked in Day-14 follow-ups) |

So SQLite is the price of persistent, multi-run introspection; the RunBus separates UI cadence from durable storage.

## 6. Narrowing the Latency Gap (Actionable Ideas)

1. **Finish credit enforcement**: wire `TelemetryAsyncHub._drain_loop` to `CreditManager.consume_credit()` / CONTROL events so slow tabs pause the UI path without dropping durable data.
2. **Tune batching intentionally**: bump `_DEFAULT_BATCH_SIZE` in `TelemetrySQLiteStore` only when the DB sink becomes the bottleneck; it already uses WAL and batched writes, so raising to 64/128 can reduce fsyncs.
3. **UI throttling knobs**: expose `set_render_throttle_for_run` defaults in the train dialog so high-FPS runs sample every N steps instead of every frame.
4. **Optional volatile cache**: if you need Coach-level immediacy, add an in-memory frame ring (shared memory or a `Topic.CONTROL` side channel) while keeping SQLite for persistence.
5. **Proxy shortcuts**: for local workers that can import the proto, bypass the JSONL proxy and stream protobufs directly to avoid the JSON encode/decode hop.

🧩 7. TL;DR

- Coach feels fast because nothing leaves the process; that design cannot scale past one interpreter.
- MOSAIC routes telemetry through `TrainerService` → `RunTelemetryBroadcaster` → `TelemetryAsyncHub` / `RunBus` → Qt tabs + SQLite. The code paths above confirm the design.
- You traded a few milliseconds of latency for durability, replay, multi-language workers, and orchestration. Finish the credit backpressure and consider a volatile fast path if ultra-low latency is still required.

Happy to help prototype the hybrid ring-buffer approach if you want to explore it next.
