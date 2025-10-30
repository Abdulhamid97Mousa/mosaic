# Resource Monitoring Strategy — Day 18 Task 4

## 1. Objective and Scope

Align the telemetry stack with real resource visibility. This plan covers CPU, GPU, and memory tracking for the GUI, trainer daemon, SPADE-BDI worker, telemetry hub, and SQLite sink. Outputs are intended for both live dashboards (UI) and historical analysis (telemetry store) without introducing heavyweight infrastructure by default.

---

## 2. Tooling Landscape

| Option | Strengths | Trade-offs | Fit for Current Stage |
|--------|-----------|------------|------------------------|
| **Python-native sampling (psutil, nvidia-ml-py)** | Lightweight, zero external services, integrates directly with existing asyncio loops. `psutil` exposes per-process CPU, memory, IO, sensors; NVML bindings expose GPU load and VRAM. | Requires manual aggregation and retention logic. GPU metrics available only when NVIDIA stack is installed. | **Primary path.** Pairs well with HealthMonitor and RunBus events. |
| **OpenTelemetry (OTel)** | Unified metrics/traces/logs, rich SDKs, exporters to Prometheus, OTLP, etc. | Considerable setup: collectors, exporters, semantic conventions. Overkill for single-host GUI, increases dependency footprint. | **Defer.** Revisit when multi-host deployment or tracing joins the roadmap. |
| **Grafana + Prometheus/InfluxDB** | Production-grade dashboards and alerting, ecosystem of plugins. | Requires metrics backend (Prometheus, InfluxDB). Operational overhead (services, storage, authentication). | **Optional extension.** Useful once baseline metrics are flowing and remote observability is needed. |
| **Glances / standalone agents** | Quick CLI/web dashboards, supports remote mode. | Another daemon to manage; overlaps with custom UI telemetry. | Use ad-hoc for profiling but not the long-term pipeline. |

---

## 3. Recommended Architecture

1. **Sample locally inside each long-lived process** using `psutil.Process` handles. Target 5 s cadence to align with HealthMonitor heartbeats.
2. **Publish ResourceSnapshot events on RunBus** (Topic.HEARTBEAT or new Topic.RESOURCE). Include run_id, component tag, windowed CPU percent, RSS, USS, GPU util, VRAM.
3. **Persist snapshots via TelemetrySQLiteStore** with bounded retention (e.g., keep last 30 minutes per run). Add aggregation hooks for rollup when writing to disk.
4. **Render lightweight charts in LiveTelemetryTab** (sparklines per component) to surface immediate pressure. Respect existing buffer-size limits to avoid UI floods.
5. **Expose optional exporter hook** so future deployments can forward metrics to Prometheus/Grafana without modifying core sampling code.

---

## 4. Component Coverage Matrix

| Component | Primary metrics | Notes |
|-----------|-----------------|-------|
| GUI (PyQt) | Process CPU%, CPU time delta, RSS, event loop lag (Qt timers) | Use `psutil.Process(os.getpid())`. Event loop lag measured via `QElapsedTimer` or asyncio loop time difference. |
| Trainer Daemon | CPU%, RSS, queue backlog length, subprocess count | Access via `psutil.Process` inside `trainer_daemon`; include telemetry queue depths (existing metrics) for correlation. |
| SPADE-BDI Worker | CPU%, RSS, GPU util (if CUDA agents added), async mailbox backlog | Worker already streams telemetry; inject resource readings alongside episode stats when enabled. |
| TelemetryAsyncHub | CPU%, RSS, backlog length (`_queue.qsize()`), dropped batch count | Capturing hub pressure helps validate new dynamic buffers. |
| SQLite Sink | CPU%, RSS, WAL size, fsync latency | Combine `psutil` for process stats and `sqlite3` pragmas for WAL checkpoints. |

---

## 5. Sampling and Retention Policy

- **Cadence:** default 5 s. Allow override via env `GYM_RESOURCE_SAMPLE_INTERVAL`. Support slower cadence (15 s) for headless CI to limit noise.
- **Averaging:** compute CPU% using `process.cpu_percent(interval=None)` with deltas stored in-memory to avoid blocking sleeps.
- **Retention:** keep 360 samples (~30 min) per run in UI buffers. Persist all samples to SQLite but provide rollup task (per-minute average) when exporting to disk to prevent unbounded growth.
- **Backpressure:** reuse dynamic buffer sizing from Day 14 fixes; apply same run-scoped capacity to resource topics.

---

## 6. Implementation Steps

1. **Create `gym_gui/telemetry/resource_sampler.py`** with utilities:
   - `collect_process_metrics(process: psutil.Process)` returning CPU%, RSS, USS, thread count, open file count.
   - Optional `collect_gpu_metrics()` using `nvidia-ml-py` when available (detect gracefully, fallback to zeros).
2. **Extend `HealthMonitor`** to schedule sampler coroutines within existing heartbeat loop. Emit `ResourceSnapshot` protobuf (new message) via RunBus.
3. **Add schema updates:**
   - Update `telemetry.proto` with `ResourceSnapshot` message; include timestamps, process_tag, cpu_percent, memory_rss_bytes, memory_uss_bytes, gpu_util_percent, gpu_mem_bytes.
   - Regenerate stubs (trainer + UI) and document CLI command in Day log.
4. **Persist snapshots:**
   - Update `TelemetrySQLiteStore` to create `resource_snapshots` table keyed by run_id, component, timestamp.
   - Batch insert using existing writer pipeline to avoid new threads.
5. **UI surfacing:**
   - In `LiveTelemetryTab`, add optional resource panel activated when snapshots present. Render line charts (CPU, RSS) with downsampled data to respect buffer limits.
6. **Configuration hooks:**
   - Introduce `GymResourceSamplingConfig` (constants module) with env overrides for interval, retention, GPU enable flag.
7. **Testing:**
   - Add unit tests simulating psutil responses to ensure sampler handles missing GPU libs and zombie processes.
   - Add integration test that publishes ResourceSnapshot and verifies persistence + UI table size obeys caps.

---

## 7. Optional Grafana Path

Once local sampling is stable, enable Prometheus-compatible exports:

- Start lightweight Prometheus exporter (e.g., `prometheus_client` HTTP server) gated by `GYM_EXPOSE_PROM_ENDPOINT`.
- Map ResourceSnapshot values into Prometheus metrics (`gym_resource_cpu_percent{component="trainer"}` etc).
- Deploy Grafana with standard Linux host dashboard as a starting point; integrate custom panels for run_id filtering.
- Document authentication and retention expectations before shipping to shared environments.

---

## 8. Profiling and Deep Dive Toolkit

Use targeted profilers when sampling indicates hotspots:

- **CPU spikes:** `py-spy top --pid <pid>` or `yappi` for coroutine-heavy sections.
- **Memory leaks:** `tracemalloc`, `memory_profiler` (line-by-line), or `objgraph` to trace Qt object churn.
- **GPU anomalies:** `nvidia-smi --loop=1` during runs, with NVML counters (`nvmlDeviceGetUtilizationRates`) for precise reads.
- **Event loop lag:** `asyncio.get_event_loop().slow_callback_duration` (Python 3.12) to log callbacks exceeding thresholds.

Record findings in the corresponding Day journal and feed adjustments back into constants.

---

## 9. Action Items

1. Implement resource sampler module with psutil baseline; document NVML optional dependency.
2. Extend telemetry protobuf and persistence layers for ResourceSnapshot events.
3. Wire HealthMonitor to emit snapshots every 5 s (configurable) and update UI presenters.
4. Prototype Prometheus exporter flag to unblock Grafana adoption when needed.
5. Update docs/Day logs after instrumentation rollout with observed headroom against Day 16 baselines.
