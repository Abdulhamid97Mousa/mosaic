# GUI Resource Usage — Day 16 Task 5

## 1. Host Baseline (Developer Workstation)

The following metrics (captured 26 Oct 2025) provide context for default allocations:

| Source | Key data |
|--------|----------|
| `free -hl` | 31 GiB RAM total, 14 GiB used, 13 GiB free, 2 GiB swap (unused). |
| `lscpu` | 32 logical CPUs (Intel i9-13900HX), 24 cores / 48 threads, base clock 800 MHz, turbo 5.4 GHz. |
| `nvidia-smi` | NVIDIA RTX 4090 16 GiB VRAM, ~494 MiB in use by desktop processes. |

Even though this host offers generous headroom, centralized constants must allow tuning for less powerful machines (e.g., laptops, CI runners).

---

## 2. Default Component Budgets (Current Behaviour)

| Component | Default resource requests | Notes |
|-----------|--------------------------|-------|
| **GUI process (Qt)** | No explicit CPU/GPU cap; follows OS scheduler. GPU usage driven by rendering workload and Qt backend. | Monitoring via `ps`, `top`, `glances`, or Qt performance tools recommended. |
| **Trainer run submission (Spade BDI)** | `cpus=2`, `memory_mb=2048`, `gpus.requested=0` (see `SpadeBdiTrainForm` & presenter). | Hard-coded defaults; should be sourced from constants loader and overridable via env. |
| **Telemetry pipeline** | RunBus queues: UI 512, DB 1024, writer 4096 (`bootstrap.py`). DB sink `batch_size=128`, `writer_queue_size=4096`, `checkpoint_interval=1024`. | These influence RAM use and write latency; central constants enable environment-specific tuning. |
| **Worker (spade_bdi_rl)** | Telemetry buffers 2048 steps / 100 episodes, step delay 0.14 s (UI-controlled), default `DEFAULT_JID`/`DEFAULT_PASSWORD`. | Worker resource needs scale with observation payloads; constants must align with UI telemetry controls. |
| **Health monitoring** | `heartbeat_interval=5s` (HealthMonitor). | Impacts log frequency and scheduler wakeups. |

---

## 3. Measuring GUI & Worker Usage

Suggested tools to profile actual consumption during runs:

- `psutil` or `top` for CPU and memory utilisation of `python` processes (GUI, trainer, worker).
- `nvidia-smi --loop=1` for GPU utilization / VRAM if future workers use CUDA.
- `sqlite3` `pragma wal_checkpoint` stats to inspect DB sink pressure.
- Custom Qt instrumentation (e.g., `QtCore.QElapsedTimer`) for render loop timings.

Collecting these metrics feeds into the constants registry: once average and peak usage are known, defaults can be set conservatively while allowing env overrides for heavier workloads.

---

## 4. Action Items

1. Annotate constants registry with recommended env keys (e.g., `GYM_RUN_CPUS`, `GYM_DB_SINK_BATCH_SIZE`, `GYM_HEARTBEAT_INTERVAL`).
2. Add scripted profiling (benchmark harness) to capture CPU/RAM/VRAM during typical runs and record findings here.
3. Integrate resource telemetry into runtime logs (e.g., periodic CPU/memory snapshots) for long sessions.
