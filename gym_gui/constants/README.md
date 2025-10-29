# Centralized Constants Package

This package consolidates all project-wide configuration constants into a single, discoverable location with consistent naming and clear ownership.

## Package Structure

### Core Organization

All constants are defined in domain-specific modules and re-exported through the central `__init__.py`:

```
gym_gui/constants/
├── __init__.py                    # Main exports, backwards compatibility wrappers
├── constants_core.py              # Episode counter, worker ID, persistence
├── constants_ui.py                # UI ranges, render delays, buffer sizing
├── constants_telemetry.py         # Queue/buffer config, credit system
├── constants_telemetry_bus.py     # RunBus, event fan-out, telemetry streaming
├── constants_telemetry_db.py      # DB sink batching, health monitoring
├── constants_trainer.py           # gRPC, daemon, trainer retry config
├── game_constants.py              # Game environment defaults (pre-existing)
├── loader.py                      # Config loading utilities (pre-existing)
└── README.md                      # This file
```

## Import Patterns

### Recommended: Import from centralized package

```python
# Import specific constants
from gym_gui.constants import (
    DEFAULT_MAX_EPISODES_PER_RUN,
    format_episode_id,
    RENDER_DELAY_MIN_MS,
)

# Import dataclass configs
from gym_gui.constants import UI_DEFAULTS, BUS_DEFAULTS, TRAINER_DEFAULTS

# Import entire submodule if you need many related constants
from gym_gui.constants import constants_telemetry
print(constants_telemetry.STEP_BUFFER_SIZE)
```

### Backwards Compatibility

Legacy modules such as `gym_gui.telemetry.constants` and
`gym_gui.services.trainer.constants` have been removed. Update any remaining
imports to reference `gym_gui.constants` directly:

```python
# ✓ Centralized interface
from gym_gui.constants import STEP_BUFFER_SIZE, TRAINER_DEFAULTS
```

## Constants by Domain

### Episode Counter (`constants_core.py`)

Episode counter bounds, formatting, worker ID configuration for distributed rollouts.

| Constant | Type | Value | Purpose |
|----------|------|-------|---------|
| `DEFAULT_MAX_EPISODES_PER_RUN` | int | 999,999 | Hard limit on episodes per run (6-digit capacity) |
| `EPISODE_COUNTER_WIDTH` | int | 6 | Number of digits used in episode ID padding |
| `MAX_COUNTER_VALUE` | int | 999,999 | Maximum representable value with counter width |
| `EPISODE_ID_SEPARATOR` | str | "-" | Separator in formatted episode IDs |
| `EPISODE_ID_PREFIX` | str | "ep" | Prefix for episode index portion of ID |
| `WORKER_ID_PREFIX` | str | "w" | Prefix for worker ID portion of ID |
| `WORKER_ID_WIDTH` | int | 6 | Worker ID field width (matches counter for consistency) |
| `COUNTER_LOCK_TIMEOUT_S` | float | 5.0 | Lock timeout for thread-safe counter access |
| `RESUME_CAPACITY_WARNING_THRESHOLD` | float | 0.95 | Warn if using >95% of counter capacity on resume |

**Dataclass:** `EpisodeCounterConfig(frozen=True)`
- `max_episodes_per_run`: int
- `counter_width`: int
- `worker_id_width`: int
- `lock_timeout_s`: float
- `resume_capacity_warning_threshold`: float
- `.max_counter_value` property

**Functions:**
- `format_episode_id(run_id, ep_index, worker_id=None)` → str
- `run_id` values are 26-character ULIDs (`01ARZ3NDEKTSV4RRFFQ69G5FAV`), generated when
  the trainer validates a submission. ULIDs remain lexicographically sortable so
  database queries keep natural chronological ordering.
- `worker_id` is optional; distributed workers append a `w{worker_id}` fragment,
  so a complete identifier looks like `run-ulid-wworker-ep000042`.
- `parse_episode_id(episode_id)` → dict[str, str | int | None]

**Error Messages:**
- `COUNTER_NOT_INITIALIZED_ERROR`
- `MAX_EPISODES_REACHED_ERROR`
- `COUNTER_EXCEEDS_MAX_ERROR`
- `INVALID_MAX_EPISODES_ERROR`
- `COUNTER_CAPACITY_EXCEEDED_ERROR`

**Database Columns:**
- `EPISODE_ID_COLUMN = "episode_id"`
- `EP_INDEX_COLUMN = "ep_index"`
- `WORKER_ID_COLUMN = "worker_id"`
- `MAX_EPISODES_COLUMN = "max_episodes_per_run"`

#### Example: Single-Process Episode ID
```
run_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"  # ULID
ep_index = 42
format_episode_id(run_id, ep_index)  # "01ARZ3NDEKTSV4RRFFQ69G5FAV-ep000042"
```

#### Example: Multi-Worker (Distributed) Episode ID
```
worker_id = "w001"
format_episode_id(run_id, ep_index, worker_id)  # "01ARZ3NDEKTSV4RRFFQ69G5FAV-ww001-ep000042"
```

**DB Constraints:**
- Single-process: `UNIQUE(run_id, ep_index)`
- Distributed: `UNIQUE(run_id, worker_id, ep_index)` (when worker_id is not NULL)

---

### UI (`constants_ui.py`)

Render delay ranges, training speed config, buffer sizing for telemetry and episodes.

| Constant | Type | Value | Purpose |
|----------|------|-------|---------|
| `RENDER_DELAY_MIN_MS` | int | 10 | Minimum render delay (100 FPS max) |
| `RENDER_DELAY_MAX_MS` | int | 500 | Maximum render delay (2 FPS min) |
| `RENDER_DELAY_TICK_INTERVAL_MS` | int | 50 | Slider tick interval |
| `DEFAULT_RENDER_DELAY_MS` | int | 100 | Default render delay (10 FPS) |
| `UI_TRAINING_SPEED_MIN` | int | 0 | Min training speed multiplier |
| `UI_TRAINING_SPEED_MAX` | int | 100 | Max training speed multiplier |
| `TRAINING_TELEMETRY_THROTTLE_MIN` | int | 1 | Min telemetry throttle factor |
| `TRAINING_TELEMETRY_THROTTLE_MAX` | int | 10 | Max telemetry throttle factor |
| `UI_RENDERING_THROTTLE_MIN` | int | 1 | Min rendering throttle factor |
| `UI_RENDERING_THROTTLE_MAX` | int | 10 | Max rendering throttle factor |
| `TELEMETRY_BUFFER_MIN` | int | 256 | Min buffer spin box value |
| `TELEMETRY_BUFFER_MAX` | int | 10,000 | Max buffer spin box value |
| `DEFAULT_TELEMETRY_BUFFER_SIZE` | int | 512 | Default telemetry buffer size |
| `EPISODE_BUFFER_MIN` | int | 10 | Min episode buffer |
| `EPISODE_BUFFER_MAX` | int | 100 | Max episode buffer |
| `DEFAULT_EPISODE_BUFFER_SIZE` | int | 50 | Default episode buffer |
| `BUFFER_BUFFER_MIN` | int | 256 | Backwards-compat alias for TELEMETRY_BUFFER_MIN |

**Dataclasses:**
- `RenderDefaults`: Render delay configuration
- `SliderDefaults`: Training speed slider configuration
- `BufferDefaults`: Buffer sizing defaults
- `UIDefaults`: Combined UI defaults (aggregates the above)

---

### Telemetry (`constants_telemetry.py`)

Queue and buffer sizes for telemetry infrastructure, credit system, logging levels.

| Constant | Type | Value | Purpose |
|----------|------|-------|---------|
| `STEP_BUFFER_SIZE` | int | 64 | Buffer for gym steps before telemetry batching |
| `EPISODE_BUFFER_SIZE` | int | 256 | Buffer for episodes before telemetry batching |
| `RUNBUS_DEFAULT_QUEUE_SIZE` | int | 2048 | Default queue size for RunBus event delivery |
| `RUNBUS_DB_PATH_QUEUE_SIZE` | int | 512 | Queue size for DB path in RunBus |
| `RUNBUS_UI_PATH_QUEUE_SIZE` | int | 512 | Queue size for UI path in RunBus |
| `LIVE_STEP_QUEUE_SIZE` | int | 128 | Queue for live step streaming to UI |
| `LIVE_EPISODE_QUEUE_SIZE` | int | 64 | Queue for live episode streaming to UI |
| `LIVE_CONTROL_QUEUE_SIZE` | int | 32 | Queue for control messages (stop, pause, etc.) |
| `DB_SINK_WRITER_QUEUE_SIZE` | int | 256 | Queue for DB sink writer tasks |
| `DB_SINK_BATCH_SIZE` | int | 64 | Batch size for DB commits |
| `DB_SINK_CHECKPOINT_INTERVAL` | int | 1000 | Checkpoint interval for WAL mode |
| `RENDER_QUEUE_SIZE` | int | 128 | Queue size for rendering tasks |
| `RENDER_BOOTSTRAP_TIMEOUT_MS` | int | 5000 | Timeout to wait for first render frame |
| `TELEMETRY_HUB_BUFFER_SIZE` | int | 512 | Hub-level buffer for telemetry events |
| `TELEMETRY_HUB_MAX_QUEUE` | int | 2048 | Max queue length before backpressure |
| `TELEMETRY_SERVICE_HISTORY_LIMIT` | int | 10000 | History limit for service queries |
| `HEALTH_MONITOR_HEARTBEAT_INTERVAL_S` | float | 2.0 | Heartbeat interval for health monitoring |
| `INITIAL_CREDITS` | int | 1000 | Initial credit tokens for backpressure |
| `MIN_CREDITS_THRESHOLD` | int | 100 | Threshold before requesting credit refill |

**Logging Levels:**
- `STEP_LOG_LEVEL = logging.DEBUG`
- `BATCH_LOG_LEVEL = logging.INFO`
- `ERROR_LOG_LEVEL = logging.ERROR`

---

### Telemetry Bus (`constants_telemetry_bus.py`)

RunBus queue configuration, event fan-out bounds, telemetry streaming, credit system.

**Dataclasses:**
- `RunBusQueueDefaults`: Queue sizing for RunBus components
- `RunEventDefaults`: Event fan-out bounds and configuration
- `TelemetryStreamDefaults`: Streaming history and buffering
- `TelemetryHubDefaults`: Hub-level queue and buffering
- `TelemetryLoggingDefaults`: Logging level guidance
- `CreditDefaults`: Credit-based backpressure configuration
- `BusDefaults`: Aggregated bus-level configuration (includes all above)

**Usage:**
```python
from gym_gui.constants import BUS_DEFAULTS

queue_size = BUS_DEFAULTS.runbus_queue_defaults.queue_size
credit_limit = BUS_DEFAULTS.credit_defaults.credit_limit
```

---

### Telemetry DB (`constants_telemetry_db.py`)

Database sink batching, health monitoring, trainer registry bounds.

**Dataclasses:**
- `TelemetryDBSinkDefaults`: DB writer queue, batch size, checkpoint interval
- `HealthMonitorDefaults`: Heartbeat interval, timeout thresholds
- `RegistryDefaults`: Trainer registry persistence bounds
- `DatabaseDefaults`: Aggregated DB-level configuration

---

### Trainer (`constants_trainer.py`)

gRPC client configuration, daemon lifecycle, retry policies, training run schema.

**Dataclasses:**
- `TrainerClientDefaults`: gRPC endpoint, timeouts, keepalive settings
- `TrainerDaemonDefaults`: Daemon startup/shutdown, event loop config
- `TrainerRetryDefaults`: Retry policy, exponential backoff, max attempts
- `TrainRunSchemaDefaults`: Schema validation defaults
- `TrainerDefaults`: Aggregated trainer configuration

---

### Game (`game_constants.py`)

Game environment configuration defaults (Frozen Lake, CartPole, etc.) — pre-existing module.

---

## Migration Guide

### For New Code

Always import from `gym_gui.constants`:

```python
from gym_gui.constants import (
    DEFAULT_MAX_EPISODES_PER_RUN,
    UI_DEFAULTS,
    BUS_DEFAULTS,
)
```

### For Existing Code (Migration Path)

1. **Identify** old import statements in your file
2. **Replace** with new centralized import
3. **Test** that functionality still works
4. **Commit** the change

Example:

```diff
- from gym_gui.telemetry.constants import STEP_BUFFER_SIZE
- from gym_gui.ui.constants import RENDER_DELAY_MIN_MS

+ from gym_gui.constants import STEP_BUFFER_SIZE, RENDER_DELAY_MIN_MS
```

## Maintenance

### Adding New Constants

1. Create or update the appropriate `constants_*.py` module
2. Add exports to the module's `__all__` list
3. Re-export from `gym_gui/constants/__init__.py`
4. Update this README.md with the new constant
5. Avoid introducing new wrappers; update call sites directly instead

### Handling Overlaps

Check the master index below before adding new constants:

- **Buffer sizes:** Use consistent naming across domains
- **Timeout values:** Prefer explicit `_S` suffix for seconds
- **Queue sizes:** Use consistent `_QUEUE_SIZE` suffix
- **Logging levels:** Prefer explicit level names (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

## Known Issues & Deprecations

### Removed Legacy Modules

The following legacy modules have been deleted. Update any straggling imports to
use `gym_gui.constants` instead:

- `gym_gui.core.constants_episode_counter`
- `gym_gui.ui.constants`
- `gym_gui.ui.constants_ui`
- `gym_gui.telemetry.constants`
- `gym_gui.telemetry.constants_bus`
- `gym_gui.telemetry.constants_db`
- `gym_gui.services.trainer.constants`

### Backwards-Compatibility Aliases

- `BUFFER_BUFFER_MIN` in `constants_ui.py`: Legacy alias for `TELEMETRY_BUFFER_MIN` (kept for compatibility with code expecting the typo)

---

## See Also

- [`docs/CONSTANTS_AUDIT_AND_CONSOLIDATION.md`](../CONSTANTS_AUDIT_AND_CONSOLIDATION.md): Audit results and consolidation plan
- [`docs/CONSTANTS_MASTER_INDEX.md`](../CONSTANTS_MASTER_INDEX.md): Full index of all constants with usage counts and dependencies
