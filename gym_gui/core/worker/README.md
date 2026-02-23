# MOSAIC Worker Core Module

Standardized infrastructure for building and integrating RL framework workers into the MOSAIC BDI-RL system.

## Overview

This module provides protocol-based standardization for all MOSAIC workers (CleanRL, Ray, BARLOG, etc.). Workers implement lightweight protocols without requiring inheritance, ensuring maximum flexibility while maintaining consistency.

## Components

### Protocol Definitions (`protocol.py`)

Define contracts that all workers must implement:

- **`WorkerConfig` Protocol**: Configuration interface requiring `run_id`, `seed`, `to_dict()`, and `from_dict()`
- **`WorkerRuntime` Protocol**: Execution interface requiring `run()` method returning `Dict[str, Any]`
- **`WorkerMetadata`**: Immutable metadata (name, version, description, author, etc.)
- **`WorkerCapabilities`**: Declarative capabilities (paradigms, env families, resource requirements)

### Telemetry System (`telemetry.py`)

Standardized lifecycle event emission in JSONL format to stdout, captured by the telemetry proxy.

**Key Classes**:
- `TelemetryEmitter`: Thread-safe emitter for lifecycle events
- `LifecycleEvent`: Structured event with run_id, timestamp, and payload
- `LifecycleEventType`: Standard event types (run_started, run_completed, run_failed, etc.)

**Lifecycle Events**:
- `run_started`: Worker execution begins
- `run_completed`: Successful completion
- `run_failed`: Unrecoverable error
- `run_cancelled`: User/daemon cancellation
- `heartbeat`: Periodic alive signal
- `checkpoint_saved`: Checkpoint written to disk
- `checkpoint_loaded`: Checkpoint restored

### Analytics Manifest (`analytics.py`)

Standardized artifact metadata for post-run analysis:

- **`WorkerAnalyticsManifest`**: JSON manifest describing all worker outputs
- **Artifact Types**: TensorBoard logs, W&B runs, checkpoints, videos, plots
- **Serialization**: Save/load from JSON with validation

### Configuration Loading (`config_loader.py`)

Utilities for loading worker configurations:

- **`load_worker_config_from_file()`**: Load config from JSON file
- **`extract_worker_config()`**: Extract config from nested GUI format
- Supports both flat and nested config formats for backwards compatibility

### Worker Discovery (`discovery.py`)

Automatic worker discovery via setuptools entry points:

- **`WorkerDiscovery`**: Main discovery class
- **`DiscoveredWorker`**: Container for discovered worker info
- **Entry Point Group**: `"mosaic.workers"`
- Discovers and caches all registered workers at runtime

### CLI Base (`cli_base.py`)

Base utilities for worker CLI scripts:

- **`WorkerCLI`**: Static utility class
- `create_base_parser()`: Standard argparse parser with common flags
- `setup_logging()`: Configure structured logging
- `load_and_validate_config()`: Load and validate worker config

## Log Constants Integration

### Overview

MOSAIC uses a centralized log constants system for structured, filterable logging. All worker lifecycle events should use log constants for consistency and debuggability.

### Worker Log Constants

Worker-specific log constants are defined in `gym_gui/logging_config/log_constants.py`:

**CleanRL Constants (LOG431-LOG445)**:
```python
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_CLEANRL_RUNTIME_STARTED,
    LOG_WORKER_CLEANRL_RUNTIME_COMPLETED,
    LOG_WORKER_CLEANRL_RUNTIME_FAILED,
    LOG_WORKER_CLEANRL_MODULE_RESOLVED,
    LOG_WORKER_CLEANRL_CONFIG_LOADED,
    LOG_WORKER_CLEANRL_TENSORBOARD_ENABLED,
    LOG_WORKER_CLEANRL_WANDB_ENABLED,
    LOG_WORKER_CLEANRL_HEARTBEAT,
    LOG_WORKER_CLEANRL_SUBPROCESS_STARTED,
    LOG_WORKER_CLEANRL_SUBPROCESS_FAILED,
    LOG_WORKER_CLEANRL_CHECKPOINT_SAVED,
    LOG_WORKER_CLEANRL_CHECKPOINT_LOADED,
    LOG_WORKER_CLEANRL_ANALYTICS_MANIFEST_CREATED,
    LOG_WORKER_CLEANRL_EVAL_MODE_STARTED,
    LOG_WORKER_CLEANRL_EVAL_MODE_COMPLETED,
)
```

**Ray Constants (LOG446-LOG460)**:
```python
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_RAY_RUNTIME_STARTED,
    LOG_WORKER_RAY_RUNTIME_COMPLETED,
    LOG_WORKER_RAY_RUNTIME_FAILED,
    LOG_WORKER_RAY_CLUSTER_STARTED,
    LOG_WORKER_RAY_CLUSTER_SHUTDOWN,
    LOG_WORKER_RAY_TUNE_STARTED,
    LOG_WORKER_RAY_TUNE_COMPLETED,
    LOG_WORKER_RAY_RLLIB_TRAINING_STARTED,
    LOG_WORKER_RAY_RLLIB_TRAINING_ITERATION,
    LOG_WORKER_RAY_CHECKPOINT_SAVED,
    LOG_WORKER_RAY_CHECKPOINT_LOADED,
    LOG_WORKER_RAY_TENSORBOARD_ENABLED,
    LOG_WORKER_RAY_WANDB_ENABLED,
    LOG_WORKER_RAY_HEARTBEAT,
    LOG_WORKER_RAY_ANALYTICS_MANIFEST_CREATED,
)
```

**BARLOG Constants (LOG1001-LOG1015)**: See `log_constants.py` for complete list

### Log Constant Naming Convention

All worker log constants follow this pattern:

```
LOG_WORKER_{WORKER_TYPE}_{EVENT_CATEGORY}[_{DETAIL}]
```

**Components**:
- `LOG_WORKER_`: Prefix for all worker constants
- `{WORKER_TYPE}`: Uppercase worker name (CLEANRL, RAY, BARLOG)
- `{EVENT_CATEGORY}`: Event type (RUNTIME, MODULE, CONFIG, CHECKPOINT, etc.)
- `_{DETAIL}`: Optional specific detail (STARTED, COMPLETED, FAILED, etc.)

**Examples**:
- `LOG_WORKER_CLEANRL_RUNTIME_STARTED` - CleanRL worker runtime started
- `LOG_WORKER_RAY_CLUSTER_SHUTDOWN` - Ray cluster shutdown
- `LOG_WORKER_BARLOG_EPISODE_COMPLETED` - BARLOG episode completed

### Using Log Constants with TelemetryEmitter

The `TelemetryEmitter` supports optional log constants on all lifecycle methods:

```python
import logging
from gym_gui.core.worker import TelemetryEmitter
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_CLEANRL_RUNTIME_STARTED,
    LOG_WORKER_CLEANRL_RUNTIME_COMPLETED,
    LOG_WORKER_CLEANRL_RUNTIME_FAILED,
)

# Create logger and emitter
logger = logging.getLogger(__name__)
emitter = TelemetryEmitter(
    run_id=config.run_id,
    logger=logger,  # Optional: enables log constant integration
)

# Emit lifecycle events with log constants
emitter.run_started(
    payload={
        "worker_type": "cleanrl",
        "algo": "ppo",
        "env_id": "CartPole-v1",
        "seed": 42,
    },
    constant=LOG_WORKER_CLEANRL_RUNTIME_STARTED,  # Structured log
)

try:
    # Training loop
    results = run_training()

    emitter.run_completed(
        payload={"episodes": 100, "final_reward": 195.0},
        constant=LOG_WORKER_CLEANRL_RUNTIME_COMPLETED,
    )
except Exception as e:
    emitter.run_failed(
        payload={
            "error": str(e),
            "error_type": type(e).__name__,
        },
        constant=LOG_WORKER_CLEANRL_RUNTIME_FAILED,
        exc_info=e,  # Include traceback
    )
```

**Dual Emission**:
When a `constant` is provided, the emitter performs **two actions**:
1. **JSONL telemetry** to stdout (captured by telemetry proxy)
2. **Structured log** via Python logging (with component/subcomponent/tags metadata)

This enables both real-time telemetry streaming AND rich log filtering/aggregation.

### Log Constant Metadata

Each log constant includes structured metadata:

```python
LogConstant(
    code="LOG431",
    level="INFO",
    message="CleanRL worker runtime started",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=("cleanrl", "worker", "runtime", "lifecycle"),
)
```

**Metadata Fields**:
- `code`: Unique log code (e.g., LOG431)
- `level`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `message`: Human-readable message
- `component`: Top-level component (always "Worker" for workers)
- `subcomponent`: Specific worker runtime (CleanRLRuntime, RayRuntime, etc.)
- `tags`: Searchable tags for filtering

## Centralized Constants System

MOSAIC uses a centralized constants system in `gym_gui/constants/` for domain-specific configuration.

### Worker Constants

Worker-specific constants are defined in `gym_gui/constants/` (either `constants_core.py` or `constants_worker.py`):

```python
from gym_gui.constants import (
    # Discovery
    WORKER_ENTRY_POINT_GROUP,  # "mosaic.workers"
    WORKER_DISCOVERY_TIMEOUT_S,  # 5.0
    WORKER_METADATA_CACHE_TTL_S,  # 300.0

    # Subprocess management
    WORKER_HEARTBEAT_INTERVAL_S,  # 30.0
    WORKER_STARTUP_TIMEOUT_S,  # 60.0
    WORKER_SHUTDOWN_TIMEOUT_S,  # 30.0

    # Telemetry
    WORKER_TELEMETRY_BUFFER_SIZE,  # 512
    WORKER_LOG_LEVEL,  # logging.INFO
)
```

### Usage in Workers

```python
from gym_gui.constants import WORKER_HEARTBEAT_INTERVAL_S
import time

# Periodic heartbeat
last_heartbeat = time.time()
while training:
    if time.time() - last_heartbeat > WORKER_HEARTBEAT_INTERVAL_S:
        emitter.heartbeat({"episode": current_episode})
        last_heartbeat = time.time()
```

## Integration with Paths and Settings

### Paths Module

Workers use `gym_gui.config.paths` for standardized directory structure:

```python
from gym_gui.config.paths import (
    get_run_dir,
    get_tensorboard_dir,
    get_checkpoints_dir,
    get_videos_dir,
)

run_dir = get_run_dir(run_id)
tensorboard_dir = get_tensorboard_dir(run_id)
checkpoint_dir = get_checkpoints_dir(run_id)
video_dir = get_videos_dir(run_id)
```

### Settings Module

Workers access global settings via `gym_gui.config.settings`:

```python
from gym_gui.config.settings import get_setting

debug_mode = get_setting("debug_mode", default=False)
default_seed = get_setting("default_seed", default=42)
```

## Worker Implementation Checklist

When creating a new worker:

- [ ] Implement `WorkerConfig` protocol with `to_dict()` and `from_dict()`
- [ ] Implement `WorkerRuntime` protocol with `run()` method
- [ ] Create `get_worker_metadata()` function returning `(WorkerMetadata, WorkerCapabilities)`
- [ ] Register entry point in `pyproject.toml`: `[project.entry-points."mosaic.workers"]`
- [ ] Add worker-specific log constants to `gym_gui/logging_config/log_constants.py`
- [ ] Use `TelemetryEmitter` with log constants for all lifecycle events
- [ ] Generate `WorkerAnalyticsManifest` after run completion
- [ ] Use centralized constants from `gym_gui/constants/`
- [ ] Use path helpers from `gym_gui.config.paths`
- [ ] Write tests for config, runtime, discovery, and telemetry

## Example: Minimal Worker Implementation

```python
# worker_package/config.py
from dataclasses import dataclass
from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol

@dataclass(frozen=True)
class MyWorkerConfig:
    run_id: str
    seed: int | None = None
    algo: str = "default"
    env_id: str = "CartPole-v1"

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "seed": self.seed, "algo": self.algo, "env_id": self.env_id}

    @classmethod
    def from_dict(cls, data: dict) -> "MyWorkerConfig":
        return cls(**data)

# worker_package/runtime.py
from gym_gui.core.worker import TelemetryEmitter
from gym_gui.logging_config.log_constants import LOG_WORKER_MYWORKER_RUNTIME_STARTED

class MyWorkerRuntime:
    def __init__(self, config, logger=None):
        self.config = config
        self.emitter = TelemetryEmitter(run_id=config.run_id, logger=logger)

    def run(self) -> dict:
        self.emitter.run_started(
            {"worker_type": "myworker", "algo": self.config.algo},
            constant=LOG_WORKER_MYWORKER_RUNTIME_STARTED,
        )

        # Training logic here
        results = {"episodes": 100, "reward": 195.0}

        self.emitter.run_completed(results)
        return results

# worker_package/__init__.py
from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

def get_worker_metadata():
    metadata = WorkerMetadata(
        name="My Worker",
        version="1.0.0",
        description="Example worker",
        author="Your Name",
        homepage="https://github.com/yourname/myworker",
        upstream_library="mylib",
        upstream_version="2.0.0",
        license="MIT",
    )
    capabilities = WorkerCapabilities(
        worker_type="myworker",
        supported_paradigms=("sequential",),
        env_families=("gymnasium",),
        action_spaces=("discrete",),
        observation_spaces=("vector",),
        max_agents=1,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=True,
        supports_pause_resume=False,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=512,
    )
    return metadata, capabilities

# pyproject.toml
[project.entry-points."mosaic.workers"]
myworker = "worker_package:get_worker_metadata"
```

## See Also

- **Worker Development Guide**: `docs/workers/WORKER_DEVELOPMENT_GUIDE.md` (comprehensive guide)
- **Migration Guide**: `docs/workers/MIGRATION_GUIDE.md` (migrating existing workers)
- **Main Architecture Doc**: `docs/Development_Progress/1.0_DAY_50/TASK_1/MOSAIC_WORKER_ARCHITECTURE.md`
- **Implementation Plan**: `docs/Development_Progress/1.0_DAY_50/TASK_2/deep-swinging-petal.md`
