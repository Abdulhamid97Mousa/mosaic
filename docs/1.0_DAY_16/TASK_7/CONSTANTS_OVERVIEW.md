# Constants Files Overview

This document explains the purpose and organization of all `constants.py` files in the codebase.

## Summary: 4 Main Constants Files

| File | Purpose | Scope |
|------|---------|-------|
| `spade_bdi_rl/constants.py` | Worker-specific defaults (SPADE agent, Q-learning, step delays) | Worker-only |
| `gym_gui/services/trainer/constants.py` | Trainer infrastructure defaults (gRPC, dispatcher, retry config) | Trainer daemon/service |
| `gym_gui/ui/constants.py` | UI widget defaults (slider ranges, buffer sizes, render delays) | GUI layer |
| `gym_gui/telemetry/constants.py` | Telemetry system constants (queue sizes, logging levels, credits) | Telemetry hub/bus |

## Additional Constants Modules

| File | Purpose | Scope |
|------|---------|-------|
| `gym_gui/constants/game_constants.py` | Game environment defaults (FrozenLake, CliffWalking, Taxi maps) | **Canonical game configs** |
| `gym_gui/ui/constants_ui.py` | Dataclass-based UI defaults aggregate | UI (typed) |
| `gym_gui/telemetry/constants_db.py` | Database sink defaults (batching, checkpoints) | DB persistence |
| `gym_gui/telemetry/constants_bus.py` | Event bus queue defaults (RunBus, hub) | Event bus |

---

## 1. Worker Constants (`spade_bdi_rl/constants.py`)

**Purpose:** Worker-specific defaults that only exist in the SPADE-BDI worker process.

### Contents

#### Agent Credentials & Networking
```python
DEFAULT_AGENT_JID = "agent@localhost"
DEFAULT_AGENT_PASSWORD = "secret"
DEFAULT_EJABBERD_HOST = "localhost"
DEFAULT_EJABBERD_PORT = 5222
DEFAULT_AGENT_START_TIMEOUT_S = 10.0
```

#### Worker Runtime
```python
DEFAULT_STEP_DELAY_S = 0.14  # Delay between steps for observation
DEFAULT_WORKER_TELEMETRY_BUFFER_SIZE = 2048
DEFAULT_WORKER_EPISODE_BUFFER_SIZE = 100
```

#### Q-Learning Algorithm Defaults
```python
DEFAULT_Q_ALPHA = 0.1  # Learning rate
DEFAULT_Q_GAMMA = 0.99  # Discount factor
DEFAULT_Q_EPSILON_INIT = 1.0  # Initial exploration rate
DEFAULT_MAX_EPISODE_STEPS = 100
```

#### Policy Epsilon
```python
DEFAULT_CACHED_POLICY_EPSILON = 0.0  # No exploration for cached policies
DEFAULT_ONLINE_POLICY_EPSILON = 0.1  # Minimal exploration during online training
```

**Why separate?** These are worker-internal implementation details that should not be imported by the GUI. The worker must have its own constants for:
- SPADE agent configuration
- Q-learning algorithm parameters
- Worker telemetry buffer sizing
- Step delays for observation

---

## 2. Game Constants (`gym_gui/constants/game_constants.py`)

**Purpose:** **Canonical source of truth** for all game environment configurations (FrozenLake, CliffWalking, Taxi).

### Contents

#### FrozenLake-v1 (4×4)
```python
FROZEN_LAKE_DEFAULTS = ToyTextDefaults(
    grid_height=4,
    grid_width=4,
    start=(0, 0),
    goal=(3, 3),
    slippery=True,
    hole_count=4,
    official_map=(
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ),
)
```

#### FrozenLake-v2 (8×8)
```python
FROZEN_LAKE_V2_DEFAULTS = ToyTextDefaults(
    grid_height=8,
    grid_width=8,
    start=(0, 0),
    goal=(7, 7),
    slippery=False,
    hole_count=10,
    random_holes=False,
    official_map=(
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ),
)
```

#### CliffWalking (4×12)
```python
CLIFF_WALKING_DEFAULTS = ToyTextDefaults(
    grid_height=4,
    grid_width=12,
    start=(3, 0),
    goal=(3, 11),
    slippery=False,
)
```

#### Taxi (5×5)
```python
TAXI_DEFAULTS = ToyTextDefaults(
    grid_height=5,
    grid_width=5,
    start=(0, 0),
    goal=(4, 0),
)
```

**Why separate?** Game configs are:
- Used by **both** GUI (for rendering/visualization) **and** worker (for training)
- The **single source of truth** for environment parameters
- Need to match Gymnasium's official maps exactly
- Should be importable from either subsystem without circular dependencies

**History:** Previously duplicated in `spade_bdi_rl/constants.py`, now centralized here ✅

---

## 3. Trainer Constants (`gym_gui/services/trainer/constants.py`)

**Purpose:** Defaults for the trainer daemon (gRPC service, dispatcher, worker orchestration).

### Contents

#### gRPC Client Defaults
```python
class TrainerClientDefaults:
    target: str = "127.0.0.1:50055"
    deadline_s: float = 10.0
    connect_timeout_s: float = 5.0
    keepalive_time_s: float = 30.0
```

#### Daemon Lifecycle
```python
class TrainerDaemonDefaults:
    poll_interval_s: float = 0.5  # How often dispatcher checks for pending runs
    startup_timeout_s: float = 10.0
    stop_timeout_s: float = 5.0
```

#### Retry Configuration
```python
class TrainerRetryDefaults:
    dispatch_interval_s: float = 2.0
    monitor_interval_s: float = 1.0
    heartbeat_timeout_s: int = 300
    stream_reconnect_attempts: int = 10
```

**Why separate?** Trainer infrastructure is:
- Completely separate from worker code
- Only used by the trainer daemon/service
- Network/retry/infrastructure concerns, not game logic

---

## 4. UI Constants (`gym_gui/ui/constants.py`)

**Purpose:** UI widget defaults (slider ranges, buffer sizes, render delays for the GUI layer).

### Contents

#### Render Delay
```python
RENDER_DELAY_MIN_MS = 10
RENDER_DELAY_MAX_MS = 500
RENDER_DELAY_TICK_INTERVAL_MS = 50
DEFAULT_RENDER_DELAY_MS = 100  # 10 FPS
```

#### Training Speed Slider
```python
UI_TRAINING_SPEED_MIN = 0
UI_TRAINING_SPEED_MAX = 100
```

#### Telemetry Throttles
```python
TRAINING_TELEMETRY_THROTTLE_MIN = 1
TRAINING_TELEMETRY_THROTTLE_MAX = 10
UI_RENDERING_THROTTLE_MIN = 1
UI_RENDERING_THROTTLE_MAX = 10
```

#### Buffer Spin Boxes
```python
TELEMETRY_BUFFER_MIN = 256
TELEMETRY_BUFFER_MAX = 10_000
DEFAULT_TELEMETRY_BUFFER_SIZE = 512
EPISODE_BUFFER_MIN = 10
EPISODE_BUFFER_MAX = 1_000
DEFAULT_EPISODE_BUFFER_SIZE = 100
```

**Why separate?** UI concerns are:
- GUI-specific (widget limits, slider ranges)
- Not relevant to the worker
- User-facing configuration parameters

---

## 5. Telemetry Constants (`gym_gui/telemetry/constants.py`)

**Purpose:** Telemetry system constants (queue sizes, logging levels, credit system).

### Contents

#### Pre-Tab Buffers
```python
STEP_BUFFER_SIZE = 64  # max steps buffered before tab created
EPISODE_BUFFER_SIZE = 32  # max episodes buffered before tab created
RENDER_QUEUE_SIZE = 32  # max visual payloads queued for rendering
```

#### RunBus Queue Sizes
```python
RUNBUS_DEFAULT_QUEUE_SIZE = 2048  # default for all subscribers
RUNBUS_UI_PATH_QUEUE_SIZE = 512  # fast path (UI updates)
RUNBUS_DB_PATH_QUEUE_SIZE = 1024  # durable path (SQLite writes)
```

#### Credit System
```python
INITIAL_CREDITS = 200  # Initial credit grant per stream
MIN_CREDITS_THRESHOLD = 10  # Threshold before starvation warning
```

#### Logging Levels
```python
STEP_LOG_LEVEL = "DEBUG"  # Per-step logging
BATCH_LOG_LEVEL = "INFO"  # Per-batch/episode logging
ERROR_LOG_LEVEL = "ERROR"  # Exception logging
```

**Why separate?** Telemetry constants are:
- Infrastructure concerns (queue sizing, backpressure)
- Shared by multiple telemetry components (hub, bus, sink)
- Independent of game logic and UI widgets

---

## Additional Constants Modules

### `gym_gui/ui/constants_ui.py` (Typed UI Defaults)

**Purpose:** Dataclass-based aggregate of all UI constants (replaces `constants.py` eventually).

```python
@dataclass(frozen=True)
class RenderDefaults:
    min_delay_ms: int = 10
    max_delay_ms: int = 500
    default_delay_ms: int = 100

UI_DEFAULTS = UIDefaults()
```

**Why?** Provides type-safe, structured defaults instead of loose constants.

### `gym_gui/telemetry/constants_db.py` (Database Persistence)

**Purpose:** Database sink configuration (batching, checkpoints, writer queues).

```python
@dataclass(frozen=True)
class TelemetryDBSinkDefaults:
    batch_size: int = 128
    checkpoint_interval: int = 1024
    writer_queue_size: int = 4096
```

**Why?** Database-specific configuration separate from telemetry bus/hub.

### `gym_gui/telemetry/constants_bus.py` (Event Bus)

**Purpose:** RunBus queue defaults and telemetry stream configuration.

```python
@dataclass(frozen=True)
class RunBusQueueDefaults:
    default: int = 2048
    ui_path: int = 512
    db_path: int = 1024
```

**Why?** Event bus routing configuration separate from database persistence.

---

## Design Principles

### 1. **Separation of Concerns**
- Worker constants = worker-internal implementation details
- UI constants = GUI widget configuration
- Telemetry constants = infrastructure queue/logging configuration
- Game constants = environment configuration (shared between GUI and worker)

### 2. **No Circular Dependencies**
- Worker never imports GUI constants (except game constants via adapters)
- GUI never imports worker constants
- Telemetry constants are independent
- Trainer constants are independent

### 3. **Single Source of Truth**
- Game configs live in `gym_gui/constants/game_constants.py` ✅
- Each subsystem has its own constants for its own concerns ✅
- No duplication between subsystems ✅

### 4. **Gradual Migration to Dataclasses**
- New constants use dataclass-based aggregates (`constants_ui.py`, `constants_db.py`, `constants_bus.py`)
- Old constants files remain for backwards compatibility
- Migration path: import structured defaults instead of loose constants

---

## Usage Patterns

### Importing Game Constants (by both GUI and worker)
```python
from gym_gui.constants.game_constants import FROZEN_LAKE_V2_DEFAULTS
```

### Importing Worker Constants (worker only)
```python
from spade_bdi_rl.constants import DEFAULT_Q_ALPHA, DEFAULT_Q_GAMMA
```

### Importing UI Constants (GUI only)
```python
from gym_gui.ui.constants import DEFAULT_RENDER_DELAY_MS, TELEMETRY_BUFFER_MIN
```

### Importing Telemetry Constants (telemetry components)
```python
from gym_gui.telemetry.constants import RUNBUS_DEFAULT_QUEUE_SIZE, INITIAL_CREDITS
```

### Importing Typed Defaults (modern code)
```python
from gym_gui.ui.constants_ui import UI_DEFAULTS
render_delay = UI_DEFAULTS.render.default_delay_ms
```

---

## Summary: Constants Organization

| Concern | Location | Imports From |
|---------|----------|--------------|
| **Game configs** | `gym_gui/constants/game_constants.py` | Both GUI and worker |
| **Worker defaults** | `spade_bdi_rl/constants.py` | Worker only |
| **Trainer infrastructure** | `gym_gui/services/trainer/constants.py` | Trainer daemon only |
| **UI widgets** | `gym_gui/ui/constants.py` | GUI only |
| **Telemetry infrastructure** | `gym_gui/telemetry/constants.py` | Telemetry components only |
| **Database persistence** | `gym_gui/telemetry/constants_db.py` | DB sink only |
| **Event bus** | `gym_gui/telemetry/constants_bus.py` | RunBus hub only |
| **Typed UI defaults** | `gym_gui/ui/constants_ui.py` | Modern GUI code |

**Total: 8 constants files, each with a clear, single purpose.**

---

## Follow-Up Checklist

- Capture `.env` override matrix for each constants module (values, env key, default) and surface it in `docs/1.0_DAY_16/TASK_7/CONSTANTS_OVERRIDES.md` (new doc, pending).
- Add quick-start snippets that demonstrate how to request typed defaults from `constants_ui.py`, `constants_db.py`, and `constants_bus.py` inside tests to avoid hard-coded literals.
- Audit unit tests that still import legacy literals (notably telemetry queue sizes) and migrate them to consume the structured defaults exported here.
- Coordinate with Day 16 Task 2 logging work so any new logging scopes reuse the same `TRAINER_DEFAULTS`/`UI_DEFAULTS` registries instead of introducing subsystem-specific constants.


