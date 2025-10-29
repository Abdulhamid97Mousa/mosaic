"""Centralized constants package for GUI BDI RL.

This package consolidates all project-wide configuration constants into a
single, discoverable location. All module-level constants are organized by
domain and exported here for easy access.

Constants Organization:
======================

1. constants_core.py
   - Episode counter configuration (bounds, formatting)
   - Worker ID configuration (for distributed rollouts)
   - Thread safety and persistence parameters
   - Functions: format_episode_id(), parse_episode_id()
   - Dataclass: EpisodeCounterConfig

2. constants_ui.py
   - Render delay configuration (slider ranges, defaults)
   - Training speed slider configuration
   - Telemetry and rendering throttle ranges
   - Buffer sizing for telemetry and episodes
   - Dataclasses: RenderDefaults, SliderDefaults, BufferDefaults, UIDefaults

3. constants_telemetry.py
   - Queue and buffer sizes for telemetry infrastructure
   - RunBus subscriber queue configuration
   - Live telemetry controller queues
   - DB sink batching configuration
   - Logging levels for telemetry
   - Credit system configuration

4. constants_telemetry_bus.py
   - RunBus queue defaults
   - Run event fan-out bounds
   - Telemetry stream history and buffering
   - Telemetry hub queue sizing
   - Logging level guidance
   - Credit-based backpressure defaults
   - Dataclasses: RunBusQueueDefaults, RunEventDefaults, TelemetryStreamDefaults,
                   TelemetryHubDefaults, TelemetryLoggingDefaults, CreditDefaults,
                   BusDefaults

5. constants_telemetry_db.py
   - Telemetry DB sink batching and queue configuration
   - Health monitor heartbeat interval
   - Trainer registry persistence bounds
   - Dataclasses: TelemetryDBSinkDefaults, HealthMonitorDefaults,
                   RegistryDefaults, DatabaseDefaults

6. constants_trainer.py
   - gRPC client configuration (endpoint, timeouts, keepalive)
   - Trainer daemon lifecycle configuration
   - Trainer retry policies and backoff configuration
   - Training run schema validation defaults
   - Dataclasses: TrainerClientDefaults, TrainerDaemonDefaults,
                   TrainerRetryDefaults, TrainRunSchemaDefaults, TrainerDefaults

7. game_constants.py (pre-existing, in this package)
   - Game environment configuration defaults

Import Examples:
================

# Option 1: Import specific constants
from gym_gui.constants import format_episode_id, DEFAULT_MAX_EPISODES_PER_RUN
from gym_gui.constants import UI_DEFAULTS, BUS_DEFAULTS

# Option 2: Import entire submodule if needed for access to multiple related constants
from gym_gui.constants import constants_core, constants_ui
# Then use: constants_core.format_episode_id(...)

# Option 3: For backwards compatibility with old import paths
# Old: from gym_gui.telemetry.constants import STEP_BUFFER_SIZE
# New: from gym_gui.constants import STEP_BUFFER_SIZE
"""

# ================================================================
# Core / Episode Counter Constants
# ================================================================

from gym_gui.constants.constants_core import (
    COUNTER_CAPACITY_EXCEEDED_ERROR,
    COUNTER_EXCEEDS_MAX_ERROR,
    COUNTER_LOCK_TIMEOUT_S,
    COUNTER_NOT_INITIALIZED_ERROR,
    DEFAULT_MAX_EPISODES_PER_RUN,
    EP_INDEX_COLUMN,
    EPISODE_COUNTER_WIDTH,
    EPISODE_ID_COLUMN,
    EPISODE_ID_PREFIX,
    EPISODE_ID_SEPARATOR,
    EpisodeCounterConfig,
    INVALID_MAX_EPISODES_ERROR,
    MAX_COUNTER_VALUE,
    MAX_EPISODES_COLUMN,
    MAX_EPISODES_REACHED_ERROR,
    RESUME_CAPACITY_WARNING_THRESHOLD,
    WORKER_ID_COLUMN,
    WORKER_ID_PREFIX,
    WORKER_ID_WIDTH,
    format_episode_id,
    parse_episode_id,
)

# ================================================================
# UI Constants
# ================================================================

from gym_gui.constants.constants_ui import (
    BUFFER_BUFFER_MIN,  # For backwards compatibility aliases
    BufferDefaults,
    DEFAULT_EPISODE_BUFFER_SIZE,
    DEFAULT_RENDER_DELAY_MS,
    DEFAULT_TELEMETRY_BUFFER_SIZE,
    EPISODE_BUFFER_MAX,
    EPISODE_BUFFER_MIN,
    RENDER_DELAY_MAX_MS,
    RENDER_DELAY_MIN_MS,
    RENDER_DELAY_TICK_INTERVAL_MS,
    RenderDefaults,
    SliderDefaults,
    TELEMETRY_BUFFER_MAX,
    TELEMETRY_BUFFER_MIN,
    TRAINING_TELEMETRY_THROTTLE_MAX,
    TRAINING_TELEMETRY_THROTTLE_MIN,
    UI_DEFAULTS,
    UI_RENDERING_THROTTLE_MAX,
    UI_RENDERING_THROTTLE_MIN,
    UI_TRAINING_SPEED_MAX,
    UI_TRAINING_SPEED_MIN,
    UIDefaults,
)

# ================================================================
# Telemetry Constants
# ================================================================

from gym_gui.constants.constants_telemetry import (
    BATCH_LOG_LEVEL,
    DB_SINK_BATCH_SIZE,
    DB_SINK_CHECKPOINT_INTERVAL,
    DB_SINK_WRITER_QUEUE_SIZE,
    EPISODE_BUFFER_SIZE,
    ERROR_LOG_LEVEL,
    HEALTH_MONITOR_HEARTBEAT_INTERVAL_S,
    INITIAL_CREDITS,
    LIVE_CONTROL_QUEUE_SIZE,
    LIVE_EPISODE_QUEUE_SIZE,
    LIVE_STEP_QUEUE_SIZE,
    MIN_CREDITS_THRESHOLD,
    RENDER_BOOTSTRAP_TIMEOUT_MS,
    RENDER_QUEUE_SIZE,
    RUNBUS_DB_PATH_QUEUE_SIZE,
    RUNBUS_DEFAULT_QUEUE_SIZE,
    RUNBUS_UI_PATH_QUEUE_SIZE,
    STEP_BUFFER_SIZE,
    STEP_LOG_LEVEL,
    TELEMETRY_HUB_BUFFER_SIZE,
    TELEMETRY_HUB_MAX_QUEUE,
    TELEMETRY_SERVICE_HISTORY_LIMIT,
)

# ================================================================
# Telemetry Bus Constants
# ================================================================

from gym_gui.constants.constants_telemetry_bus import (
    BUS_DEFAULTS,
    BusDefaults,
    CreditDefaults,
    RunBusQueueDefaults,
    RunEventDefaults,
    TelemetryHubDefaults,
    TelemetryLoggingDefaults,
    TelemetryStreamDefaults,
)

# ================================================================
# Telemetry DB Constants
# ================================================================

from gym_gui.constants.constants_telemetry_db import (
    DB_DEFAULTS,
    DatabaseDefaults,
    HealthMonitorDefaults,
    RegistryDefaults,
    TelemetryDBSinkDefaults,
)

# ================================================================
# Trainer Constants
# ================================================================

from gym_gui.constants.constants_trainer import (
    TRAINER_DEFAULTS,
    TrainerClientDefaults,
    TrainerDaemonDefaults,
    TrainerDefaults,
    TrainerRetryDefaults,
    TrainRunSchemaDefaults,
)

# ================================================================
# Pre-existing Constants (game_constants.py)
# ================================================================

# Note: game_constants.py is pre-existing and may not expose __all__
# We handle its exports dynamically but list them explicitly below
try:
    from gym_gui.constants import constants_game as _constants_game
except ImportError:
    _constants_game = None  # type: ignore

if _constants_game is not None:
    _game_exports = getattr(_constants_game, "__all__", ())
    for _name in _game_exports:
        globals()[_name] = getattr(_constants_game, _name)
else:
    _game_exports = ()

__all__: list[str] = [
    # Core / Episode Counter
    "COUNTER_CAPACITY_EXCEEDED_ERROR",
    "COUNTER_EXCEEDS_MAX_ERROR",
    "COUNTER_LOCK_TIMEOUT_S",
    "COUNTER_NOT_INITIALIZED_ERROR",
    "DEFAULT_MAX_EPISODES_PER_RUN",
    "EP_INDEX_COLUMN",
    "EPISODE_COUNTER_WIDTH",
    "EPISODE_ID_COLUMN",
    "EPISODE_ID_PREFIX",
    "EPISODE_ID_SEPARATOR",
    "EpisodeCounterConfig",
    "INVALID_MAX_EPISODES_ERROR",
    "MAX_COUNTER_VALUE",
    "MAX_EPISODES_COLUMN",
    "MAX_EPISODES_REACHED_ERROR",
    "RESUME_CAPACITY_WARNING_THRESHOLD",
    "WORKER_ID_COLUMN",
    "WORKER_ID_PREFIX",
    "WORKER_ID_WIDTH",
    "format_episode_id",
    "parse_episode_id",
    # UI
    "BufferDefaults",
    "DEFAULT_EPISODE_BUFFER_SIZE",
    "DEFAULT_RENDER_DELAY_MS",
    "DEFAULT_TELEMETRY_BUFFER_SIZE",
    "EPISODE_BUFFER_MAX",
    "EPISODE_BUFFER_MIN",
    "RENDER_DELAY_MAX_MS",
    "RENDER_DELAY_MIN_MS",
    "RENDER_DELAY_TICK_INTERVAL_MS",
    "RenderDefaults",
    "SliderDefaults",
    "TELEMETRY_BUFFER_MAX",
    "TELEMETRY_BUFFER_MIN",
    "TRAINING_TELEMETRY_THROTTLE_MAX",
    "TRAINING_TELEMETRY_THROTTLE_MIN",
    "UI_DEFAULTS",
    "UI_RENDERING_THROTTLE_MAX",
    "UI_RENDERING_THROTTLE_MIN",
    "UI_TRAINING_SPEED_MAX",
    "UI_TRAINING_SPEED_MIN",
    "UIDefaults",
    # Telemetry
    "BATCH_LOG_LEVEL",
    "DB_SINK_BATCH_SIZE",
    "DB_SINK_CHECKPOINT_INTERVAL",
    "DB_SINK_WRITER_QUEUE_SIZE",
    "EPISODE_BUFFER_SIZE",
    "ERROR_LOG_LEVEL",
    "HEALTH_MONITOR_HEARTBEAT_INTERVAL_S",
    "INITIAL_CREDITS",
    "LIVE_CONTROL_QUEUE_SIZE",
    "LIVE_EPISODE_QUEUE_SIZE",
    "LIVE_STEP_QUEUE_SIZE",
    "MIN_CREDITS_THRESHOLD",
    "RENDER_BOOTSTRAP_TIMEOUT_MS",
    "RENDER_QUEUE_SIZE",
    "RUNBUS_DB_PATH_QUEUE_SIZE",
    "RUNBUS_DEFAULT_QUEUE_SIZE",
    "RUNBUS_UI_PATH_QUEUE_SIZE",
    "STEP_BUFFER_SIZE",
    "STEP_LOG_LEVEL",
    "TELEMETRY_HUB_BUFFER_SIZE",
    "TELEMETRY_HUB_MAX_QUEUE",
    "TELEMETRY_SERVICE_HISTORY_LIMIT",
    # Telemetry Bus
    "BUS_DEFAULTS",
    "BusDefaults",
    "CreditDefaults",
    "RunBusQueueDefaults",
    "RunEventDefaults",
    "TelemetryHubDefaults",
    "TelemetryLoggingDefaults",
    "TelemetryStreamDefaults",
    # Telemetry DB
    "DB_DEFAULTS",
    "DatabaseDefaults",
    "HealthMonitorDefaults",
    "RegistryDefaults",
    "TelemetryDBSinkDefaults",
    # Trainer
    "TRAINER_DEFAULTS",
    "TrainerClientDefaults",
    "TrainerDaemonDefaults",
    "TrainerDefaults",
    "TrainerRetryDefaults",
    "TrainRunSchemaDefaults",
    # Game (will be extended if game_constants has __all__)
]

# Append game constants if they're available
if _game_exports:  # pragma: no cover
    __all__.extend(_game_exports)  # type: ignore[attr-defined]

# Clean up temporary variables
del _constants_game, _game_exports, _name
