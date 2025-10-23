"""Centralized telemetry configuration constants."""

# ================================================================
# Queue and Buffer Sizes
# ================================================================

# Pre-tab buffers in LiveTelemetryController (per run/agent pair)
STEP_BUFFER_SIZE = 64  # max steps buffered before tab created
EPISODE_BUFFER_SIZE = 32  # max episodes buffered before tab created

# Rendering regulator queue size
RENDER_QUEUE_SIZE = 32  # max visual payloads queued for rendering

# RunBus subscriber queue sizes
RUNBUS_DEFAULT_QUEUE_SIZE = 2048  # default for all subscribers
RUNBUS_UI_PATH_QUEUE_SIZE = 512  # fast path (UI updates)
RUNBUS_DB_PATH_QUEUE_SIZE = 1024  # durable path (SQLite writes)

# ================================================================
# Logging Levels for Telemetry
# ================================================================

# Per-step telemetry logging: use DEBUG to avoid spam
STEP_LOG_LEVEL = "DEBUG"

# Per-batch/episode telemetry logging: use INFO
BATCH_LOG_LEVEL = "INFO"

# Exception/failure telemetry logging: use ERROR
ERROR_LOG_LEVEL = "ERROR"

# ================================================================
# Credit System Configuration
# ================================================================

# Initial credit grant for each (run_id, agent_id) stream
INITIAL_CREDITS = 200

# Minimum credits before emitting CONTROL.STARVATION message
MIN_CREDITS_THRESHOLD = 10

# ================================================================
# Rendering Configuration
# ================================================================

# Default rendering delay in milliseconds
DEFAULT_RENDER_DELAY_MS = 100  # 10 FPS

# Time to wait before auto-starting rendering regulator (ms)
RENDER_BOOTSTRAP_TIMEOUT_MS = 500

__all__ = [
    "STEP_BUFFER_SIZE",
    "EPISODE_BUFFER_SIZE",
    "RENDER_QUEUE_SIZE",
    "RUNBUS_DEFAULT_QUEUE_SIZE",
    "RUNBUS_UI_PATH_QUEUE_SIZE",
    "RUNBUS_DB_PATH_QUEUE_SIZE",
    "STEP_LOG_LEVEL",
    "BATCH_LOG_LEVEL",
    "ERROR_LOG_LEVEL",
    "INITIAL_CREDITS",
    "MIN_CREDITS_THRESHOLD",
    "DEFAULT_RENDER_DELAY_MS",
    "RENDER_BOOTSTRAP_TIMEOUT_MS",
]
