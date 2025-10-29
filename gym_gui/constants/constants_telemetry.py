"""Centralized telemetry configuration constants.

Consolidated from:
- Original: gym_gui/telemetry/constants.py

Defines queue sizes, buffer configuration, logging levels, and credit system
for the telemetry event streaming infrastructure.
"""

from __future__ import annotations

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

# Live telemetry controller queue sizes
LIVE_STEP_QUEUE_SIZE = 64
LIVE_EPISODE_QUEUE_SIZE = 64
LIVE_CONTROL_QUEUE_SIZE = 32

# Telemetry hub defaults
TELEMETRY_HUB_MAX_QUEUE = 4096
TELEMETRY_HUB_BUFFER_SIZE = 2048

# Durable sink defaults
DB_SINK_BATCH_SIZE = 128
DB_SINK_CHECKPOINT_INTERVAL = 1024
DB_SINK_WRITER_QUEUE_SIZE = 4096

# Telemetry service history
TELEMETRY_SERVICE_HISTORY_LIMIT = 512

# Health monitor configuration
HEALTH_MONITOR_HEARTBEAT_INTERVAL_S = 5.0

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

# Time to wait before auto-starting rendering regulator (ms)
RENDER_BOOTSTRAP_TIMEOUT_MS = 500

__all__ = [
    "STEP_BUFFER_SIZE",
    "EPISODE_BUFFER_SIZE",
    "RENDER_QUEUE_SIZE",
    "RUNBUS_DEFAULT_QUEUE_SIZE",
    "RUNBUS_UI_PATH_QUEUE_SIZE",
    "RUNBUS_DB_PATH_QUEUE_SIZE",
    "LIVE_STEP_QUEUE_SIZE",
    "LIVE_EPISODE_QUEUE_SIZE",
    "LIVE_CONTROL_QUEUE_SIZE",
    "TELEMETRY_HUB_MAX_QUEUE",
    "TELEMETRY_HUB_BUFFER_SIZE",
    "DB_SINK_BATCH_SIZE",
    "DB_SINK_CHECKPOINT_INTERVAL",
    "DB_SINK_WRITER_QUEUE_SIZE",
    "TELEMETRY_SERVICE_HISTORY_LIMIT",
    "HEALTH_MONITOR_HEARTBEAT_INTERVAL_S",
    "STEP_LOG_LEVEL",
    "BATCH_LOG_LEVEL",
    "ERROR_LOG_LEVEL",
    "INITIAL_CREDITS",
    "MIN_CREDITS_THRESHOLD",
    "RENDER_BOOTSTRAP_TIMEOUT_MS",
]
