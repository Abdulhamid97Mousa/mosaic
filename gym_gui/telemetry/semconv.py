"""Telemetry semantic conventions.

This module centralizes naming for telemetry paths, modes, metrics, log IDs,
and environment variables so the rest of the code base can reference a single
source of truth. The strings follow OpenTelemetry-inspired conventions
(dot-separated, lowercase) so they remain stable if we later export them to an
OTel backend.
"""

from __future__ import annotations

from dataclasses import dataclass


class TelemetryPaths:
    """Canonical identifiers for telemetry fan-out paths."""

    FASTLANE = "fastlane"
    RUNBUS_UI = "runbus.ui"
    RUNBUS_DB = "runbus.db"


class TelemetryModes:
    """Supported telemetry modes exposed to operators."""

    FASTLANE_ONLY = "fastlane_only"
    UI_AND_DB = "ui_and_db"
    DB_ONLY = "db_only"


class VideoModes:
    """Fast Lane video rendering modes."""

    SINGLE = "single"
    GRID = "grid"
    OFF = "off"


class TelemetryMetrics:
    """OTel-style metric/attribute names."""

    FASTLANE_FRAME_RATE = "fastlane.frame_rate_hz"
    FASTLANE_READER_BACKLOG = "fastlane.reader_backlog"
    FASTLANE_RECONNECT_COUNT = "fastlane.reconnect_count"
    RUNBUS_UI_QUEUE_DEPTH = "runbus.ui.queue_depth"
    RUNBUS_DB_QUEUE_DEPTH = "runbus.db.queue_depth"
    RUNBUS_DB_DROPPED_EVENTS = "runbus.db.dropped_events_total"


class TelemetryLogIds:
    """Log constant names reserved for telemetry events."""

    FASTLANE_CONNECTED = "LOG_FASTLANE_CONNECTED"
    FASTLANE_UNAVAILABLE = "LOG_FASTLANE_UNAVAILABLE"
    FASTLANE_QUEUE_DEPTH = "LOG_FASTLANE_QUEUE_DEPTH"
    FASTLANE_READER_LAG = "LOG_FASTLANE_READER_LAG"
    FASTLANE_HEADER_INVALID = "LOG_FASTLANE_HEADER_INVALID"
    FASTLANE_FRAME_READ_ERROR = "LOG_FASTLANE_FRAME_READ_ERROR"
    RUNBUS_UI_QUEUE_DEPTH = "LOG_RUNBUS_UI_QUEUE_DEPTH"
    RUNBUS_DB_QUEUE_DEPTH = "LOG_RUNBUS_DB_QUEUE_DEPTH"


class TelemetryEnv:
    """Environment variable names shared across workers."""

    FASTLANE_ONLY = "GYM_GUI_FASTLANE_ONLY"
    FASTLANE_SLOT = "GYM_GUI_FASTLANE_SLOT"
    FASTLANE_VIDEO_MODE = "GYM_GUI_FASTLANE_VIDEO_MODE"
    FASTLANE_GRID_LIMIT = "GYM_GUI_FASTLANE_GRID_LIMIT"
    CLEANRL_INTERVAL_MS = "CLEANRL_FASTLANE_INTERVAL_MS"
    CLEANRL_MAX_DIM = "CLEANRL_FASTLANE_MAX_DIM"
    LEGACY_FASTLANE_ENABLED = "FASTLANE_ENABLED"
    LEGACY_FASTLANE_SLOT = "FASTLANE_SLOT"


@dataclass(frozen=True)
class TelemetryModeDescriptor:
    """Helper describing how a mode should be presented to operators."""

    name: str
    label: str
    description: str


TELEMETRY_MODE_DESCRIPTORS: dict[str, TelemetryModeDescriptor] = {
    TelemetryModes.FASTLANE_ONLY: TelemetryModeDescriptor(
        name=TelemetryModes.FASTLANE_ONLY,
        label="Fast Lane Only",
        description="Streams frames via shared memory only; disables RunBus durable writes.",
    ),
    TelemetryModes.UI_AND_DB: TelemetryModeDescriptor(
        name=TelemetryModes.UI_AND_DB,
        label="Dual Path (RunBus UI + SQLite)",
        description="Keeps the RunBus fast path for UI plus the durable SQLite writer enabled.",
    ),
    TelemetryModes.DB_ONLY: TelemetryModeDescriptor(
        name=TelemetryModes.DB_ONLY,
        label="Durable Only",
        description="(Reserved) Persist telemetry to SQLite without rendering in the UI.",
    ),
}

VIDEO_MODE_DESCRIPTORS: dict[str, TelemetryModeDescriptor] = {
    VideoModes.SINGLE: TelemetryModeDescriptor(
        name=VideoModes.SINGLE,
        label="Single Env (probe)",
        description="Stream one vectorized environment index to Fast Lane.",
    ),
    VideoModes.GRID: TelemetryModeDescriptor(
        name=VideoModes.GRID,
        label="Grid (first N envs)",
        description="Composite the first N env slots into a tiled frame before streaming.",
    ),
    VideoModes.OFF: TelemetryModeDescriptor(
        name=VideoModes.OFF,
        label="Video Off",
        description="Do not publish Fast Lane frames (metrics only).",
    ),
}
