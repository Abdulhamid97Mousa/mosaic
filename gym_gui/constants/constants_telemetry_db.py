"""Persistence-related defaults for telemetry and trainer state.

Consolidated from:
- Original: gym_gui/telemetry/constants_db.py

Defines batch sizes, checkpoint intervals, and database sink configuration
for durable telemetry persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TelemetryDBSinkDefaults:
    """Batching and queue capacities for the telemetry DB sink."""

    batch_size: int = 128
    checkpoint_interval: int = 1024
    writer_queue_size: int = 4096


@dataclass(frozen=True)
class RegistryDefaults:
    """Trainer registry persistence bounds."""

    gpu_slot_capacity: int = 8


@dataclass(frozen=True)
class DatabaseDefaults:
    """Aggregated persistence defaults."""

    sink: TelemetryDBSinkDefaults = field(default_factory=TelemetryDBSinkDefaults)
    registry: RegistryDefaults = field(default_factory=RegistryDefaults)


DB_DEFAULTS = DatabaseDefaults()

__all__ = [
    "DB_DEFAULTS",
    "DatabaseDefaults",
    "TelemetryDBSinkDefaults",

    "RegistryDefaults",
]
