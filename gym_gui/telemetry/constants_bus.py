from __future__ import annotations

"""Domain-scoped defaults for the telemetry event bus and hub."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunBusQueueDefaults:
    """Queue capacities for RunBus subscribers."""

    default: int = 2048
    ui_path: int = 512
    db_path: int = 1024


@dataclass(frozen=True)
class RunEventDefaults:
    """Bounds for run-level event fan-out."""

    queue_size: int = 1024
    history_limit: int = 2048


@dataclass(frozen=True)
class TelemetryStreamDefaults:
    """History and buffering for telemetry fan-out."""

    step_queue_size: int = 2048
    step_history_limit: int = 4096
    episode_history_limit: int = 1024


@dataclass(frozen=True)
class TelemetryHubDefaults:
    """Async hub queue sizing and service history."""

    max_queue: int = 4096
    buffer_size: int = 2048
    service_history_limit: int = 512


@dataclass(frozen=True)
class TelemetryLoggingDefaults:
    """Log level guidance for telemetry producers/consumers."""

    step_level: str = "DEBUG"
    batch_level: str = "INFO"
    error_level: str = "ERROR"


@dataclass(frozen=True)
class CreditDefaults:
    """Credit-based backpressure defaults."""

    initial_credits: int = 200
    starvation_threshold: int = 10


@dataclass(frozen=True)
class BusDefaults:
    """Aggregated defaults for telemetry bus orchestration."""

    run_bus: RunBusQueueDefaults = field(default_factory=RunBusQueueDefaults)
    run_events: RunEventDefaults = field(default_factory=RunEventDefaults)
    telemetry_streams: TelemetryStreamDefaults = field(default_factory=TelemetryStreamDefaults)
    hub: TelemetryHubDefaults = field(default_factory=TelemetryHubDefaults)
    logging: TelemetryLoggingDefaults = field(default_factory=TelemetryLoggingDefaults)
    credit: CreditDefaults = field(default_factory=CreditDefaults)


BUS_DEFAULTS = BusDefaults()

__all__ = [
    "BUS_DEFAULTS",
    "BusDefaults",
    "RunBusQueueDefaults",
    "RunEventDefaults",
    "TelemetryStreamDefaults",
    "TelemetryHubDefaults",
    "TelemetryLoggingDefaults",
    "CreditDefaults",
]
