"""Worker discovery, subprocess management, and telemetry defaults.

Defines standardized constants for:
- Worker discovery via setuptools entry points
- Worker subprocess lifecycle management
- Worker telemetry and heartbeat configuration
- Worker resource limits and timeouts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkerDiscoveryDefaults:
    """Worker discovery and registry configuration."""

    entry_point_group: str = "mosaic.workers"
    """setuptools entry point group for worker plugins."""

    discovery_timeout_s: float = 5.0
    """Maximum time to wait for worker discovery to complete."""

    metadata_cache_ttl_s: float = 300.0
    """Time-to-live for cached worker metadata (5 minutes)."""

    enable_discovery_cache: bool = True
    """Whether to cache discovered workers."""

    reload_on_import_error: bool = False
    """Whether to retry discovery on import errors."""


@dataclass(frozen=True)
class WorkerSubprocessDefaults:
    """Worker subprocess management and lifecycle."""

    startup_timeout_s: float = 60.0
    """Maximum time to wait for worker subprocess to start."""

    shutdown_timeout_s: float = 30.0
    """Maximum time to wait for graceful worker shutdown."""

    heartbeat_interval_s: float = 30.0
    """Expected interval between worker heartbeat signals."""

    heartbeat_timeout_s: float = 300.0
    """Maximum time without heartbeat before declaring worker dead."""

    sigterm_timeout_s: float = 5.0
    """Time to wait after SIGTERM before sending SIGKILL."""

    restart_on_failure: bool = False
    """Whether to automatically restart failed workers."""

    max_restart_attempts: int = 3
    """Maximum number of automatic restart attempts."""

    restart_cooldown_s: float = 10.0
    """Minimum time between restart attempts."""


@dataclass(frozen=True)
class WorkerTelemetryDefaults:
    """Worker telemetry and logging configuration."""

    buffer_size: int = 512
    """Size of telemetry event buffer for batching."""

    jsonl_output: bool = True
    """Whether to emit JSONL telemetry to stdout."""

    structured_logging: bool = True
    """Whether to emit structured logs via Python logging."""

    log_level: int = logging.INFO
    """Default log level for worker logging."""

    emit_heartbeats: bool = True
    """Whether workers should emit periodic heartbeats."""

    emit_lifecycle_events: bool = True
    """Whether to emit lifecycle events (started, completed, failed)."""

    emit_checkpoint_events: bool = True
    """Whether to emit checkpoint save/load events."""

    flush_on_heartbeat: bool = False
    """Whether to flush telemetry buffer on each heartbeat."""


@dataclass(frozen=True)
class WorkerResourceDefaults:
    """Worker resource limits and requirements."""

    default_cpu_cores: int = 1
    """Default number of CPU cores requested per worker."""

    default_memory_mb: int = 512
    """Default memory allocation in MB per worker."""

    default_gpu_memory_mb: int | None = None
    """Default GPU memory allocation in MB (None = no GPU)."""

    max_cpu_cores: int = 32
    """Maximum CPU cores a single worker can request."""

    max_memory_mb: int = 32 * 1024
    """Maximum memory in MB a single worker can request (32 GB)."""

    max_gpu_memory_mb: int = 24 * 1024
    """Maximum GPU memory in MB per worker (24 GB)."""

    enforce_limits: bool = True
    """Whether to enforce resource limits on workers."""


@dataclass(frozen=True)
class WorkerDefaults:
    """Aggregate view of worker defaults for easier consumption."""

    discovery: WorkerDiscoveryDefaults = field(default_factory=WorkerDiscoveryDefaults)
    subprocess: WorkerSubprocessDefaults = field(default_factory=WorkerSubprocessDefaults)
    telemetry: WorkerTelemetryDefaults = field(default_factory=WorkerTelemetryDefaults)
    resources: WorkerResourceDefaults = field(default_factory=WorkerResourceDefaults)


WORKER_DEFAULTS = WorkerDefaults()


# Convenience aliases for backwards compatibility
WORKER_ENTRY_POINT_GROUP = WORKER_DEFAULTS.discovery.entry_point_group
WORKER_DISCOVERY_TIMEOUT_S = WORKER_DEFAULTS.discovery.discovery_timeout_s
WORKER_METADATA_CACHE_TTL_S = WORKER_DEFAULTS.discovery.metadata_cache_ttl_s

WORKER_HEARTBEAT_INTERVAL_S = WORKER_DEFAULTS.subprocess.heartbeat_interval_s
WORKER_STARTUP_TIMEOUT_S = WORKER_DEFAULTS.subprocess.startup_timeout_s
WORKER_SHUTDOWN_TIMEOUT_S = WORKER_DEFAULTS.subprocess.shutdown_timeout_s

WORKER_TELEMETRY_BUFFER_SIZE = WORKER_DEFAULTS.telemetry.buffer_size
WORKER_LOG_LEVEL = WORKER_DEFAULTS.telemetry.log_level


__all__ = [
    "WORKER_DEFAULTS",
    "WorkerDefaults",
    "WorkerDiscoveryDefaults",
    "WorkerSubprocessDefaults",
    "WorkerTelemetryDefaults",
    "WorkerResourceDefaults",
    # Convenience aliases
    "WORKER_ENTRY_POINT_GROUP",
    "WORKER_DISCOVERY_TIMEOUT_S",
    "WORKER_METADATA_CACHE_TTL_S",
    "WORKER_HEARTBEAT_INTERVAL_S",
    "WORKER_STARTUP_TIMEOUT_S",
    "WORKER_SHUTDOWN_TIMEOUT_S",
    "WORKER_TELEMETRY_BUFFER_SIZE",
    "WORKER_LOG_LEVEL",
]
