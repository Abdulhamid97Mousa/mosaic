"""Shared trainer defaults organised by client, daemon, retry, and schema.

Consolidated from:
- Original: gym_gui/services/trainer/constants.py

Defines gRPC client configuration, daemon lifecycle, retry policies,
and training run validation defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainerClientDefaults:
    """gRPC client tuning for the GUI ↔ trainer bridge."""

    target: str = "127.0.0.1:50055"
    deadline_s: float = 10.0
    connect_timeout_s: float = 5.0
    keepalive_time_s: float = 30.0
    keepalive_timeout_s: float = 10.0
    max_message_bytes: int = 64 * 1024 * 1024
    http2_min_ping_interval_ms: int = 10_000


@dataclass(frozen=True)
class TrainerDaemonDefaults:
    """Lifecycle timings for the trainer daemon process."""

    poll_interval_s: float = 0.5
    startup_timeout_s: float = 10.0
    stop_timeout_s: float = 5.0
    port_probe_timeout_s: float = 1.0
    wal_checkpoint_interval_s: int = 300
    telemetry_batch_size: int = 64
    telemetry_checkpoint_interval: int = 1000
    telemetry_writer_queue_size: int = 512
    lock_file_name: str = "trainer.lock"
    pid_file_name: str = "trainer.pid"
    default_log_level: str = "INFO"


@dataclass(frozen=True)
class TrainerRetryDefaults:
    """Dispatch, heartbeat, and stream retry configuration."""

    dispatch_interval_s: float = 2.0
    monitor_interval_s: float = 1.0
    heartbeat_timeout_s: int = 300
    heartbeat_poll_interval_s: float = 30.0
    worker_sigterm_timeout_s: float = 5.0
    stream_reconnect_attempts: int = 10
    stream_reconnect_delay_s: float = 0.5
    stream_backoff_initial_s: float = 1.0
    stream_backoff_multiplier: float = 1.5
    stream_backoff_cap_s: float = 30.0
    stream_drain_timeout_s: float = 0.5


@dataclass(frozen=True)
class TrainRunSchemaDefaults:
    """Validation defaults for trainer run payloads."""

    run_name_max_length: int = 120
    resources_min_cpus: int = 1
    resources_min_memory_mb: int = 256
    resources_max_requested_gpus: int = 8
    resources_default_gpu_mandatory: bool = False
    artifacts_default_persist_logs: bool = True
    artifacts_default_keep_checkpoints: bool = True
    schedule_min_duration_s: int = 1
    schedule_min_steps: int = 1


@dataclass(frozen=True)
class TrainerDefaults:
    """Aggregate view of trainer defaults for easier consumption."""

    client: TrainerClientDefaults = field(default_factory=TrainerClientDefaults)
    daemon: TrainerDaemonDefaults = field(default_factory=TrainerDaemonDefaults)
    retry: TrainerRetryDefaults = field(default_factory=TrainerRetryDefaults)
    schema: TrainRunSchemaDefaults = field(default_factory=TrainRunSchemaDefaults)


TRAINER_DEFAULTS = TrainerDefaults()


__all__ = [
    "TRAINER_DEFAULTS",
    "TrainerDefaults",
    "TrainerClientDefaults",
    "TrainerDaemonDefaults",
    "TrainerRetryDefaults",
    "TrainRunSchemaDefaults",
]
