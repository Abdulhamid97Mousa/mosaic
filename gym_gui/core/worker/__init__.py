"""Worker standardization interfaces and base implementations.

This module provides the standard protocols and base classes that all MOSAIC
workers must implement to ensure consistent integration with the trainer daemon.

Usage:
    from gym_gui.core.worker import (
        WorkerConfig,
        WorkerRuntime,
        TelemetryEmitter,
        WorkerAnalyticsManifest,
        WorkerCLI,
        discover_workers,
    )

Workers should:
1. Implement WorkerConfig protocol for configuration
2. Implement WorkerRuntime protocol for execution
3. Use TelemetryEmitter for lifecycle events
4. Generate WorkerAnalyticsManifest for artifacts
5. Inherit from WorkerCLI for standard CLI interface
6. Register via entry point for automatic discovery
"""

from .analytics import (
    ArtifactsMetadata,
    CheckpointMetadata,
    TensorBoardMetadata,
    WandBMetadata,
    WorkerAnalyticsManifest,
)
from .cli_base import WorkerCLI
from .config_loader import (
    extract_worker_config,
    load_worker_config_from_file,
)
from .discovery import (
    WORKER_ENTRY_POINT_GROUP,
    DiscoveredWorker,
    WorkerDiscovery,
    discover_workers,
)
from .protocol import (
    WorkerCapabilities,
    WorkerConfig,
    WorkerMetadata,
    WorkerRuntime,
)
from .telemetry import (
    LifecycleEvent,
    LifecycleEventType,
    TelemetryEmitter,
)

__all__ = [
    # Protocol definitions
    "WorkerConfig",
    "WorkerRuntime",
    "WorkerCapabilities",
    "WorkerMetadata",
    # Telemetry
    "TelemetryEmitter",
    "LifecycleEvent",
    "LifecycleEventType",
    # Analytics
    "WorkerAnalyticsManifest",
    "ArtifactsMetadata",
    "TensorBoardMetadata",
    "WandBMetadata",
    "CheckpointMetadata",
    # CLI utilities
    "WorkerCLI",
    # Config loading
    "load_worker_config_from_file",
    "extract_worker_config",
    # Worker discovery
    "WorkerDiscovery",
    "DiscoveredWorker",
    "discover_workers",
    "WORKER_ENTRY_POINT_GROUP",
]

__version__ = "1.0.0"
