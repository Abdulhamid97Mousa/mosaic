# MOSAIC Worker Development Guide

**Version**: 1.0
**Last Updated**: 2024-12-30
**Status**: Production Ready

This guide provides comprehensive instructions for developing standardized workers for the MOSAIC BDI-RL framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Testing Your Worker](#testing-your-worker)
6. [Best Practices](#best-practices)
7. [Common Pitfalls](#common-pitfalls)
8. [Reference Examples](#reference-examples)

---

## Overview

### What is a MOSAIC Worker?

A **MOSAIC Worker** is a standardized subprocess wrapper around a reinforcement learning library or framework (e.g., CleanRL, Ray RLlib, XuanCe, BARLOG). Workers provide:

- **Uniform Configuration Interface**: Standardized config dataclasses
- **Lifecycle Management**: Standardized telemetry emission (run_started, heartbeat, run_completed, run_failed)
- **Analytics Integration**: Automatic generation of analytics manifests
- **Discoverability**: Entry point registration for automatic discovery
- **Protocol Compliance**: Type-safe interfaces via Python Protocols

### Why Standardize Workers?

1. **Interoperability**: All workers work with the same GUI and CLI
2. **Discoverability**: Workers are automatically discovered via entry points
3. **Observability**: Uniform telemetry and logging
4. **Testability**: Standard test patterns and protocols
5. **Maintainability**: Consistent structure across all workers

### Core Components

Every MOSAIC worker consists of:

1. **Config Module** (`config.py`): Configuration dataclass implementing `WorkerConfig` protocol
2. **Runtime Module** (`runtime.py`): Execution logic with lifecycle event emission
3. **Analytics Module** (`analytics.py`): Analytics manifest generation
4. **Package Module** (`__init__.py`): Worker metadata and capabilities
5. **Entry Point** (`pyproject.toml`): Registration for discovery
6. **Tests** (`tests/test_*_standardization.py`): Protocol compliance and integration tests

---

## Quick Start

### Prerequisites

- Python 3.10+
- MOSAIC framework installed (`gym_gui` package)
- Your RL library/framework installed

### Minimal Worker Template

```python
# my_worker/config.py
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class MyWorkerConfig:
    run_id: str
    env_id: str
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")

        # Protocol compliance check
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
            assert isinstance(self, WorkerConfigProtocol)
        except ImportError:
            pass  # gym_gui not available, skip check

    def to_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "env_id": self.env_id, "seed": self.seed}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MyWorkerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# my_worker/runtime.py
from typing import Dict, Any

try:
    from gym_gui.core.worker import TelemetryEmitter
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    TelemetryEmitter = None

class MyWorkerRuntime:
    def __init__(self, config: MyWorkerConfig):
        self.config = config
        if _HAS_GYM_GUI:
            self._emitter = TelemetryEmitter(run_id=config.run_id)
        else:
            self._emitter = None

    def run(self) -> Dict[str, Any]:
        # Emit lifecycle events
        if self._emitter:
            self._emitter.run_started({"env_id": self.config.env_id})

        try:
            # Your training logic here
            result = self._train()

            summary = {"status": "completed", "result": result}

            if self._emitter:
                self._emitter.run_completed(summary)

            return summary
        except Exception as e:
            error_summary = {"status": "failed", "error": str(e)}
            if self._emitter:
                self._emitter.run_failed(error_summary)
            raise

    def _train(self):
        # Implement your training logic
        pass


# my_worker/__init__.py
def get_worker_metadata() -> tuple:
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="My Worker",
        version="1.0.0",
        description="My RL worker",
        author="Your Name",
        homepage="https://github.com/...",
        upstream_library="mylib",
        upstream_version="1.0.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="myworker",
        supported_paradigms=("sequential",),
        env_families=("gymnasium",),
        action_spaces=("discrete", "continuous"),
        observation_spaces=("vector", "image"),
        max_agents=1,
        supports_checkpointing=True,
        requires_gpu=False,
    )

    return metadata, capabilities


# pyproject.toml
[project.entry-points."mosaic.workers"]
myworker = "my_worker:get_worker_metadata"
```

---

## Architecture Overview

### Protocol-Based Design

MOSAIC uses **Python Protocols** (structural subtyping) instead of inheritance:

```python
from typing import Protocol, Any, Dict

class WorkerConfig(Protocol):
    """Protocol that all worker configs must implement."""
    run_id: str
    seed: int | None

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerConfig": ...
```

**Advantages**:
- No forced inheritance hierarchy
- Duck typing with type safety
- Easier to retrofit existing code
- Better for third-party integrations

### Worker Lifecycle

```
┌─────────────────┐
│   Discovery     │  Entry point loaded
│  (Entry Point)  │  get_worker_metadata() called
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Config Loading  │  load_worker_config(path)
│   & Validation  │  Protocol compliance checked
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Runtime Init    │  TelemetryEmitter created
│                 │  Resources initialized
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  run_started    │  ━━━━━━━━━━━━━━━━━━━━━━┓
└────────┬────────┘                        ┃
         │                                 ┃
         ▼                                 ┃
┌─────────────────┐                        ┃
│  Training Loop  │  heartbeat events      ┃  Lifecycle
│                 │  emitted periodically  ┃  Events
└────────┬────────┘                        ┃
         │                                 ┃
         ▼                                 ┃
┌─────────────────┐                        ┃
│ run_completed   │  ━━━━━━━━━━━━━━━━━━━━━━┛
│   or failed     │  Analytics manifest written
└─────────────────┘
```

### Directory Structure

```
3rd_party/my_worker/
├── pyproject.toml              # Package metadata + entry point
├── README.md
├── my_worker/
│   ├── __init__.py             # get_worker_metadata()
│   ├── config.py               # WorkerConfig implementation
│   ├── runtime.py              # Training orchestration
│   ├── analytics.py            # Analytics manifest generation
│   └── cli.py                  # Optional CLI interface
└── tests/
    └── test_my_worker_standardization.py
```

---

## Step-by-Step Implementation

### Step 1: Create Config Module

**File**: `my_worker/config.py`

```python
"""Configuration dataclass for My Worker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

@dataclass
class MyWorkerConfig:
    """Configuration for My Worker training runs.

    This dataclass implements the WorkerConfig protocol.

    Attributes:
        run_id: Unique identifier for this run (REQUIRED by protocol)
        env_id: Environment to train on
        algorithm: Algorithm name
        total_steps: Total training steps
        seed: Random seed (REQUIRED by protocol, can be None)
        learning_rate: Learning rate
        batch_size: Batch size
    """

    # Protocol-required fields
    run_id: str
    seed: int | None = None

    # Worker-specific fields
    env_id: str = "CartPole-v1"
    algorithm: str = "dqn"
    total_steps: int = 100_000
    learning_rate: float = 1e-4
    batch_size: int = 32

    def __post_init__(self) -> None:
        """Validate configuration and assert protocol compliance."""
        # Validate required fields
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.env_id:
            raise ValueError("env_id is required")

        # Validate ranges
        if self.total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        # Protocol compliance assertion (graceful fallback if gym_gui unavailable)
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
            assert isinstance(self, WorkerConfigProtocol), (
                "MyWorkerConfig must implement WorkerConfig protocol"
            )
        except ImportError:
            pass  # gym_gui not available, skip protocol check

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (REQUIRED by protocol)."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "env_id": self.env_id,
            "algorithm": self.algorithm,
            "total_steps": self.total_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MyWorkerConfig":
        """Create config from dictionary (REQUIRED by protocol)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def load_worker_config(config_path: str) -> MyWorkerConfig:
    """Load worker configuration from JSON file.

    Handles both direct config format and nested metadata.worker.config structure.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Parsed MyWorkerConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw_data = json.load(f)

    # Handle nested metadata.worker.config structure from GUI
    if "metadata" in raw_data and "worker" in raw_data["metadata"]:
        worker_data = raw_data["metadata"]["worker"]
        config_data = worker_data.get("config", {})
    else:
        # Direct config format
        config_data = raw_data

    return MyWorkerConfig.from_dict(config_data)


__all__ = ["MyWorkerConfig", "load_worker_config"]
```

**Key Points**:
- Use `@dataclass` for automatic `__init__`, `__repr__`, etc.
- `run_id` and `seed` are REQUIRED by protocol
- Add `__post_init__()` for validation
- Graceful fallback if `gym_gui` not available
- Support both direct and nested config formats

---

### Step 2: Create Runtime Module

**File**: `my_worker/runtime.py`

```python
"""Runtime orchestration for My Worker."""

from __future__ import annotations

import logging
from typing import Any, Dict

from .config import MyWorkerConfig

# Import standardized telemetry
try:
    from gym_gui.core.worker import TelemetryEmitter
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    TelemetryEmitter = None

# Import analytics
try:
    from .analytics import write_analytics_manifest
    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False
    write_analytics_manifest = None

LOGGER = logging.getLogger(__name__)


class MyWorkerRuntime:
    """Orchestrate My Worker training execution.

    Example:
        >>> config = MyWorkerConfig(run_id="test", env_id="CartPole-v1")
        >>> runtime = MyWorkerRuntime(config)
        >>> summary = runtime.run()
        >>> print(summary["status"])
        'completed'
    """

    def __init__(self, config: MyWorkerConfig, *, dry_run: bool = False):
        """Initialize runtime.

        Args:
            config: Worker configuration
            dry_run: If True, validate without executing
        """
        self.config = config
        self._dry_run = dry_run

        # Create lifecycle telemetry emitter
        if _HAS_GYM_GUI:
            self._emitter = TelemetryEmitter(run_id=config.run_id)
        else:
            self._emitter = None

    def run(self) -> Dict[str, Any]:
        """Execute training run.

        Returns:
            Dictionary with execution results (compatible with standardized interface)

        Raises:
            Exception: Propagated from training on failure
        """
        # Emit run_started lifecycle event
        if self._emitter:
            self._emitter.run_started({
                "env_id": self.config.env_id,
                "algorithm": self.config.algorithm,
                "total_steps": self.config.total_steps,
            })

        if self._dry_run:
            LOGGER.info("Dry-run mode | env_id=%s", self.config.env_id)
            summary = {
                "status": "dry-run",
                "env_id": self.config.env_id,
                "config": self.config.to_dict(),
            }
            if self._emitter:
                self._emitter.run_completed(summary)
            return summary

        try:
            LOGGER.info(
                "Starting training | env_id=%s algorithm=%s steps=%d",
                self.config.env_id,
                self.config.algorithm,
                self.config.total_steps,
            )

            # Execute your training logic
            result = self._train()

            LOGGER.info("Training completed | env_id=%s", self.config.env_id)

            # Generate analytics manifest
            manifest_path = None
            if _HAS_ANALYTICS:
                try:
                    manifest_path = write_analytics_manifest(
                        self.config,
                        notes=f"Training {self.config.algorithm} on {self.config.env_id}",
                    )
                    LOGGER.info("Analytics manifest written to: %s", manifest_path)
                except Exception as e:
                    LOGGER.warning("Failed to write analytics manifest: %s", e)

            summary = {
                "status": "completed",
                "env_id": self.config.env_id,
                "result": result,
                "config": self.config.to_dict(),
                "analytics_manifest": str(manifest_path) if manifest_path else None,
            }

            # Emit run_completed lifecycle event
            if self._emitter:
                self._emitter.run_completed(summary)

            return summary

        except Exception as e:
            LOGGER.error("Training failed: %s", e, exc_info=True)

            error_summary = {
                "status": "failed",
                "env_id": self.config.env_id,
                "error": str(e),
                "config": self.config.to_dict(),
            }

            # Emit run_failed lifecycle event
            if self._emitter:
                self._emitter.run_failed(error_summary)

            raise

    def _train(self) -> Dict[str, Any]:
        """Execute training loop.

        Returns:
            Training results dictionary
        """
        # TODO: Implement your training logic here
        # Example:
        # - Create environment
        # - Initialize agent
        # - Run training loop
        # - Emit heartbeat events periodically
        # - Return final metrics

        # Emit heartbeat during training
        if self._emitter:
            self._emitter.heartbeat({
                "step": 1000,
                "episode_reward": 42.0,
            })

        return {
            "final_reward": 100.0,
            "episodes": 50,
        }


__all__ = ["MyWorkerRuntime"]
```

**Key Points**:
- Return `Dict[str, Any]` (not a custom dataclass)
- Emit lifecycle events: `run_started`, `heartbeat`, `run_completed`, `run_failed`
- Graceful handling of missing `gym_gui`
- Generate analytics manifest on completion
- Handle errors and emit `run_failed`

---

### Step 3: Create Analytics Module

**File**: `my_worker/analytics.py`

```python
"""Analytics manifest generation for My Worker."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

try:
    from gym_gui.core.worker import (
        WorkerAnalyticsManifest,
        ArtifactsMetadata,
        CheckpointMetadata,
    )
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    WorkerAnalyticsManifest = None
    ArtifactsMetadata = None
    CheckpointMetadata = None

if TYPE_CHECKING:
    from my_worker.config import MyWorkerConfig


def write_analytics_manifest(
    config: "MyWorkerConfig",
    *,
    notes: Optional[str] = None,
    tensorboard_dir: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
) -> Path:
    """Build and write analytics manifest for a training run.

    Args:
        config: Worker configuration
        notes: Optional notes about the run
        tensorboard_dir: Optional TensorBoard directory (relative to run dir)
        checkpoints_dir: Optional checkpoints directory (relative to run dir)

    Returns:
        Path to the written manifest file
    """
    if not _HAS_GYM_GUI:
        raise ImportError(
            "gym_gui.core.worker not available. "
            "Cannot generate standardized analytics manifest."
        )

    # Build checkpoint metadata
    checkpoints = CheckpointMetadata(
        directory=checkpoints_dir or "checkpoints",
        format="pytorch",  # or "tensorflow", "jax", etc.
    )

    # Build artifacts metadata
    artifacts = ArtifactsMetadata(
        tensorboard=tensorboard_dir,
        wandb=None,
        checkpoints=checkpoints,
        logs_dir="logs",
        videos_dir=None,
    )

    # Build worker-specific metadata
    metadata = {
        "worker_type": "myworker",
        "env_id": config.env_id,
        "algorithm": config.algorithm,
        "total_steps": config.total_steps,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
    }

    if notes:
        metadata["notes"] = notes

    # Determine manifest path
    manifest_path = Path.cwd() / f"{config.run_id}_analytics.json"

    # Create manifest
    manifest = WorkerAnalyticsManifest(
        run_id=config.run_id,
        worker_type="myworker",
        artifacts=artifacts,
        metadata=metadata,
    )

    # Save manifest
    manifest.save(manifest_path)
    return manifest_path


__all__ = ["write_analytics_manifest"]
```

---

### Step 4: Add Worker Metadata

**File**: `my_worker/__init__.py`

```python
"""My Worker - MOSAIC integration."""

from .config import MyWorkerConfig, load_worker_config
from .runtime import MyWorkerRuntime

__version__ = "1.0.0"


def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="My Worker",
        version=__version__,
        description="My awesome RL worker",
        author="Your Name",
        homepage="https://github.com/your/repo",
        upstream_library="mylib",
        upstream_version="1.0.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="myworker",
        supported_paradigms=(
            "sequential",  # Single-agent training
        ),
        env_families=(
            "gymnasium",
            "atari",
        ),
        action_spaces=("discrete", "continuous"),
        observation_spaces=("vector", "image"),
        max_agents=1,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=True,
        supports_pause_resume=False,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=512,
    )

    return metadata, capabilities


__all__ = [
    "__version__",
    "MyWorkerConfig",
    "load_worker_config",
    "MyWorkerRuntime",
    "get_worker_metadata",
]
```

---

### Step 5: Register Entry Point

**File**: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-worker"
version = "1.0.0"
description = "My RL Worker for MOSAIC"
requires-python = ">=3.10"
license = {text = "MIT"}

[project.scripts]
my-worker = "my_worker.cli:main"

# CRITICAL: Register worker for discovery
[project.entry-points."mosaic.workers"]
myworker = "my_worker:get_worker_metadata"

[tool.setuptools.packages.find]
where = ["."]
include = ["my_worker*"]
```

**Install in editable mode**:

```bash
cd 3rd_party/my_worker
pip install -e .
```

---

## Testing Your Worker

### Create Test Suite

**File**: `tests/test_my_worker_standardization.py`

```python
"""Tests for My Worker standardization."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from my_worker.config import MyWorkerConfig


class TestConfigCompliance:
    """Test config implements WorkerConfig protocol."""

    def test_config_has_required_fields(self):
        """Config must have run_id and seed."""
        config = MyWorkerConfig(run_id="test", seed=42)
        assert hasattr(config, "run_id")
        assert hasattr(config, "seed")
        assert config.run_id == "test"
        assert config.seed == 42

    def test_config_implements_protocol(self):
        """Config must implement WorkerConfig protocol."""
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
        except ImportError:
            pytest.skip("gym_gui not available")

        config = MyWorkerConfig(run_id="test")
        assert isinstance(config, WorkerConfigProtocol)


class TestWorkerMetadata:
    """Test worker metadata."""

    def test_get_worker_metadata_exists(self):
        """Worker must export get_worker_metadata()."""
        from my_worker import get_worker_metadata
        assert callable(get_worker_metadata)

    def test_get_worker_metadata_returns_tuple(self):
        """get_worker_metadata() must return (WorkerMetadata, WorkerCapabilities)."""
        try:
            from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from my_worker import get_worker_metadata

        result = get_worker_metadata()
        assert isinstance(result, tuple)
        assert len(result) == 2

        metadata, capabilities = result
        assert isinstance(metadata, WorkerMetadata)
        assert isinstance(capabilities, WorkerCapabilities)


class TestEntryPointDiscovery:
    """Test worker is discoverable."""

    def test_entry_point_registered(self):
        """Worker must be registered in mosaic.workers group."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points

        if sys.version_info >= (3, 10):
            eps = entry_points(group="mosaic.workers")
        else:
            eps = entry_points().get("mosaic.workers", [])

        my_eps = [ep for ep in eps if ep.name == "myworker"]
        assert len(my_eps) > 0, "myworker entry point not found"
```

### Run Tests

```bash
pytest tests/test_my_worker_standardization.py -v
```

---

## Best Practices

### 1. Configuration Design

✅ **DO**:
- Use `@dataclass` for automatic methods
- Make `run_id` and `seed` required protocol fields
- Add validation in `__post_init__()`
- Support both dict and JSON serialization
- Handle nested config formats from GUI

❌ **DON'T**:
- Use `frozen=True` if you need `__post_init__()`
- Hard-code default values that vary by environment
- Raise errors for missing optional fields

### 2. Runtime Design

✅ **DO**:
- Return `Dict[str, Any]` from `run()`
- Emit all lifecycle events (`run_started`, `run_completed`, `run_failed`)
- Handle exceptions and emit `run_failed` before re-raising
- Generate analytics manifest on success
- Use graceful fallbacks if `gym_gui` unavailable

❌ **DON'T**:
- Return custom dataclasses (breaks serialization)
- Skip lifecycle events
- Swallow exceptions silently
- Assume `gym_gui` is always available

### 3. Telemetry

✅ **DO**:
- Emit `heartbeat` events during long training loops
- Include meaningful metrics in events
- Use structured data (dicts) in events
- Emit to stdout as JSONL

❌ **DON'T**:
- Emit events too frequently (< 1 second intervals)
- Include sensitive data (API keys, passwords)
- Use unstructured strings

### 4. Testing

✅ **DO**:
- Test protocol compliance
- Test entry point registration
- Test config validation
- Test dict serialization roundtrip
- Test with and without `gym_gui`

❌ **DON'T**:
- Skip tests because "it works on my machine"
- Test only happy paths
- Assume dependencies are installed

---

## Common Pitfalls

### Pitfall 1: Frozen Dataclass with `__post_init__`

❌ **WRONG**:
```python
@dataclass(frozen=True)  # frozen=True prevents __post_init__ modifications
class MyConfig:
    run_id: str

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("run_id required")  # This works
        self.validated = True  # ERROR: can't modify frozen dataclass
```

✅ **CORRECT**:
```python
@dataclass  # Don't use frozen=True if you need validation
class MyConfig:
    run_id: str

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("run_id required")
```

### Pitfall 2: Assuming gym_gui is Available

❌ **WRONG**:
```python
from gym_gui.core.worker import TelemetryEmitter  # ImportError if not installed

class MyRuntime:
    def __init__(self, config):
        self.emitter = TelemetryEmitter(run_id=config.run_id)  # Crashes
```

✅ **CORRECT**:
```python
try:
    from gym_gui.core.worker import TelemetryEmitter
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    TelemetryEmitter = None

class MyRuntime:
    def __init__(self, config):
        if _HAS_GYM_GUI:
            self.emitter = TelemetryEmitter(run_id=config.run_id)
        else:
            self.emitter = None
```

### Pitfall 3: Returning Custom Objects from run()

❌ **WRONG**:
```python
@dataclass
class RunSummary:
    status: str
    reward: float

def run(self) -> RunSummary:
    return RunSummary(status="completed", reward=100.0)  # Not JSON-serializable
```

✅ **CORRECT**:
```python
def run(self) -> Dict[str, Any]:
    return {"status": "completed", "reward": 100.0}  # JSON-serializable
```

### Pitfall 4: Forgetting Entry Point

❌ **WRONG**:
```toml
# pyproject.toml - No entry point!
[project]
name = "my-worker"
```

✅ **CORRECT**:
```toml
[project]
name = "my-worker"

[project.entry-points."mosaic.workers"]
myworker = "my_worker:get_worker_metadata"  # Required for discovery!
```

### Pitfall 5: Not Installing Package

❌ **WRONG**:
```bash
# Just running tests without installing
pytest tests/
# Entry point not registered!
```

✅ **CORRECT**:
```bash
# Install in editable mode first
pip install -e .
# Now entry point is registered
pytest tests/
```

---

## Reference Examples

### CleanRL Worker
Location: `3rd_party/cleanrl_worker/`

**Good for**: Simple single-file algorithms, minimal dependencies

### Ray Worker
Location: `3rd_party/ray_worker/`

**Good for**: Multi-agent, distributed training, complex configs

### BARLOG Worker
Location: `3rd_party/barlog_worker/`

**Good for**: LLM-based agents, text environments, VLM support

### XuanCe Worker
Location: `3rd_party/xuance_worker/`

**Good for**: Multi-backend (PyTorch/TensorFlow), 46+ algorithms, multi-agent

---

## Additional Resources

- **MOSAIC Architecture**: `docs/Development_Progress/1.0_DAY_50/TASK_1/MOSAIC_WORKER_ARCHITECTURE.md`
- **Migration Guide**: `docs/workers/MIGRATION_GUIDE.md`
- **Protocol Definitions**: `gym_gui/core/worker/protocols.py`
- **Telemetry Spec**: `gym_gui/core/worker/telemetry.py`
- **Analytics Spec**: `gym_gui/core/worker/analytics.py`

---

## Support

For questions or issues:
1. Check existing worker implementations in `3rd_party/`
2. Read the migration guide for before/after examples
3. Open an issue on GitHub
4. Contact the MOSAIC team

---

**Last Updated**: 2024-12-30
**Maintained by**: MOSAIC Team
