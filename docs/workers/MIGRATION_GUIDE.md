# MOSAIC Worker Migration Guide

**Version**: 1.0
**Last Updated**: 2024-12-30
**Status**: Production Ready

This guide helps you migrate existing workers to the MOSAIC standardized worker architecture.

---

## Table of Contents

1. [Overview](#overview)
2. [Migration Checklist](#migration-checklist)
3. [Before & After Examples](#before--after-examples)
4. [Step-by-Step Migration](#step-by-step-migration)
5. [Backwards Compatibility](#backwards-compatibility)
6. [Testing After Migration](#testing-after-migration)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Migrate?

**Benefits of Migration**:
- ✅ **Automatic Discovery**: Workers appear in GUI automatically
- ✅ **Unified Telemetry**: Standardized lifecycle events and logging
- ✅ **Analytics Integration**: Automatic manifest generation
- ✅ **Type Safety**: Protocol-based interfaces with static type checking
- ✅ **Better Testing**: Standard test patterns and compliance checks

**What Changes**:
- Config: Add `__post_init__()` validation and protocol check
- Runtime: Emit lifecycle events, return `Dict[str, Any]`
- Analytics: Generate `WorkerAnalyticsManifest`
- Package: Add `get_worker_metadata()` function
- Build: Add entry point registration

**What Stays the Same**:
- Your training logic
- Environment integration
- Hyperparameters
- Model architectures
- Existing config files (backwards compatible!)

### Migration Difficulty

| Worker Complexity | Estimated Time | Notes |
|-------------------|----------------|-------|
| **Simple** (CleanRL-like) | 2-3 hours | Single-file algorithms, minimal config |
| **Medium** (BARLOG-like) | 4-6 hours | Custom telemetry, specialized envs |
| **Complex** (Ray-like) | 6-8 hours | Nested configs, multiple paradigms |

---

## Migration Checklist

Use this checklist to track your migration progress:

### Phase 1: Configuration

- [ ] Add `__post_init__()` method for validation
- [ ] Add protocol compliance assertion
- [ ] Ensure `run_id` and `seed` fields exist
- [ ] Implement `to_dict()` method
- [ ] Implement `from_dict()` class method
- [ ] Create `load_worker_config()` function
- [ ] Test with both direct and nested config formats

### Phase 2: Runtime

- [ ] Import `TelemetryEmitter` with graceful fallback
- [ ] Create `_lifecycle_emitter` in `__init__()`
- [ ] Update `run()` return type to `Dict[str, Any]`
- [ ] Emit `run_started` event
- [ ] Emit `heartbeat` events during training
- [ ] Emit `run_completed` event on success
- [ ] Emit `run_failed` event on error
- [ ] Generate analytics manifest

### Phase 3: Analytics

- [ ] Create `analytics.py` module
- [ ] Implement `write_analytics_manifest()` function
- [ ] Use `WorkerAnalyticsManifest` class
- [ ] Include worker-specific metadata
- [ ] Handle missing `gym_gui` gracefully

### Phase 4: Discovery

- [ ] Add `get_worker_metadata()` to `__init__.py`
- [ ] Return `WorkerMetadata` instance
- [ ] Return `WorkerCapabilities` instance
- [ ] Declare supported paradigms
- [ ] Declare environment families
- [ ] Export `get_worker_metadata` in `__all__`

### Phase 5: Entry Point

- [ ] Add `[project.entry-points."mosaic.workers"]` section to `pyproject.toml`
- [ ] Register entry point: `<name> = "<package>:get_worker_metadata"`
- [ ] Run `pip install -e .` to register

### Phase 6: Testing

- [ ] Create `test_*_standardization.py`
- [ ] Test config protocol compliance
- [ ] Test worker metadata
- [ ] Test entry point discovery
- [ ] Test analytics standardization
- [ ] Test config loading (nested/direct)
- [ ] Run all tests: `pytest tests/ -v`
- [ ] Verify 100% test pass rate

---

## Before & After Examples

### Example 1: CleanRL Worker Config

#### BEFORE (Old Config)

```python
# cleanrl_worker/config.py
from dataclasses import dataclass

@dataclass(frozen=True)  # ❌ frozen prevents __post_init__
class WorkerConfig:  # ❌ Naming conflict with protocol
    run_id: str
    algo: str
    env_id: str
    total_timesteps: int = 1_000_000
    seed: int = 1

    # ❌ No validation
    # ❌ No protocol compliance check
```

#### AFTER (Standardized Config)

```python
# cleanrl_worker/config.py
from dataclasses import dataclass
from typing import Any, Dict

@dataclass  # ✅ Not frozen (allows __post_init__)
class CleanRLWorkerConfig:  # ✅ Renamed to avoid collision
    run_id: str
    algo: str
    env_id: str
    total_timesteps: int = 1_000_000
    seed: int | None = 1

    def __post_init__(self) -> None:  # ✅ Added validation
        """Validate configuration and assert protocol compliance."""
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.algo:
            raise ValueError("algo is required")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be >= 1")

        # ✅ Protocol compliance check
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
            assert isinstance(self, WorkerConfigProtocol)
        except ImportError:
            pass  # gym_gui not available, skip check

    def to_dict(self) -> Dict[str, Any]:  # ✅ Added serialization
        return {
            "run_id": self.run_id,
            "algo": self.algo,
            "env_id": self.env_id,
            "total_timesteps": self.total_timesteps,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CleanRLWorkerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ✅ Added config loader
def load_worker_config(config_path: str) -> CleanRLWorkerConfig:
    """Load worker configuration from JSON file."""
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
        config_data = raw_data

    return CleanRLWorkerConfig.from_dict(config_data)
```

### Example 2: BARLOG Worker Runtime

#### BEFORE (Old Runtime)

```python
# barlog_worker/runtime.py
class BarlogWorkerRuntime:
    def __init__(self, config):
        self.config = config
        # ❌ No standardized telemetry

    def run(self) -> RuntimeSummary:  # ❌ Custom return type
        # ❌ No lifecycle events
        result = self._train()

        # ❌ No analytics manifest
        return RuntimeSummary(
            status="completed",
            episodes=result.episodes,
        )
```

#### AFTER (Standardized Runtime)

```python
# barlog_worker/runtime.py
from typing import Dict, Any

# ✅ Import standardized telemetry
try:
    from gym_gui.core.worker import TelemetryEmitter as StandardTelemetryEmitter
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    StandardTelemetryEmitter = None

class BarlogWorkerRuntime:
    def __init__(self, config):
        self.config = config

        # ✅ Create lifecycle telemetry emitter
        if _HAS_GYM_GUI:
            self._lifecycle_emitter = StandardTelemetryEmitter(run_id=config.run_id)
        else:
            self._lifecycle_emitter = None

    def run(self) -> Dict[str, Any]:  # ✅ Standardized return type
        # ✅ Emit run_started lifecycle event
        if self._lifecycle_emitter:
            self._lifecycle_emitter.run_started({
                "env_name": self.config.env_name,
                "task": self.config.task,
                "agent_type": self.config.agent_type,
            })

        try:
            result = self._train()

            # ✅ Generate analytics manifest
            manifest_path = write_analytics_manifest(
                self.config,
                notes=f"BARLOG {self.config.agent_type} on {self.config.task}",
            )

            summary = {
                "status": "completed",
                "episodes": result.episodes,
                "analytics_manifest": str(manifest_path),
            }

            # ✅ Emit run_completed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(summary)

            return summary

        except Exception as e:
            error_summary = {
                "status": "failed",
                "error": str(e),
            }

            # ✅ Emit run_failed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_failed(error_summary)

            raise
```

### Example 3: XuanCe Worker Discovery

#### BEFORE (No Discovery)

```python
# xuance_worker/__init__.py
from .config import XuanCeWorkerConfig
from .runtime import XuanCeWorkerRuntime

__version__ = "0.1.0"

__all__ = [
    "XuanCeWorkerConfig",
    "XuanCeWorkerRuntime",
]

# ❌ No get_worker_metadata()
# ❌ Worker not discoverable
```

#### AFTER (With Discovery)

```python
# xuance_worker/__init__.py
from .config import XuanCeWorkerConfig
from .runtime import XuanCeWorkerRuntime

__version__ = "0.1.0"


# ✅ Added get_worker_metadata()
def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery."""
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="XuanCe Worker",
        version=__version__,
        description="Comprehensive deep RL library with 46+ algorithms",
        author="MOSAIC Team",
        homepage="https://github.com/agi-brain/xuance",
        upstream_library="xuance",
        upstream_version="2.0.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="xuance",
        supported_paradigms=("sequential", "parameter_sharing", "independent"),
        env_families=("gymnasium", "atari", "mujoco", "pettingzoo", "smac"),
        action_spaces=("discrete", "continuous", "multi_discrete"),
        observation_spaces=("vector", "image", "dict"),
        max_agents=100,
        supports_checkpointing=True,
    )

    return metadata, capabilities


__all__ = [
    "XuanCeWorkerConfig",
    "XuanCeWorkerRuntime",
    "get_worker_metadata",  # ✅ Exported
]
```

**pyproject.toml BEFORE**:
```toml
[project]
name = "mosaic-xuance-worker"
version = "0.1.0"

[project.scripts]
xuance-worker = "xuance_worker.cli:main"

# ❌ No entry point!
```

**pyproject.toml AFTER**:
```toml
[project]
name = "mosaic-xuance-worker"
version = "0.1.0"

[project.scripts]
xuance-worker = "xuance_worker.cli:main"

# ✅ Added entry point
[project.entry-points."mosaic.workers"]
xuance = "xuance_worker:get_worker_metadata"
```

---

## Step-by-Step Migration

### Step 1: Update Configuration

**Time**: 30-45 minutes

1. **Remove `frozen=True`** if you need `__post_init__()`:
   ```python
   @dataclass  # Remove frozen=True
   class MyWorkerConfig:
       ...
   ```

2. **Add validation in `__post_init__()`**:
   ```python
   def __post_init__(self) -> None:
       if not self.run_id:
           raise ValueError("run_id is required")

       # Protocol compliance check
       try:
           from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
           assert isinstance(self, WorkerConfigProtocol)
       except ImportError:
           pass
   ```

3. **Ensure `to_dict()` and `from_dict()` exist**:
   ```python
   def to_dict(self) -> Dict[str, Any]:
       return {field.name: getattr(self, field.name) for field in fields(self)}

   @classmethod
   def from_dict(cls, data: Dict[str, Any]) -> "MyWorkerConfig":
       return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
   ```

4. **Add config loader**:
   ```python
   def load_worker_config(config_path: str) -> MyWorkerConfig:
       # ... (see example above)
   ```

5. **Test**:
   ```bash
   python -c "from my_worker.config import MyWorkerConfig; c = MyWorkerConfig(run_id='test'); print(c.to_dict())"
   ```

### Step 2: Update Runtime

**Time**: 45-60 minutes

1. **Import telemetry with fallback**:
   ```python
   try:
       from gym_gui.core.worker import TelemetryEmitter
       _HAS_GYM_GUI = True
   except ImportError:
       _HAS_GYM_GUI = False
       TelemetryEmitter = None
   ```

2. **Create emitter in `__init__()`**:
   ```python
   def __init__(self, config):
       self.config = config
       if _HAS_GYM_GUI:
           self._emitter = TelemetryEmitter(run_id=config.run_id)
       else:
           self._emitter = None
   ```

3. **Change return type of `run()`**:
   ```python
   # Before
   def run(self) -> RuntimeSummary:
       ...

   # After
   def run(self) -> Dict[str, Any]:
       ...
   ```

4. **Emit lifecycle events**:
   ```python
   def run(self) -> Dict[str, Any]:
       # START
       if self._emitter:
           self._emitter.run_started({"env_id": self.config.env_id})

       try:
           result = self._train()

           summary = {"status": "completed", "result": result}

           # SUCCESS
           if self._emitter:
               self._emitter.run_completed(summary)

           return summary

       except Exception as e:
           error_summary = {"status": "failed", "error": str(e)}

           # FAILURE
           if self._emitter:
               self._emitter.run_failed(error_summary)

           raise
   ```

5. **Add heartbeat in training loop**:
   ```python
   def _train(self):
       for step in range(self.config.total_steps):
           # ... training logic ...

           if step % 1000 == 0 and self._emitter:
               self._emitter.heartbeat({"step": step, "reward": reward})
   ```

### Step 3: Create Analytics Module

**Time**: 20-30 minutes

1. **Create `analytics.py`**:
   ```python
   """Analytics manifest generation."""

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

   if TYPE_CHECKING:
       from my_worker.config import MyWorkerConfig


   def write_analytics_manifest(
       config: "MyWorkerConfig",
       *,
       notes: Optional[str] = None,
   ) -> Path:
       if not _HAS_GYM_GUI:
           raise ImportError("gym_gui.core.worker not available")

       checkpoints = CheckpointMetadata(directory="checkpoints", format="pytorch")
       artifacts = ArtifactsMetadata(checkpoints=checkpoints, logs_dir="logs")

       metadata = {
           "worker_type": "myworker",
           "env_id": config.env_id,
           "algorithm": config.algorithm,
       }
       if notes:
           metadata["notes"] = notes

       manifest_path = Path.cwd() / f"{config.run_id}_analytics.json"

       manifest = WorkerAnalyticsManifest(
           run_id=config.run_id,
           worker_type="myworker",
           artifacts=artifacts,
           metadata=metadata,
       )

       manifest.save(manifest_path)
       return manifest_path
   ```

2. **Call from runtime**:
   ```python
   from .analytics import write_analytics_manifest

   def run(self) -> Dict[str, Any]:
       # ... training ...

       manifest_path = write_analytics_manifest(self.config)

       return {
           "status": "completed",
           "analytics_manifest": str(manifest_path),
       }
   ```

### Step 4: Add Worker Metadata

**Time**: 15-20 minutes

1. **Add to `__init__.py`**:
   ```python
   def get_worker_metadata() -> tuple:
       from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

       metadata = WorkerMetadata(
           name="My Worker",
           version="1.0.0",
           description="My awesome worker",
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
       )

       return metadata, capabilities
   ```

2. **Export in `__all__`**:
   ```python
   __all__ = [
       "MyWorkerConfig",
       "MyWorkerRuntime",
       "get_worker_metadata",  # Add this
   ]
   ```

### Step 5: Register Entry Point

**Time**: 5 minutes

1. **Edit `pyproject.toml`**:
   ```toml
   [project.entry-points."mosaic.workers"]
   myworker = "my_worker:get_worker_metadata"
   ```

2. **Install package**:
   ```bash
   pip install -e .
   ```

3. **Verify registration**:
   ```python
   from importlib.metadata import entry_points
   eps = entry_points(group="mosaic.workers")
   print([ep.name for ep in eps])  # Should include 'myworker'
   ```

### Step 6: Create Tests

**Time**: 30-45 minutes

See **Testing After Migration** section below.

---

## Backwards Compatibility

### Config File Compatibility

The standardized config loader supports **both** old and new formats:

**Old Direct Format** (still works):
```json
{
  "run_id": "test_run",
  "env_id": "CartPole-v1",
  "seed": 42
}
```

**New Nested Format** (from GUI):
```json
{
  "metadata": {
    "worker": {
      "config": {
        "run_id": "test_run",
        "env_id": "CartPole-v1",
        "seed": 42
      }
    }
  }
}
```

Both are handled by `load_worker_config()` transparently.

### API Compatibility

**Return Types**:
- Old: Custom dataclasses
- New: `Dict[str, Any]` (JSON-serializable)

If you have code that depends on custom return types:

```python
# Before
summary = runtime.run()
print(summary.final_reward)  # Attribute access

# After
summary = runtime.run()
print(summary["final_reward"])  # Dict access
```

**Migration Strategy**:
1. Update callers to use dict access
2. OR: Add a helper to convert dict → dataclass

```python
@dataclass
class RuntimeSummary:
    status: str
    final_reward: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeSummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

# Usage
summary_dict = runtime.run()
summary = RuntimeSummary.from_dict(summary_dict)  # Backwards compat
```

---

## Testing After Migration

### Create Test Suite

**File**: `tests/test_my_worker_standardization.py`

```python
"""Tests for My Worker standardization."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from my_worker.config import MyWorkerConfig, load_worker_config


class TestConfigCompliance:
    """Test config implements WorkerConfig protocol."""

    def test_config_has_required_fields(self):
        config = MyWorkerConfig(run_id="test", seed=42)
        assert hasattr(config, "run_id")
        assert hasattr(config, "seed")

    def test_config_implements_protocol(self):
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
        except ImportError:
            pytest.skip("gym_gui not available")

        config = MyWorkerConfig(run_id="test")
        assert isinstance(config, WorkerConfigProtocol)

    def test_config_to_dict_from_dict_roundtrip(self):
        original = MyWorkerConfig(run_id="test", seed=42)
        config_dict = original.to_dict()
        restored = MyWorkerConfig.from_dict(config_dict)
        assert restored.run_id == original.run_id
        assert restored.seed == original.seed


class TestWorkerMetadata:
    """Test worker metadata."""

    def test_get_worker_metadata_exists(self):
        from my_worker import get_worker_metadata
        assert callable(get_worker_metadata)

    def test_get_worker_metadata_returns_tuple(self):
        try:
            from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from my_worker import get_worker_metadata
        result = get_worker_metadata()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestEntryPointDiscovery:
    """Test entry point registration."""

    def test_entry_point_registered(self):
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points

        if sys.version_info >= (3, 10):
            eps = entry_points(group="mosaic.workers")
        else:
            eps = entry_points().get("mosaic.workers", [])

        my_eps = [ep for ep in eps if ep.name == "myworker"]
        assert len(my_eps) > 0


class TestConfigLoading:
    """Test config loading."""

    def test_load_worker_config_from_direct_format(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"run_id": "test", "seed": 42}))

        config = load_worker_config(str(config_file))
        assert config.run_id == "test"
        assert config.seed == 42

    def test_load_worker_config_from_nested_format(self, tmp_path):
        config_file = tmp_path / "config.json"
        nested = {
            "metadata": {
                "worker": {
                    "config": {"run_id": "nested_test", "seed": 123}
                }
            }
        }
        config_file.write_text(json.dumps(nested))

        config = load_worker_config(str(config_file))
        assert config.run_id == "nested_test"
        assert config.seed == 123
```

### Run Tests

```bash
# Install package first
pip install -e .

# Run all tests
pytest tests/test_my_worker_standardization.py -v

# Expected output: 100% passing
```

---

## Troubleshooting

### Issue: Entry Point Not Found

**Symptom**:
```python
eps = entry_points(group="mosaic.workers")
# myworker not in list
```

**Solution**:
1. Check `pyproject.toml` has entry point section
2. Run `pip install -e .` to register
3. Restart Python interpreter
4. Verify: `pip show my-worker`

### Issue: Protocol Compliance Failure

**Symptom**:
```
AssertionError: MyWorkerConfig must implement WorkerConfig protocol
```

**Solution**:
1. Ensure `run_id: str` and `seed: int | None` fields exist
2. Implement `to_dict()` method
3. Implement `from_dict()` class method
4. Check field types match protocol

### Issue: Frozen Dataclass Error

**Symptom**:
```
dataclasses.FrozenInstanceError: cannot assign to field
```

**Solution**:
Remove `frozen=True` from `@dataclass`:
```python
# Before
@dataclass(frozen=True)

# After
@dataclass
```

### Issue: Import Error for gym_gui

**Symptom**:
```
ModuleNotFoundError: No module named 'gym_gui'
```

**Solution**:
Use graceful fallback:
```python
try:
    from gym_gui.core.worker import TelemetryEmitter
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    TelemetryEmitter = None

# Then check before using
if _HAS_GYM_GUI:
    emitter = TelemetryEmitter(...)
```

### Issue: Tests Fail After Migration

**Symptom**:
```
TypeError: run() should return Dict[str, Any], not RuntimeSummary
```

**Solution**:
1. Change return type: `def run(self) -> Dict[str, Any]:`
2. Return dict instead of dataclass: `return {"status": "completed", ...}`
3. Update tests to expect dict

---

## Migration Checklist Summary

Print this and check off as you go:

```
Configuration:
☐ Add __post_init__() validation
☐ Add protocol compliance check
☐ Implement to_dict() and from_dict()
☐ Create load_worker_config()
☐ Test with nested/direct configs

Runtime:
☐ Import TelemetryEmitter with fallback
☐ Create lifecycle emitter
☐ Change run() return type to Dict[str, Any]
☐ Emit run_started
☐ Emit heartbeat (in training loop)
☐ Emit run_completed
☐ Emit run_failed

Analytics:
☐ Create analytics.py
☐ Implement write_analytics_manifest()
☐ Call from runtime.run()

Discovery:
☐ Add get_worker_metadata() to __init__.py
☐ Export in __all__
☐ Add entry point to pyproject.toml
☐ Run pip install -e .

Testing:
☐ Create test_*_standardization.py
☐ Test config protocol
☐ Test worker metadata
☐ Test entry point
☐ Test config loading
☐ Run pytest -v
☐ Verify 100% pass rate

Documentation:
☐ Update README with new features
☐ Add migration notes
☐ Update examples
```

---

## Additional Resources

- **Development Guide**: `docs/workers/WORKER_DEVELOPMENT_GUIDE.md`
- **Example Workers**: `3rd_party/{cleanrl,ray,barlog,xuance}_worker/`
- **Protocol Definitions**: `gym_gui/core/worker/protocols.py`

---

**Last Updated**: 2024-12-30
**Maintained by**: MOSAIC Team
