# TASK 2: HDF5 Replay Storage Architecture

## Summary

This task designs a new storage architecture that splits RL replay data between:
- **SQLite**: Scalars (rewards, actions, timestamps) + frame references
- **HDF5**: Large arrays (frames, observations)

The key insight is that SQLite is excellent for queryable scalar data but terrible for large binary blobs. HDF5 is the opposite - excellent for numpy arrays, limited query capability. By using both, we get the best of both worlds.

## Problem Statement

Current bottleneck in `gym_gui/telemetry/sqlite_store.py`:
- Serializing numpy arrays (84×84×4 = 28KB) to JSON strings
- Writing JSON BLOBs to SQLite
- Result: ~100 frames/sec write speed (too slow for Atari games)

## Solution

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEW ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SQLite (var/telemetry/telemetry.sqlite)                        │
│  ├── steps table                                                │
│  │   ├── reward, action, terminated (scalars)                   │
│  │   ├── frame_ref = "h5://run_abc123/frames/1523" (reference)  │
│  │   └── obs_ref = "h5://run_abc123/observations/1523"          │
│  └── episodes table (metadata)                                  │
│                                                                  │
│  HDF5 (var/replay/run_abc123.h5)                                │
│  ├── /frames (100000, 210, 160, 3) uint8                        │
│  ├── /observations (100000, 84, 84, 4) uint8                    │
│  ├── /actions (100000,) int32                                   │
│  ├── /rewards (100000,) float32                                 │
│  └── /episodes/starts, /episodes/lengths                        │
│                                                                  │
│  Replay: Query SQLite → Resolve frame_ref → Read HDF5           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Documentation Files

| File | Description |
|------|-------------|
| [hdf5_replay_storage_design.md](./hdf5_replay_storage_design.md) | Complete architecture design with write/read flows |
| [hdf5_fundamentals.md](./hdf5_fundamentals.md) | HDF5 concepts, API, and RL-specific patterns |
| [sqlite_hdf5_reference_system.md](./sqlite_hdf5_reference_system.md) | How SQLite references HDF5 data (frame_ref) |
| [implementation_roadmap.md](./implementation_roadmap.md) | Phased implementation plan with code examples |

## Key Concepts

### Frame Reference URI

```
h5://{run_id}/{dataset}/{index}

Examples:
  h5://run_abc123/frames/1523
  h5://run_abc123/observations/42
```

Stored in SQLite's `frame_ref` column, resolved to numpy array via `FrameResolver`.

### Performance Improvement

| Metric | Current | Proposed | Improvement |
|--------|---------|----------|-------------|
| Write speed | 100 fps | 2000 fps | **20x** |
| Read (1000 frames) | 5000ms | 50ms | **100x** |
| Storage (100K frames) | 10 GB | 4 GB | **2.5x** |

## New Components

### gym_gui/replay/ package

```python
from gym_gui.replay import ReplayWriter, ReplayReader, FrameResolver

# Write to HDF5
writer = ReplayWriter(run_id, replay_dir)
writer.start()
frame_ref = writer.record_step(frame, obs, action, reward, done)
# frame_ref = "h5://run_id/frames/123"
writer.close()

# Read from HDF5
with ReplayReader(path) as reader:
    episode = reader.get_episode(0)

# Resolve SQLite refs to HDF5 data
with FrameResolver(replay_dir) as resolver:
    frame = resolver.resolve("h5://run_id/frames/123")
```

## Implementation Phases

1. **Phase 1: Foundation** - Add `gym_gui/replay/` package (non-breaking)
2. **Phase 2: Integration** - Wire into `TelemetryDBSink` (non-breaking)
3. **Phase 3: Migration** - Stop storing BLOBs in SQLite (breaking for new runs)
4. **Phase 4: Cleanup** - Drop unused SQLite columns

## Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "h5py>=3.0.0",
]
```

## Related Files

### Current Implementation (to be modified)

- `gym_gui/telemetry/db_sink.py` - Add ReplayWriter integration
- `gym_gui/telemetry/sqlite_store.py` - Remove BLOB storage
- `gym_gui/core/data_model/telemetry_core.py` - Add obs_ref field

### Architecture Documentation

- `gym_gui/fastlane/fastlane_slowlane.md` - Existing fast/slow lane design
- `docs/1.0_DAY_38/TASK_1/` - Ray architecture comparison
