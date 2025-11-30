# SQLite ↔ HDF5 Reference System

## Overview

This document details how SQLite stores references to HDF5 frame data, enabling:
- **Fast queries** on scalar data (rewards, actions, timestamps)
- **Fast storage** of large arrays (frames, observations)
- **Unified replay** by joining both data sources

## Reference URI Format

### Specification

```
h5://{run_id}/{dataset}/{index}
```

| Component | Description | Example |
|-----------|-------------|---------|
| Scheme | Always "h5" | `h5` |
| run_id | Unique run identifier | `run_abc123` |
| dataset | HDF5 dataset path | `frames`, `observations` |
| index | Zero-based step index | `1523` |

### Examples

```
h5://run_abc123/frames/0          # First frame
h5://run_abc123/frames/1523       # Frame at step 1523
h5://run_abc123/observations/42   # Observation at step 42
h5://run_def456/frames/99999      # Different run
```

## SQLite Schema

### Steps Table (Modified)

```sql
CREATE TABLE steps (
    -- Primary key
    episode_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,

    -- Scalar data (stored in SQLite)
    action INTEGER,
    reward REAL NOT NULL,
    terminated INTEGER NOT NULL,
    truncated INTEGER NOT NULL,
    timestamp TEXT NOT NULL,

    -- References to HDF5 (NEW - replaces BLOBs)
    frame_ref TEXT,      -- "h5://run_id/frames/index"
    obs_ref TEXT,        -- "h5://run_id/observations/index"

    -- Small metadata (kept as BLOB)
    info BLOB,           -- JSON-serialized dict (usually small)

    -- REMOVED (now in HDF5):
    -- observation BLOB,      -- Was 28KB+ per step
    -- render_payload BLOB,   -- Was 100KB+ per step

    -- Identifiers
    agent_id TEXT,
    run_id TEXT,
    worker_id TEXT,

    -- Constraints
    PRIMARY KEY (episode_id, step_index)
);

-- Indexes for common queries
CREATE INDEX idx_steps_run_id ON steps(run_id);
CREATE INDEX idx_steps_frame_ref ON steps(frame_ref);
CREATE INDEX idx_steps_timestamp ON steps(timestamp);
```

### Episodes Table (Unchanged)

```sql
CREATE TABLE episodes (
    episode_id TEXT PRIMARY KEY,
    total_reward REAL NOT NULL,
    steps INTEGER NOT NULL,
    terminated INTEGER NOT NULL,
    truncated INTEGER NOT NULL,
    metadata BLOB,        -- JSON (small, kept in SQLite)
    timestamp TEXT NOT NULL,
    agent_id TEXT,
    run_id TEXT,
    worker_id TEXT
);
```

## Data Flow

### Write Path

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           WRITE PATH                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Worker sends step data                                              │
│     {                                                                    │
│       "step_index": 1523,                                               │
│       "action": 2,                                                      │
│       "reward": 1.0,                                                    │
│       "observation": np.ndarray(84, 84, 4),   ← 28KB                    │
│       "frame": np.ndarray(210, 160, 3),       ← 100KB                   │
│       "terminated": false                                               │
│     }                                                                    │
│                              │                                           │
│                              ▼                                           │
│  2. TelemetryDBSink processes event                                     │
│     ┌────────────────────────────────────────────────────────────────┐  │
│     │ # Extract arrays                                                │  │
│     │ frame = payload["frame"]                                        │  │
│     │ observation = payload["observation"]                            │  │
│     │                                                                 │  │
│     │ # Write to HDF5 (background thread)                            │  │
│     │ frame_ref = replay_writer.record_step(                          │  │
│     │     frame=frame,                                                │  │
│     │     observation=observation,                                    │  │
│     │     action=2,                                                   │  │
│     │     reward=1.0,                                                 │  │
│     │     done=False,                                                 │  │
│     │ )                                                               │  │
│     │ # Returns: "h5://run_abc123/frames/1523"                        │  │
│     │                                                                 │  │
│     │ # Create SQLite record with reference (not data)               │  │
│     │ step = StepRecord(                                              │  │
│     │     step_index=1523,                                            │  │
│     │     action=2,                                                   │  │
│     │     reward=1.0,                                                 │  │
│     │     observation=None,       ← NOT STORED                        │  │
│     │     render_payload=None,    ← NOT STORED                        │  │
│     │     frame_ref="h5://run_abc123/frames/1523",  ← REFERENCE       │  │
│     │     obs_ref="h5://run_abc123/observations/1523",                │  │
│     │ )                                                               │  │
│     └────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│              ┌───────────────┴───────────────┐                          │
│              ▼                               ▼                          │
│                                                                          │
│  3a. SQLite receives scalars + refs    3b. HDF5 receives arrays         │
│                                                                          │
│  ┌─────────────────────────────┐       ┌─────────────────────────────┐  │
│  │ INSERT INTO steps           │       │ frames[1523] = frame        │  │
│  │ (step_index, action,        │       │ observations[1523] = obs    │  │
│  │  reward, frame_ref, obs_ref)│       │ actions[1523] = 2           │  │
│  │ VALUES (                    │       │ rewards[1523] = 1.0         │  │
│  │   1523, 2, 1.0,             │       │                             │  │
│  │   'h5://.../frames/1523',   │       │ # Batched, chunked write    │  │
│  │   'h5://.../observations/...'│       │ # ~130KB total              │  │
│  │ )                           │       │                             │  │
│  │                             │       │                             │  │
│  │ # ~100 bytes per row        │       │                             │  │
│  └─────────────────────────────┘       └─────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Read Path (Replay)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           READ PATH (REPLAY)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. User requests episode replay                                        │
│     replay_viewer.load_episode("run_abc123-ep0042")                     │
│                              │                                           │
│                              ▼                                           │
│  2. Query SQLite for episode steps                                      │
│     ┌────────────────────────────────────────────────────────────────┐  │
│     │ SELECT step_index, action, reward, terminated, truncated,       │  │
│     │        frame_ref, obs_ref                                       │  │
│     │ FROM steps                                                      │  │
│     │ WHERE episode_id = 'run_abc123-ep0042'                          │  │
│     │ ORDER BY step_index                                             │  │
│     │                                                                 │  │
│     │ Result (fast - no BLOBs!):                                      │  │
│     │ ┌────────┬────────┬────────┬───────────────────────────────┐    │  │
│     │ │step_idx│ action │ reward │ frame_ref                     │    │  │
│     │ ├────────┼────────┼────────┼───────────────────────────────┤    │  │
│     │ │ 0      │ 1      │ 0.0    │ h5://run_abc123/frames/50000  │    │  │
│     │ │ 1      │ 2      │ 0.0    │ h5://run_abc123/frames/50001  │    │  │
│     │ │ 2      │ 0      │ 1.0    │ h5://run_abc123/frames/50002  │    │  │
│     │ │ ...    │ ...    │ ...    │ ...                           │    │  │
│     │ │ 1522   │ 3      │ 5.0    │ h5://run_abc123/frames/51522  │    │  │
│     │ └────────┴────────┴────────┴───────────────────────────────┘    │  │
│     │                                                                 │  │
│     │ Query time: ~20ms (vs ~5000ms with BLOBs)                      │  │
│     └────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  3. Parse frame references and group by run                             │
│     ┌────────────────────────────────────────────────────────────────┐  │
│     │ refs = [step.frame_ref for step in sqlite_steps]               │  │
│     │                                                                 │  │
│     │ # Parse: "h5://run_abc123/frames/50000" → FrameRef(...)        │  │
│     │ parsed = [FrameRef.parse(ref) for ref in refs]                 │  │
│     │                                                                 │  │
│     │ # Group by run_id for batch read                               │  │
│     │ # All 1523 refs point to same run → one HDF5 file              │  │
│     └────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  4. Batch read from HDF5                                                │
│     ┌────────────────────────────────────────────────────────────────┐  │
│     │ with h5py.File("var/replay/run_abc123.h5", "r") as f:          │  │
│     │                                                                 │  │
│     │     # These indices are contiguous: 50000-51522                │  │
│     │     # HDF5 reads this as ONE I/O operation!                    │  │
│     │     frames = f["frames"][50000:51523]                          │  │
│     │                                                                 │  │
│     │     # Shape: (1523, 210, 160, 3)                               │  │
│     │     # Read time: ~30ms (sequential disk read)                  │  │
│     │                                                                 │  │
│     │     # vs SQLite: 1523 separate BLOB reads + JSON decode        │  │
│     │     # Would take: ~15 seconds                                  │  │
│     └────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  5. Combine and render                                                  │
│     ┌────────────────────────────────────────────────────────────────┐  │
│     │ for i, step in enumerate(sqlite_steps):                        │  │
│     │     frame = frames[i]  # Already numpy array                   │  │
│     │                                                                 │  │
│     │     # Render frame                                             │  │
│     │     display(frame)                                             │  │
│     │                                                                 │  │
│     │     # Show HUD with scalar data from SQLite                    │  │
│     │     show_hud(                                                  │  │
│     │         step=step.step_index,                                  │  │
│     │         action=step.action,                                    │  │
│     │         reward=step.reward,                                    │  │
│     │     )                                                          │  │
│     │                                                                 │  │
│     │     time.sleep(1/60)  # 60 FPS playback                        │  │
│     └────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Reference Resolution

### Single Reference

```python
from gym_gui.replay import FrameResolver

resolver = FrameResolver(Path("var/replay"))

# Resolve single reference
frame_ref = "h5://run_abc123/frames/1523"
frame = resolver.resolve(frame_ref)

# frame is now numpy array shape (210, 160, 3)
```

### Batch Resolution

```python
# More efficient for multiple references
frame_refs = [
    "h5://run_abc123/frames/50000",
    "h5://run_abc123/frames/50001",
    "h5://run_abc123/frames/50002",
    # ... 1000 more refs
]

# Single batch read internally
frames = resolver.resolve_batch(frame_refs)

# frames is list of 1003 numpy arrays
```

### Query + Resolve Pattern

```python
from gym_gui.telemetry.sqlite_store import TelemetrySQLiteStore
from gym_gui.replay import FrameResolver

def load_episode_for_replay(
    store: TelemetrySQLiteStore,
    resolver: FrameResolver,
    episode_id: str,
) -> dict:
    """Load episode with frames from both SQLite and HDF5."""

    # 1. Query scalars from SQLite (fast)
    steps = store.episode_steps(episode_id)

    if not steps:
        return {"episode_id": episode_id, "steps": [], "frames": []}

    # 2. Extract frame references
    frame_refs = [s.frame_ref for s in steps if s.frame_ref]

    # 3. Batch resolve from HDF5 (efficient)
    frames = resolver.resolve_batch(frame_refs) if frame_refs else []

    # 4. Combine into unified structure
    return {
        "episode_id": episode_id,
        "steps": [
            {
                "step_index": step.step_index,
                "action": step.action,
                "reward": step.reward,
                "terminated": step.terminated,
                "truncated": step.truncated,
                "timestamp": step.timestamp,
            }
            for step in steps
        ],
        "frames": frames,  # Numpy arrays from HDF5
        "total_reward": sum(s.reward for s in steps),
        "length": len(steps),
    }
```

## File Layout

### Directory Structure

```
var/
├── telemetry/
│   ├── telemetry.sqlite          # Scalars + references
│   ├── telemetry.sqlite-wal      # WAL file
│   └── telemetry.sqlite-shm      # Shared memory
│
└── replay/
    ├── run_abc123.h5             # HDF5 for run abc123
    │   ├── /frames               # (100000, 210, 160, 3) uint8
    │   ├── /observations         # (100000, 84, 84, 4) uint8
    │   ├── /actions              # (100000,) int32
    │   ├── /rewards              # (100000,) float32
    │   ├── /dones                # (100000,) bool
    │   └── /episodes/
    │       ├── starts            # [0, 1523, 3102, ...]
    │       └── lengths           # [1523, 1579, ...]
    │
    ├── run_def456.h5             # Another run
    └── run_ghi789.h5             # Another run
```

### Correspondence

```
SQLite (steps table)                 HDF5 (run_abc123.h5)
─────────────────────               ──────────────────────

episode_id: "run_abc123-ep0042"
run_id: "run_abc123"                 File: run_abc123.h5

step_index: 1523
frame_ref: "h5://run_abc123/frames/51523"  →  /frames[51523]
obs_ref: "h5://run_abc123/observations/51523"  →  /observations[51523]

action: 2                            →  /actions[51523] = 2
reward: 1.0                          →  /rewards[51523] = 1.0
terminated: 0                        →  /dones[51523] = False
```

## Query Examples

### Get Episode Statistics (SQLite Only)

```sql
-- Fast: no frame data needed
SELECT
    episode_id,
    COUNT(*) as steps,
    SUM(reward) as total_reward,
    MAX(terminated) as terminated,
    MIN(timestamp) as started_at,
    MAX(timestamp) as ended_at
FROM steps
WHERE run_id = 'run_abc123'
GROUP BY episode_id
ORDER BY started_at;
```

### Get Specific Step with Frame

```python
# Query SQLite for step
step = store.get_step(episode_id="run_abc123-ep0042", step_index=100)

# Resolve frame if needed
if step.frame_ref:
    frame = resolver.resolve(step.frame_ref)
    # Now have both scalar data and frame
```

### Get Best Episodes by Reward

```sql
-- Query SQLite for top episodes
SELECT episode_id, total_reward, steps
FROM episodes
WHERE run_id = 'run_abc123'
ORDER BY total_reward DESC
LIMIT 10;
```

```python
# Then load frames for selected episodes from HDF5
for episode_id in top_episode_ids:
    episode = load_episode_for_replay(store, resolver, episode_id)
    save_as_video(episode)
```

### Training: Sample Random Batch

```python
# 1. Query random step indices from SQLite
random_steps = store.sample_steps(run_id="run_abc123", n=32)

# 2. Resolve frames
frame_refs = [s.frame_ref for s in random_steps]
frames = resolver.resolve_batch(frame_refs)

# 3. Create training batch
batch = {
    "observations": np.stack(frames),
    "actions": np.array([s.action for s in random_steps]),
    "rewards": np.array([s.reward for s in random_steps]),
}
```

## Performance Comparison

### Write Performance

| Operation | SQLite + JSON | SQLite + HDF5 Ref | Improvement |
|-----------|--------------|-------------------|-------------|
| Serialize frame | 5ms | 0ms | - |
| Write to DB | 10ms | 0.1ms (ref only) | 100x |
| Write to HDF5 | - | 0.5ms | - |
| **Total per step** | **15ms** | **0.6ms** | **25x** |
| **Steps/second** | **66** | **1600+** | **24x** |

### Read Performance (1000-step episode)

| Operation | SQLite + JSON | SQLite + HDF5 | Improvement |
|-----------|--------------|---------------|-------------|
| Query SQLite | 50ms | 20ms (no BLOBs) | 2.5x |
| Deserialize | 5000ms | 0ms | - |
| Read HDF5 | - | 30ms (batch) | - |
| **Total** | **5050ms** | **50ms** | **100x** |

### Storage Size (100K frames)

| Storage | Size | Notes |
|---------|------|-------|
| SQLite + JSON BLOBs | ~10 GB | Uncompressed JSON |
| SQLite (refs only) | ~50 MB | Just scalars + refs |
| HDF5 (uncompressed) | ~10 GB | Raw arrays |
| HDF5 (lzf compressed) | ~4 GB | 2.5x compression |
| HDF5 (gzip-4) | ~2 GB | 5x compression |
| **Combined (refs + lzf)** | **~4 GB** | **2.5x total** |

## Error Handling

### Missing HDF5 File

```python
def resolve_with_fallback(resolver, frame_ref, fallback=None):
    """Resolve frame with graceful fallback."""
    try:
        frame = resolver.resolve(frame_ref)
        if frame is None:
            _LOGGER.warning(f"Frame not found: {frame_ref}")
            return fallback
        return frame
    except Exception as e:
        _LOGGER.error(f"Failed to resolve {frame_ref}: {e}")
        return fallback
```

### Orphaned References

```python
def validate_references(store, resolver, run_id):
    """Check that all frame_refs in SQLite exist in HDF5."""
    steps = store.steps_for_run(run_id)
    missing = []

    for step in steps:
        if step.frame_ref and not resolver.has_frame(step.frame_ref):
            missing.append(step.frame_ref)

    if missing:
        _LOGGER.warning(f"Found {len(missing)} orphaned frame refs in {run_id}")

    return missing
```

## Summary

The SQLite ↔ HDF5 reference system provides:

| Benefit | Description |
|---------|-------------|
| **Fast queries** | SQLite handles scalar queries without BLOB overhead |
| **Fast writes** | HDF5 handles streaming array writes efficiently |
| **Fast reads** | Batch HDF5 reads for replay (100x faster) |
| **Space efficient** | HDF5 compression (2.5-5x smaller) |
| **Flexible** | Can query scalars without loading frames |
| **Unified** | Single API joins both data sources |
