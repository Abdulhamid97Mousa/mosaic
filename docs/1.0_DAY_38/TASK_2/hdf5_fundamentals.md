# HDF5 Fundamentals for RL Applications

## What is HDF5?

**HDF5** (Hierarchical Data Format version 5) is a file format and library designed for storing large numerical datasets. It is:

- **Self-describing**: Metadata embedded in the file
- **Hierarchical**: Folder-like structure with groups and datasets
- **Efficient**: Native numpy support, chunking, compression
- **Portable**: Cross-platform, cross-language

### Who Uses HDF5?

| Organization | Use Case |
|--------------|----------|
| NASA | Satellite imagery, scientific data |
| CERN | Particle physics experimental data |
| DeepMind | RL replay buffers, experience storage |
| OpenAI | Training data, model checkpoints |
| Most ML Frameworks | Dataset storage (PyTorch, TensorFlow) |

## HDF5 vs Other Formats

### HDF5 vs SQLite

| Aspect | HDF5 | SQLite |
|--------|------|--------|
| **Primary use** | Large arrays | Relational data |
| **Array storage** | Native, zero-copy | JSON/BLOB serialization |
| **Write speed (arrays)** | ~2000 frames/sec | ~100 frames/sec |
| **Query capability** | Limited (by index) | Full SQL |
| **Best for** | Frames, observations | Rewards, actions, metadata |

### HDF5 vs Video Files (MP4/WebM)

| Aspect | HDF5 | Video |
|--------|------|-------|
| **Random access** | O(1) any frame | Decode from keyframe |
| **Lossless** | Yes | No (lossy compression) |
| **Store observations** | Yes (any array) | No |
| **Store actions/rewards** | Yes (same file) | No |
| **Python integration** | h5py (numpy native) | cv2/ffmpeg decode |

### HDF5 vs Raw Binary / NPY Files

| Aspect | HDF5 | Raw Binary / NPY |
|--------|------|------------------|
| **Multiple datasets** | Yes | One per file |
| **Metadata** | Built-in attributes | Separate file |
| **Chunking** | Built-in | Manual |
| **Compression** | Built-in | Manual |
| **Random access** | Built-in | Manual offset math |

## HDF5 File Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HDF5 FILE STRUCTURE                              │
│                  (var/replay/run_abc123.h5)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  /                                    ← Root group                   │
│  │                                                                   │
│  ├── metadata                         ← Group (like a folder)        │
│  │   │                                                               │
│  │   ├── run_id: "abc123"             ← Attribute (small value)      │
│  │   ├── agent_id: "ppo-agent"                                       │
│  │   ├── env_name: "Breakout-v5"                                     │
│  │   ├── created_at: "2024-01-15"                                    │
│  │   └── total_steps: 100000                                         │
│  │                                                                   │
│  ├── frames                           ← Dataset (large array)        │
│  │   shape: (100000, 210, 160, 3)       100K RGB frames              │
│  │   dtype: uint8                                                    │
│  │   chunks: (100, 210, 160, 3)         100 frames per chunk         │
│  │   compression: lzf                   Fast compression             │
│  │                                                                   │
│  ├── observations                     ← Dataset                      │
│  │   shape: (100000, 84, 84, 4)         Preprocessed observations    │
│  │   dtype: uint8                                                    │
│  │   chunks: (100, 84, 84, 4)                                        │
│  │   compression: lzf                                                │
│  │                                                                   │
│  ├── actions                          ← Dataset                      │
│  │   shape: (100000,)                   One action per step          │
│  │   dtype: int32                                                    │
│  │   chunks: (1000,)                    Larger chunks (small data)   │
│  │                                                                   │
│  ├── rewards                          ← Dataset                      │
│  │   shape: (100000,)                                                │
│  │   dtype: float32                                                  │
│  │   chunks: (1000,)                                                 │
│  │                                                                   │
│  ├── dones                            ← Dataset                      │
│  │   shape: (100000,)                                                │
│  │   dtype: bool                                                     │
│  │   chunks: (1000,)                                                 │
│  │                                                                   │
│  └── episodes/                        ← Group for episode info       │
│      ├── starts                       ← Dataset                      │
│      │   data: [0, 1523, 3102, ...]     Episode start indices        │
│      │   dtype: int64                                                │
│      │                                                               │
│      └── lengths                      ← Dataset                      │
│          data: [1523, 1579, ...]        Episode lengths              │
│          dtype: int64                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Groups (Like Folders)

```python
import h5py

with h5py.File("data.h5", "w") as f:
    # Create groups
    metadata = f.create_group("metadata")
    episodes = f.create_group("episodes")

    # Nested groups
    train_data = f.create_group("data/train")
    test_data = f.create_group("data/test")
```

### 2. Datasets (Arrays)

```python
import h5py
import numpy as np

with h5py.File("data.h5", "w") as f:
    # Create dataset from existing array
    frames = np.random.randint(0, 255, (1000, 210, 160, 3), dtype=np.uint8)
    f.create_dataset("frames", data=frames)

    # Create empty dataset with max size
    f.create_dataset(
        "observations",
        shape=(0, 84, 84, 4),           # Start empty
        maxshape=(None, 84, 84, 4),     # Unlimited first dim
        dtype=np.uint8,
    )
```

### 3. Attributes (Metadata)

```python
with h5py.File("data.h5", "w") as f:
    # File-level attributes
    f.attrs["run_id"] = "abc123"
    f.attrs["created_at"] = "2024-01-15T10:30:00"

    # Dataset-level attributes
    frames = f.create_dataset("frames", ...)
    frames.attrs["frame_rate"] = 60
    frames.attrs["pixel_format"] = "RGB"
```

### 4. Chunking (Critical for Performance)

```
WITHOUT CHUNKING (contiguous storage):
┌──────────────────────────────────────────────────────────────────┐
│ Frame 0 │ Frame 1 │ Frame 2 │ Frame 3 │ ... │ Frame N │          │
└──────────────────────────────────────────────────────────────────┘
Problem: Appending requires rewriting the entire dataset!
Problem: Reading frame 50000 requires scanning from start

WITH CHUNKING:
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│  Chunk 0  │ │  Chunk 1  │ │  Chunk 2  │ │  Chunk 3  │ ...
│ Frames    │ │ Frames    │ │ Frames    │ │ Frames    │
│ 0-99      │ │ 100-199   │ │ 200-299   │ │ 300-399   │
└───────────┘ └───────────┘ └───────────┘ └───────────┘
     ↓             ↓             ↓             ↓
Can be read/written independently!
```

```python
# Optimal chunking for RL frames
f.create_dataset(
    "frames",
    shape=(0, 210, 160, 3),
    maxshape=(None, 210, 160, 3),
    chunks=(100, 210, 160, 3),    # 100 frames per chunk
    dtype=np.uint8,
)

# Why 100 frames?
# - Single chunk ~10MB (100 * 210 * 160 * 3 bytes)
# - Fits in memory easily
# - Good balance between I/O efficiency and granularity
# - Can read any 100-frame block in one I/O operation
```

### 5. Compression

```python
# Available compression filters
f.create_dataset(
    "frames",
    shape=(100000, 210, 160, 3),
    dtype=np.uint8,
    chunks=(100, 210, 160, 3),

    # Option A: gzip (best compression, slower)
    compression="gzip",
    compression_opts=4,  # Level 1-9

    # Option B: lzf (fast, moderate compression) - RECOMMENDED
    # compression="lzf",

    # Option C: szip (good for scientific data)
    # compression="szip",
)
```

| Compression | Ratio | Write Speed | Read Speed | Best For |
|-------------|-------|-------------|------------|----------|
| None | 1.0x | Fastest | Fastest | SSDs, temp data |
| lzf | 2-3x | Fast | Fast | **RL replay** |
| gzip-1 | 3-4x | Medium | Medium | Archival |
| gzip-9 | 5-6x | Slow | Medium | Long-term storage |

## Basic Operations

### Writing Data

```python
import h5py
import numpy as np

# Create new file
with h5py.File("replay.h5", "w") as f:
    # Create datasets
    f.create_dataset("frames", data=frames_array)
    f.create_dataset("actions", data=actions_array)

    # Add metadata
    f.attrs["run_id"] = "abc123"

# Append to existing file
with h5py.File("replay.h5", "a") as f:
    frames = f["frames"]
    current_size = frames.shape[0]

    # Resize and append
    frames.resize(current_size + 100, axis=0)
    frames[current_size:current_size + 100] = new_frames
```

### Reading Data

```python
import h5py

with h5py.File("replay.h5", "r") as f:
    # Read single frame (O(1) access)
    frame_100 = f["frames"][100]

    # Read slice
    batch = f["frames"][1000:1032]  # 32 frames

    # Read with fancy indexing
    specific_frames = f["frames"][[10, 50, 100, 500]]

    # Iterate without loading all
    for i in range(0, len(f["frames"]), 32):
        batch = f["frames"][i:i+32]
        process(batch)
```

### Memory-Mapped Access

```python
# HDF5 doesn't load data until accessed
with h5py.File("replay.h5", "r") as f:
    frames = f["frames"]  # NOT in memory yet!

    print(frames.shape)   # (100000, 210, 160, 3) - just metadata
    print(frames.dtype)   # uint8

    # Only this single frame is loaded
    frame = frames[50000]

    # Only 32 frames loaded
    batch = frames[0:32]
```

## RL-Specific Patterns

### Pattern 1: Streaming Writer

```python
class StreamingHDF5Writer:
    """Write RL experiences as they arrive."""

    def __init__(self, path: str, frame_shape: tuple):
        self.file = h5py.File(path, "w")
        self.frame_shape = frame_shape

        # Create resizable datasets
        self.frames = self.file.create_dataset(
            "frames",
            shape=(0, *frame_shape),
            maxshape=(None, *frame_shape),
            chunks=(100, *frame_shape),
            dtype=np.uint8,
            compression="lzf",
        )

        self.rewards = self.file.create_dataset(
            "rewards",
            shape=(0,),
            maxshape=(None,),
            chunks=(1000,),
            dtype=np.float32,
        )

        self._buffer_frames = []
        self._buffer_rewards = []
        self._batch_size = 100

    def add(self, frame: np.ndarray, reward: float):
        self._buffer_frames.append(frame)
        self._buffer_rewards.append(reward)

        if len(self._buffer_frames) >= self._batch_size:
            self._flush()

    def _flush(self):
        if not self._buffer_frames:
            return

        n = len(self._buffer_frames)
        current = self.frames.shape[0]

        # Resize
        self.frames.resize(current + n, axis=0)
        self.rewards.resize(current + n, axis=0)

        # Write
        self.frames[current:current + n] = np.array(self._buffer_frames)
        self.rewards[current:current + n] = np.array(self._buffer_rewards)

        # Clear buffers
        self._buffer_frames.clear()
        self._buffer_rewards.clear()

    def close(self):
        self._flush()
        self.file.close()
```

### Pattern 2: Episode-Based Reader

```python
class EpisodeReader:
    """Read episodes from HDF5 file."""

    def __init__(self, path: str):
        self.file = h5py.File(path, "r")
        self.episode_starts = self.file["episodes/starts"][:]
        self.episode_lengths = self.file["episodes/lengths"][:]

    def __len__(self):
        return len(self.episode_starts)

    def __getitem__(self, episode_idx: int) -> dict:
        start = self.episode_starts[episode_idx]
        length = self.episode_lengths[episode_idx]
        end = start + length

        return {
            "frames": self.file["frames"][start:end],
            "observations": self.file["observations"][start:end],
            "actions": self.file["actions"][start:end],
            "rewards": self.file["rewards"][start:end],
            "dones": self.file["dones"][start:end],
        }

    def iter_batches(self, batch_size: int = 32):
        """Iterate through all steps in batches."""
        total = self.file["frames"].shape[0]

        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            yield {
                "observations": self.file["observations"][i:end],
                "actions": self.file["actions"][i:end],
                "rewards": self.file["rewards"][i:end],
            }
```

### Pattern 3: Random Sampling for Training

```python
class ReplayBuffer:
    """HDF5-backed replay buffer for training."""

    def __init__(self, path: str):
        self.file = h5py.File(path, "r")
        self.size = self.file["observations"].shape[0]

    def sample(self, batch_size: int) -> dict:
        # Random indices
        indices = np.random.randint(0, self.size, size=batch_size)

        # Sort for efficient HDF5 read (contiguous access)
        sorted_indices = np.sort(indices)

        # Read batch
        obs = self.file["observations"][sorted_indices]
        actions = self.file["actions"][sorted_indices]
        rewards = self.file["rewards"][sorted_indices]
        next_obs = self.file["observations"][sorted_indices + 1]
        dones = self.file["dones"][sorted_indices]

        return {
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_obs,
            "dones": dones,
        }
```

## Performance Tips

### 1. Choose Right Chunk Size

```python
# Rule of thumb: chunk ~1-10 MB
# For frames (210, 160, 3) = 100KB each
# Chunk 100 frames = 10MB per chunk - GOOD

# For observations (84, 84, 4) = 28KB each
# Chunk 100 observations = 2.8MB per chunk - GOOD

# For scalars (rewards, actions)
# Chunk 1000-10000 values - small anyway
```

### 2. Use Appropriate Compression

```python
# For RL replay (balance speed + size): lzf
compression="lzf"

# For archival (maximize compression): gzip
compression="gzip", compression_opts=6

# For fastest writes (temp data): none
compression=None
```

### 3. Batch Operations

```python
# BAD: One write per step
for frame in frames:
    dataset.resize(dataset.shape[0] + 1, axis=0)
    dataset[-1] = frame  # Many small I/O operations

# GOOD: Batch writes
batch = []
for frame in frames:
    batch.append(frame)
    if len(batch) >= 100:
        n = len(batch)
        current = dataset.shape[0]
        dataset.resize(current + n, axis=0)
        dataset[current:current + n] = np.array(batch)  # One I/O
        batch.clear()
```

### 4. Sort Indices for Random Access

```python
# BAD: Random order reads
indices = [500, 100, 300, 200, 400]
data = dataset[indices]  # May require 5 I/O operations

# GOOD: Sorted reads
indices = [500, 100, 300, 200, 400]
sorted_indices = np.sort(indices)
data = dataset[sorted_indices]  # Fewer I/O operations
```

## Installation

```bash
# Via pip
pip install h5py

# Via conda (recommended for HPC)
conda install h5py

# With parallel HDF5 (for MPI)
conda install h5py=*=mpi*
```

## Summary

| Feature | HDF5 Capability |
|---------|-----------------|
| Large arrays | Native numpy, zero-copy |
| Streaming writes | Chunked, resizable datasets |
| Compression | Built-in (lzf, gzip, szip) |
| Random access | O(1) by index |
| Memory efficiency | Lazy loading, memory-mapped |
| Metadata | Attributes on files/datasets |
| Organization | Hierarchical groups |
| Python API | h5py (numpy-like) |

**For RL replay storage, HDF5 provides 30x faster writes and 100x faster reads compared to SQLite + JSON serialization.**
