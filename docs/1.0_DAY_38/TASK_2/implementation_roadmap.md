# HDF5 Replay Storage Implementation Roadmap

## Overview

This document outlines the step-by-step implementation plan for integrating HDF5 replay storage into gym_gui while maintaining backward compatibility.

## Implementation Phases

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     IMPLEMENTATION TIMELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Foundation (Non-Breaking)                                     │
│  ├── Create gym_gui/replay/ package                                     │
│  ├── Implement ReplayWriter                                             │
│  ├── Implement ReplayReader                                             │
│  └── Add h5py dependency                                                │
│                                                                          │
│  Phase 2: Integration (Non-Breaking)                                    │
│  ├── Add frame_ref field to StepRecord                                  │
│  ├── Wire ReplayWriter into TelemetryDBSink                            │
│  ├── Create FrameResolver                                               │
│  └── Run SQLite + HDF5 in parallel for validation                       │
│                                                                          │
│  Phase 3: Migration (Breaking for new runs)                             │
│  ├── Stop storing observation/render_payload BLOBs                      │
│  ├── Update TelemetryService to resolve refs                            │
│  └── Update replay UI to use FrameResolver                              │
│                                                                          │
│  Phase 4: Cleanup                                                       │
│  ├── Add SQLite migration to drop unused columns                        │
│  ├── Compact existing databases                                         │
│  └── Document new architecture                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Foundation

### 1.1 Create Package Structure

```
gym_gui/
├── replay/
│   ├── __init__.py
│   ├── constants.py
│   ├── replay_store.py
│   └── frame_resolver.py
```

### 1.2 Add Dependency

```toml
# pyproject.toml
[project]
dependencies = [
    # ... existing deps
    "h5py>=3.0.0",
]

# Or requirements/base.txt
h5py>=3.0.0
```

### 1.3 Implement Constants

```python
# gym_gui/replay/constants.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ReplayStorageDefaults:
    """Default configuration for HDF5 replay storage."""

    # Dataset sizes
    max_steps_per_run: int = 10_000_000  # 10M steps max per run

    # Frame dimensions (can be overridden per-run)
    default_frame_shape: tuple = (210, 160, 3)  # Atari RGB
    default_obs_shape: tuple = (84, 84, 4)      # Preprocessed

    # Chunking (affects I/O performance)
    frame_chunk_size: int = 100      # Frames per chunk
    scalar_chunk_size: int = 1000    # Scalars per chunk

    # Compression
    compression: str = "lzf"         # Fast compression
    compression_opts: int | None = None  # lzf doesn't use opts

    # Batching (for background writer)
    write_batch_size: int = 100      # Frames to buffer before write
    write_queue_size: int = 1000     # Max items in write queue

    # Paths
    replay_subdir: str = "replay"    # Under var/


@dataclass(frozen=True)
class FrameRefDefaults:
    """Defaults for frame reference URIs."""

    scheme: str = "h5"
    frame_dataset: str = "frames"
    obs_dataset: str = "observations"


REPLAY_DEFAULTS = ReplayStorageDefaults()
FRAME_REF_DEFAULTS = FrameRefDefaults()

__all__ = [
    "REPLAY_DEFAULTS",
    "FRAME_REF_DEFAULTS",
    "ReplayStorageDefaults",
    "FrameRefDefaults",
]
```

### 1.4 Implement ReplayWriter

```python
# gym_gui/replay/replay_store.py

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from gym_gui.replay.constants import REPLAY_DEFAULTS, FRAME_REF_DEFAULTS

_LOGGER = logging.getLogger(__name__)


class ReplayWriter:
    """Writes frames and observations to HDF5 for replay.

    This class provides:
    - Background thread for non-blocking writes
    - Automatic batching for efficient I/O
    - Chunked, compressed HDF5 storage
    - Episode boundary tracking

    Usage:
        writer = ReplayWriter(run_id, config, replay_dir)
        writer.start()

        for step in episode:
            frame_ref = writer.record_step(
                frame=frame,
                observation=obs,
                action=action,
                reward=reward,
                done=done,
            )
            # frame_ref = "h5://run_id/frames/123"

        writer.mark_episode_end()
        writer.close()
    """

    def __init__(
        self,
        run_id: str,
        replay_dir: Path,
        *,
        frame_shape: tuple | None = None,
        obs_shape: tuple | None = None,
        max_steps: int = REPLAY_DEFAULTS.max_steps_per_run,
        chunk_size: int = REPLAY_DEFAULTS.frame_chunk_size,
        compression: str = REPLAY_DEFAULTS.compression,
        batch_size: int = REPLAY_DEFAULTS.write_batch_size,
        queue_size: int = REPLAY_DEFAULTS.write_queue_size,
    ) -> None:
        self._run_id = run_id
        self._path = Path(replay_dir) / f"{run_id}.h5"
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Configuration
        self._frame_shape = frame_shape
        self._obs_shape = obs_shape
        self._max_steps = max_steps
        self._chunk_size = chunk_size
        self._compression = compression
        self._batch_size = batch_size

        # Background writer
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        # State
        self._file: Optional[h5py.File] = None
        self._step_count = 0
        self._episode_starts: list[int] = [0]
        self._initialized = False

        _LOGGER.debug(
            "ReplayWriter created",
            extra={"run_id": run_id, "path": str(self._path)},
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def step_count(self) -> int:
        return self._step_count

    def start(self) -> None:
        """Initialize HDF5 file and start background writer."""
        if self._initialized:
            return

        self._file = h5py.File(self._path, "w")
        self._worker = threading.Thread(
            target=self._writer_loop,
            name=f"replay-writer-{self._run_id}",
            daemon=True,
        )
        self._worker.start()
        self._initialized = True

        _LOGGER.info(
            "ReplayWriter started",
            extra={"run_id": self._run_id, "path": str(self._path)},
        )

    def record_step(
        self,
        frame: Optional[np.ndarray],
        observation: Optional[np.ndarray],
        action: int,
        reward: float,
        done: bool,
    ) -> str:
        """Queue a step for background writing.

        Args:
            frame: RGB frame array, shape (H, W, 3) or (H, W, 4)
            observation: Preprocessed observation array
            action: Action taken
            reward: Reward received
            done: Whether episode ended

        Returns:
            Frame reference URI: "h5://{run_id}/frames/{index}"
        """
        if not self._initialized:
            raise RuntimeError("ReplayWriter not started. Call start() first.")

        step_index = self._step_count
        frame_ref = self._make_ref(FRAME_REF_DEFAULTS.frame_dataset, step_index)

        # Initialize datasets on first step if shapes not provided
        if step_index == 0:
            if frame is not None and self._frame_shape is None:
                self._frame_shape = frame.shape
            if observation is not None and self._obs_shape is None:
                self._obs_shape = observation.shape

        self._queue.put({
            "type": "step",
            "frame": frame,
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "step_index": step_index,
        })

        self._step_count += 1
        return frame_ref

    def mark_episode_end(self) -> None:
        """Mark current position as episode boundary."""
        self._episode_starts.append(self._step_count)

    def flush(self) -> None:
        """Block until all queued writes complete."""
        self._queue.join()

    def close(self) -> None:
        """Stop writer and finalize HDF5 file."""
        if not self._initialized:
            return

        # Signal stop and wait
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=10.0)

        # Finalize file
        if self._file is not None:
            self._write_episode_index()
            self._write_metadata()
            self._file.close()
            self._file = None

        self._initialized = False

        _LOGGER.info(
            "ReplayWriter closed",
            extra={
                "run_id": self._run_id,
                "total_steps": self._step_count,
                "episodes": len(self._episode_starts) - 1,
            },
        )

    def _make_ref(self, dataset: str, index: int) -> str:
        """Create frame reference URI."""
        return f"{FRAME_REF_DEFAULTS.scheme}://{self._run_id}/{dataset}/{index}"

    def _writer_loop(self) -> None:
        """Background thread that writes to HDF5."""
        batch_frames: list = []
        batch_obs: list = []
        batch_actions: list = []
        batch_rewards: list = []
        batch_dones: list = []

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                # Flush partial batch on timeout
                if batch_frames or batch_actions:
                    self._flush_batch(
                        batch_frames, batch_obs,
                        batch_actions, batch_rewards, batch_dones
                    )
                    batch_frames, batch_obs = [], []
                    batch_actions, batch_rewards, batch_dones = [], [], []
                continue

            try:
                if item["type"] == "step":
                    if item["frame"] is not None:
                        batch_frames.append(item["frame"])
                    if item["observation"] is not None:
                        batch_obs.append(item["observation"])
                    batch_actions.append(item["action"])
                    batch_rewards.append(item["reward"])
                    batch_dones.append(item["done"])

                    # Flush when batch is full
                    if len(batch_actions) >= self._batch_size:
                        self._flush_batch(
                            batch_frames, batch_obs,
                            batch_actions, batch_rewards, batch_dones
                        )
                        batch_frames, batch_obs = [], []
                        batch_actions, batch_rewards, batch_dones = [], [], []
            finally:
                self._queue.task_done()

        # Flush remaining
        if batch_frames or batch_actions:
            self._flush_batch(
                batch_frames, batch_obs,
                batch_actions, batch_rewards, batch_dones
            )

    def _flush_batch(
        self,
        frames: list,
        observations: list,
        actions: list,
        rewards: list,
        dones: list,
    ) -> None:
        """Write a batch to HDF5."""
        if not actions:
            return
        if self._file is None:
            return

        # Ensure datasets exist
        self._ensure_datasets()

        n = len(actions)

        # Write frames if present
        if frames and "frames" in self._file:
            frames_ds = self._file["frames"]
            current = frames_ds.shape[0]
            frames_ds.resize(current + len(frames), axis=0)
            frames_ds[current:current + len(frames)] = np.array(frames)

        # Write observations if present
        if observations and "observations" in self._file:
            obs_ds = self._file["observations"]
            current = obs_ds.shape[0]
            obs_ds.resize(current + len(observations), axis=0)
            obs_ds[current:current + len(observations)] = np.array(observations)

        # Write scalars
        for name, data in [
            ("actions", actions),
            ("rewards", rewards),
            ("dones", dones),
        ]:
            if name in self._file:
                ds = self._file[name]
                current = ds.shape[0]
                ds.resize(current + n, axis=0)
                ds[current:current + n] = np.array(data)

        # Flush to disk
        self._file.flush()

    def _ensure_datasets(self) -> None:
        """Create datasets on first write."""
        if self._file is None:
            return

        # Frames dataset
        if "frames" not in self._file and self._frame_shape is not None:
            self._file.create_dataset(
                "frames",
                shape=(0, *self._frame_shape),
                maxshape=(self._max_steps, *self._frame_shape),
                chunks=(self._chunk_size, *self._frame_shape),
                dtype=np.uint8,
                compression=self._compression,
            )

        # Observations dataset
        if "observations" not in self._file and self._obs_shape is not None:
            self._file.create_dataset(
                "observations",
                shape=(0, *self._obs_shape),
                maxshape=(self._max_steps, *self._obs_shape),
                chunks=(self._chunk_size, *self._obs_shape),
                dtype=np.uint8,
                compression=self._compression,
            )

        # Scalar datasets
        scalar_chunk = self._chunk_size * 10
        for name, dtype in [
            ("actions", np.int32),
            ("rewards", np.float32),
            ("dones", np.bool_),
        ]:
            if name not in self._file:
                self._file.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(self._max_steps,),
                    chunks=(scalar_chunk,),
                    dtype=dtype,
                )

    def _write_episode_index(self) -> None:
        """Write episode boundaries to file."""
        if self._file is None:
            return

        episodes = self._file.create_group("episodes")
        episodes.create_dataset("starts", data=np.array(self._episode_starts[:-1]))

        # Calculate lengths
        lengths = []
        for i in range(len(self._episode_starts) - 1):
            lengths.append(self._episode_starts[i + 1] - self._episode_starts[i])
        if self._step_count > self._episode_starts[-1]:
            lengths.append(self._step_count - self._episode_starts[-1])

        episodes.create_dataset("lengths", data=np.array(lengths))

    def _write_metadata(self) -> None:
        """Write run metadata to file."""
        if self._file is None:
            return

        self._file.attrs["run_id"] = self._run_id
        self._file.attrs["total_steps"] = self._step_count
        self._file.attrs["total_episodes"] = len(self._episode_starts) - 1
        self._file.attrs["created_at"] = datetime.now().isoformat()
        self._file.attrs["version"] = "1.0"


class ReplayReader:
    """Reads replay data from HDF5 for playback or training."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: Optional[h5py.File] = None

    def open(self) -> None:
        self._file = h5py.File(self._path, "r")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> "ReplayReader":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def num_steps(self) -> int:
        if self._file is None:
            return 0
        if "actions" in self._file:
            return self._file["actions"].shape[0]
        return 0

    @property
    def num_episodes(self) -> int:
        if self._file is None:
            return 0
        if "episodes/starts" in self._file:
            return len(self._file["episodes/starts"])
        return 0

    @property
    def metadata(self) -> dict:
        if self._file is None:
            return {}
        return dict(self._file.attrs)

    def get_step(self, index: int) -> dict:
        """Get a single step by index."""
        if self._file is None:
            raise RuntimeError("Reader not open")

        result = {
            "action": self._file["actions"][index],
            "reward": self._file["rewards"][index],
            "done": self._file["dones"][index],
        }

        if "frames" in self._file:
            result["frame"] = self._file["frames"][index]
        if "observations" in self._file:
            result["observation"] = self._file["observations"][index]

        return result

    def get_episode(self, episode_idx: int) -> dict:
        """Get all data for an episode."""
        if self._file is None:
            raise RuntimeError("Reader not open")

        starts = self._file["episodes/starts"][:]
        lengths = self._file["episodes/lengths"][:]

        start = int(starts[episode_idx])
        length = int(lengths[episode_idx])
        end = start + length

        result = {
            "actions": self._file["actions"][start:end],
            "rewards": self._file["rewards"][start:end],
            "dones": self._file["dones"][start:end],
            "episode_index": episode_idx,
            "start_step": start,
            "length": length,
        }

        if "frames" in self._file:
            result["frames"] = self._file["frames"][start:end]
        if "observations" in self._file:
            result["observations"] = self._file["observations"][start:end]

        return result

    def get_frame(self, step_index: int) -> Optional[np.ndarray]:
        """Get a single frame by step index."""
        if self._file is None:
            return None
        if "frames" not in self._file:
            return None
        if step_index >= self._file["frames"].shape[0]:
            return None
        return self._file["frames"][step_index]


__all__ = ["ReplayWriter", "ReplayReader"]
```

### 1.5 Implement FrameResolver

```python
# gym_gui/replay/frame_resolver.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from gym_gui.replay.constants import FRAME_REF_DEFAULTS

_LOGGER = logging.getLogger(__name__)


@dataclass
class FrameRef:
    """Parsed frame reference."""

    run_id: str
    dataset: str
    index: int

    @classmethod
    def parse(cls, ref: str) -> Optional["FrameRef"]:
        """Parse 'h5://run_id/dataset/index' format.

        Examples:
            h5://run_abc123/frames/1523
            h5://run_abc123/observations/42
        """
        pattern = rf"{FRAME_REF_DEFAULTS.scheme}://([^/]+)/([^/]+)/(\d+)"
        match = re.match(pattern, ref)
        if not match:
            return None
        return cls(
            run_id=match.group(1),
            dataset=match.group(2),
            index=int(match.group(3)),
        )

    def to_uri(self) -> str:
        """Convert back to URI string."""
        return f"{FRAME_REF_DEFAULTS.scheme}://{self.run_id}/{self.dataset}/{self.index}"


class FrameResolver:
    """Resolves frame references from SQLite to HDF5 data.

    This class manages HDF5 file handles and provides efficient
    batch resolution of frame references.

    Usage:
        resolver = FrameResolver(replay_dir)

        # Single resolution
        frame = resolver.resolve("h5://run_abc123/frames/1523")

        # Batch resolution (more efficient)
        frames = resolver.resolve_batch([
            "h5://run_abc123/frames/0",
            "h5://run_abc123/frames/1",
            "h5://run_abc123/frames/2",
        ])

        resolver.close()
    """

    def __init__(self, replay_dir: Path) -> None:
        self._replay_dir = Path(replay_dir)
        self._open_files: dict[str, h5py.File] = {}

    def resolve(self, frame_ref: str) -> Optional[np.ndarray]:
        """Resolve a frame reference to actual numpy array.

        Args:
            frame_ref: URI like "h5://run_id/dataset/index"

        Returns:
            Numpy array if found, None otherwise
        """
        ref = FrameRef.parse(frame_ref)
        if ref is None:
            _LOGGER.warning("Invalid frame ref format", extra={"ref": frame_ref})
            return None

        h5_file = self._get_file(ref.run_id)
        if h5_file is None:
            return None

        try:
            dataset = h5_file[ref.dataset]
            if ref.index >= len(dataset):
                _LOGGER.warning(
                    "Frame index out of range",
                    extra={"ref": frame_ref, "max": len(dataset)},
                )
                return None
            return dataset[ref.index]
        except KeyError:
            _LOGGER.warning(
                "Dataset not found",
                extra={"ref": frame_ref, "dataset": ref.dataset},
            )
            return None

    def resolve_batch(
        self,
        frame_refs: list[str],
    ) -> list[Optional[np.ndarray]]:
        """Resolve multiple references efficiently.

        Groups references by run_id and dataset for optimal
        HDF5 batch reads.

        Args:
            frame_refs: List of URI strings

        Returns:
            List of arrays (or None) in same order as input
        """
        if not frame_refs:
            return []

        # Parse all refs
        parsed: list[tuple[int, Optional[FrameRef]]] = []
        for i, ref_str in enumerate(frame_refs):
            parsed.append((i, FrameRef.parse(ref_str)))

        # Group by run_id -> dataset -> (original_idx, h5_idx)
        grouped: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for orig_idx, ref in parsed:
            if ref is None:
                continue
            if ref.run_id not in grouped:
                grouped[ref.run_id] = {}
            if ref.dataset not in grouped[ref.run_id]:
                grouped[ref.run_id][ref.dataset] = []
            grouped[ref.run_id][ref.dataset].append((orig_idx, ref.index))

        # Initialize results
        results: list[Optional[np.ndarray]] = [None] * len(frame_refs)

        # Batch read per run_id/dataset
        for run_id, datasets in grouped.items():
            h5_file = self._get_file(run_id)
            if h5_file is None:
                continue

            for dataset_name, indices in datasets.items():
                try:
                    dataset = h5_file[dataset_name]

                    # Sort by h5 index for efficient read
                    sorted_indices = sorted(indices, key=lambda x: x[1])
                    h5_indices = [idx for _, idx in sorted_indices]

                    # Filter valid indices
                    valid_h5_indices = [i for i in h5_indices if i < len(dataset)]
                    if not valid_h5_indices:
                        continue

                    # Batch read
                    data = dataset[valid_h5_indices]

                    # Map back to results
                    data_idx = 0
                    for orig_idx, h5_idx in sorted_indices:
                        if h5_idx < len(dataset):
                            results[orig_idx] = data[data_idx]
                            data_idx += 1

                except KeyError:
                    continue

        return results

    def has_run(self, run_id: str) -> bool:
        """Check if HDF5 file exists for run."""
        path = self._replay_dir / f"{run_id}.h5"
        return path.exists()

    def get_run_info(self, run_id: str) -> Optional[dict]:
        """Get metadata for a run's HDF5 file."""
        h5_file = self._get_file(run_id)
        if h5_file is None:
            return None

        return {
            "run_id": run_id,
            "total_steps": h5_file.attrs.get("total_steps", 0),
            "total_episodes": h5_file.attrs.get("total_episodes", 0),
            "created_at": h5_file.attrs.get("created_at", ""),
            "has_frames": "frames" in h5_file,
            "has_observations": "observations" in h5_file,
        }

    def _get_file(self, run_id: str) -> Optional[h5py.File]:
        """Get or open HDF5 file for run."""
        if run_id in self._open_files:
            return self._open_files[run_id]

        path = self._replay_dir / f"{run_id}.h5"
        if not path.exists():
            _LOGGER.debug("HDF5 file not found", extra={"path": str(path)})
            return None

        try:
            f = h5py.File(path, "r")
            self._open_files[run_id] = f
            return f
        except Exception as e:
            _LOGGER.error(
                "Failed to open HDF5 file",
                extra={"path": str(path), "error": str(e)},
            )
            return None

    def close(self) -> None:
        """Close all open HDF5 files."""
        for f in self._open_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_files.clear()

    def __enter__(self) -> "FrameResolver":
        return self

    def __exit__(self, *args) -> None:
        self.close()


__all__ = ["FrameRef", "FrameResolver"]
```

### 1.6 Package Init

```python
# gym_gui/replay/__init__.py

"""HDF5-based replay storage for RL experiences.

This package provides efficient storage and retrieval of RL replay data,
splitting storage between SQLite (scalars) and HDF5 (arrays).

Components:
    - ReplayWriter: Writes frames/observations to HDF5
    - ReplayReader: Reads episodes from HDF5
    - FrameResolver: Resolves SQLite frame_ref to HDF5 data

Usage:
    from gym_gui.replay import ReplayWriter, ReplayReader, FrameResolver

    # Writing
    writer = ReplayWriter(run_id, replay_dir)
    writer.start()
    frame_ref = writer.record_step(frame, obs, action, reward, done)
    writer.close()

    # Reading
    with ReplayReader(path) as reader:
        episode = reader.get_episode(0)

    # Resolving refs from SQLite
    with FrameResolver(replay_dir) as resolver:
        frame = resolver.resolve("h5://run_id/frames/123")
"""

from gym_gui.replay.constants import (
    REPLAY_DEFAULTS,
    FRAME_REF_DEFAULTS,
    ReplayStorageDefaults,
    FrameRefDefaults,
)
from gym_gui.replay.replay_store import ReplayWriter, ReplayReader
from gym_gui.replay.frame_resolver import FrameRef, FrameResolver

__all__ = [
    # Constants
    "REPLAY_DEFAULTS",
    "FRAME_REF_DEFAULTS",
    "ReplayStorageDefaults",
    "FrameRefDefaults",
    # Writer/Reader
    "ReplayWriter",
    "ReplayReader",
    # Resolver
    "FrameRef",
    "FrameResolver",
]
```

## Phase 2: Integration

### 2.1 Update StepRecord

```python
# gym_gui/core/data_model/telemetry_core.py

@dataclass
class StepRecord:
    # Existing fields...
    episode_id: str
    step_index: int
    action: Any
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    render_payload: Any
    timestamp: datetime
    agent_id: Optional[str]
    render_hint: Any
    frame_ref: Optional[str]  # EXISTING - now used for HDF5 refs
    payload_version: int
    run_id: Optional[str]
    worker_id: Optional[str]
    space_signature: Any
    vector_metadata: Any
    time_step: Optional[int]

    # NEW field for observation reference
    obs_ref: Optional[str] = None
```

### 2.2 Wire ReplayWriter into TelemetryDBSink

```python
# gym_gui/telemetry/db_sink.py - MODIFIED

from gym_gui.replay import ReplayWriter

class TelemetryDBSink(LogConstantMixin):
    def __init__(
        self,
        store: TelemetrySQLiteStore,
        bus: RunBus,
        *,
        replay_writer: Optional[ReplayWriter] = None,  # NEW
        batch_size: int = DB_SINK_BATCH_SIZE,
        # ... other params
    ) -> None:
        # ... existing init
        self._replay_writer = replay_writer

    def _process_step_queue(self, q: queue.Queue) -> None:
        while True:
            try:
                evt = q.get_nowait()
                if not isinstance(evt, TelemetryEvent):
                    continue

                payload = evt.payload

                # NEW: Extract arrays for HDF5
                frame = None
                observation = None
                frame_ref = None
                obs_ref = None

                if self._replay_writer is not None:
                    # Extract frame from render_payload
                    render_payload = payload.get("render_payload")
                    if isinstance(render_payload, dict):
                        frame = render_payload.get("frame")
                    elif isinstance(render_payload, np.ndarray):
                        frame = render_payload

                    # Extract observation
                    observation = payload.get("observation")
                    if isinstance(observation, np.ndarray):
                        pass  # Already numpy
                    elif observation is not None:
                        observation = None  # Skip non-array observations

                    # Write to HDF5 and get references
                    if frame is not None or observation is not None:
                        frame_ref = self._replay_writer.record_step(
                            frame=frame,
                            observation=observation,
                            action=payload.get("action", 0),
                            reward=payload.get("reward", 0.0),
                            done=payload.get("terminated", False) or payload.get("truncated", False),
                        )
                        # obs_ref uses same index
                        if observation is not None:
                            obs_ref = frame_ref.replace("/frames/", "/observations/")

                # Create StepRecord with refs instead of data
                step = StepRecord(
                    episode_id=payload.get("episode_id", ""),
                    step_index=payload.get("step_index", 0),
                    action=payload.get("action"),
                    observation=None if self._replay_writer else payload.get("observation"),
                    reward=payload.get("reward", 0.0),
                    terminated=payload.get("terminated", False),
                    truncated=payload.get("truncated", False),
                    info=payload.get("info", {}),
                    render_payload=None if self._replay_writer else payload.get("render_payload"),
                    frame_ref=frame_ref or payload.get("frame_ref"),
                    obs_ref=obs_ref,
                    # ... other fields
                )

                self._step_batch.append(step)
            except queue.Empty:
                break
```

## Phase 3: Migration

### 3.1 SQLite Schema Changes

```sql
-- Add new columns (if not exist)
ALTER TABLE steps ADD COLUMN obs_ref TEXT;

-- Create index for frame lookups
CREATE INDEX IF NOT EXISTS idx_steps_frame_ref ON steps(frame_ref);
CREATE INDEX IF NOT EXISTS idx_steps_obs_ref ON steps(obs_ref);
```

### 3.2 Update TelemetryService

```python
# gym_gui/services/telemetry.py - Add frame resolution

class TelemetryService:
    def __init__(
        self,
        *,
        frame_resolver: Optional[FrameResolver] = None,  # NEW
        # ... existing params
    ):
        self._frame_resolver = frame_resolver

    def get_step_with_frame(self, step: StepRecord) -> StepRecord:
        """Resolve frame_ref to actual frame data."""
        if not self._frame_resolver or not step.frame_ref:
            return step

        frame = self._frame_resolver.resolve(step.frame_ref)
        if frame is not None:
            return replace(step, render_payload={"frame": frame})
        return step
```

## Testing Plan

### Unit Tests

```python
# gym_gui/tests/test_replay_store.py

import pytest
import numpy as np
from pathlib import Path
from gym_gui.replay import ReplayWriter, ReplayReader, FrameResolver


class TestReplayWriter:
    def test_write_and_read_episode(self, tmp_path):
        run_id = "test-run"
        writer = ReplayWriter(run_id, tmp_path, frame_shape=(84, 84, 3))
        writer.start()

        # Write 100 steps
        for i in range(100):
            frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
            writer.record_step(frame, None, i % 4, float(i), i == 99)

        writer.mark_episode_end()
        writer.close()

        # Read back
        with ReplayReader(tmp_path / f"{run_id}.h5") as reader:
            assert reader.num_steps == 100
            assert reader.num_episodes == 1

            episode = reader.get_episode(0)
            assert episode["length"] == 100
            assert len(episode["frames"]) == 100


class TestFrameResolver:
    def test_resolve_single_frame(self, tmp_path):
        # Create test file
        run_id = "test-run"
        writer = ReplayWriter(run_id, tmp_path)
        writer.start()
        frame = np.ones((84, 84, 3), dtype=np.uint8) * 42
        ref = writer.record_step(frame, None, 0, 0.0, False)
        writer.close()

        # Resolve
        resolver = FrameResolver(tmp_path)
        resolved = resolver.resolve(ref)
        resolver.close()

        assert resolved is not None
        assert np.array_equal(resolved, frame)
```

## Summary

| Phase | Changes | Breaking? |
|-------|---------|-----------|
| 1 | Add replay package, h5py | No |
| 2 | Wire ReplayWriter, add frame_ref | No |
| 3 | Stop storing arrays in SQLite | Yes (new runs) |
| 4 | Drop unused columns | Yes (migration) |

The implementation maintains backward compatibility until Phase 3, allowing gradual rollout and validation.
