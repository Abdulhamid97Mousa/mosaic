"""HDF5-based replay storage for RL experiences.

This module provides efficient storage and retrieval of RL replay data using HDF5,
which is optimized for large numerical arrays (frames, observations).

The key classes are:
- ReplayWriter: Writes frames/observations to HDF5 with background threading
- ReplayReader: Reads episodes from HDF5 for replay or training
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Any

import numpy as np

from gym_gui.constants import (
    REPLAY_DEFAULTS,
    FRAME_REF_SCHEME,
    FRAME_REF_FRAMES_DATASET,
    FRAME_REF_OBS_DATASET,
)

_LOGGER = logging.getLogger(__name__)

# Lazy import h5py to avoid hard dependency at module load time
_h5py = None


def _get_h5py():
    """Lazy import of h5py."""
    global _h5py
    if _h5py is None:
        try:
            import h5py
            _h5py = h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required for HDF5 replay storage. "
                "Install with: pip install h5py"
            ) from e
    return _h5py


class ReplayWriter:
    """Writes frames and observations to HDF5 for replay.

    This class provides:
    - Background thread for non-blocking writes
    - Automatic batching for efficient I/O
    - Chunked, compressed HDF5 storage
    - Episode boundary tracking

    Usage:
        writer = ReplayWriter(run_id, replay_dir)
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
        frame_shape: Optional[tuple[int, ...]] = None,
        obs_shape: Optional[tuple[int, ...]] = None,
        max_steps: int = REPLAY_DEFAULTS.writer.max_steps_per_run,
        chunk_size: int = REPLAY_DEFAULTS.writer.frame_chunk_size,
        compression: Optional[str] = REPLAY_DEFAULTS.writer.compression,
        batch_size: int = REPLAY_DEFAULTS.writer.write_batch_size,
        queue_size: int = REPLAY_DEFAULTS.writer.write_queue_size,
    ) -> None:
        """Initialize the replay writer.

        Args:
            run_id: Unique identifier for the run
            replay_dir: Directory to store HDF5 files
            frame_shape: Shape of frames (H, W, C). Auto-detected if None.
            obs_shape: Shape of observations. Auto-detected if None.
            max_steps: Maximum steps to store per run
            chunk_size: HDF5 chunk size (frames per chunk)
            compression: Compression algorithm ("lzf", "gzip", or None)
            batch_size: Frames to buffer before writing
            queue_size: Max items in background write queue
        """
        self._run_id = run_id
        self._path = Path(replay_dir) / f"{run_id}.h5"
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Configuration
        self._frame_shape = frame_shape
        self._obs_shape = obs_shape
        self._obs_dtype: Optional[np.dtype] = None  # Auto-detected from first obs
        self._action_shape: Optional[tuple[int, ...]] = None  # For continuous actions
        self._action_dtype: Optional[np.dtype] = None  # int32 for discrete, float32 for continuous
        self._max_steps = max_steps
        self._chunk_size = chunk_size
        self._compression = compression
        self._batch_size = batch_size

        # Background writer
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        # State
        self._file: Any = None  # h5py.File
        self._step_count = 0
        self._episode_starts: list[int] = [0]
        self._initialized = False
        self._datasets_created = False

        _LOGGER.debug(
            "ReplayWriter created",
            extra={"run_id": run_id, "path": str(self._path)},
        )

    @property
    def run_id(self) -> str:
        """Return the run ID."""
        return self._run_id

    @property
    def path(self) -> Path:
        """Return the HDF5 file path."""
        return self._path

    @property
    def step_count(self) -> int:
        """Return the number of steps written."""
        return self._step_count

    def start(self) -> None:
        """Initialize HDF5 file and start background writer."""
        if self._initialized:
            return

        h5py = _get_h5py()
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
        action: Any,  # int for discrete, np.ndarray for continuous
        reward: float,
        done: bool,
    ) -> str:
        """Queue a step for background writing.

        Args:
            frame: RGB frame array, shape (H, W, C)
            observation: Preprocessed observation array
            action: Action taken (int for discrete, ndarray for continuous)
            reward: Reward received
            done: Whether episode ended

        Returns:
            Frame reference URI: "h5://{run_id}/frames/{index}"
        """
        if not self._initialized:
            raise RuntimeError("ReplayWriter not started. Call start() first.")

        step_index = self._step_count
        frame_ref = self._make_ref(FRAME_REF_FRAMES_DATASET, step_index)

        # Initialize shapes and dtypes on first step if not provided
        if step_index == 0:
            if frame is not None and self._frame_shape is None:
                self._frame_shape = frame.shape
            if observation is not None:
                if self._obs_shape is None:
                    self._obs_shape = observation.shape
                if self._obs_dtype is None:
                    self._obs_dtype = observation.dtype
            # Detect action type (discrete vs continuous)
            if isinstance(action, np.ndarray):
                self._action_shape = action.shape
                self._action_dtype = action.dtype
            elif isinstance(action, (list, tuple)):
                arr = np.array(action)
                self._action_shape = arr.shape
                self._action_dtype = arr.dtype
            else:
                self._action_shape = ()  # Scalar
                self._action_dtype = np.dtype(np.int32)

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
        return f"{FRAME_REF_SCHEME}://{self._run_id}/{dataset}/{index}"

    def _writer_loop(self) -> None:
        """Background thread that writes to HDF5."""
        batch_frames: list = []
        batch_obs: list = []
        batch_actions: list = []
        batch_rewards: list = []
        batch_dones: list = []

        timeout = REPLAY_DEFAULTS.writer.write_timeout_s

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=timeout)
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

        # Write actions (may be scalar or array per step)
        if "actions" in self._file:
            ds = self._file["actions"]
            current = ds.shape[0]
            ds.resize(current + n, axis=0)
            # Convert actions to proper array format
            if len(ds.shape) > 1:
                # Continuous actions (2D: steps x action_dim)
                ds[current:current + n] = np.array(actions)
            else:
                # Discrete actions (1D: steps)
                ds[current:current + n] = np.array(actions)

        # Write rewards and dones (always scalar per step)
        for name, data in [
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
        if self._datasets_created:
            return
        if self._file is None:
            return

        scalar_chunk = self._chunk_size * 10

        # Frames dataset (always uint8 for RGB images)
        if self._frame_shape is not None:
            self._file.create_dataset(
                "frames",
                shape=(0, *self._frame_shape),
                maxshape=(self._max_steps, *self._frame_shape),
                chunks=(self._chunk_size, *self._frame_shape),
                dtype=np.uint8,
                compression=self._compression,
            )

        # Observations dataset - use detected dtype (float32 for continuous, uint8 for images)
        if self._obs_shape is not None:
            obs_dtype = self._obs_dtype if self._obs_dtype is not None else np.float32
            self._file.create_dataset(
                "observations",
                shape=(0, *self._obs_shape),
                maxshape=(self._max_steps, *self._obs_shape),
                chunks=(self._chunk_size, *self._obs_shape),
                dtype=obs_dtype,
                compression=self._compression,
            )

        # Actions dataset - handle both discrete (scalar int) and continuous (float array)
        action_dtype = self._action_dtype if self._action_dtype is not None else np.int32
        if self._action_shape and len(self._action_shape) > 0:
            # Continuous action space (e.g., CarRacing: shape=(3,) for [steering, gas, brake])
            self._file.create_dataset(
                "actions",
                shape=(0, *self._action_shape),
                maxshape=(self._max_steps, *self._action_shape),
                chunks=(scalar_chunk, *self._action_shape),
                dtype=action_dtype,
            )
        else:
            # Discrete action space (scalar int)
            self._file.create_dataset(
                "actions",
                shape=(0,),
                maxshape=(self._max_steps,),
                chunks=(scalar_chunk,),
                dtype=action_dtype,
            )

        self._file.create_dataset(
            "rewards",
            shape=(0,),
            maxshape=(self._max_steps,),
            chunks=(scalar_chunk,),
            dtype=np.float32,
        )
        self._file.create_dataset(
            "dones",
            shape=(0,),
            maxshape=(self._max_steps,),
            chunks=(scalar_chunk,),
            dtype=np.bool_,
        )

        self._datasets_created = True

    def _write_episode_index(self) -> None:
        """Write episode boundaries to file."""
        if self._file is None:
            return

        episodes = self._file.create_group("episodes")

        # Episode start indices (excluding the trailing entry)
        starts = self._episode_starts[:-1] if len(self._episode_starts) > 1 else [0]
        episodes.create_dataset("starts", data=np.array(starts, dtype=np.int64))

        # Calculate lengths
        lengths = []
        for i in range(len(self._episode_starts) - 1):
            lengths.append(self._episode_starts[i + 1] - self._episode_starts[i])
        # Handle last episode if it has steps
        if self._step_count > self._episode_starts[-1]:
            if len(self._episode_starts) == 1:
                # Only one episode, no mark_episode_end called
                lengths.append(self._step_count)
            else:
                lengths.append(self._step_count - self._episode_starts[-1])

        episodes.create_dataset("lengths", data=np.array(lengths, dtype=np.int64))

    def _write_metadata(self) -> None:
        """Write run metadata to file."""
        if self._file is None:
            return

        self._file.attrs["run_id"] = self._run_id
        self._file.attrs["total_steps"] = self._step_count
        self._file.attrs["total_episodes"] = max(1, len(self._episode_starts) - 1)
        self._file.attrs["created_at"] = datetime.now().isoformat()
        self._file.attrs["version"] = "1.0"

    def __enter__(self) -> "ReplayWriter":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.close()


class ReplayReader:
    """Reads replay data from HDF5 for playback or training.

    Usage:
        with ReplayReader(path) as reader:
            # Get metadata
            print(reader.num_steps, reader.num_episodes)

            # Get single step
            step = reader.get_step(100)

            # Get entire episode
            episode = reader.get_episode(0)

            # Iterate in batches for training
            for batch in reader.iter_batches(batch_size=32):
                train(batch)
    """

    def __init__(self, path: Path) -> None:
        """Initialize the replay reader.

        Args:
            path: Path to the HDF5 file
        """
        self._path = Path(path)
        self._file: Any = None  # h5py.File

    def open(self) -> None:
        """Open the HDF5 file for reading."""
        h5py = _get_h5py()
        self._file = h5py.File(self._path, "r")

    def close(self) -> None:
        """Close the HDF5 file."""
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
        """Return total number of steps."""
        if self._file is None:
            return 0
        if "actions" in self._file:
            return self._file["actions"].shape[0]
        return 0

    @property
    def num_episodes(self) -> int:
        """Return total number of episodes."""
        if self._file is None:
            return 0
        if "episodes/starts" in self._file:
            return len(self._file["episodes/starts"])
        return 0

    @property
    def metadata(self) -> dict:
        """Return file metadata as dict."""
        if self._file is None:
            return {}
        return dict(self._file.attrs)

    @property
    def has_frames(self) -> bool:
        """Return True if file contains frames dataset."""
        return self._file is not None and "frames" in self._file

    @property
    def has_observations(self) -> bool:
        """Return True if file contains observations dataset."""
        return self._file is not None and "observations" in self._file

    def get_step(self, index: int) -> dict:
        """Get a single step by index.

        Args:
            index: Step index (0-based)

        Returns:
            Dict with action, reward, done, and optionally frame/observation
        """
        if self._file is None:
            raise RuntimeError("Reader not open")

        result = {
            "action": int(self._file["actions"][index]),
            "reward": float(self._file["rewards"][index]),
            "done": bool(self._file["dones"][index]),
        }

        if "frames" in self._file:
            result["frame"] = self._file["frames"][index]
        if "observations" in self._file:
            result["observation"] = self._file["observations"][index]

        return result

    def get_episode(self, episode_idx: int) -> dict:
        """Get all data for an episode.

        Args:
            episode_idx: Episode index (0-based)

        Returns:
            Dict with arrays for the entire episode
        """
        if self._file is None:
            raise RuntimeError("Reader not open")

        starts = self._file["episodes/starts"][:]
        lengths = self._file["episodes/lengths"][:]

        if episode_idx >= len(starts):
            raise IndexError(f"Episode {episode_idx} not found (max: {len(starts)-1})")

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
        """Get a single frame by step index.

        Args:
            step_index: Step index (0-based)

        Returns:
            Frame array or None if not found
        """
        if self._file is None:
            return None
        if "frames" not in self._file:
            return None
        if step_index >= self._file["frames"].shape[0]:
            return None
        return self._file["frames"][step_index]

    def get_observation(self, step_index: int) -> Optional[np.ndarray]:
        """Get a single observation by step index.

        Args:
            step_index: Step index (0-based)

        Returns:
            Observation array or None if not found
        """
        if self._file is None:
            return None
        if "observations" not in self._file:
            return None
        if step_index >= self._file["observations"].shape[0]:
            return None
        return self._file["observations"][step_index]

    def iter_batches(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        include_frames: bool = False,
    ) -> Iterator[dict]:
        """Iterate through data in batches for training.

        Args:
            batch_size: Number of steps per batch
            shuffle: Whether to shuffle indices
            include_frames: Whether to include frame data

        Yields:
            Dict with batch arrays
        """
        if self._file is None:
            raise RuntimeError("Reader not open")

        indices = np.arange(self.num_steps)
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            # Sort for efficient HDF5 read (contiguous access)
            sorted_indices = np.sort(batch_indices)

            batch = {
                "actions": self._file["actions"][sorted_indices],
                "rewards": self._file["rewards"][sorted_indices],
                "dones": self._file["dones"][sorted_indices],
            }

            if "observations" in self._file:
                batch["observations"] = self._file["observations"][sorted_indices]

            if include_frames and "frames" in self._file:
                batch["frames"] = self._file["frames"][sorted_indices]

            yield batch


__all__ = ["ReplayWriter", "ReplayReader"]
