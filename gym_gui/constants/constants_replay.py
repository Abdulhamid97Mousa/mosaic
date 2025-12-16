"""HDF5 replay storage configuration defaults.

Defines configuration for the replay storage system that splits RL data between:
- SQLite: Scalars (rewards, actions, timestamps) + frame references
- HDF5: Large arrays (frames, observations)

The key innovation is frame references stored in SQLite that point to HDF5 locations,
enabling fast queries on scalars while efficiently storing large numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ReplayWriterDefaults:
    """Configuration for HDF5 replay writer."""

    # Dataset size limits
    max_steps_per_run: int = 10_000_000  # 10M steps max per run

    # Chunking (affects I/O performance)
    # Rule of thumb: chunk ~1-10 MB
    # For frames (210, 160, 3) = 100KB each, 100 frames = 10MB
    frame_chunk_size: int = 100
    scalar_chunk_size: int = 1000

    # Compression options:
    # - "lzf": Fast compression, moderate ratio (RECOMMENDED)
    # - "gzip": Better compression, slower
    # - None: No compression, fastest writes
    compression: Optional[str] = "lzf"
    compression_opts: Optional[int] = None  # lzf doesn't use opts

    # Background writer batching
    write_batch_size: int = 100  # Frames to buffer before write
    write_queue_size: int = 1000  # Max items in write queue
    write_timeout_s: float = 0.1  # Queue poll timeout


@dataclass(frozen=True)
class ReplayReaderDefaults:
    """Configuration for HDF5 replay reader."""

    # Batch reading
    default_batch_size: int = 32

    # Cache settings
    max_open_files: int = 10  # Max HDF5 files to keep open


@dataclass(frozen=True)
class FrameRefDefaults:
    """Defaults for frame reference URIs.

    Frame references use URI format: h5://{run_id}/{dataset}/{index}

    Examples:
        h5://run_abc123/frames/1523
        h5://run_abc123/observations/42
    """

    scheme: str = "h5"
    frame_dataset: str = "frames"
    obs_dataset: str = "observations"


@dataclass(frozen=True)
class ReplayPathDefaults:
    """Path configuration for replay storage."""

    # Subdirectory under var/ for HDF5 files
    replay_subdir: str = "replay"

    # File extension
    file_extension: str = ".h5"


@dataclass(frozen=True)
class ReplayDefaults:
    """Aggregated replay storage defaults."""

    writer: ReplayWriterDefaults = field(default_factory=ReplayWriterDefaults)
    reader: ReplayReaderDefaults = field(default_factory=ReplayReaderDefaults)
    frame_ref: FrameRefDefaults = field(default_factory=FrameRefDefaults)
    paths: ReplayPathDefaults = field(default_factory=ReplayPathDefaults)


REPLAY_DEFAULTS = ReplayDefaults()

# Convenience constants for direct access
REPLAY_MAX_STEPS_PER_RUN = REPLAY_DEFAULTS.writer.max_steps_per_run
REPLAY_FRAME_CHUNK_SIZE = REPLAY_DEFAULTS.writer.frame_chunk_size
REPLAY_COMPRESSION = REPLAY_DEFAULTS.writer.compression
REPLAY_WRITE_BATCH_SIZE = REPLAY_DEFAULTS.writer.write_batch_size
REPLAY_WRITE_QUEUE_SIZE = REPLAY_DEFAULTS.writer.write_queue_size

FRAME_REF_SCHEME = REPLAY_DEFAULTS.frame_ref.scheme
FRAME_REF_FRAMES_DATASET = REPLAY_DEFAULTS.frame_ref.frame_dataset
FRAME_REF_OBS_DATASET = REPLAY_DEFAULTS.frame_ref.obs_dataset

REPLAY_SUBDIR = REPLAY_DEFAULTS.paths.replay_subdir


__all__ = [
    # Main defaults object
    "REPLAY_DEFAULTS",
    # Dataclasses
    "ReplayDefaults",
    "ReplayWriterDefaults",
    "ReplayReaderDefaults",
    "FrameRefDefaults",
    "ReplayPathDefaults",
    # Convenience constants
    "REPLAY_MAX_STEPS_PER_RUN",
    "REPLAY_FRAME_CHUNK_SIZE",
    "REPLAY_COMPRESSION",
    "REPLAY_WRITE_BATCH_SIZE",
    "REPLAY_WRITE_QUEUE_SIZE",
    "FRAME_REF_SCHEME",
    "FRAME_REF_FRAMES_DATASET",
    "FRAME_REF_OBS_DATASET",
    "REPLAY_SUBDIR",
]
