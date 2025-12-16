"""HDF5-based replay storage for RL experiences.

This package provides efficient storage and retrieval of RL replay data,
splitting storage between SQLite (scalars) and HDF5 (arrays).

Architecture:
    SQLite stores: rewards, actions, terminated, step_index, frame_ref
    HDF5 stores: frames, observations (large numpy arrays)

    The frame_ref field in SQLite contains a URI like "h5://run_id/frames/123"
    that can be resolved to the actual numpy array using FrameResolver.

Components:
    - ReplayWriter: Writes frames/observations to HDF5 with background threading
    - ReplayReader: Reads episodes from HDF5 for replay or training
    - FrameRef: Parsed frame reference dataclass
    - FrameResolver: Resolves SQLite frame_ref to HDF5 data
    - make_frame_ref: Helper to create frame reference URIs

Usage:
    # Writing to HDF5
    from gym_gui.replay import ReplayWriter

    writer = ReplayWriter(run_id, replay_dir)
    writer.start()
    frame_ref = writer.record_step(frame, obs, action, reward, done)
    # frame_ref = "h5://run_id/frames/123" - store this in SQLite
    writer.close()

    # Reading from HDF5
    from gym_gui.replay import ReplayReader

    with ReplayReader(path) as reader:
        episode = reader.get_episode(0)
        for batch in reader.iter_batches(batch_size=32):
            train(batch)

    # Resolving refs from SQLite
    from gym_gui.replay import FrameResolver

    with FrameResolver(replay_dir) as resolver:
        # Single resolution
        frame = resolver.resolve("h5://run_id/frames/123")

        # Batch resolution (more efficient)
        frames = resolver.resolve_batch(frame_refs)

Configuration:
    Constants are defined in gym_gui.constants.constants_replay:
    - REPLAY_DEFAULTS: Main configuration object
    - REPLAY_MAX_STEPS_PER_RUN: Max steps per HDF5 file
    - REPLAY_COMPRESSION: Compression algorithm ("lzf" recommended)
    - FRAME_REF_SCHEME: URI scheme ("h5")
"""

from gym_gui.replay.replay_store import ReplayWriter, ReplayReader
from gym_gui.replay.frame_resolver import FrameRef, FrameResolver, make_frame_ref

__all__ = [
    # Writer/Reader
    "ReplayWriter",
    "ReplayReader",
    # Resolver
    "FrameRef",
    "FrameResolver",
    # Helpers
    "make_frame_ref",
]
