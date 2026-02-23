"""Helpers for composing multi-env FastLane frames."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def tile_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Tile multiple RGB frames into a single composite image.

    Mirrors Stable-Baselines3's VecEnv tiling logic so we can stream several
    vectorized environments inside a single Fast Lane frame.

    Args:
        frames: Sequence of RGB images shaped (H, W, C).

    Returns:
        Composite RGB image shaped (rows * H, cols * W, C) where rows/cols are
        chosen to form a near-square grid.

    Raises:
        ValueError: If no frames are provided or shapes differ.
    """

    if not frames:
        raise ValueError("tile_frames requires at least one frame")

    np_frames = [np.asarray(frame) for frame in frames]
    first_shape = np_frames[0].shape
    if len(first_shape) != 3:
        raise ValueError("Frames must be HWC RGB arrays")

    for frame in np_frames:
        if frame.shape != first_shape:
            raise ValueError("All frames must share the same shape")

    n_images = len(np_frames)
    rows = int(math.ceil(math.sqrt(n_images)))
    cols = int(math.ceil(n_images / rows))
    total_slots = rows * cols

    if total_slots > n_images:
        pad_count = total_slots - n_images
        pad_frame = np.zeros_like(np_frames[0])
        np_frames.extend([pad_frame] * pad_count)

    stacked = np.stack(np_frames, axis=0)
    h, w, c = first_shape
    reshaped = stacked.reshape(rows, cols, h, w, c)
    transposed = reshaped.swapaxes(1, 2)
    composite = transposed.reshape(rows * h, cols * w, c)
    return composite
