"""Frame reference resolution from SQLite to HDF5.

This module resolves frame references stored in SQLite (as URIs like
"h5://run_id/frames/123") to actual numpy arrays stored in HDF5 files.

The key classes are:
- FrameRef: Parsed frame reference dataclass
- FrameResolver: Resolves frame references to numpy arrays
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np

from gym_gui.constants import (
    FRAME_REF_SCHEME,
    REPLAY_DEFAULTS,
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


@dataclass
class FrameRef:
    """Parsed frame reference.

    Frame references use URI format: h5://{run_id}/{dataset}/{index}

    Examples:
        h5://run_abc123/frames/1523
        h5://run_abc123/observations/42

    Attributes:
        run_id: Run identifier
        dataset: Dataset name (e.g., "frames", "observations")
        index: Step index (0-based)
    """

    run_id: str
    dataset: str
    index: int

    @classmethod
    def parse(cls, ref: str) -> Optional["FrameRef"]:
        """Parse a frame reference URI.

        Args:
            ref: URI string like "h5://run_id/dataset/index"

        Returns:
            FrameRef instance or None if parsing fails
        """
        if not ref:
            return None

        pattern = rf"{FRAME_REF_SCHEME}://([^/]+)/([^/]+)/(\d+)"
        match = re.match(pattern, ref)
        if not match:
            _LOGGER.debug("Invalid frame ref format: %s", ref)
            return None

        return cls(
            run_id=match.group(1),
            dataset=match.group(2),
            index=int(match.group(3)),
        )

    def to_uri(self) -> str:
        """Convert back to URI string."""
        return f"{FRAME_REF_SCHEME}://{self.run_id}/{self.dataset}/{self.index}"


def make_frame_ref(run_id: str, dataset: str, index: int) -> str:
    """Create a frame reference URI.

    Args:
        run_id: Run identifier
        dataset: Dataset name (e.g., "frames", "observations")
        index: Step index

    Returns:
        URI like "h5://run_abc123/frames/1523"
    """
    return f"{FRAME_REF_SCHEME}://{run_id}/{dataset}/{index}"


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

    Or as context manager:
        with FrameResolver(replay_dir) as resolver:
            frame = resolver.resolve(ref)
    """

    def __init__(
        self,
        replay_dir: Path,
        max_open_files: int = REPLAY_DEFAULTS.reader.max_open_files,
    ) -> None:
        """Initialize the frame resolver.

        Args:
            replay_dir: Directory containing HDF5 files (var/replay/)
            max_open_files: Maximum number of HDF5 files to keep open
        """
        self._replay_dir = Path(replay_dir)
        self._max_open_files = max_open_files
        self._open_files: dict[str, Any] = {}  # run_id -> h5py.File

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
                    dataset_len = len(dataset)

                    # Sort by h5 index for efficient read
                    sorted_indices = sorted(indices, key=lambda x: x[1])

                    # Filter valid indices
                    valid_pairs = [
                        (orig_idx, h5_idx)
                        for orig_idx, h5_idx in sorted_indices
                        if h5_idx < dataset_len
                    ]
                    if not valid_pairs:
                        continue

                    # Extract h5 indices for batch read
                    h5_indices = [h5_idx for _, h5_idx in valid_pairs]

                    # Batch read from HDF5
                    data = dataset[h5_indices]

                    # Map back to results
                    for i, (orig_idx, _) in enumerate(valid_pairs):
                        results[orig_idx] = data[i]

                except KeyError:
                    _LOGGER.debug(
                        "Dataset not found during batch resolve",
                        extra={"run_id": run_id, "dataset": dataset_name},
                    )
                    continue

        return results

    def has_run(self, run_id: str) -> bool:
        """Check if HDF5 file exists for run.

        Args:
            run_id: Run identifier

        Returns:
            True if file exists
        """
        path = self._replay_dir / f"{run_id}.h5"
        return path.exists()

    def get_run_info(self, run_id: str) -> Optional[dict]:
        """Get metadata for a run's HDF5 file.

        Args:
            run_id: Run identifier

        Returns:
            Dict with metadata or None if file not found
        """
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

    def list_runs(self) -> list[str]:
        """List all run IDs with HDF5 files.

        Returns:
            List of run IDs
        """
        if not self._replay_dir.exists():
            return []

        runs = []
        for path in self._replay_dir.glob("*.h5"):
            runs.append(path.stem)
        return sorted(runs)

    def _get_file(self, run_id: str) -> Optional[Any]:
        """Get or open HDF5 file for run.

        Args:
            run_id: Run identifier

        Returns:
            h5py.File instance or None
        """
        if run_id in self._open_files:
            return self._open_files[run_id]

        path = self._replay_dir / f"{run_id}.h5"
        if not path.exists():
            _LOGGER.debug("HDF5 file not found", extra={"path": str(path)})
            return None

        try:
            h5py = _get_h5py()
            f = h5py.File(path, "r")

            # Evict oldest if at capacity
            if len(self._open_files) >= self._max_open_files:
                oldest_key = next(iter(self._open_files))
                self._open_files[oldest_key].close()
                del self._open_files[oldest_key]

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


__all__ = ["FrameRef", "FrameResolver", "make_frame_ref"]
