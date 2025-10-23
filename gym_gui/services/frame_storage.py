"""Frame storage service for persisting rendered frames to disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from gym_gui.config.paths import VAR_RECORDS_DIR, ensure_var_directories


class FrameStorageService:
    """Service for storing and retrieving rendered frames."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """Initialize frame storage service.
        
        Args:
            base_dir: Base directory for frame storage. Defaults to VAR_RECORDS_DIR.
        """
        self._base_dir = (base_dir or VAR_RECORDS_DIR).expanduser().resolve()
        self._logger = logging.getLogger("gym_gui.services.frame_storage")
        ensure_var_directories()

    def save_frame(
        self,
        frame_data: Any,
        frame_ref: str,
        run_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Save a frame to disk.
        
        Args:
            frame_data: Frame data (numpy array for RGB, dict for grid)
            frame_ref: Frame reference string (e.g., "frames/2024-01-15_10-30-45_123.png")
            run_id: Optional run ID for organizing frames by training run
            
        Returns:
            Path to saved frame file, or None if save failed
        """
        try:
            # Determine storage directory
            if run_id:
                frame_dir = self._base_dir / run_id / frame_ref.rsplit("/", 1)[0]
            else:
                frame_dir = self._base_dir / frame_ref.rsplit("/", 1)[0]
            
            frame_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file path
            frame_filename = frame_ref.rsplit("/", 1)[-1]
            frame_path = frame_dir / frame_filename
            
            # Save based on frame data type
            if isinstance(frame_data, np.ndarray):
                # RGB frame from Box2D environments
                self._save_rgb_frame(frame_data, frame_path)
            elif isinstance(frame_data, dict):
                # Grid frame from ToyText environments
                self._save_grid_frame(frame_data, frame_path)
            else:
                self._logger.warning(f"Unsupported frame data type: {type(frame_data)}")
                return None
            
            self._logger.debug(f"Saved frame to {frame_path}")
            return frame_path
        except Exception as e:
            self._logger.error(f"Failed to save frame {frame_ref}: {e}", exc_info=True)
            return None

    def get_frame(
        self,
        frame_ref: str,
        run_id: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Load a frame from disk.
        
        Args:
            frame_ref: Frame reference string
            run_id: Optional run ID for organizing frames by training run
            
        Returns:
            Frame data as numpy array, or None if load failed
        """
        try:
            # Determine file path
            if run_id:
                frame_path = self._base_dir / run_id / frame_ref
            else:
                frame_path = self._base_dir / frame_ref
            
            if not frame_path.exists():
                self._logger.debug(f"Frame file not found: {frame_path}")
                return None
            
            # Load frame
            image = Image.open(frame_path)
            frame_array = np.array(image)
            
            self._logger.debug(f"Loaded frame from {frame_path}")
            return frame_array
        except Exception as e:
            self._logger.error(f"Failed to load frame {frame_ref}: {e}", exc_info=True)
            return None

    def delete_frame(
        self,
        frame_ref: str,
        run_id: Optional[str] = None,
    ) -> bool:
        """Delete a frame from disk.
        
        Args:
            frame_ref: Frame reference string
            run_id: Optional run ID for organizing frames by training run
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            # Determine file path
            if run_id:
                frame_path = self._base_dir / run_id / frame_ref
            else:
                frame_path = self._base_dir / frame_ref
            
            if frame_path.exists():
                frame_path.unlink()
                self._logger.debug(f"Deleted frame: {frame_path}")
                return True
            
            return False
        except Exception as e:
            self._logger.error(f"Failed to delete frame {frame_ref}: {e}", exc_info=True)
            return False

    def cleanup_run(self, run_id: str) -> bool:
        """Clean up all frames for a training run.
        
        Args:
            run_id: Training run ID
            
        Returns:
            True if cleanup succeeded, False otherwise
        """
        try:
            run_dir = self._base_dir / run_id
            if run_dir.exists():
                import shutil
                shutil.rmtree(run_dir)
                self._logger.debug(f"Cleaned up frames for run: {run_id}")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to cleanup run {run_id}: {e}", exc_info=True)
            return False

    @staticmethod
    def _save_rgb_frame(frame_data: np.ndarray, frame_path: Path) -> None:
        """Save RGB frame as PNG.
        
        Args:
            frame_data: RGB frame as numpy array (H, W, 3)
            frame_path: Path to save frame
        """
        # Ensure frame is uint8
        if frame_data.dtype != np.uint8:
            frame_data = np.clip(frame_data, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image and save
        image = Image.fromarray(frame_data, mode="RGB")
        image.save(frame_path, format="PNG")

    @staticmethod
    def _save_grid_frame(frame_data: dict[str, Any], frame_path: Path) -> None:
        """Save grid frame as PNG.
        
        Args:
            frame_data: Grid frame data (dict with 'grid' key)
            frame_path: Path to save frame
        """
        # For now, save as a simple text representation
        # In the future, this could render the grid as an image
        grid = frame_data.get("grid", [])
        ansi = frame_data.get("ansi", "")
        
        # Save ANSI representation as text
        text_path = frame_path.with_suffix(".txt")
        text_path.write_text(ansi, encoding="utf-8")


__all__ = ["FrameStorageService"]

