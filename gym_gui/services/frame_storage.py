"""Frame storage service for persisting rendered frames to disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from gym_gui.config.paths import VAR_RECORDS_DIR, ensure_var_directories
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_FRAME_DEBUG,
    LOG_SERVICE_FRAME_INFO,
    LOG_SERVICE_FRAME_WARNING,
    LOG_SERVICE_FRAME_ERROR,
)


class FrameStorageService(LogConstantMixin):
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
                self.log_constant(
                    LOG_SERVICE_FRAME_WARNING,
                    message="unsupported_frame_data",
                    extra={"data_type": type(frame_data).__name__},
                )
                return None

            self.log_constant(
                LOG_SERVICE_FRAME_INFO,
                message="frame_saved",
                extra={
                    "frame_path": str(frame_path),
                    "frame_ref": frame_ref,
                    "run_id": run_id or "-",
                    "data_type": type(frame_data).__name__,
                },
            )
            return frame_path
        except Exception as e:
            self.log_constant(
                LOG_SERVICE_FRAME_ERROR,
                message="frame_save_failed",
                extra={"frame_ref": frame_ref, "run_id": run_id or "-"},
                exc_info=e,
            )
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
                self.log_constant(
                    LOG_SERVICE_FRAME_DEBUG,
                    message="frame_not_found",
                    extra={"frame_path": str(frame_path), "frame_ref": frame_ref, "run_id": run_id or "-"},
                )
                return None
            
            # Load frame
            image = Image.open(frame_path)
            frame_array = np.array(image)
            
            self.log_constant(
                LOG_SERVICE_FRAME_INFO,
                message="frame_loaded",
                extra={"frame_path": str(frame_path), "frame_ref": frame_ref, "run_id": run_id or "-"},
            )
            return frame_array
        except Exception as e:
            self.log_constant(
                LOG_SERVICE_FRAME_ERROR,
                message="frame_load_failed",
                extra={"frame_ref": frame_ref, "run_id": run_id or "-"},
                exc_info=e,
            )
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
                self.log_constant(
                    LOG_SERVICE_FRAME_INFO,
                    message="frame_deleted",
                    extra={"frame_path": str(frame_path), "frame_ref": frame_ref, "run_id": run_id or "-"},
                )
                return True
            
            return False
        except Exception as e:
            self.log_constant(
                LOG_SERVICE_FRAME_ERROR,
                message="frame_delete_failed",
                extra={"frame_ref": frame_ref, "run_id": run_id or "-"},
                exc_info=e,
            )
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
                self.log_constant(
                    LOG_SERVICE_FRAME_INFO,
                    message="run_cleanup_completed",
                    extra={"run_id": run_id, "path": str(run_dir)},
                )
                return True
            return False
        except Exception as e:
            self.log_constant(
                LOG_SERVICE_FRAME_ERROR,
                message="run_cleanup_failed",
                extra={"run_id": run_id},
                exc_info=e,
            )
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
