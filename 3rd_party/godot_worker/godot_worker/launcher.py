"""Godot Launcher - Launch and manage Godot processes.

This module provides utilities to:
1. Check if Godot binary is available
2. Launch Godot instances with various configurations
3. Manage running Godot processes
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from godot_worker.config import GodotConfig, GodotRenderMode

_LOGGER = logging.getLogger(__name__)

# Paths relative to this file
_THIS_DIR = Path(__file__).parent
_WORKER_DIR = _THIS_DIR.parent
_BIN_DIR = _WORKER_DIR / "bin"


@dataclass
class GodotProcess:
    """Represents a running Godot process."""

    instance_id: int
    process: subprocess.Popen
    project_path: Optional[str] = None
    port: Optional[int] = None

    @property
    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.process.poll() is None

    def terminate(self) -> None:
        """Terminate the process."""
        if self.is_running:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class GodotLauncher:
    """Launcher for Godot game engine instances."""

    def __init__(self) -> None:
        self._processes: dict[int, GodotProcess] = {}
        self._next_instance_id: int = 0

    @property
    def godot_binary_path(self) -> Path:
        """Path to the Godot binary."""
        return _BIN_DIR / "godot"

    def is_available(self) -> bool:
        """Check if Godot binary is available."""
        godot_path = self.godot_binary_path
        return godot_path.exists() and os.access(godot_path, os.X_OK)

    def get_version(self) -> Optional[str]:
        """Get Godot version string."""
        if not self.is_available():
            return None
        try:
            result = subprocess.run(
                [str(self.godot_binary_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def get_status(self) -> dict:
        """Get detailed status information."""
        return {
            "is_available": self.is_available(),
            "godot_binary": str(self.godot_binary_path),
            "godot_exists": self.godot_binary_path.exists(),
            "version": self.get_version(),
            "bin_dir": str(_BIN_DIR),
            "worker_dir": str(_WORKER_DIR),
        }

    def launch(
        self,
        config: Optional[GodotConfig] = None,
    ) -> tuple[Optional[GodotProcess], str]:
        """Launch a new Godot instance.

        Args:
            config: Configuration for the Godot instance

        Returns:
            Tuple of (process or None, message)
        """
        if not self.is_available():
            return None, (
                f"Godot binary not found at {self.godot_binary_path}\n"
                "Please ensure the Godot binary is in the bin/ directory."
            )

        if config is None:
            config = GodotConfig()

        self._next_instance_id += 1
        instance_id = self._next_instance_id

        try:
            # Build command
            cmd = [str(self.godot_binary_path)]

            # Project path
            if config.project_path:
                cmd.extend(["--path", config.project_path])

            # Scene
            if config.scene_path:
                cmd.append(config.scene_path)

            # Rendering mode
            if config.render_mode == GodotRenderMode.OPENGL3:
                cmd.append("--rendering-driver")
                cmd.append("opengl3")
            elif config.render_mode == GodotRenderMode.HEADLESS:
                cmd.append("--headless")

            # Headless mode (for training)
            if config.headless and config.render_mode != GodotRenderMode.HEADLESS:
                cmd.append("--headless")

            # Resolution
            cmd.append(f"--resolution")
            cmd.append(f"{config.resolution[0]}x{config.resolution[1]}")

            # Fixed FPS
            if config.fixed_fps > 0:
                cmd.append(f"--fixed-fps")
                cmd.append(str(config.fixed_fps))

            # Verbose output
            if config.verbose:
                cmd.append("--verbose")

            # Set environment
            env = os.environ.copy()

            _LOGGER.info(f"Launching Godot instance {instance_id}: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            godot_process = GodotProcess(
                instance_id=instance_id,
                process=process,
                project_path=config.project_path,
                port=config.port,
            )
            self._processes[instance_id] = godot_process

            return godot_process, f"Launched Godot instance {instance_id}"

        except Exception as e:
            return None, f"Failed to launch Godot: {e}"

    def launch_editor(
        self,
        project_path: Optional[str] = None,
    ) -> tuple[Optional[GodotProcess], str]:
        """Launch Godot in editor mode.

        Args:
            project_path: Path to project to open in editor

        Returns:
            Tuple of (process or None, message)
        """
        if not self.is_available():
            return None, f"Godot binary not found at {self.godot_binary_path}"

        self._next_instance_id += 1
        instance_id = self._next_instance_id

        try:
            cmd = [str(self.godot_binary_path), "--editor"]
            if project_path:
                cmd.extend(["--path", project_path])

            _LOGGER.info(f"Launching Godot editor {instance_id}: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            godot_process = GodotProcess(
                instance_id=instance_id,
                process=process,
                project_path=project_path,
            )
            self._processes[instance_id] = godot_process

            return godot_process, f"Launched Godot editor instance {instance_id}"

        except Exception as e:
            return None, f"Failed to launch Godot editor: {e}"

    def get_process(self, instance_id: int) -> Optional[GodotProcess]:
        """Get a running process by ID."""
        return self._processes.get(instance_id)

    def list_processes(self) -> list[GodotProcess]:
        """List all managed processes (running or not)."""
        return list(self._processes.values())

    def list_running(self) -> list[GodotProcess]:
        """List only running processes."""
        return [p for p in self._processes.values() if p.is_running]

    def terminate(self, instance_id: int) -> bool:
        """Terminate a specific Godot instance."""
        process = self._processes.get(instance_id)
        if process:
            process.terminate()
            return True
        return False

    def terminate_all(self) -> int:
        """Terminate all Godot instances. Returns count terminated."""
        count = 0
        for process in self._processes.values():
            if process.is_running:
                process.terminate()
                count += 1
        return count

    def cleanup(self) -> None:
        """Clean up all processes and resources."""
        self.terminate_all()
        self._processes.clear()


# Global launcher instance
_launcher: Optional[GodotLauncher] = None


def get_launcher() -> GodotLauncher:
    """Get or create the global Godot launcher."""
    global _launcher
    if _launcher is None:
        _launcher = GodotLauncher()
    return _launcher
