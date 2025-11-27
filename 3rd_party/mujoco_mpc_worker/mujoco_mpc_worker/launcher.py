"""MuJoCo MPC Launcher - Build and launch MJPC processes.

This module provides utilities to:
1. Check if MJPC is built
2. Build MJPC if needed
3. Launch MJPC GUI instances
4. Manage running MJPC processes
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_LOGGER = logging.getLogger(__name__)

# Paths relative to this file
_THIS_DIR = Path(__file__).parent
_WORKER_DIR = _THIS_DIR
_MUJOCO_MPC_DIR = _THIS_DIR.parent / "mujoco_mpc"
# MJPC is built inside the vendored mujoco_mpc directory
_BUILD_DIR = _MUJOCO_MPC_DIR / "build"
_BIN_DIR = _BUILD_DIR / "bin"


@dataclass
class MJPCProcess:
    """Represents a running MJPC process."""

    instance_id: int
    process: subprocess.Popen
    task_id: Optional[str] = None
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


class MJPCLauncher:
    """Launcher for MuJoCo MPC GUI instances."""

    def __init__(self) -> None:
        self._processes: dict[int, MJPCProcess] = {}
        self._next_instance_id: int = 0

    @property
    def mjpc_binary_path(self) -> Path:
        """Path to the MJPC GUI binary."""
        return _BIN_DIR / "mjpc"

    @property
    def agent_server_path(self) -> Path:
        """Path to the agent_server binary."""
        return _BIN_DIR / "agent_server"

    def is_built(self) -> bool:
        """Check if MJPC binaries are built."""
        mjpc_path = self.mjpc_binary_path
        return mjpc_path.exists() and os.access(mjpc_path, os.X_OK)

    def get_build_status(self) -> dict:
        """Get detailed build status information."""
        return {
            "is_built": self.is_built(),
            "mjpc_binary": str(self.mjpc_binary_path),
            "mjpc_exists": self.mjpc_binary_path.exists(),
            "agent_server_exists": self.agent_server_path.exists(),
            "build_dir": str(_BUILD_DIR),
            "build_dir_exists": _BUILD_DIR.exists(),
            "source_dir": str(_MUJOCO_MPC_DIR),
            "source_exists": _MUJOCO_MPC_DIR.exists(),
        }

    def build(self, num_jobs: Optional[int] = None) -> tuple[bool, str]:
        """Build MJPC from source.

        Args:
            num_jobs: Number of parallel build jobs (None = auto-detect)

        Returns:
            Tuple of (success, message)
        """
        if not _MUJOCO_MPC_DIR.exists():
            return False, (
                f"MuJoCo MPC source not found at {_MUJOCO_MPC_DIR}\n"
                "Please initialize the submodule:\n"
                "  git submodule update --init --recursive"
            )

        # Check for cmake
        if not shutil.which("cmake"):
            return False, "CMake not found. Please install CMake first."

        # Check for make or ninja
        build_tool = "make"
        if shutil.which("ninja"):
            build_tool = "ninja"
        elif not shutil.which("make"):
            return False, "Neither 'make' nor 'ninja' found. Please install a build tool."

        # Create build directory
        _BUILD_DIR.mkdir(parents=True, exist_ok=True)

        # Determine number of jobs
        if num_jobs is None:
            num_jobs = os.cpu_count() or 4

        try:
            # Configure with CMake
            _LOGGER.info("Configuring MJPC with CMake...")
            cmake_args = [
                "cmake",
                "-DCMAKE_BUILD_TYPE=Release",
                str(_WORKER_DIR),
            ]
            if build_tool == "ninja":
                cmake_args.insert(1, "-GNinja")

            result = subprocess.run(
                cmake_args,
                cwd=_BUILD_DIR,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, f"CMake configuration failed:\n{result.stderr}"

            # Build
            _LOGGER.info(f"Building MJPC with {num_jobs} jobs...")
            build_args = [build_tool, f"-j{num_jobs}"]
            result = subprocess.run(
                build_args,
                cwd=_BUILD_DIR,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, f"Build failed:\n{result.stderr}"

            if self.is_built():
                return True, "MJPC built successfully!"
            else:
                return False, "Build completed but MJPC binary not found."

        except Exception as e:
            return False, f"Build error: {e}"

    def launch(self, task_id: Optional[str] = None) -> tuple[Optional[MJPCProcess], str]:
        """Launch a new MJPC GUI instance.

        Args:
            task_id: Optional task to load on startup

        Returns:
            Tuple of (process or None, message)
        """
        if not self.is_built():
            return None, (
                "MJPC is not built. Please build first:\n"
                "  cd 3rd_party/mujoco_mpc_worker/mujoco_mpc_worker\n"
                "  mkdir -p build && cd build\n"
                "  cmake .. && make -j$(nproc)"
            )

        self._next_instance_id += 1
        instance_id = self._next_instance_id

        try:
            # Build command
            cmd = [str(self.mjpc_binary_path)]
            if task_id:
                cmd.extend(["--task", task_id])

            # Set environment for proper rendering
            env = os.environ.copy()
            # Ensure MJPC can find MuJoCo resources
            if "MUJOCO_GL" not in env:
                env["MUJOCO_GL"] = "egl"  # or "glx" for X11

            _LOGGER.info(f"Launching MJPC instance {instance_id}: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            mjpc_process = MJPCProcess(
                instance_id=instance_id,
                process=process,
                task_id=task_id,
            )
            self._processes[instance_id] = mjpc_process

            return mjpc_process, f"Launched MJPC instance {instance_id}"

        except Exception as e:
            return None, f"Failed to launch MJPC: {e}"

    def get_process(self, instance_id: int) -> Optional[MJPCProcess]:
        """Get a running process by ID."""
        return self._processes.get(instance_id)

    def list_processes(self) -> list[MJPCProcess]:
        """List all managed processes (running or not)."""
        return list(self._processes.values())

    def list_running(self) -> list[MJPCProcess]:
        """List only running processes."""
        return [p for p in self._processes.values() if p.is_running]

    def terminate(self, instance_id: int) -> bool:
        """Terminate a specific MJPC instance."""
        process = self._processes.get(instance_id)
        if process:
            process.terminate()
            return True
        return False

    def terminate_all(self) -> int:
        """Terminate all MJPC instances. Returns count terminated."""
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
_launcher: Optional[MJPCLauncher] = None


def get_launcher() -> MJPCLauncher:
    """Get or create the global MJPC launcher."""
    global _launcher
    if _launcher is None:
        _launcher = MJPCLauncher()
    return _launcher
