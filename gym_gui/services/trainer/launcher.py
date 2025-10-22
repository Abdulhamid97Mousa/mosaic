# /home/hamid/Desktop/Projects/GUI_BDI_RL/gym_gui/services/trainer/launcher.py

from __future__ import annotations

"""Utilities for launching and supervising the trainer daemon process."""

import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

from gym_gui.config.paths import VAR_LOGS_DIR, ensure_var_directories

LOGGER = logging.getLogger("gym_gui.trainer.launcher")

_DEFAULT_TARGET = "127.0.0.1:50055"
_POLL_INTERVAL = 0.5  # seconds


class TrainerDaemonLaunchError(RuntimeError):
    """Raised when the trainer daemon could not be started or contacted."""


@dataclass(slots=True)
class TrainerDaemonHandle:
    """Tracks whether we spawned the daemon and stores its process handle."""

    reused: bool
    process: Optional[subprocess.Popen[str]]
    log_path: Optional[Path]
    _log_file: Optional[IO[str]]

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the daemon if we spawned it."""

        if self.reused or self.process is None:
            self._close_log()
            return
        if self.process.poll() is not None:
            self._close_log()
            return

        LOGGER.debug("Stopping trainer daemon (pid=%s)", self.process.pid)
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            LOGGER.warning("Trainer daemon did not exit on SIGTERM – forcing kill", extra={"pid": self.process.pid})
            self.process.kill()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                LOGGER.error("Trainer daemon failed to terminate after SIGKILL", extra={"pid": self.process.pid})
        finally:
            self._close_log()

    def _close_log(self) -> None:
        if self._log_file and not self._log_file.closed:
            try:
                self._log_file.flush()
            except Exception:
                pass
            self._log_file.close()


def ensure_trainer_daemon_running(
    *,
    target: str = _DEFAULT_TARGET,
    python_executable: Optional[str] = None,
    startup_timeout: float = 10.0,
) -> TrainerDaemonHandle:
    """Ensure the trainer daemon is accepting connections.

    Returns a handle that can be used to terminate the daemon on shutdown.
    Raises ``TrainerDaemonLaunchError`` if the daemon cannot be reached.
    """

    host, port = _split_target(target)
    if _is_port_open(host, port, timeout=1.0):
        LOGGER.debug("Trainer daemon already reachable at %s", target)
        return TrainerDaemonHandle(reused=True, process=None, log_path=None, _log_file=None)

    ensure_var_directories()
    VAR_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = VAR_LOGS_DIR / "trainer_daemon.log"
    log_file = log_path.open("a", encoding="utf-8")

    python_executable = python_executable or sys.executable
    env = os.environ.copy()
    env.setdefault("QT_DEBUG_PLUGINS", "0")

    LOGGER.info("Spawning trainer daemon via %s", python_executable)
    process = subprocess.Popen(
        [python_executable, "-m", "gym_gui.services.trainer_daemon"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    handle = TrainerDaemonHandle(reused=False, process=process, log_path=log_path, _log_file=log_file)
    try:
        _wait_for_daemon_ready(process, host, port, startup_timeout)
    except TrainerDaemonLaunchError:
        handle.stop()
        raise
    LOGGER.info("Trainer daemon is accepting connections at %s", target)
    return handle


def _split_target(target: str) -> tuple[str, int]:
    host, sep, port = target.rpartition(":")
    if not sep:
        raise TrainerDaemonLaunchError(f"Invalid gRPC target '{target}'")
    try:
        port_num = int(port)
    except ValueError as exc:  # pragma: no cover - defensive
        raise TrainerDaemonLaunchError(f"Invalid trainer daemon port '{port}'") from exc
    return host or "127.0.0.1", port_num


def _is_port_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_daemon_ready(
    process: subprocess.Popen[str],
    host: str,
    port: int,
    timeout: float,
) -> None:
    """Poll until the daemon port opens or the timeout elapses."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise TrainerDaemonLaunchError(
                "Trainer daemon exited before becoming ready. "
                "Check var/logs/trainer_daemon.log for details."
            )
        if _is_port_open(host, port, timeout=_POLL_INTERVAL):
            return
        time.sleep(_POLL_INTERVAL)

    raise TrainerDaemonLaunchError(
        "Trainer daemon did not become ready in time. "
        "See var/logs/trainer_daemon.log for diagnostics."
    )


__all__ = [
    "TrainerDaemonHandle",
    "TrainerDaemonLaunchError",
    "ensure_trainer_daemon_running",
]
