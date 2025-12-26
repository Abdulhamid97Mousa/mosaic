"""vLLM Server management widget for per-operator local inference.

This module provides a widget for managing vLLM server instances that run
local LLM/VLM models for operators. Each operator can have its own dedicated
vLLM server running on a different port.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from PyQt6.QtCore import QTimer, pyqtSignal
from qtpy import QtCore, QtWidgets

from gym_gui.config.paths import VAR_MODELS_HF_CACHE, VAR_VLLM_DIR
from gym_gui.config.settings import get_settings

_LOGGER = logging.getLogger(__name__)

# Base port for vLLM servers (Server 1 = 8000, Server 2 = 8001, etc.)
VLLM_BASE_PORT = 8000


@dataclass
class VLLMServerState:
    """State of a vLLM server instance."""

    server_id: int  # 1-indexed (Server 1, Server 2, etc.)
    port: int
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    status: str = "stopped"  # "stopped", "starting", "running", "error"
    memory_gb: float = 0.0
    error_message: Optional[str] = None


class VLLMServerRow(QtWidgets.QWidget):
    """Single row representing one vLLM server."""

    start_requested = pyqtSignal(int)  # server_id
    stop_requested = pyqtSignal(int)  # server_id

    def __init__(
        self,
        server_id: int,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._server_id = server_id
        self._state = VLLMServerState(
            server_id=server_id,
            port=VLLM_BASE_PORT + server_id - 1,
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Server label
        self._server_label = QtWidgets.QLabel(f"Server {self._server_id}", self)
        self._server_label.setFixedWidth(60)
        self._server_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._server_label)

        # Port label
        self._port_label = QtWidgets.QLabel(f":{self._state.port}", self)
        self._port_label.setFixedWidth(45)
        self._port_label.setStyleSheet("color: #666;")
        layout.addWidget(self._port_label)

        # Model dropdown
        self._model_combo = QtWidgets.QComboBox(self)
        self._model_combo.setMinimumWidth(180)
        self._model_combo.setToolTip("Select model to load")
        self._populate_model_combo()
        layout.addWidget(self._model_combo)

        # Status indicator
        self._status_indicator = QtWidgets.QLabel("○", self)
        self._status_indicator.setFixedWidth(16)
        self._status_indicator.setStyleSheet("color: #888; font-size: 14px;")
        self._status_indicator.setToolTip("Server status")
        layout.addWidget(self._status_indicator)

        # Status text
        self._status_label = QtWidgets.QLabel("Stopped", self)
        self._status_label.setFixedWidth(60)
        self._status_label.setStyleSheet("color: #666;")
        layout.addWidget(self._status_label)

        # Memory usage
        self._memory_label = QtWidgets.QLabel("-", self)
        self._memory_label.setFixedWidth(50)
        self._memory_label.setStyleSheet("color: #888;")
        self._memory_label.setToolTip("GPU memory usage")
        layout.addWidget(self._memory_label)

        layout.addStretch()

        # Start button
        self._start_btn = QtWidgets.QPushButton("Start", self)
        self._start_btn.setFixedWidth(50)
        self._start_btn.setToolTip("Start vLLM server with selected model")
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 3px; padding: 3px 8px; }"
            "QPushButton:hover { background-color: #388E3C; }"
            "QPushButton:disabled { background-color: #A5D6A7; color: #E8F5E9; }"
        )
        self._start_btn.clicked.connect(lambda: self.start_requested.emit(self._server_id))
        layout.addWidget(self._start_btn)

        # Stop button
        self._stop_btn = QtWidgets.QPushButton("Stop", self)
        self._stop_btn.setFixedWidth(50)
        self._stop_btn.setToolTip("Stop vLLM server")
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; border-radius: 3px; padding: 3px 8px; }"
            "QPushButton:hover { background-color: #D32F2F; }"
            "QPushButton:disabled { background-color: #EF9A9A; color: #FFEBEE; }"
        )
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(lambda: self.stop_requested.emit(self._server_id))
        layout.addWidget(self._stop_btn)

    def _populate_model_combo(self) -> None:
        """Populate the model dropdown with locally available models."""
        self._model_combo.clear()
        self._model_combo.addItem("(Select model)", None)

        hf_cache = VAR_MODELS_HF_CACHE
        if not hf_cache.exists():
            return

        for model_dir in sorted(hf_cache.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_dir.name.startswith("."):
                continue
            # Skip non-model directories
            if model_dir.name in ("hub", "xet", "datasets"):
                continue

            dir_name = model_dir.name

            # Convert directory name to model ID
            if dir_name.startswith("models--"):
                remainder = dir_name[len("models--"):]
                if "--" in remainder:
                    org, model_name = remainder.split("--", 1)
                    model_id = f"{org}/{model_name}"
                else:
                    continue
            elif "--" in dir_name:
                org, model_name = dir_name.split("--", 1)
                model_id = f"{org}/{model_name}"
            else:
                model_id = dir_name
                model_name = dir_name

            # Store model path for vLLM
            model_path = str(model_dir)
            display_name = model_id.split("/")[-1] if "/" in model_id else model_id

            self._model_combo.addItem(display_name, {"model_id": model_id, "model_path": model_path})

    def update_state(self, state: VLLMServerState) -> None:
        """Update the UI to reflect the server state."""
        self._state = state

        # Update status indicator and text
        if state.status == "running":
            self._status_indicator.setText("●")
            self._status_indicator.setStyleSheet("color: #4CAF50; font-size: 14px;")
            self._status_label.setText("Running")
            self._status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._model_combo.setEnabled(False)
        elif state.status == "starting":
            self._status_indicator.setText("◐")
            self._status_indicator.setStyleSheet("color: #FF9800; font-size: 14px;")
            self._status_label.setText("Starting...")
            self._status_label.setStyleSheet("color: #FF9800;")
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            self._model_combo.setEnabled(False)
        elif state.status == "error":
            self._status_indicator.setText("✕")
            self._status_indicator.setStyleSheet("color: #F44336; font-size: 14px;")
            self._status_label.setText("Error")
            self._status_label.setStyleSheet("color: #F44336;")
            self._status_label.setToolTip(state.error_message or "Unknown error")
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._model_combo.setEnabled(True)
        else:  # stopped
            self._status_indicator.setText("○")
            self._status_indicator.setStyleSheet("color: #888; font-size: 14px;")
            self._status_label.setText("Stopped")
            self._status_label.setStyleSheet("color: #666;")
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._model_combo.setEnabled(True)

        # Update memory display
        if state.memory_gb > 0:
            self._memory_label.setText(f"{state.memory_gb:.1f}GB")
        else:
            self._memory_label.setText("-")

    def get_selected_model(self) -> Optional[Dict[str, str]]:
        """Get the currently selected model info."""
        data = self._model_combo.currentData()
        return data if data else None

    @property
    def server_id(self) -> int:
        return self._server_id

    @property
    def state(self) -> VLLMServerState:
        return self._state


class VLLMServerWidget(QtWidgets.QGroupBox):
    """Widget for managing vLLM servers for operators.

    Shows per-operator vLLM servers with status indicators and controls.
    Each server runs on a different port (8000, 8001, etc.) and uses
    GPU memory utilization limits to allow multiple servers on one GPU.
    """

    # Emitted when a server's status changes (server_id, status, base_url)
    server_status_changed = pyqtSignal(int, str, str)

    def __init__(
        self,
        max_servers: Optional[int] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__("vLLM Servers (Local Inference)", parent)
        settings = get_settings()
        self._max_servers = max_servers if max_servers is not None else settings.vllm_max_servers
        self._gpu_memory_utilization = settings.vllm_gpu_memory_utilization
        self._servers: Dict[int, VLLMServerRow] = {}
        self._server_states: Dict[int, VLLMServerState] = {}
        self._processes: Dict[int, subprocess.Popen] = {}

        # Timer for checking server health
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_server_health)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(4)

        # Info text
        info_label = QtWidgets.QLabel(
            f"<small>Each server uses ~{self._gpu_memory_utilization*100:.0f}% GPU memory. "
            f"Server 1 → Operator 1, Server 2 → Operator 2</small>",
            self
        )
        info_label.setStyleSheet("color: #666;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Server rows
        for i in range(1, self._max_servers + 1):
            row = VLLMServerRow(i, self)
            row.start_requested.connect(self._on_start_requested)
            row.stop_requested.connect(self._on_stop_requested)
            self._servers[i] = row
            self._server_states[i] = VLLMServerState(
                server_id=i,
                port=VLLM_BASE_PORT + i - 1,
            )
            layout.addWidget(row)

        # Control buttons row
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(8)

        btn_layout.addStretch()

        self._start_all_btn = QtWidgets.QPushButton("Start All", self)
        self._start_all_btn.setToolTip("Start all configured servers")
        self._start_all_btn.clicked.connect(self._on_start_all)
        btn_layout.addWidget(self._start_all_btn)

        self._stop_all_btn = QtWidgets.QPushButton("Stop All", self)
        self._stop_all_btn.setToolTip("Stop all running servers")
        self._stop_all_btn.clicked.connect(self._on_stop_all)
        btn_layout.addWidget(self._stop_all_btn)

        self._refresh_btn = QtWidgets.QPushButton("↻ Refresh", self)
        self._refresh_btn.setToolTip("Refresh model list from disk")
        self._refresh_btn.clicked.connect(self._refresh_models)
        btn_layout.addWidget(self._refresh_btn)

        layout.addLayout(btn_layout)

    def _on_start_requested(self, server_id: int) -> None:
        """Handle start request for a specific server."""
        self._start_server(server_id)

    def _on_stop_requested(self, server_id: int) -> None:
        """Handle stop request for a specific server."""
        self._stop_server(server_id)

    def _on_start_all(self) -> None:
        """Start all servers that have models selected."""
        for server_id, row in self._servers.items():
            model_info = row.get_selected_model()
            if model_info and self._server_states[server_id].status == "stopped":
                self._start_server(server_id)

    def _on_stop_all(self) -> None:
        """Stop all running servers."""
        for server_id in list(self._processes.keys()):
            self._stop_server(server_id)

    def _refresh_models(self) -> None:
        """Refresh model list in all server rows."""
        for row in self._servers.values():
            if row.state.status == "stopped":
                row._populate_model_combo()

    def _start_server(self, server_id: int) -> None:
        """Start a vLLM server for the given server ID."""
        row = self._servers.get(server_id)
        if not row:
            return

        model_info = row.get_selected_model()
        if not model_info:
            QtWidgets.QMessageBox.warning(
                self,
                "No Model Selected",
                f"Please select a model for Server {server_id} before starting."
            )
            return

        state = self._server_states[server_id]
        state.status = "starting"
        state.model_id = model_info["model_id"]
        state.model_path = model_info["model_path"]
        row.update_state(state)

        # Build vLLM command
        port = VLLM_BASE_PORT + server_id - 1
        model_path = model_info["model_path"]
        model_id = model_info["model_id"]

        # Use served-model-name for proper API compatibility
        cmd = [
            "vllm", "serve", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--served-model-name", model_id,
            "--gpu-memory-utilization", str(self._gpu_memory_utilization),
            "--enforce-eager",  # More stable, slightly slower
        ]

        _LOGGER.info(f"Starting vLLM server {server_id}: {' '.join(cmd)}")

        try:
            # Ensure log directory exists
            VAR_VLLM_DIR.mkdir(parents=True, exist_ok=True)
            log_path = VAR_VLLM_DIR / f"server_{server_id}.log"
            log_file = open(log_path, "w")

            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from terminal
            )

            self._processes[server_id] = process
            state.process = process

            # Start health check timer
            if not self._health_timer.isActive():
                self._health_timer.start(2000)  # Check every 2 seconds

            # Schedule a check for startup
            QTimer.singleShot(5000, lambda: self._check_server_startup(server_id))

        except Exception as e:
            _LOGGER.error(f"Failed to start vLLM server {server_id}: {e}")
            state.status = "error"
            state.error_message = str(e)
            row.update_state(state)

    def _stop_server(self, server_id: int) -> None:
        """Stop a vLLM server with full decommissioning.

        This method ensures all child processes are killed and GPU memory is freed.
        vLLM spawns child processes (EngineCore, etc.) that must be terminated.
        """
        process = self._processes.get(server_id)
        if not process:
            # Even if we don't have a tracked process, try to clean up any orphans
            self._kill_orphan_vllm_processes(server_id)
            return

        _LOGGER.info(f"Decommissioning vLLM server {server_id} (PID: {process.pid})")

        try:
            # First, kill all child processes using psutil (more reliable)
            self._kill_process_tree(process.pid)

            # Also try the process group approach as fallback
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass

        except (ProcessLookupError, OSError) as e:
            _LOGGER.debug(f"Process already terminated: {e}")

        # Kill any orphan vLLM processes that might still be using the port
        self._kill_orphan_vllm_processes(server_id)

        # Verify GPU memory is freed
        self._verify_gpu_memory_freed(server_id)

        # Clean up internal state
        self._processes.pop(server_id, None)
        state = self._server_states[server_id]
        state.status = "stopped"
        state.process = None
        state.memory_gb = 0.0
        state.error_message = None
        self._servers[server_id].update_state(state)

        # Emit status change
        self.server_status_changed.emit(server_id, "stopped", "")

        # Stop health timer if no servers running
        if not self._processes:
            self._health_timer.stop()

    def _kill_process_tree(self, pid: int) -> None:
        """Kill a process and all its children recursively."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Kill children first
            for child in children:
                try:
                    _LOGGER.debug(f"Killing child process {child.pid} ({child.name()})")
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Wait for children to terminate
            gone, alive = psutil.wait_procs(children, timeout=3)

            # Force kill any remaining
            for p in alive:
                try:
                    _LOGGER.debug(f"Force killing child process {p.pid}")
                    p.kill()
                except psutil.NoSuchProcess:
                    pass

            # Kill parent
            try:
                parent.terminate()
                parent.wait(timeout=3)
            except psutil.NoSuchProcess:
                pass
            except psutil.TimeoutExpired:
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

        except psutil.NoSuchProcess:
            _LOGGER.debug(f"Process {pid} already terminated")

    def _kill_orphan_vllm_processes(self, server_id: int) -> None:
        """Kill any orphan vLLM processes that might be using resources.

        This catches processes that escaped the normal shutdown, like
        EngineCore subprocesses that hold GPU memory.
        """
        port = VLLM_BASE_PORT + server_id - 1
        killed_count = 0

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                name = proc.info['name'] or ''
                cmdline = proc.info['cmdline'] or []
                cmdline_str = ' '.join(cmdline)

                # Look for vLLM-related processes
                is_vllm = (
                    'vllm' in name.lower() or
                    'VLLM' in name or
                    'EngineCore' in name or
                    'vllm' in cmdline_str.lower() or
                    f'--port {port}' in cmdline_str or
                    f':{port}' in cmdline_str
                )

                if is_vllm:
                    _LOGGER.info(f"Killing orphan vLLM process: {proc.pid} ({name})")
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    killed_count += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if killed_count > 0:
            _LOGGER.info(f"Killed {killed_count} orphan vLLM process(es) for server {server_id}")
            # Give GPU memory time to be freed
            time.sleep(1)

    def _verify_gpu_memory_freed(self, server_id: int, max_retries: int = 3) -> bool:
        """Verify that GPU memory has been freed after stopping a server.

        Returns True if GPU memory appears to be freed, False otherwise.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            for attempt in range(max_retries):
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_gb = info.free / (1024**3)
                total_gb = info.total / (1024**3)
                used_gb = info.used / (1024**3)

                _LOGGER.debug(
                    f"GPU memory check (attempt {attempt + 1}): "
                    f"{free_gb:.1f}GB free / {total_gb:.1f}GB total"
                )

                # Consider memory freed if we have reasonable free space
                expected_free = total_gb * (1 - self._gpu_memory_utilization * 0.5)
                if free_gb >= expected_free:
                    _LOGGER.info(f"GPU memory verified freed for server {server_id}")
                    pynvml.nvmlShutdown()
                    return True

                # Wait and retry
                time.sleep(1)

            _LOGGER.warning(
                f"GPU memory may not be fully freed for server {server_id}: "
                f"{free_gb:.1f}GB free"
            )
            pynvml.nvmlShutdown()
            return False

        except ImportError:
            _LOGGER.debug("pynvml not available, skipping GPU memory verification")
            return True
        except Exception as e:
            _LOGGER.debug(f"GPU memory verification failed: {e}")
            return True

    def _check_server_startup(self, server_id: int) -> None:
        """Check if server has started successfully."""
        state = self._server_states.get(server_id)
        if not state or state.status != "starting":
            return

        process = self._processes.get(server_id)
        if not process:
            return

        # Check if process is still running
        if process.poll() is not None:
            # Process exited
            state.status = "error"
            state.error_message = "Server process exited unexpectedly"
            self._servers[server_id].update_state(state)
            self._processes.pop(server_id, None)
            return

        # Try to connect to the server
        import urllib.request
        port = VLLM_BASE_PORT + server_id - 1
        url = f"http://127.0.0.1:{port}/health"

        try:
            response = urllib.request.urlopen(url, timeout=2)
            if response.status == 200:
                state.status = "running"
                state.memory_gb = self._gpu_memory_utilization * 16  # Estimated
                self._servers[server_id].update_state(state)

                # Emit status change with base URL
                base_url = f"http://127.0.0.1:{port}/v1"
                self.server_status_changed.emit(server_id, "running", base_url)
                _LOGGER.info(f"vLLM server {server_id} is running on port {port}")
        except Exception as e:
            # Server not ready yet, will be checked by health timer
            _LOGGER.debug(f"Server {server_id} startup check failed: {e}")

    def _check_server_health(self) -> None:
        """Periodically check health of all running servers.

        Uses HTTP health endpoint as primary indicator since vLLM spawns
        child processes (EngineCore) that may outlive the parent process.
        """
        import urllib.request

        for server_id, state in self._server_states.items():
            if state.status not in ("running", "starting"):
                continue

            port = VLLM_BASE_PORT + server_id - 1
            url = f"http://127.0.0.1:{port}/health"

            # Try health endpoint first (more reliable than process.poll())
            server_responding = False
            try:
                response = urllib.request.urlopen(url, timeout=2)
                server_responding = (response.status == 200)
            except Exception:
                server_responding = False

            if server_responding:
                # Server is healthy
                if state.status == "starting":
                    state.status = "running"
                    state.memory_gb = self._gpu_memory_utilization * 16
                    state.error_message = None
                    self._servers[server_id].update_state(state)
                    base_url = f"http://127.0.0.1:{port}/v1"
                    self.server_status_changed.emit(server_id, "running", base_url)
                    _LOGGER.info(f"vLLM server {server_id} is now running on port {port}")
            else:
                # Server not responding - check if process is dead
                process = self._processes.get(server_id)
                if process and process.poll() is not None:
                    # Process exited and health endpoint not responding
                    state.status = "error"
                    state.error_message = "Server process exited"
                    self._servers[server_id].update_state(state)
                    self._processes.pop(server_id, None)
                    self.server_status_changed.emit(server_id, "error", "")
                    _LOGGER.warning(f"vLLM server {server_id} process exited")
                elif state.status == "running":
                    # Was running but now not responding
                    state.status = "error"
                    state.error_message = "Server not responding"
                    self._servers[server_id].update_state(state)
                    self.server_status_changed.emit(server_id, "error", "")
                    _LOGGER.warning(f"vLLM server {server_id} stopped responding")

    def get_server_base_url(self, server_id: int) -> Optional[str]:
        """Get the base URL for a running server."""
        state = self._server_states.get(server_id)
        if state and state.status == "running":
            port = VLLM_BASE_PORT + server_id - 1
            return f"http://127.0.0.1:{port}/v1"
        return None

    def get_server_model_id(self, server_id: int) -> Optional[str]:
        """Get the model ID loaded on a server."""
        state = self._server_states.get(server_id)
        if state:
            return state.model_id
        return None

    def is_server_running(self, server_id: int) -> bool:
        """Check if a server is running."""
        state = self._server_states.get(server_id)
        return state is not None and state.status == "running"

    def cleanup(self) -> None:
        """Stop all servers on widget destruction."""
        self._health_timer.stop()
        self._on_stop_all()

    def closeEvent(self, event) -> None:
        """Handle widget close."""
        self.cleanup()
        super().closeEvent(event)


__all__ = [
    "VLLMServerWidget",
    "VLLMServerRow",
    "VLLMServerState",
    "VLLM_BASE_PORT",
    "GPU_MEMORY_UTILIZATION",
]
