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
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_VLLM_SERVER_COUNT_CHANGED,
    LOG_VLLM_SERVER_STARTING,
    LOG_VLLM_SERVER_RUNNING,
    LOG_VLLM_SERVER_STOPPING,
    LOG_VLLM_SERVER_START_FAILED,
    LOG_VLLM_SERVER_PROCESS_EXITED,
    LOG_VLLM_SERVER_NOT_RESPONDING,
    LOG_VLLM_ORPHAN_PROCESS_KILLED,
    LOG_VLLM_GPU_MEMORY_FREED,
    LOG_VLLM_GPU_MEMORY_NOT_FREED,
)

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


@dataclass
class ServerRowWidgets:
    """Widgets for a single server row in the grid."""

    server_id: int
    server_label: QtWidgets.QLabel
    port_label: QtWidgets.QLabel
    model_combo: QtWidgets.QComboBox
    status_label: QtWidgets.QLabel
    start_btn: QtWidgets.QPushButton
    stop_btn: QtWidgets.QPushButton


class VLLMServerWidget(QtWidgets.QGroupBox):
    """Widget for managing vLLM servers for operators.

    Shows per-operator vLLM servers with status indicators and controls.
    Each server runs on a different port (8000, 8001, etc.) and uses
    GPU memory utilization limits to allow multiple servers on one GPU.
    """

    # Emitted when a server's status changes (server_id, status, base_url)
    server_status_changed = pyqtSignal(int, str, str)

    # Grid column indices
    COL_SERVER = 0
    COL_PORT = 1
    COL_MODEL = 2
    COL_STATUS = 3
    COL_START = 4
    COL_STOP = 5

    # Row offset for server rows (after header row)
    HEADER_ROW = 0
    SERVER_ROW_OFFSET = 1

    def __init__(
        self,
        max_servers: int = 2,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__("vLLM Servers (Local Inference)", parent)
        self._max_servers = max(1, min(8, max_servers))  # Clamp 1-8
        self._current_server_count = self._max_servers
        self._server_rows: Dict[int, ServerRowWidgets] = {}
        self._server_states: Dict[int, VLLMServerState] = {}
        self._processes: Dict[int, subprocess.Popen] = {}

        # Track GPU memory usage per server (measured after loading)
        self._server_gpu_usage: Dict[int, float] = {}  # server_id -> GB used

        # Timer for checking server health
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_server_health)

        # Info label reference for dynamic updates
        self._info_label: Optional[QtWidgets.QLabel] = None

        self._build_ui()

    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in GB."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = info.total / (1024**3)
            pynvml.nvmlShutdown()
            return total_gb
        except Exception:
            return 16.0  # Default assumption

    def _get_gpu_free_memory(self) -> float:
        """Get free GPU memory in GB."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = info.free / (1024**3)
            pynvml.nvmlShutdown()
            return free_gb
        except Exception:
            return 0.0

    def _get_gpu_info_text(self) -> str:
        """Get the GPU memory info text for the header."""
        per_server_pct = (0.8 / max(1, self._current_server_count)) * 100
        total_used = sum(self._server_gpu_usage.values())
        free_gb = self._get_gpu_free_memory()
        if total_used > 0:
            return f"GPU: {total_used:.1f}GB used, {free_gb:.1f}GB free ({per_server_pct:.0f}%/server)"
        return f"GPU: {free_gb:.1f}GB free ({per_server_pct:.0f}% allocated per server)"

    def _update_info_label(self) -> None:
        """Update the info label with current GPU usage."""
        if self._info_label:
            self._info_label.setText(self._get_gpu_info_text())

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 12, 8, 8)
        main_layout.setSpacing(8)

        # Header row with info and server count spinbox
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(8)

        self._info_label = QtWidgets.QLabel(self._get_gpu_info_text(), self)
        self._info_label.setStyleSheet("color: #666; font-size: 11px;")
        header_layout.addWidget(self._info_label)

        header_layout.addStretch()

        # Server count spinbox
        servers_label = QtWidgets.QLabel("Servers:", self)
        servers_label.setStyleSheet("color: #666;")
        header_layout.addWidget(servers_label)

        self._server_count_spinbox = QtWidgets.QSpinBox(self)
        self._server_count_spinbox.setRange(1, 8)
        self._server_count_spinbox.setValue(self._current_server_count)
        self._server_count_spinbox.setToolTip("Number of vLLM server instances (1-8)")
        self._server_count_spinbox.setFixedWidth(50)
        self._server_count_spinbox.valueChanged.connect(self._on_server_count_changed)
        header_layout.addWidget(self._server_count_spinbox)

        main_layout.addLayout(header_layout)

        # Grid for server rows
        self._grid_widget = QtWidgets.QWidget(self)
        self._grid_layout = QtWidgets.QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(6)
        self._grid_layout.setColumnStretch(self.COL_MODEL, 1)  # Model column expands

        # Add header row
        self._add_header_row()

        # Add server rows
        for i in range(1, self._current_server_count + 1):
            self._add_server_row(i)

        main_layout.addWidget(self._grid_widget)

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

        main_layout.addLayout(btn_layout)

    def _add_header_row(self) -> None:
        """Add the header row to the grid."""
        header_style = "font-weight: bold; color: #444; font-size: 11px;"

        server_header = QtWidgets.QLabel("Server", self)
        server_header.setStyleSheet(header_style)
        self._grid_layout.addWidget(server_header, self.HEADER_ROW, self.COL_SERVER)

        port_header = QtWidgets.QLabel("Port", self)
        port_header.setStyleSheet(header_style)
        self._grid_layout.addWidget(port_header, self.HEADER_ROW, self.COL_PORT)

        model_header = QtWidgets.QLabel("Model", self)
        model_header.setStyleSheet(header_style)
        self._grid_layout.addWidget(model_header, self.HEADER_ROW, self.COL_MODEL)

        status_header = QtWidgets.QLabel("Status", self)
        status_header.setStyleSheet(header_style)
        self._grid_layout.addWidget(status_header, self.HEADER_ROW, self.COL_STATUS)

        actions_header = QtWidgets.QLabel("Actions", self)
        actions_header.setStyleSheet(header_style)
        self._grid_layout.addWidget(actions_header, self.HEADER_ROW, self.COL_START, 1, 2)

    def _add_server_row(self, server_id: int) -> None:
        """Add a server row to the grid."""
        row = self.SERVER_ROW_OFFSET + server_id - 1
        port = VLLM_BASE_PORT + server_id - 1

        # Server label
        server_label = QtWidgets.QLabel(f"{server_id}", self)
        server_label.setStyleSheet("font-weight: bold;")
        server_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._grid_layout.addWidget(server_label, row, self.COL_SERVER)

        # Port label
        port_label = QtWidgets.QLabel(f":{port}", self)
        port_label.setStyleSheet("color: #666;")
        self._grid_layout.addWidget(port_label, row, self.COL_PORT)

        # Model dropdown
        model_combo = QtWidgets.QComboBox(self)
        model_combo.setMinimumWidth(150)
        model_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed
        )
        model_combo.setToolTip("Select model to load")
        self._populate_model_combo(model_combo)
        self._grid_layout.addWidget(model_combo, row, self.COL_MODEL)

        # Status label (combined indicator + text)
        status_label = QtWidgets.QLabel("○ Stopped", self)
        status_label.setStyleSheet("color: #666;")
        status_label.setMinimumWidth(80)
        self._grid_layout.addWidget(status_label, row, self.COL_STATUS)

        # Start button
        start_btn = QtWidgets.QPushButton("Start", self)
        start_btn.setFixedWidth(55)
        start_btn.setToolTip("Start vLLM server with selected model")
        start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 3px; padding: 4px 8px; }"
            "QPushButton:hover { background-color: #388E3C; }"
            "QPushButton:disabled { background-color: #A5D6A7; color: #E8F5E9; }"
        )
        start_btn.clicked.connect(lambda checked, sid=server_id: self._on_start_requested(sid))
        self._grid_layout.addWidget(start_btn, row, self.COL_START)

        # Stop button
        stop_btn = QtWidgets.QPushButton("Stop", self)
        stop_btn.setFixedWidth(55)
        stop_btn.setToolTip("Stop vLLM server")
        stop_btn.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; border-radius: 3px; padding: 4px 8px; }"
            "QPushButton:hover { background-color: #D32F2F; }"
            "QPushButton:disabled { background-color: #EF9A9A; color: #FFEBEE; }"
        )
        stop_btn.setEnabled(False)
        stop_btn.clicked.connect(lambda checked, sid=server_id: self._on_stop_requested(sid))
        self._grid_layout.addWidget(stop_btn, row, self.COL_STOP)

        # Store widgets
        self._server_rows[server_id] = ServerRowWidgets(
            server_id=server_id,
            server_label=server_label,
            port_label=port_label,
            model_combo=model_combo,
            status_label=status_label,
            start_btn=start_btn,
            stop_btn=stop_btn,
        )

        # Initialize state
        self._server_states[server_id] = VLLMServerState(
            server_id=server_id,
            port=port,
        )

    def _remove_server_row(self, server_id: int) -> None:
        """Remove a server row from the grid."""
        if server_id not in self._server_rows:
            return

        # Stop server if running
        if server_id in self._processes:
            self._stop_server(server_id)

        # Remove widgets from grid and delete them
        row_widgets = self._server_rows[server_id]
        for widget in [
            row_widgets.server_label,
            row_widgets.port_label,
            row_widgets.model_combo,
            row_widgets.status_label,
            row_widgets.start_btn,
            row_widgets.stop_btn,
        ]:
            self._grid_layout.removeWidget(widget)
            widget.deleteLater()

        # Clean up state
        del self._server_rows[server_id]
        del self._server_states[server_id]

    def _on_server_count_changed(self, new_count: int) -> None:
        """Handle server count spinbox change."""
        if new_count == self._current_server_count:
            return

        # Check if any servers to be removed are running
        if new_count < self._current_server_count:
            running_to_remove = [
                sid for sid in range(new_count + 1, self._current_server_count + 1)
                if sid in self._processes
            ]
            if running_to_remove:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Stop Running Servers?",
                    f"Server(s) {', '.join(map(str, running_to_remove))} are running. "
                    "Stop them to reduce server count?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    # Revert spinbox
                    self._server_count_spinbox.blockSignals(True)
                    self._server_count_spinbox.setValue(self._current_server_count)
                    self._server_count_spinbox.blockSignals(False)
                    return

        self.set_server_count(new_count)

    def set_server_count(self, count: int) -> None:
        """Dynamically adjust the number of server rows.

        Args:
            count: New number of servers (1-8)
        """
        count = max(1, min(8, count))

        if count == self._current_server_count:
            return

        if count > self._current_server_count:
            # Add new rows
            for i in range(self._current_server_count + 1, count + 1):
                self._add_server_row(i)
        else:
            # Remove rows (from highest to lowest)
            for i in range(self._current_server_count, count, -1):
                self._remove_server_row(i)

        self._current_server_count = count

        # Update spinbox if called programmatically
        self._server_count_spinbox.blockSignals(True)
        self._server_count_spinbox.setValue(count)
        self._server_count_spinbox.blockSignals(False)

        # Update info label to show new per-server allocation
        self._update_info_label()

        per_server_pct = (0.8 / max(1, count)) * 100
        log_constant(
            _LOGGER, LOG_VLLM_SERVER_COUNT_CHANGED,
            message=f"Server count changed to {count}, {per_server_pct:.0f}% GPU per server",
            extra={"server_count": count, "per_server_gpu_pct": per_server_pct},
        )

    def _populate_model_combo(self, combo: QtWidgets.QComboBox) -> None:
        """Populate a model dropdown with locally available models."""
        combo.clear()
        combo.addItem("(Select model)", None)

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

            combo.addItem(display_name, {"model_id": model_id, "model_path": model_path})

    def _update_row_state(self, server_id: int, state: VLLMServerState) -> None:
        """Update the UI for a server row to reflect its state."""
        if server_id not in self._server_rows:
            return

        row = self._server_rows[server_id]
        self._server_states[server_id] = state

        # Update status label and buttons based on state
        if state.status == "running":
            status_text = "● Running"
            if state.memory_gb > 0:
                status_text += f" ({state.memory_gb:.1f}GB)"
            row.status_label.setText(status_text)
            row.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            row.start_btn.setEnabled(False)
            row.stop_btn.setEnabled(True)
            row.model_combo.setEnabled(False)
        elif state.status == "starting":
            row.status_label.setText("◐ Starting...")
            row.status_label.setStyleSheet("color: #FF9800;")
            row.start_btn.setEnabled(False)
            row.stop_btn.setEnabled(False)
            row.model_combo.setEnabled(False)
        elif state.status == "error":
            row.status_label.setText("✕ Error")
            row.status_label.setStyleSheet("color: #F44336;")
            row.status_label.setToolTip(state.error_message or "Unknown error")
            row.start_btn.setEnabled(True)
            row.stop_btn.setEnabled(False)
            row.model_combo.setEnabled(True)
        else:  # stopped
            row.status_label.setText("○ Stopped")
            row.status_label.setStyleSheet("color: #666;")
            row.status_label.setToolTip("")
            row.start_btn.setEnabled(True)
            row.stop_btn.setEnabled(False)
            row.model_combo.setEnabled(True)

    def _on_start_requested(self, server_id: int) -> None:
        """Handle start request for a specific server."""
        self._start_server(server_id)

    def _on_stop_requested(self, server_id: int) -> None:
        """Handle stop request for a specific server."""
        self._stop_server(server_id)

    def _on_start_all(self) -> None:
        """Start all servers that have models selected."""
        for server_id, row_widgets in self._server_rows.items():
            model_info = row_widgets.model_combo.currentData()
            if model_info and self._server_states[server_id].status == "stopped":
                self._start_server(server_id)

    def _on_stop_all(self) -> None:
        """Stop all running servers."""
        for server_id in list(self._processes.keys()):
            self._stop_server(server_id)

    def _refresh_models(self) -> None:
        """Refresh model list in all server rows."""
        for server_id, row_widgets in self._server_rows.items():
            if self._server_states[server_id].status == "stopped":
                self._populate_model_combo(row_widgets.model_combo)

    def _start_server(self, server_id: int) -> None:
        """Start a vLLM server for the given server ID."""
        row_widgets = self._server_rows.get(server_id)
        if not row_widgets:
            return

        model_info = row_widgets.model_combo.currentData()
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
        self._update_row_state(server_id, state)

        # Build vLLM command
        port = VLLM_BASE_PORT + server_id - 1
        model_path = model_info["model_path"]
        model_id = model_info["model_id"]

        # Record GPU memory before starting to calculate delta later
        gpu_before = self._get_gpu_free_memory()
        self._server_gpu_usage[server_id] = 0.0  # Will be updated after loading

        # Calculate per-server GPU allocation
        # vLLM pre-allocates for KV cache, so we MUST divide GPU among servers
        # Use 0.8 total (leaving 20% for system), divided by server count
        # e.g., 2 servers = 0.4 each, 3 servers = 0.27 each
        per_server_gpu = 0.8 / max(1, self._current_server_count)

        # Use served-model-name for proper API compatibility
        cmd = [
            "vllm", "serve", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--served-model-name", model_id,
            "--gpu-memory-utilization", str(per_server_gpu),
            "--enforce-eager",  # More stable, slightly slower
        ]

        # Store GPU baseline for measuring actual usage
        state.memory_gb = gpu_before  # Temporarily store baseline

        log_constant(
            _LOGGER, LOG_VLLM_SERVER_STARTING,
            message=f"Starting vLLM server {server_id} on port {port} (GPU alloc: {per_server_gpu*100:.0f}%)",
            extra={
                "server_id": server_id,
                "port": port,
                "model_id": model_id,
                "gpu_alloc_pct": per_server_gpu * 100,
                "command": " ".join(cmd),
            },
        )

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
            log_constant(
                _LOGGER, LOG_VLLM_SERVER_START_FAILED,
                message=f"Failed to start vLLM server {server_id}: {e}",
                extra={"server_id": server_id, "error": str(e)},
            )
            state.status = "error"
            state.error_message = str(e)
            self._update_row_state(server_id, state)

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

        log_constant(
            _LOGGER, LOG_VLLM_SERVER_STOPPING,
            message=f"Decommissioning vLLM server {server_id} (PID: {process.pid})",
            extra={"server_id": server_id, "pid": process.pid},
        )

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
        self._server_gpu_usage.pop(server_id, None)
        state = self._server_states.get(server_id)
        if state:
            state.status = "stopped"
            state.process = None
            state.memory_gb = 0.0
            state.error_message = None
            self._update_row_state(server_id, state)

        # Update GPU info label
        self._update_info_label()

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
                    log_constant(
                        _LOGGER, LOG_VLLM_ORPHAN_PROCESS_KILLED,
                        message=f"Killing orphan vLLM process: {proc.pid} ({name})",
                        extra={"pid": proc.pid, "process_name": name, "server_id": server_id},
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    killed_count += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if killed_count > 0:
            log_constant(
                _LOGGER, LOG_VLLM_ORPHAN_PROCESS_KILLED,
                message=f"Killed {killed_count} orphan vLLM process(es) for server {server_id}",
                extra={"killed_count": killed_count, "server_id": server_id},
            )
            # Give GPU memory time to be freed
            time.sleep(1)

    def _verify_gpu_memory_freed(self, server_id: int, max_retries: int = 3) -> bool:
        """Verify that GPU memory has been freed after stopping a server.

        Returns True if GPU memory appears to be freed, False otherwise.
        """
        # Get the amount of memory this server was using
        expected_freed = self._server_gpu_usage.get(server_id, 0.0)
        if expected_freed <= 0:
            return True  # No tracked usage, assume freed

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get initial free memory
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            initial_free = info.free / (1024**3)

            for attempt in range(max_retries):
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_gb = info.free / (1024**3)
                total_gb = info.total / (1024**3)

                _LOGGER.debug(
                    f"GPU memory check (attempt {attempt + 1}): "
                    f"{free_gb:.1f}GB free / {total_gb:.1f}GB total"
                )

                # Consider memory freed if we gained back most of what was used
                if free_gb >= initial_free + (expected_freed * 0.8):
                    log_constant(
                        _LOGGER, LOG_VLLM_GPU_MEMORY_FREED,
                        message=f"GPU memory freed for server {server_id}: {free_gb:.1f}GB now free",
                        extra={"server_id": server_id, "free_gb": free_gb, "total_gb": total_gb},
                    )
                    pynvml.nvmlShutdown()
                    return True

                # Wait and retry
                time.sleep(1)

            log_constant(
                _LOGGER, LOG_VLLM_GPU_MEMORY_NOT_FREED,
                message=f"GPU memory may not be fully freed for server {server_id}: {free_gb:.1f}GB free",
                extra={"server_id": server_id, "free_gb": free_gb},
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
            self._update_row_state(server_id, state)
            self._processes.pop(server_id, None)
            return

        # Try to connect to the server
        import urllib.request
        port = VLLM_BASE_PORT + server_id - 1
        url = f"http://127.0.0.1:{port}/health"

        try:
            response = urllib.request.urlopen(url, timeout=2)
            if response.status == 200:
                # Measure actual GPU memory used by this server
                gpu_before = state.memory_gb  # Baseline stored at start
                gpu_after = self._get_gpu_free_memory()
                actual_usage = max(0.0, gpu_before - gpu_after)

                state.status = "running"
                state.memory_gb = actual_usage
                self._server_gpu_usage[server_id] = actual_usage
                self._update_row_state(server_id, state)
                self._update_info_label()

                # Emit status change with base URL
                base_url = f"http://127.0.0.1:{port}/v1"
                self.server_status_changed.emit(server_id, "running", base_url)
                log_constant(
                    _LOGGER, LOG_VLLM_SERVER_RUNNING,
                    message=f"vLLM server {server_id} running on port {port}, using {actual_usage:.1f}GB GPU",
                    extra={"server_id": server_id, "port": port, "gpu_usage_gb": actual_usage},
                )
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
                    # Measure actual GPU memory used
                    gpu_before = state.memory_gb  # Baseline stored at start
                    gpu_after = self._get_gpu_free_memory()
                    actual_usage = max(0.0, gpu_before - gpu_after)

                    state.status = "running"
                    state.memory_gb = actual_usage
                    state.error_message = None
                    self._server_gpu_usage[server_id] = actual_usage
                    self._update_row_state(server_id, state)
                    self._update_info_label()

                    base_url = f"http://127.0.0.1:{port}/v1"
                    self.server_status_changed.emit(server_id, "running", base_url)
                    log_constant(
                        _LOGGER, LOG_VLLM_SERVER_RUNNING,
                        message=f"vLLM server {server_id} running on port {port}, using {actual_usage:.1f}GB GPU",
                        extra={"server_id": server_id, "port": port, "gpu_usage_gb": actual_usage},
                    )
            else:
                # Server not responding - check if process is dead
                process = self._processes.get(server_id)
                if process and process.poll() is not None:
                    # Process exited and health endpoint not responding
                    state.status = "error"
                    state.error_message = "Server process exited"
                    self._update_row_state(server_id, state)
                    self._processes.pop(server_id, None)
                    self.server_status_changed.emit(server_id, "error", "")
                    log_constant(
                        _LOGGER, LOG_VLLM_SERVER_PROCESS_EXITED,
                        message=f"vLLM server {server_id} process exited",
                        extra={"server_id": server_id},
                    )
                elif state.status == "running":
                    # Was running but now not responding
                    state.status = "error"
                    state.error_message = "Server not responding"
                    self._update_row_state(server_id, state)
                    self.server_status_changed.emit(server_id, "error", "")
                    log_constant(
                        _LOGGER, LOG_VLLM_SERVER_NOT_RESPONDING,
                        message=f"vLLM server {server_id} stopped responding",
                        extra={"server_id": server_id},
                    )

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
    "ServerRowWidgets",
    "VLLMServerState",
    "VLLM_BASE_PORT",
]
