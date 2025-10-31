from __future__ import annotations

"""Qt widget that surfaces TensorBoard artifact locations inside the GUI."""

from pathlib import Path
from typing import Iterable
import logging
import socket
import subprocess
import urllib.error
import urllib.request

from qtpy import QtCore, QtGui, QtWidgets

try:  # Optional dependency for embedded browser support
    from qtpy.QtWebEngineWidgets import QWebEngineView  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional feature not available in tests
    QWebEngineView = None

from gym_gui.logging_config.helpers import LogConstantMixin


_LOGGER = logging.getLogger(__name__)


class TensorboardArtifactTab(QtWidgets.QWidget, LogConstantMixin):
    """Present a TensorBoard log directory with quick actions."""

    _STATUS_REFRESH_MS = 4000

    def __init__(self, run_id: str, agent_id: str, log_dir: Path, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self._run_id = run_id
        self._agent_id = agent_id
        self._log_dir = Path(log_dir)
        self._cli_command = self._build_cli_command(self._log_dir)
        self._status_timer = QtCore.QTimer(self)
        self._path_field: QtWidgets.QLineEdit | None = None
        self._command_field: QtWidgets.QLineEdit | None = None
        self._status_label: QtWidgets.QLabel | None = None
        self._embedded_button: QtWidgets.QPushButton | None = None
        self._web_container: QtWidgets.QGroupBox | None = None
        self._web_placeholder: QtWidgets.QLabel | None = None
        self._web_view: QWebEngineView | None = None  # type: ignore[assignment]
        self._tensorboard_process: subprocess.Popen[str] | None = None
        self._tensorboard_port: int | None = None
        self._server_probe_timer: QtCore.QTimer | None = None
        self._setup_ui()
        self._configure_timer()
        self._refresh_status()

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

        header = QtWidgets.QLabel(
            f"TensorBoard metrics for run {self._run_id[:12]}… (agent {self._agent_id})"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        layout.addWidget(QtWidgets.QLabel("Log directory"))

        path_field = QtWidgets.QLineEdit(str(self._log_dir))
        path_field.setReadOnly(True)
        path_field.setCursorPosition(0)
        path_field.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        self._path_field = path_field
        layout.addWidget(path_field)

        button_row = QtWidgets.QHBoxLayout()
        copy_path_btn = QtWidgets.QPushButton("Copy Path")
        copy_path_btn.clicked.connect(lambda: self._copy_to_clipboard(str(self._log_dir)))
        button_row.addWidget(copy_path_btn)

        open_folder_btn = QtWidgets.QPushButton("Open Folder")
        open_folder_btn.clicked.connect(self._open_folder)
        button_row.addWidget(open_folder_btn)

        copy_cmd_btn = QtWidgets.QPushButton("Copy CLI Command")
        copy_cmd_btn.clicked.connect(lambda: self._copy_to_clipboard(self._cli_command))
        button_row.addWidget(copy_cmd_btn)

        button_row.addStretch(1)
        layout.addLayout(button_row)

        status_label = QtWidgets.QLabel()
        status_label.setWordWrap(True)
        self._status_label = status_label
        layout.addWidget(status_label)

        instructions_text = "Launch TensorBoard using the CLI command below or open the embedded viewer."
        if QWebEngineView is None:
            instructions_text = (
                "Launch TensorBoard using the command below and open http://localhost:6006 in your browser."
            )
        instructions = QtWidgets.QLabel(instructions_text)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        if QWebEngineView is not None:
            launch_btn = QtWidgets.QPushButton("Open Embedded TensorBoard")
            launch_btn.clicked.connect(self._launch_embedded_viewer)
            layout.addWidget(launch_btn)
            self._embedded_button = launch_btn

        command_field = QtWidgets.QLineEdit(self._cli_command)
        command_field.setReadOnly(True)
        command_field.setCursorPosition(0)
        mono_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        command_field.setFont(mono_font)
        command_field.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        self._command_field = command_field
        layout.addWidget(command_field)

        if QWebEngineView is not None:
            web_group = QtWidgets.QGroupBox("Embedded Viewer")
            web_layout = QtWidgets.QVBoxLayout(web_group)
            placeholder = QtWidgets.QLabel(
                "TensorBoard will open here once the embedded server is ready."
            )
            placeholder.setWordWrap(True)
            web_layout.addWidget(placeholder)
            web_group.setVisible(False)
            layout.addWidget(web_group)
            self._web_container = web_group
            self._web_placeholder = placeholder

        layout.addStretch(1)

    # ------------------------------------------------------------------
    def _configure_timer(self) -> None:
        self._status_timer.setInterval(self._STATUS_REFRESH_MS)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

    # ------------------------------------------------------------------
    def _refresh_status(self) -> None:
        exists = self._log_dir.exists()
        event_count = self._count_event_files(self._log_dir) if exists else 0
        if self._status_label is not None:
            if exists:
                message = (
                    f"Found TensorBoard directory at {self._log_dir}. "
                    f"Detected {event_count} event file(s)."
                )
            else:
                message = (
                    f"Waiting for logs… Directory {self._log_dir} does not exist yet. "
                    "It will appear once the worker emits TensorBoard data."
                )
            self._status_label.setText(message)

    # ------------------------------------------------------------------
    @staticmethod
    def _count_event_files(root: Path) -> int:
        def iter_events() -> Iterable[Path]:
            pattern = "events.out.tfevents.*"
            for candidate in root.rglob(pattern):
                yield candidate

        return sum(1 for _ in iter_events())

    # ------------------------------------------------------------------
    def _copy_to_clipboard(self, value: str) -> None:
        QtWidgets.QApplication.clipboard().setText(value)

    # ------------------------------------------------------------------
    def _open_folder(self) -> None:
        if not self._log_dir.exists():
            QtWidgets.QMessageBox.information(
                self,
                "TensorBoard Log Directory",
                f"Directory does not exist yet:\n{self._log_dir}",
            )
            return
        url = QtCore.QUrl.fromLocalFile(str(self._log_dir))
        if not QtGui.QDesktopServices.openUrl(url):
            QtWidgets.QMessageBox.warning(
                self,
                "TensorBoard Log Directory",
                "Could not open the directory with the default file browser.",
            )

    # ------------------------------------------------------------------
    def set_log_dir(self, log_dir: Path | str) -> None:
        self._log_dir = Path(log_dir)
        self._cli_command = self._build_cli_command(self._log_dir)
        if self._path_field is not None:
            self._path_field.setText(str(self._log_dir))
        if self._command_field is not None:
            self._command_field.setText(self._cli_command)
            self._command_field.setCursorPosition(0)
        self._refresh_status()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self._refresh_status()
        if self._web_view is not None and self._tensorboard_port is not None:
            self._web_view.setUrl(QtCore.QUrl(f"http://127.0.0.1:{self._tensorboard_port}/"))

    # ------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt slot signature
        if self._status_timer.isActive():
            self._status_timer.stop()
        if self._server_probe_timer is not None and self._server_probe_timer.isActive():
            self._server_probe_timer.stop()
        if self._tensorboard_process is not None:
            self._tensorboard_process.terminate()
            try:
                self._tensorboard_process.wait(timeout=5)
            except Exception:  # pragma: no cover - defensive
                self._tensorboard_process.kill()
        self._tensorboard_process = None
        super().closeEvent(event)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_cli_command(log_dir: Path) -> str:
        return f"tensorboard --logdir \"{log_dir}\""

    # ------------------------------------------------------------------
    def _launch_embedded_viewer(self) -> None:
        if QWebEngineView is None:
            return
        if self._tensorboard_process is None:
            self._start_tensorboard_process()
        if self._tensorboard_process is None:
            return
        if self._server_probe_timer is None:
            timer = QtCore.QTimer(self)
            timer.setInterval(750)
            timer.timeout.connect(self._probe_tensorboard_server)
            self._server_probe_timer = timer
        if not self._server_probe_timer.isActive():
            self._server_probe_timer.start()
        if self._web_container is not None:
            self._web_container.setVisible(True)

    # ------------------------------------------------------------------
    def _start_tensorboard_process(self) -> None:
        port = self._tensorboard_port or self._allocate_port()
        command = [
            "tensorboard",
            "--logdir",
            str(self._log_dir),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(
                self,
                "TensorBoard",
                "Could not find the `tensorboard` executable. Install TensorBoard or adjust your PATH.",
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            QtWidgets.QMessageBox.warning(
                self,
                "TensorBoard",
                f"Failed to launch TensorBoard: {exc}",
            )
            return

        self._tensorboard_process = process
        self._tensorboard_port = port
        if self._status_label is not None:
            self._status_label.setText(
                f"Launching embedded TensorBoard on http://127.0.0.1:{port} …"
            )

    # ------------------------------------------------------------------
    def _probe_tensorboard_server(self) -> None:
        if self._tensorboard_port is None:
            return
        if self._tensorboard_process and self._tensorboard_process.poll() is not None:
            if self._status_label is not None:
                self._status_label.setText("TensorBoard process exited unexpectedly.")
            if self._server_probe_timer is not None:
                self._server_probe_timer.stop()
            return

        url = f"http://127.0.0.1:{self._tensorboard_port}/"
        try:
            urllib.request.urlopen(url, timeout=0.5)
        except urllib.error.URLError:
            return
        except Exception:
            return

        if self._server_probe_timer is not None and self._server_probe_timer.isActive():
            self._server_probe_timer.stop()

        if QWebEngineView is None:
            return

        if self._web_view is None:
            self._web_view = QWebEngineView(self)
            self._web_view.setUrl(QtCore.QUrl(url))
            if self._web_container is not None:
                layout = self._web_container.layout()
                if layout is not None:
                    if self._web_placeholder is not None:
                        layout.removeWidget(self._web_placeholder)
                        self._web_placeholder.deleteLater()
                        self._web_placeholder = None
                    layout.addWidget(self._web_view)
        else:
            self._web_view.setUrl(QtCore.QUrl(url))

        if self._status_label is not None:
            self._status_label.setText(
                f"Embedded TensorBoard running on http://127.0.0.1:{self._tensorboard_port}"
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _allocate_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]


__all__ = ["TensorboardArtifactTab"]
