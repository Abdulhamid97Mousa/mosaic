from __future__ import annotations

"""Qt widget that surfaces TensorBoard artifact locations inside the GUI."""

import logging
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from qtpy import QtCore, QtGui, QtWidgets  # type: ignore[import]
from PyQt6.QtCore import pyqtSlot, pyqtSignal

try:  # Optional dependency for embedded browser support
    from qtpy.QtWebEngineWidgets import QWebEngineView  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional feature not available in tests
    QWebEngineView = None

try:
    from gym_gui.ui.widgets.filtered_web_engine import FilteredWebEnginePage
except ImportError:  # pragma: no cover - optional feature
    FilteredWebEnginePage = None  # type: ignore[assignment, misc]

WEB_ENGINE_AVAILABLE = QWebEngineView is not None

from gym_gui.constants.constants_tensorboard import DEFAULT_TENSORBOARD
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_ERROR,
    LOG_UI_RENDER_TABS_INFO,
    LOG_UI_RENDER_TABS_TRACE,
    LOG_UI_RENDER_TABS_WARNING,
    LOG_UI_RENDER_TABS_TENSORBOARD_STATUS,
    LOG_UI_TENSORBOARD_KILL_WARNING,
    LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
)


_LOGGER = logging.getLogger(__name__)


class TensorboardArtifactTab(QtWidgets.QWidget, LogConstantMixin):
    """Present a TensorBoard log directory with quick actions."""

    statusChanged = pyqtSignal(str, int, bool)

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
        self._web_scroll_area: QtWidgets.QScrollArea | None = None
        self._web_container: QtWidgets.QGroupBox | None = None
        self._web_placeholder: QtWidgets.QLabel | None = None
        self._web_view: QWebEngineView | None = None  # type: ignore[assignment]
        self._tensorboard_process: subprocess.Popen[str] | None = None
        self._tensorboard_port: int | None = None
        self._server_probe_timer: QtCore.QTimer | None = None
        self._instructions_label: QtWidgets.QLabel | None = None
        self._nav_bar: QtWidgets.QWidget | None = None
        self._url_field: QtWidgets.QLineEdit | None = None
        self._nav_back_btn: QtWidgets.QAbstractButton | None = None
        self._nav_forward_btn: QtWidgets.QAbstractButton | None = None
        self._nav_reload_btn: QtWidgets.QAbstractButton | None = None
        self._nav_copy_btn: QtWidgets.QAbstractButton | None = None
        self._nav_external_btn: QtWidgets.QAbstractButton | None = None
        self._pending_launch = False
        self._last_status: tuple[str, int, bool] | None = None
        self.statusChanged.connect(self._handle_status_change)
        self._setup_ui()
        self._configure_timer()
        self._refresh_status()

    # ------------------------------------------------------------------
    def set_log_dir(self, log_dir: Path | str) -> None:
        """Update the TensorBoard log directory and refresh status."""

        new_path = Path(log_dir)
        if new_path == self._log_dir:
            return
        self._stop_tensorboard_process()
        self._log_dir = new_path
        self._cli_command = self._build_cli_command(self._log_dir)
        if self._path_field is not None:
            self._path_field.setText(str(self._log_dir))
            self._path_field.setCursorPosition(0)
        self._pending_launch = False
        self._last_status = None
        self._refresh_status()

    def refresh(self) -> None:
        self._refresh_status()

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.addStretch(1)
        details_toggle = QtWidgets.QToolButton()
        details_toggle.setText("Hide Details")
        details_toggle.setCheckable(True)
        details_toggle.setChecked(True)
        details_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        details_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        details_toggle.toggled.connect(self._toggle_details_section)
        toggle_row.addWidget(details_toggle)
        layout.addLayout(toggle_row)
        self._details_toggle = details_toggle

        details_container = QtWidgets.QWidget(self)
        details_layout = QtWidgets.QVBoxLayout(details_container)
        details_layout.setContentsMargins(12, 12, 12, 12)
        details_layout.setSpacing(6)
        layout.addWidget(details_container)
        self._details_container = details_container

        header = QtWidgets.QLabel(
            f"TensorBoard metrics for run {self._run_id[:12]}… ({self._agent_id})"
        )
        header.setWordWrap(True)
        details_layout.addWidget(header)

        details_layout.addWidget(QtWidgets.QLabel("Log directory"))

        path_field = QtWidgets.QLineEdit(str(self._log_dir))
        path_field.setReadOnly(True)
        path_field.setCursorPosition(0)
        path_field.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        self._path_field = path_field
        details_layout.addWidget(path_field)

        button_row = QtWidgets.QHBoxLayout()
        copy_path_btn = QtWidgets.QPushButton("Copy Path")
        copy_path_btn.clicked.connect(lambda: self._copy_to_clipboard(str(self._log_dir)))
        button_row.addWidget(copy_path_btn)

        open_folder_btn = QtWidgets.QPushButton("Open Folder")
        open_folder_btn.clicked.connect(self._open_folder)
        button_row.addWidget(open_folder_btn)

        copy_cmd_btn = QtWidgets.QPushButton("Copy CLI Command")
        copy_cmd_btn.clicked.connect(lambda: self._copy_to_clipboard(self._cli_command))
        if not WEB_ENGINE_AVAILABLE:
            button_row.addWidget(copy_cmd_btn)

        button_row.addStretch(1)
        details_layout.addLayout(button_row)

        status_label = QtWidgets.QLabel()
        status_label.setWordWrap(True)
        self._status_label = status_label
        details_layout.addWidget(status_label)

        if WEB_ENGINE_AVAILABLE:
            instructions = QtWidgets.QLabel(
                "Click 'Open Embedded TensorBoard' to launch TensorBoard inside the GUI."
            )
        else:
            instructions = QtWidgets.QLabel(
                "Launch TensorBoard using the command below or open it in your default browser."
            )
        instructions.setWordWrap(True)
        details_layout.addWidget(instructions)
        self._instructions_label = instructions

        if WEB_ENGINE_AVAILABLE:
            nav_bar = QtWidgets.QWidget()
            nav_layout = QtWidgets.QHBoxLayout(nav_bar)
            nav_layout.setContentsMargins(0, 0, 0, 0)
            nav_layout.setSpacing(6)
            back_btn = QtWidgets.QToolButton()
            back_btn.setText("←")
            back_btn.setEnabled(False)
            back_btn.clicked.connect(lambda: self._web_view.back() if self._web_view else None)
            forward_btn = QtWidgets.QToolButton()
            forward_btn.setText("→")
            forward_btn.setEnabled(False)
            forward_btn.clicked.connect(lambda: self._web_view.forward() if self._web_view else None)
            reload_btn = QtWidgets.QToolButton()
            reload_btn.setText("↻")
            reload_btn.setEnabled(False)
            reload_btn.clicked.connect(lambda: self._web_view.reload() if self._web_view else None)
            url_field = QtWidgets.QLineEdit(
                self._tensorboard_url(DEFAULT_TENSORBOARD.default_port)
            )
            url_field.setReadOnly(True)
            url_field.setCursorPosition(0)
            copy_btn = QtWidgets.QToolButton()
            copy_btn.setText("Copy URL")
            copy_btn.setEnabled(False)
            copy_btn.clicked.connect(lambda checked, field=url_field: self._copy_to_clipboard(field.text()))
            external_btn = QtWidgets.QToolButton()
            external_btn.setText("↗")
            external_btn.setEnabled(False)
            external_btn.setToolTip("Open in default browser")
            external_btn.clicked.connect(lambda checked, url_field=url_field: QtGui.QDesktopServices.openUrl(QtCore.QUrl(url_field.text())) if url_field else None)
            nav_layout.addWidget(back_btn)
            nav_layout.addWidget(forward_btn)
            nav_layout.addWidget(reload_btn)
            nav_layout.addWidget(url_field, 1)
            nav_layout.addWidget(copy_btn)
            nav_layout.addWidget(external_btn)
            details_layout.addWidget(nav_bar)
            self._nav_bar = nav_bar
            self._url_field = url_field
            self._nav_back_btn = back_btn
            self._nav_forward_btn = forward_btn
            self._nav_reload_btn = reload_btn
            self._nav_copy_btn = copy_btn
            self._nav_external_btn = external_btn
            launch_btn = QtWidgets.QPushButton("Open Embedded TensorBoard")
            launch_btn.clicked.connect(self._launch_embedded_viewer)
            details_layout.addWidget(launch_btn)
            self._embedded_button = launch_btn
            self._command_field = None
        else:
            command_field = QtWidgets.QLineEdit(self._cli_command)
            command_field.setReadOnly(True)
            command_field.setCursorPosition(0)
            mono_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
            command_field.setFont(mono_font)
            command_field.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
            self._command_field = command_field
            details_layout.addWidget(command_field)
            launch_btn = QtWidgets.QPushButton("Open TensorBoard in Browser")
            launch_btn.clicked.connect(self._launch_embedded_viewer)
            details_layout.addWidget(launch_btn)
            self._embedded_button = launch_btn

        if WEB_ENGINE_AVAILABLE:
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

            web_group = QtWidgets.QGroupBox("")
            web_group.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            web_layout = QtWidgets.QVBoxLayout(web_group)
            placeholder = QtWidgets.QLabel(
                "TensorBoard will open here once the embedded server is ready."
            )
            placeholder.setWordWrap(True)
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            web_layout.addWidget(placeholder, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            web_group.setVisible(False)

            scroll_area.setWidget(web_group)
            layout.addWidget(scroll_area, 1)
            self._web_scroll_area = scroll_area
            self._web_container = web_group
            self._web_placeholder = placeholder
        else:
            layout.addStretch(1)

    # ------------------------------------------------------------------
    def _configure_timer(self) -> None:
        self._status_timer.setInterval(DEFAULT_TENSORBOARD.status_refresh_ms)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

    # ------------------------------------------------------------------
    def _refresh_status(self) -> None:
        exists = self._log_dir.exists()
        event_count = self._count_event_files(self._log_dir) if exists else 0
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
        snapshot = (message, event_count, exists)
        if self._last_status != snapshot:
            self.statusChanged.emit(message, event_count, exists)
            self._last_status = snapshot
        if self._status_label is not None:
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
        QtWidgets.QApplication.clipboard().setText(value)  # type: ignore[attr-defined]

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
        self._last_status = None
        self._refresh_status()

    @pyqtSlot(bool)
    def _toggle_details_section(self, checked: bool) -> None:
        if hasattr(self, "_details_container") and self._details_container is not None:
            self._details_container.setVisible(checked)
        arrow = QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        self._details_toggle.setArrowType(arrow)
        self._details_toggle.setText("Hide Details" if checked else "Show Details")

    @pyqtSlot(str, int, bool)
    def _handle_status_change(self, message: str, event_count: int, exists: bool) -> None:
        constant = (
            LOG_UI_RENDER_TABS_TENSORBOARD_STATUS if exists else LOG_UI_RENDER_TABS_TENSORBOARD_WAITING
        )
        self.log_constant(
            constant,
            message=message,
            extra={
                "run_id": self._run_id,
                "agent_id": self._agent_id,
                "event_files": event_count,
                "log_dir": str(self._log_dir),
            },
        )

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self._refresh_status()
        if self._web_view is not None and self._tensorboard_port is not None:
            self._web_view.setUrl(QtCore.QUrl(self._tensorboard_url(self._tensorboard_port)))

    # ------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt slot signature
        if self._status_timer.isActive():
            self._status_timer.stop()
        if self._server_probe_timer is not None and self._server_probe_timer.isActive():
            self._server_probe_timer.stop()
        self._stop_tensorboard_process()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_cli_command(log_dir: Path) -> str:
        return (
            f"{DEFAULT_TENSORBOARD.cli_executable} --logdir \"{log_dir}\" "
            f"--host {DEFAULT_TENSORBOARD.server_host} "
            f"--port {DEFAULT_TENSORBOARD.default_port} "
            f"--reload_multifile=true"
        )

    # ------------------------------------------------------------------
    def _launch_embedded_viewer(self, *, auto: bool = False) -> None:
        current_thread = QtCore.QThread.currentThread()
        widget_thread = self.thread()
        self.log_constant(
            LOG_UI_RENDER_TABS_TRACE,
            message="TensorBoard launch requested",
            extra={
                "run_id": self._run_id,
                "agent_id": self._agent_id,
                "auto": auto,
                "current_thread": repr(current_thread),
                "widget_thread": repr(widget_thread),
            },
        )
        if QtCore.QThread.currentThread() == self.thread():
            self._queued_launch_embedded_viewer(auto)
            return
        queued = QtCore.QMetaObject.invokeMethod(
            self,
            "_queued_launch_embedded_viewer",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(bool, auto),
        )
        if not queued:
            self.log_constant(
                LOG_UI_RENDER_TABS_WARNING,
                message="Failed to queue TensorBoard launch on GUI thread",
                extra={"run_id": self._run_id, "agent_id": self._agent_id, "auto": auto},
            )
            # Fall back to direct execution to avoid dropping the request entirely.
            self.log_constant(
                LOG_UI_RENDER_TABS_TRACE,
                message="TensorBoard launch fallback executing directly",
                extra={"run_id": self._run_id, "agent_id": self._agent_id, "auto": auto},
            )
            self._queued_launch_embedded_viewer(auto)

    @pyqtSlot(bool)
    def _queued_launch_embedded_viewer(self, auto: bool) -> None:
        self.log_constant(
            LOG_UI_RENDER_TABS_TRACE,
            message="TensorBoard launch executing on GUI thread",
            extra={"run_id": self._run_id, "agent_id": self._agent_id, "auto": auto},
        )
        if self._embedded_button is not None and not auto:
            self._embedded_button.setEnabled(False)
        if WEB_ENGINE_AVAILABLE:
            QtCore.QTimer.singleShot(0, lambda: self._perform_embedded_launch(auto=auto))
        else:
            QtCore.QTimer.singleShot(0, self._perform_browser_launch)

    def _tensorboard_url(self, port: int) -> str:
        host = DEFAULT_TENSORBOARD.server_host
        return f"http://{host}:{port}/"

    def _perform_browser_launch(self) -> None:
        if self._tensorboard_process is None:
            self._start_tensorboard_process()
        if self._tensorboard_process is None:
            if self._embedded_button is not None:
                self._embedded_button.setEnabled(True)
            return
        if self._tensorboard_port is not None:
            self.log_constant(
                LOG_UI_RENDER_TABS_INFO,
                message="Opening TensorBoard in system browser",
                extra={
                    "run_id": self._run_id,
                    "agent_id": self._agent_id,
                    "port": self._tensorboard_port,
                },
            )
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._tensorboard_url(self._tensorboard_port)))
        if self._embedded_button is not None:
            self._embedded_button.setEnabled(True)

    def _perform_embedded_launch(self, *, auto: bool = False) -> None:
        if self._tensorboard_process is None:
            self._start_tensorboard_process()
        if self._tensorboard_process is None:
            if self._embedded_button is not None and not auto:
                self._embedded_button.setEnabled(True)
            return
        self.log_constant(
            LOG_UI_RENDER_TABS_TRACE,
            message="Launching embedded TensorBoard",
            extra={"run_id": self._run_id, "agent_id": self._agent_id, "auto": auto},
        )
        if self._server_probe_timer is None:
            timer = QtCore.QTimer(self)
            timer.setInterval(DEFAULT_TENSORBOARD.server_probe_interval_ms)
            timer.timeout.connect(self._probe_tensorboard_server)
            self._server_probe_timer = timer
        if self._server_probe_timer is not None and not self._server_probe_timer.isActive():
            self._server_probe_timer.start()
        if self._web_container is not None:
            self._web_container.setVisible(True)
        if self._web_scroll_area is not None:
            self._web_scroll_area.setVisible(True)

    # ------------------------------------------------------------------
    def _start_tensorboard_process(self) -> None:
        preferred_port = self._tensorboard_port or DEFAULT_TENSORBOARD.default_port
        try:
            port = self._allocate_port(
                preferred=preferred_port,
                attempts=DEFAULT_TENSORBOARD.port_probe_attempts,
            )
        except RuntimeError as exc:  # pragma: no cover - defensive
            self.log_constant(
                LOG_UI_RENDER_TABS_ERROR,
                message="Unable to allocate port for TensorBoard",
                extra={
                    "run_id": self._run_id,
                    "agent_id": self._agent_id,
                    "preferred_port": preferred_port,
                    "attempts": DEFAULT_TENSORBOARD.port_probe_attempts,
                    "error": str(exc),
                },
                exc_info=exc,
            )
            QtWidgets.QMessageBox.warning(
                self,
                "TensorBoard",
                (
                    "Could not find a free TCP port for TensorBoard. "
                    "Close other TensorBoard instances or choose a different port."
                ),
            )
            return

        if port != preferred_port:
            self.log_constant(
                LOG_UI_RENDER_TABS_INFO,
                message="TensorBoard port busy; selecting alternative port",
                extra={
                    "run_id": self._run_id,
                    "agent_id": self._agent_id,
                    "requested_port": preferred_port,
                    "selected_port": port,
                },
            )

        command = [
            DEFAULT_TENSORBOARD.cli_executable,
            "--logdir",
            str(self._log_dir),
            "--port",
            str(port),
            "--host",
            DEFAULT_TENSORBOARD.server_host,
            # Enable multifile loading for XuanCe's nested event file structure
            "--reload_multifile=true",
        ]
        try:
            # nosemgrep: python.lang.security.audit.subprocess-shell-true.subprocess-shell-true
            # Safe: Command built from trusted constants and validated internal data:
            # - cli_executable: hardcoded constant "tensorboard"
            # - logdir: internal Path object from config
            # - port: validated integer from port scanner
            # - host: hardcoded constant
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            self.log_constant(
                LOG_UI_RENDER_TABS_WARNING,
                message="TensorBoard executable not found",
                extra={"run_id": self._run_id, "agent_id": self._agent_id},
            )
            QtWidgets.QMessageBox.warning(
                self,
                "TensorBoard",
                "Could not find the `tensorboard` executable. Install TensorBoard or adjust your PATH.",
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            self.log_constant(
                LOG_UI_RENDER_TABS_ERROR,
                message="Failed to launch TensorBoard",
                extra={"run_id": self._run_id, "agent_id": self._agent_id, "error": str(exc)},
                exc_info=exc,
            )
            QtWidgets.QMessageBox.warning(
                self,
                "TensorBoard",
                f"Failed to launch TensorBoard: {exc}",
            )
            return

        self._tensorboard_process = process
        self._tensorboard_port = port
        self.log_constant(
            LOG_UI_RENDER_TABS_INFO,
            message="TensorBoard process started",
            extra={"run_id": self._run_id, "agent_id": self._agent_id, "port": port},
        )
        if self._status_label is not None:
            self._status_label.setText(
                f"Launching embedded TensorBoard on {self._tensorboard_url(port)} …"
            )

    # ------------------------------------------------------------------
    def _probe_tensorboard_server(self) -> None:
        if self._tensorboard_port is None:
            return
        if self._tensorboard_process and self._tensorboard_process.poll() is not None:
            self.log_constant(
                LOG_UI_RENDER_TABS_WARNING,
                message="TensorBoard process exited unexpectedly",
                extra={"run_id": self._run_id, "agent_id": self._agent_id, "port": self._tensorboard_port},
            )
            if self._status_label is not None:
                self._status_label.setText("TensorBoard process exited unexpectedly.")
            if self._server_probe_timer is not None:
                self._server_probe_timer.stop()
            return

        url = self._tensorboard_url(self._tensorboard_port)
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
            web_view = QWebEngineView(self)
            if FilteredWebEnginePage is not None:
                web_view.setPage(FilteredWebEnginePage(web_view))
            web_view.setUrl(QtCore.QUrl(url))
            web_view.urlChanged.connect(self._update_nav_controls)
            web_view.loadFinished.connect(lambda _: self._update_nav_controls())
            if self._web_container is not None:
                layout = self._web_container.layout()
                if layout is not None:
                    if self._web_placeholder is not None:
                        layout.removeWidget(self._web_placeholder)
                        self._web_placeholder.deleteLater()
                        self._web_placeholder = None
                    layout.addWidget(web_view)
            if self._web_scroll_area is not None:
                self._web_scroll_area.ensureVisible(0, 0)
            self._web_view = web_view
        else:
            web_view = self._web_view
            if web_view is not None:
                web_view.setUrl(QtCore.QUrl(url))

        if self._status_label is not None:
            self._status_label.setText(
                f"Embedded TensorBoard running on {self._tensorboard_url(self._tensorboard_port)}"
            )
        if self._instructions_label is not None:
            self._instructions_label.hide()
        if self._command_field is not None:
            self._command_field.hide()
        if self._embedded_button is not None:
            self._embedded_button.hide()
        if self._nav_bar is not None:
            self._nav_bar.setVisible(True)
        if self._url_field is not None and self._tensorboard_port is not None:
            self._url_field.setText(self._tensorboard_url(self._tensorboard_port))
        self._update_nav_controls()
        self.log_constant(
            LOG_UI_RENDER_TABS_INFO,
            message="TensorBoard embedded viewer ready",
            extra={"run_id": self._run_id, "agent_id": self._agent_id, "port": self._tensorboard_port},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _allocate_port(*, preferred: int | None = None, attempts: int = 1) -> int:
        """Find an available localhost TCP port, preferring a specific range."""

        candidates: list[int] = []
        if preferred is not None:
            attempts = max(1, attempts)
            for offset in range(attempts):
                candidates.append(preferred + offset)
        candidates.append(0)

        last_error: OSError | None = None
        for candidate in candidates:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind((DEFAULT_TENSORBOARD.server_host, candidate))
                except OSError as err:
                    last_error = err
                    continue
                return sock.getsockname()[1]

        raise RuntimeError("No free TCP ports available for TensorBoard") from last_error

    # ------------------------------------------------------------------
    def _stop_tensorboard_process(self) -> None:
        if self._tensorboard_process is None:
            return
        try:
            self._tensorboard_process.terminate()
            self._tensorboard_process.wait(timeout=5)
        except Exception:  # pragma: no cover - defensive
            try:
                self._tensorboard_process.kill()
            except Exception as exc:
                self.log_constant(
                    LOG_UI_TENSORBOARD_KILL_WARNING,
                    message="Failed to kill TensorBoard process during cleanup",
                    extra={
                        "run_id": self._run_id,
                        "agent_id": self._agent_id,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=exc,
                )
        finally:
            self._tensorboard_process = None
            self.log_constant(
                LOG_UI_RENDER_TABS_TRACE,
                message="TensorBoard process stopped",
                extra={"run_id": self._run_id, "agent_id": self._agent_id},
            )

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Clean up resources and optionally delete TensorBoard logs."""
        self._stop_tensorboard_process()
        if self._log_dir.exists():
            response = QtWidgets.QMessageBox.question(
                self,
                "Remove TensorBoard Logs",
                (
                    "Do you want to delete the TensorBoard log directory?\n\n"
                    f"{self._log_dir}"
                ),
                QtWidgets.QMessageBox.StandardButton.Yes,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if response == QtWidgets.QMessageBox.StandardButton.Yes:
                try:
                    shutil.rmtree(self._log_dir, ignore_errors=False)
                    self.log_constant(
                        LOG_UI_RENDER_TABS_INFO,
                        message="TensorBoard log directory deleted",
                        extra={"run_id": self._run_id, "agent_id": self._agent_id, "path": str(self._log_dir)},
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    self.log_constant(
                        LOG_UI_RENDER_TABS_ERROR,
                        message="Failed to delete TensorBoard log directory",
                        extra={"run_id": self._run_id, "agent_id": self._agent_id, "path": str(self._log_dir)},
                        exc_info=exc,
                    )
                    QtWidgets.QMessageBox.warning(
                        self,
                        "TensorBoard",
                        f"Failed to delete log directory: {exc}",
                    )

    # ------------------------------------------------------------------
    def _update_nav_controls(self) -> None:
        if not WEB_ENGINE_AVAILABLE:
            return
        if self._url_field is not None and self._web_view is not None:
            self._url_field.setText(self._web_view.url().toString())
        for btn, enabled in (
            (self._nav_back_btn, bool(self._web_view and self._web_view.history().canGoBack())),
            (self._nav_forward_btn, bool(self._web_view and self._web_view.history().canGoForward())),
        ):
            if btn is not None:
                btn.setEnabled(enabled)
        if self._nav_reload_btn is not None:
            self._nav_reload_btn.setEnabled(self._web_view is not None)
        if self._nav_copy_btn is not None:
            self._nav_copy_btn.setEnabled(self._web_view is not None)
        if self._nav_external_btn is not None:
            self._nav_external_btn.setEnabled(self._web_view is not None)


__all__ = ["TensorboardArtifactTab"]
