from __future__ import annotations

"""Qt widget that surfaces TensorBoard artifact locations inside the GUI."""

from pathlib import Path
from typing import Iterable
import logging
import shutil
import socket
import subprocess
import urllib.error
import urllib.request

from qtpy import QtCore, QtGui, QtWidgets  # type: ignore[import]

try:  # Optional dependency for embedded browser support
    from qtpy.QtWebEngineWidgets import QWebEngineView  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional feature not available in tests
    QWebEngineView = None

WEB_ENGINE_AVAILABLE = QWebEngineView is not None

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_ERROR,
    LOG_UI_RENDER_TABS_INFO,
    LOG_UI_RENDER_TABS_TRACE,
    LOG_UI_RENDER_TABS_WARNING,
)
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_ERROR,
    LOG_UI_RENDER_TABS_INFO,
    LOG_UI_RENDER_TABS_TRACE,
    LOG_UI_RENDER_TABS_WARNING,
)


_LOGGER = logging.getLogger(__name__)


class TensorboardArtifactTab(QtWidgets.QWidget, LogConstantMixin):
    """Present a TensorBoard log directory with quick actions."""

    _STATUS_REFRESH_MS = 4000
    _DEFAULT_TENSORBOARD_PORT = 6006
    _PORT_PROBE_ATTEMPTS = 12

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
        if not WEB_ENGINE_AVAILABLE:
            button_row.addWidget(copy_cmd_btn)

        button_row.addStretch(1)
        layout.addLayout(button_row)

        status_label = QtWidgets.QLabel()
        status_label.setWordWrap(True)
        self._status_label = status_label
        layout.addWidget(status_label)

        if WEB_ENGINE_AVAILABLE:
            instructions = QtWidgets.QLabel(
                "Click 'Open Embedded TensorBoard' to launch TensorBoard inside the GUI."
            )
        else:
            instructions = QtWidgets.QLabel(
                "Launch TensorBoard using the command below or open it in your default browser."
            )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
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
            url_field = QtWidgets.QLineEdit("http://127.0.0.1")
            url_field.setReadOnly(True)
            url_field.setCursorPosition(0)
            copy_btn = QtWidgets.QToolButton()
            copy_btn.setText("Copy URL")
            copy_btn.setEnabled(False)
            copy_btn.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(url_field.text()))
            external_btn = QtWidgets.QToolButton()
            external_btn.setText("↗")
            external_btn.setEnabled(False)
            external_btn.setToolTip("Open in default browser")
            external_btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(url_field.text())) if self._url_field else None)
            nav_layout.addWidget(back_btn)
            nav_layout.addWidget(forward_btn)
            nav_layout.addWidget(reload_btn)
            nav_layout.addWidget(url_field, 1)
            nav_layout.addWidget(copy_btn)
            nav_layout.addWidget(external_btn)
            layout.addWidget(nav_bar)
            self._nav_bar = nav_bar
            self._url_field = url_field
            self._nav_back_btn = back_btn
            self._nav_forward_btn = forward_btn
            self._nav_reload_btn = reload_btn
            self._nav_copy_btn = copy_btn
            self._nav_external_btn = external_btn
            launch_btn = QtWidgets.QPushButton("Open Embedded TensorBoard")
            launch_btn.clicked.connect(self._launch_embedded_viewer)
            layout.addWidget(launch_btn)
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
            layout.addWidget(command_field)
            launch_btn = QtWidgets.QPushButton("Open TensorBoard in Browser")
            launch_btn.clicked.connect(self._launch_embedded_viewer)
            layout.addWidget(launch_btn)
            self._embedded_button = launch_btn

        if WEB_ENGINE_AVAILABLE:
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            )

            web_group = QtWidgets.QGroupBox("Embedded Viewer")
            web_group.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            )
            web_layout = QtWidgets.QVBoxLayout(web_group)
            placeholder = QtWidgets.QLabel(
                "TensorBoard will open here once the embedded server is ready."
            )
            placeholder.setWordWrap(True)
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
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
        self._stop_tensorboard_process()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_cli_command(log_dir: Path) -> str:
        return f"tensorboard --logdir \"{log_dir}\""

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

    @QtCore.Slot(bool)
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
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(f"http://127.0.0.1:{self._tensorboard_port}/"))
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
            timer.setInterval(750)
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
        preferred_port = self._tensorboard_port or self._DEFAULT_TENSORBOARD_PORT
        try:
            port = self._allocate_port(preferred=preferred_port, attempts=self._PORT_PROBE_ATTEMPTS)
        except RuntimeError as exc:  # pragma: no cover - defensive
            self.log_constant(
                LOG_UI_RENDER_TABS_ERROR,
                message="Unable to allocate port for TensorBoard",
                extra={
                    "run_id": self._run_id,
                    "agent_id": self._agent_id,
                    "preferred_port": preferred_port,
                    "attempts": self._PORT_PROBE_ATTEMPTS,
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
                f"Launching embedded TensorBoard on http://127.0.0.1:{port} …"
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
            web_view = QWebEngineView(self)
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
                f"Embedded TensorBoard running on http://127.0.0.1:{self._tensorboard_port}"
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
            self._url_field.setText(f"http://127.0.0.1:{self._tensorboard_port}/")
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
                    sock.bind(("127.0.0.1", candidate))
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
            except Exception:
                pass
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
