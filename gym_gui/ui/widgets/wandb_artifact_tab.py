"""Qt widget for presenting Weights & Biases (W&B) run metadata."""

from __future__ import annotations

import logging
from typing import Optional

from qtpy import QtCore, QtGui, QtWidgets  # type: ignore[import]
from PyQt6.QtCore import pyqtSignal

try:  # Optional dependency for embedded browser support
    from qtpy.QtWebEngineWidgets import QWebEngineView  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional feature not available in tests
    QWebEngineView = None

WEB_ENGINE_AVAILABLE = QWebEngineView is not None

from gym_gui.constants.constants_wandb import DEFAULT_WANDB, WandbDefaults, build_wandb_run_url
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_WANDB_ERROR,
    LOG_UI_RENDER_TABS_WANDB_STATUS,
    LOG_UI_RENDER_TABS_WANDB_WARNING,
)

_LOGGER = logging.getLogger(__name__)


class WandbArtifactTab(QtWidgets.QWidget, LogConstantMixin):
    """Present W&B run metadata with quick actions."""

    statusChanged = pyqtSignal(str, bool)

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        run_path: str,
        *,
        base_url: Optional[str] = None,
        defaults: WandbDefaults = DEFAULT_WANDB,
        parent: Optional[QtWidgets.QWidget] = None,
        status_message: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self._run_id = run_id
        self._agent_id = agent_id
        self._defaults = defaults
        self._base_url = base_url or defaults.app_base_url
        self._run_path = run_path.strip()
        self._run_url = build_wandb_run_url(self._run_path, base_url=self._base_url)
        self._web_view: Optional[QWebEngineView] = None  # type: ignore[assignment]
        self._url_field: Optional[QtWidgets.QLineEdit] = None
        self._status_area: Optional[QtWidgets.QPlainTextEdit] = None

        self.statusChanged.connect(self._handle_status_changed)
        self._setup_ui(initial_status=status_message)
        self._emit_status("initialized", success=True)

    # ------------------------------------------------------------------
    def _setup_ui(self, *, initial_status: Optional[str]) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QtWidgets.QLabel(
            f"Weights & Biases metrics for run <b>{self._run_id[:12]}…</b> "
            f"(agent <b>{self._agent_id}</b>)"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header.setWordWrap(True)
        layout.addWidget(header)

        url_label = QtWidgets.QLabel("Run URL")
        layout.addWidget(url_label)

        url_field = QtWidgets.QLineEdit(self._run_url)
        url_field.setReadOnly(True)
        url_field.setCursorPosition(0)
        url_field.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        layout.addWidget(url_field)
        self._url_field = url_field

        action_row = QtWidgets.QHBoxLayout()
        copy_btn = QtWidgets.QPushButton("Copy URL")
        copy_btn.clicked.connect(lambda: self._copy_to_clipboard(self._run_url))
        action_row.addWidget(copy_btn)

        open_btn = QtWidgets.QPushButton("Open in Browser")
        open_btn.clicked.connect(self._open_in_browser)
        action_row.addWidget(open_btn)

        if WEB_ENGINE_AVAILABLE:
            embed_btn = QtWidgets.QPushButton("Open Embedded View")
            embed_btn.clicked.connect(self._open_embedded)
            action_row.addWidget(embed_btn)

        action_row.addStretch(1)
        layout.addLayout(action_row)

        info = QtWidgets.QLabel(
            "W&B dashboards require internet access. If the link fails to open, "
            "verify that this workstation can reach <code>wandb.ai</code>."
        )
        info.setWordWrap(True)
        info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(info)

        status_area = QtWidgets.QPlainTextEdit(self)
        status_area.setPlaceholderText(
            "W&B connection log (e.g., paste `wandb login` output or API key status here)."
        )
        status_area.setReadOnly(False)
        status_area.setVisible(True)
        if initial_status:
            status_area.setPlainText(initial_status)
        layout.addWidget(status_area)
        self._status_area = status_area

        if WEB_ENGINE_AVAILABLE:
            placeholder = QtWidgets.QLabel(
                "Embedded view will appear here when launched."
            )
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            placeholder.setMinimumHeight(320)
            layout.addWidget(placeholder)
            self._placeholder = placeholder
        else:
            self._placeholder = None

        layout.addStretch(1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Recompute run URL (in case metadata changed)."""
        self._run_url = build_wandb_run_url(self._run_path, base_url=self._base_url)
        if self._url_field is not None:
            self._url_field.setText(self._run_url)
            self._url_field.setCursorPosition(0)
        self._emit_status("refreshed", success=True)

    def set_run_path(self, run_path: str) -> None:
        """Update the W&B run path and refresh the UI."""
        self._run_path = run_path.strip()
        self.refresh()

    def set_status_text(self, message: str) -> None:
        """Replace the status text contents displayed beneath the metadata."""
        if self._status_area is not None:
            self._status_area.setPlainText(message)

    def append_status_line(self, message: str) -> None:
        """Append a log line to the status area."""
        if self._status_area is not None:
            self._status_area.appendPlainText(message)

    # ------------------------------------------------------------------
    def _emit_status(self, message: str, *, success: bool) -> None:
        code = LOG_UI_RENDER_TABS_WANDB_STATUS if success else LOG_UI_RENDER_TABS_WANDB_WARNING
        self.log_constant(
            code,
            extra={
                "run_id": self._run_id,
                "agent_id": self._agent_id,
                "message": message,
                "url": self._run_url,
            },
        )
        self.statusChanged.emit(message, success)

    def _handle_status_changed(self, message: str, success: bool) -> None:
        _LOGGER.debug(
            "W&B tab status changed",
            extra={
                "run_id": self._run_id,
                "agent_id": self._agent_id,
                "message": message,
                "success": success,
            },
        )

    def _copy_to_clipboard(self, value: str) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(value)
            self._emit_status("copied_url", success=True)
        else:
            self._emit_status("clipboard_unavailable", success=False)

    def _open_in_browser(self) -> None:
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._run_url))
        except Exception as exc:  # noqa: BLE001
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_ERROR,
                extra={
                    "run_id": self._run_id,
                    "agent_id": self._agent_id,
                    "error": str(exc),
                },
                exc_info=exc,
            )
            self._emit_status("open_browser_failed", success=False)
        else:
            self._emit_status("open_browser", success=True)

    def _open_embedded(self) -> None:
        if not WEB_ENGINE_AVAILABLE:
            self._emit_status("embedded_unavailable", success=False)
            return
        try:
            if self._web_view is None:
                self._web_view = QWebEngineView(self)  # type: ignore[assignment]
                self.layout().replaceWidget(self._placeholder, self._web_view)  # type: ignore[arg-type]
                if self._placeholder is not None:
                    self._placeholder.deleteLater()
                    self._placeholder = None
            if self._web_view is not None:
                self._web_view.setUrl(QtCore.QUrl(self._run_url))
        except Exception as exc:  # noqa: BLE001
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_ERROR,
                extra={
                    "run_id": self._run_id,
                    "agent_id": self._agent_id,
                    "error": str(exc),
                },
                exc_info=exc,
            )
            self._emit_status("embedded_failed", success=False)
        else:
            self._emit_status("embedded_opened", success=True)


__all__ = ["WandbArtifactTab"]
