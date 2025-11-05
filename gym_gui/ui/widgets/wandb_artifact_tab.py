"""Qt widget for presenting Weights & Biases (WANDB) run metadata."""

from __future__ import annotations

import logging
import os
from typing import Optional

from qtpy import QtCore, QtGui, QtWidgets  # type: ignore[import]
from PyQt6.QtCore import pyqtSignal

try:  # Optional dependency for embedded browser support
    from qtpy.QtWebEngineWidgets import QWebEngineView  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional feature not available in tests
    QWebEngineView = None

WEB_ENGINE_AVAILABLE = QWebEngineView is not None
AUTO_EMBED_ENABLED = os.environ.get("GYM_GUI_DISABLE_WANDB_AUTO_EMBED", "0") != "1"

from gym_gui.constants.constants_wandb import DEFAULT_WANDB, WandbDefaults, build_wandb_run_url
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_WANDB_ERROR,
    LOG_UI_RENDER_TABS_WANDB_STATUS,
    LOG_UI_RENDER_TABS_WANDB_WARNING,
)

_LOGGER = logging.getLogger(__name__)


class WandbArtifactTab(QtWidgets.QWidget, LogConstantMixin):
    """Present WANDB run metadata with quick actions."""

    statusChanged = pyqtSignal(str, bool)

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        run_path: Optional[str],
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
        self._run_path = (run_path or "").strip()
        self._run_url = build_wandb_run_url(self._run_path, base_url=self._base_url) if self._run_path else ""
        self._web_view: Optional[QWebEngineView] = None  # type: ignore[assignment]
        self._url_field: Optional[QtWidgets.QLineEdit] = None
        self._status_area: Optional[QtWidgets.QPlainTextEdit] = None
        self._copy_btn: Optional[QtWidgets.QPushButton] = None
        self._open_btn: Optional[QtWidgets.QPushButton] = None
        self._embed_btn: Optional[QtWidgets.QPushButton] = None
        self._details_toggle: Optional[QtWidgets.QToolButton] = None
        self._details_container: Optional[QtWidgets.QWidget] = None
        self._entity_label: Optional[QtWidgets.QLabel] = None
        self._project_label: Optional[QtWidgets.QLabel] = None
        self._nav_bar: Optional[QtWidgets.QWidget] = None
        self._nav_back_btn: Optional[QtWidgets.QToolButton] = None
        self._nav_forward_btn: Optional[QtWidgets.QToolButton] = None
        self._nav_reload_btn: Optional[QtWidgets.QToolButton] = None
        self._nav_copy_btn: Optional[QtWidgets.QToolButton] = None
        self._nav_external_btn: Optional[QtWidgets.QToolButton] = None
        self._nav_url_field: Optional[QtWidgets.QLineEdit] = None
        self._placeholder: Optional[QtWidgets.QWidget] = None
        self._entity: Optional[str] = None
        self._project: Optional[str] = None
        self._auto_embedded = False

        self.statusChanged.connect(self._handle_status_changed)
        self._setup_ui(initial_status=status_message)
        self._emit_status("initialized", success=bool(self._run_path))

    # ------------------------------------------------------------------
    def _setup_ui(self, *, initial_status: Optional[str]) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        toggle_layout = QtWidgets.QHBoxLayout()
        toggle_layout.addStretch(1)
        details_toggle = QtWidgets.QToolButton(self)
        details_toggle.setText("Hide Details")
        details_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        details_toggle.setCheckable(True)
        details_toggle.setChecked(True)
        details_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        details_toggle.toggled.connect(self._toggle_details_section)
        toggle_layout.addWidget(details_toggle)
        layout.addLayout(toggle_layout)
        self._details_toggle = details_toggle

        details_container = QtWidgets.QWidget(self)
        details_layout = QtWidgets.QVBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)
        layout.addWidget(details_container)
        self._details_container = details_container

        header = QtWidgets.QLabel(
            f"Weights & Biases metrics for run <b>{self._run_id[:12]}…</b> "
            f"(agent <b>{self._agent_id}</b>)"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header.setWordWrap(True)
        details_layout.addWidget(header)

        url_label = QtWidgets.QLabel("Run URL")
        details_layout.addWidget(url_label)

        url_field = QtWidgets.QLineEdit(self._run_url)
        url_field.setReadOnly(True)
        url_field.setCursorPosition(0)
        url_field.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        details_layout.addWidget(url_field)
        self._url_field = url_field

        action_row = QtWidgets.QHBoxLayout()
        copy_btn = QtWidgets.QPushButton("Copy URL")
        copy_btn.clicked.connect(lambda: self._copy_to_clipboard(self._run_url))
        action_row.addWidget(copy_btn)
        self._copy_btn = copy_btn

        open_btn = QtWidgets.QPushButton("Open in Browser")
        open_btn.clicked.connect(self._open_in_browser)
        action_row.addWidget(open_btn)
        self._open_btn = open_btn

        if WEB_ENGINE_AVAILABLE:
            embed_btn = QtWidgets.QPushButton("Open Embedded WANDB")
            embed_btn.clicked.connect(lambda: self._open_embedded())
            action_row.addWidget(embed_btn)
            self._embed_btn = embed_btn

        action_row.addStretch(1)
        details_layout.addLayout(action_row)

        identity_form = QtWidgets.QFormLayout()
        entity_label = QtWidgets.QLabel("—")
        project_label = QtWidgets.QLabel("—")
        identity_form.addRow("Entity", entity_label)
        identity_form.addRow("Project", project_label)
        details_layout.addLayout(identity_form)
        self._entity_label = entity_label
        self._project_label = project_label

        info = QtWidgets.QLabel(
            "WANDB dashboards require internet access. If the link fails to open, "
            "verify that this workstation can reach <code>wandb.ai</code>."
        )
        info.setWordWrap(True)
        info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        details_layout.addWidget(info)

        status_area = QtWidgets.QPlainTextEdit(self)
        status_area.setPlaceholderText(
            "WANDB connection log (e.g., paste `wandb login` output or API key status here)."
        )
        status_area.setReadOnly(False)
        status_area.setVisible(True)
        if initial_status:
            status_area.setPlainText(initial_status)
        details_layout.addWidget(status_area)
        self._status_area = status_area

        if WEB_ENGINE_AVAILABLE:
            nav_bar = QtWidgets.QWidget(self)
            nav_layout = QtWidgets.QHBoxLayout(nav_bar)
            nav_layout.setContentsMargins(0, 0, 0, 0)
            nav_layout.setSpacing(6)

            back_btn = QtWidgets.QToolButton(nav_bar)
            back_btn.setText("←")
            back_btn.clicked.connect(lambda: self._web_view.back() if self._web_view else None)
            nav_layout.addWidget(back_btn)
            self._nav_back_btn = back_btn

            forward_btn = QtWidgets.QToolButton(nav_bar)
            forward_btn.setText("→")
            forward_btn.clicked.connect(lambda: self._web_view.forward() if self._web_view else None)
            nav_layout.addWidget(forward_btn)
            self._nav_forward_btn = forward_btn

            reload_btn = QtWidgets.QToolButton(nav_bar)
            reload_btn.setText("↻")
            reload_btn.clicked.connect(lambda: self._web_view.reload() if self._web_view else None)
            nav_layout.addWidget(reload_btn)
            self._nav_reload_btn = reload_btn

            nav_url = QtWidgets.QLineEdit(self._run_url)
            nav_url.setReadOnly(True)
            nav_layout.addWidget(nav_url, 1)
            self._nav_url_field = nav_url

            nav_copy_btn = QtWidgets.QToolButton(nav_bar)
            nav_copy_btn.setText("Copy URL")
            nav_copy_btn.clicked.connect(lambda: self._copy_to_clipboard(self._run_url))
            nav_layout.addWidget(nav_copy_btn)
            self._nav_copy_btn = nav_copy_btn

            nav_external_btn = QtWidgets.QToolButton(nav_bar)
            nav_external_btn.setText("↗")
            nav_external_btn.clicked.connect(self._open_in_browser)
            nav_layout.addWidget(nav_external_btn)
            self._nav_external_btn = nav_external_btn

            details_layout.addWidget(nav_bar)
            self._nav_bar = nav_bar

        if WEB_ENGINE_AVAILABLE:
            placeholder = QtWidgets.QLabel(
                "Embedded view will appear here when launched."
            )
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            placeholder.setMinimumHeight(320)
            layout.addWidget(placeholder)
            self._placeholder = placeholder

        layout.addStretch(1)
        self._update_action_states(status_message=initial_status)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Recompute run URL (in case metadata changed)."""
        self._run_url = build_wandb_run_url(self._run_path, base_url=self._base_url) if self._run_path else ""
        if self._url_field is not None and self._run_url:
            self._url_field.setCursorPosition(0)
        self._update_action_states()
        self._emit_status("refreshed", success=bool(self._run_path))

    def set_run_path(self, run_path: str) -> None:
        """Update the WANDB run path and refresh the UI."""
        self._run_path = run_path.strip()
        if self._run_path:
            parts = [p for p in self._run_path.split("/") if p]
            if len(parts) >= 4 and parts[-2] == "runs":
                entity = parts[0]
                project = parts[1] if len(parts) > 1 else None
                self.set_wandb_identity(entity, project)
        self.refresh()
        if WEB_ENGINE_AVAILABLE and AUTO_EMBED_ENABLED and self._run_path and not self._auto_embedded:
            self._open_embedded(auto=True)
            self._auto_embedded = True

    def has_run_path(self) -> bool:
        return bool(self._run_path)

    def _update_action_states(self, *, status_message: Optional[str] = None) -> None:
        has_run = bool(self._run_path)
        if self._copy_btn is not None:
            self._copy_btn.setEnabled(has_run)
        if self._open_btn is not None:
            self._open_btn.setEnabled(has_run)
        if self._embed_btn is not None:
            self._embed_btn.setEnabled(has_run and WEB_ENGINE_AVAILABLE)
        if self._url_field is not None:
            if has_run:
                self._url_field.setText(self._run_url)
            else:
                self._url_field.setText("WANDB run path not yet available.")
        if self._nav_url_field is not None:
            self._nav_url_field.setText(self._run_url if self._run_url else "WANDB run path not yet available.")
        if self._nav_copy_btn is not None:
            self._nav_copy_btn.setEnabled(has_run)
        if self._nav_external_btn is not None:
            self._nav_external_btn.setEnabled(has_run)
        if self._status_area is not None:
            if not has_run:
                if status_message:
                    self._status_area.setPlainText(status_message)
                elif not self._status_area.toPlainText().strip():
                    self._status_area.setPlainText(
                        "Waiting for WANDB run metadata to become available..."
                    )
            elif status_message:
                self._status_area.appendPlainText(status_message)
        self._update_identity_labels()
        self._update_nav_buttons()

    def set_wandb_identity(self, entity: Optional[str], project: Optional[str]) -> None:
        if entity:
            self._entity = entity
        if project:
            self._project = project
        self._update_identity_labels()

    def _update_identity_labels(self) -> None:
        if self._entity_label is not None:
            self._entity_label.setText(self._entity or "—")
        if self._project_label is not None:
            self._project_label.setText(self._project or "—")

    def _toggle_details_section(self, checked: bool) -> None:
        if self._details_container is not None:
            self._details_container.setVisible(checked)
        if self._details_toggle is not None:
            self._details_toggle.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
            )
            self._details_toggle.setText("Hide Details" if checked else "Show Details")

    def _update_nav_buttons(self) -> None:
        can_embed = self._web_view is not None
        if self._nav_back_btn is not None:
            self._nav_back_btn.setEnabled(can_embed and bool(self._web_view.history().canGoBack() if self._web_view else False))
        if self._nav_forward_btn is not None:
            self._nav_forward_btn.setEnabled(can_embed and bool(self._web_view.history().canGoForward() if self._web_view else False))
        if self._nav_reload_btn is not None:
            self._nav_reload_btn.setEnabled(can_embed)
        if self._nav_bar is not None:
            self._nav_bar.setVisible(WEB_ENGINE_AVAILABLE)
        if self._nav_url_field is not None:
            self._nav_url_field.setText(self._run_url if self._run_url else "WANDB run path not yet available.")

    def _on_web_url_changed(self, url: QtCore.QUrl) -> None:  # pragma: no cover - Qt signal
        if self._nav_url_field is not None:
            self._nav_url_field.setText(url.toString())
        if self._status_area is not None:
            self._status_area.appendPlainText(f"Embedded view navigated to {url.toString()}")
        self._update_nav_buttons()

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
                "status_message": message,
                "url": self._run_url,
            },
        )
        self.statusChanged.emit(message, success)

    def _handle_status_changed(self, message: str, success: bool) -> None:
        _LOGGER.debug(
            "WANDB tab status changed",
            extra={
                "run_id": self._run_id,
                "agent_id": self._agent_id,
                "status": message,
                "success": success,
            },
        )

    def _copy_to_clipboard(self, value: str) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        if not value:
            self._emit_status("copy_url_waiting_for_run_path", success=False)
            return
        if clipboard is not None:
            clipboard.setText(value)
            self._emit_status("copied_url", success=True)
        else:
            self._emit_status("clipboard_unavailable", success=False)

    def _open_in_browser(self) -> None:
        if not self._run_url:
            self._emit_status("open_browser_waiting_for_run_path", success=False)
            return
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

    def _open_embedded(self, *, auto: bool = False) -> None:
        if not WEB_ENGINE_AVAILABLE:
            self._emit_status("embedded_unavailable", success=False)
            return
        if not self._run_url:
            if not auto:
                self._emit_status("embedded_waiting_for_run_path", success=False)
            return
        try:
            if self._web_view is None:
                self._web_view = QWebEngineView(self)  # type: ignore[assignment]
                self.layout().replaceWidget(self._placeholder, self._web_view)  # type: ignore[arg-type]
                if self._placeholder is not None:
                    self._placeholder.deleteLater()
                    self._placeholder = None
                if self._web_view is not None:
                    self._web_view.urlChanged.connect(self._on_web_url_changed)
                    self._web_view.loadFinished.connect(lambda _: self._update_nav_buttons())
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
            self._emit_status("embedded_auto_opened" if auto else "embedded_opened", success=True)


__all__ = ["WandbArtifactTab"]
