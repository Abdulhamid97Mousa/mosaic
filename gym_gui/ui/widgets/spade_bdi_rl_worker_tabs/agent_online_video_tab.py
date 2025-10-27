"""Live video/RGB frame tab for agent training runs."""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.ui.widgets.base_telemetry_tab import BaseTelemetryTab


class AgentOnlineVideoTab(BaseTelemetryTab):
    """Displays live RGB frames if telemetry includes frame data."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        self._frame_count = 0
        self._seed: Optional[int] = None
        self._control_mode: Optional[str] = None

        super().__init__(run_id, agent_id, parent=parent)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Use inherited header builder and extend it
        header = self._build_header()
        self._frame_count_label = QtWidgets.QLabel("<b>Frames:</b> 0")
        self._metadata_label = QtWidgets.QLabel("Seed: — • Mode: —")
        self._metadata_label.setStyleSheet("color: gray;")
        header.addWidget(self._frame_count_label)
        header.addWidget(self._metadata_label)
        layout.addLayout(header)

        # Frame display
        self._label = QtWidgets.QLabel("Waiting for RGB frames…", self)
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(400, 300)
        self._label.setStyleSheet("background-color: #2b2b2b; color: #aaa; padding: 20px;")
        layout.addWidget(self._label, 1)

        # Info footer
        self._info_label = QtWidgets.QLabel(
            "Frames will appear if environment supports rgb_array rendering"
        )
        self._info_label.setStyleSheet("color: gray; font-size: 9pt;")
        self._info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._info_label)

    def on_step(self, step: Dict[str, Any], *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update frame display from step payload."""
        # Try multiple possible field names for RGB data
        b64_data = step.get("rgb_b64") or step.get("frame_b64") or step.get("render_b64")
        
        if not b64_data:
            # Check for frame_ref pointing to file
            frame_ref = step.get("frame_ref")
            if frame_ref:
                self._load_frame_from_file(frame_ref)
            return
        
        try:
            img_bytes = base64.b64decode(b64_data)
            img = QtGui.QImage.fromData(img_bytes)
            if not img.isNull():
                self._frame_count += 1
                self._frame_count_label.setText(f"<b>Frames:</b> {self._frame_count}")
                
                # Scale to fit label while maintaining aspect ratio
                pixmap = QtGui.QPixmap.fromImage(img).scaled(
                    self._label.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
                self._label.setPixmap(pixmap)
                self._label.setStyleSheet("background-color: black;")
        except Exception as e:
            self._info_label.setText(f"Frame decode error: {e}")

        if metadata:
            self.update_metadata(metadata)

    def _load_frame_from_file(self, frame_ref: str) -> None:
        """Load frame from file path reference."""
        try:
            pixmap = QtGui.QPixmap(frame_ref)
            if not pixmap.isNull():
                self._frame_count += 1
                self._frame_count_label.setText(f"<b>Frames:</b> {self._frame_count}")
                
                pixmap = pixmap.scaled(
                    self._label.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
                self._label.setPixmap(pixmap)
                self._label.setStyleSheet("background-color: black;")
        except Exception as e:
            self._info_label.setText(f"Frame load error: {e}")

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        seed = metadata.get("seed")
        if seed is not None:
            try:
                self._seed = int(seed)
            except (TypeError, ValueError):
                self._seed = None
        control_mode = metadata.get("control_mode") or metadata.get("mode")
        if control_mode:
            self._control_mode = str(control_mode)
        seed_text = f"{self._seed}" if self._seed is not None else "—"
        mode_text = self._control_mode or "—"
        self._metadata_label.setText(f"Seed: {seed_text} • Mode: {mode_text}")


__all__ = ["AgentOnlineVideoTab"]
