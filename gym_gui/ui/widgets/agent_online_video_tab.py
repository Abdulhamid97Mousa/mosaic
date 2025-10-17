"""Live video/RGB frame tab for agent training runs."""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from qtpy import QtCore, QtGui, QtWidgets


class AgentOnlineVideoTab(QtWidgets.QWidget):
    """Displays live RGB frames if telemetry includes frame data."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.run_id = run_id
        self.agent_id = agent_id
        self._frame_count = 0

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QtWidgets.QHBoxLayout()
        self._run_label = QtWidgets.QLabel(f"<b>Run:</b> {self.run_id[:12]}...")
        self._agent_label = QtWidgets.QLabel(f"<b>Agent:</b> {self.agent_id}")
        self._frame_count_label = QtWidgets.QLabel("<b>Frames:</b> 0")
        header.addWidget(self._run_label)
        header.addWidget(self._agent_label)
        header.addStretch()
        header.addWidget(self._frame_count_label)
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

    def on_step(self, step: Dict[str, Any]) -> None:
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


__all__ = ["AgentOnlineVideoTab"]
