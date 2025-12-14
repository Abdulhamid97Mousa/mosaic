"""Grid-based FastLane tab for Ray multi-worker visualization.

Displays all Ray rollout workers in a single tab with a grid layout.
Each cell shows the FastLane stream from one worker.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Tuple

from PyQt6 import QtCore, QtWidgets, QtQuickWidgets, QtGui

from gym_gui.ui.fastlane_consumer import FastLaneConsumer, FastLaneFrameEvent


_LOGGER = logging.getLogger(__name__)


def _compute_grid_dimensions(num_workers: int) -> Tuple[int, int]:
    """Compute optimal grid dimensions (rows, cols) for given number of workers.

    Returns dimensions that create a roughly square grid:
    - 1 worker: 1x1
    - 2 workers: 1x2
    - 3-4 workers: 2x2
    - 5-6 workers: 2x3
    - 7-9 workers: 3x3
    - etc.
    """
    if num_workers <= 0:
        return (1, 1)
    if num_workers == 1:
        return (1, 1)
    if num_workers == 2:
        return (1, 2)

    cols = math.ceil(math.sqrt(num_workers))
    rows = math.ceil(num_workers / cols)
    return (rows, cols)


class WorkerCell(QtWidgets.QFrame):
    """A single cell in the grid displaying one worker's FastLane stream."""

    def __init__(
        self,
        stream_id: str,
        worker_idx: int,
        *,
        mode_label: str = "Fast lane",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._stream_id = stream_id
        self._worker_idx = worker_idx
        self._mode_label = mode_label

        # Frame styling
        self.setFrameStyle(QtWidgets.QFrame.Shape.Box | QtWidgets.QFrame.Shadow.Plain)
        self.setLineWidth(1)

        # Consumer for this worker's stream
        self._consumer = FastLaneConsumer(stream_id, parent=self)
        self._consumer.frame_ready.connect(self._on_frame_ready)
        self._consumer.status_changed.connect(self._on_status_changed)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Status label at top
        self._status_label = QtWidgets.QLabel(f"W{worker_idx}: connecting...", self)
        self._status_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        self._status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        # QML view for rendering
        self._quick = QtQuickWidgets.QQuickWidget(self)
        self._quick.setResizeMode(QtQuickWidgets.QQuickWidget.ResizeMode.SizeRootObjectToView)
        qml_path = Path(__file__).resolve().parent.parent / "qml" / "FastLaneView.qml"
        self._quick.engine().addImportPath(str(qml_path.parent))
        self._quick.setSource(QtCore.QUrl.fromLocalFile(str(qml_path)))
        layout.addWidget(self._quick, 1)

        self._root_obj = self._quick.rootObject()

    def _on_status_changed(self, status: str) -> None:
        self._status_label.setText(f"W{self._worker_idx}: {status}")

    def _on_frame_ready(self, event: FastLaneFrameEvent) -> None:
        if self._root_obj is None:
            self._root_obj = self._quick.rootObject()
        if self._root_obj is None:
            return

        # Add worker index to HUD
        hud_text = f"W{self._worker_idx}\n{event.hud_text}"
        self._root_obj.setProperty("hudText", hud_text)

        canvas = self._root_obj.findChild(QtCore.QObject, "fastlaneCanvas")
        if canvas is None:
            return
        QtCore.QMetaObject.invokeMethod(
            canvas,
            "setFrame",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(QtGui.QImage, event.image),
        )

    def cleanup(self) -> None:
        self._consumer.stop()
        self._quick.setSource(QtCore.QUrl())


class RayMultiWorkerFastLaneTab(QtWidgets.QWidget):
    """Grid-based FastLane tab showing active Ray rollout workers.

    RLlib architecture:
    - Worker 0 (local) = coordinator, doesn't sample
    - Workers 1..N (remote) = active samplers

    We only show the ACTIVE workers:
    - num_workers=0: W0 is the only worker and it samples → show W0
    - num_workers=2: W1, W2 are active → show W1, W2 (1x2 grid)
    - num_workers=3: W1, W2, W3 are active → show W1, W2, W3 (2x2 grid)

    Layout example with num_workers=2:
    +----------+----------+
    |    W1    |    W2    |
    +----------+----------+
    """

    def __init__(
        self,
        run_id: str,
        num_workers: int,
        *,
        env_id: str = "",
        run_mode: str = "training",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_id = run_id
        self._num_workers = num_workers
        self._env_id = env_id
        self._run_mode = run_mode
        self._cells: List[WorkerCell] = []

        # Only show ACTIVE workers:
        # - num_workers=0: W0 does all sampling (local mode)
        # - num_workers>0: W1..WN do sampling (W0 is coordinator, skip it)
        if num_workers == 0:
            worker_indices = [0]
        else:
            worker_indices = list(range(1, num_workers + 1))

        active_count = len(worker_indices)
        rows, cols = _compute_grid_dimensions(active_count)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Header label
        mode_text = "Evaluation" if run_mode == "policy_eval" else "Training"
        env_label = env_id or "MultiAgent"
        header_text = f"Ray {mode_text} - {env_label} - {active_count} Workers"
        header = QtWidgets.QLabel(header_text, self)
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        # Grid layout for worker cells
        grid_widget = QtWidgets.QWidget(self)
        grid_layout = QtWidgets.QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(4)

        mode_label = "Fast lane (eval)" if run_mode == "policy_eval" else "Fast lane"

        # Create cells for each active worker
        for cell_idx, worker_idx in enumerate(worker_indices):
            stream_id = f"{run_id}-worker-{worker_idx}"
            cell = WorkerCell(
                stream_id,
                worker_idx,
                mode_label=mode_label,
                parent=grid_widget,
            )
            self._cells.append(cell)

            row = cell_idx // cols
            col = cell_idx % cols
            grid_layout.addWidget(cell, row, col)

        # Add empty placeholder cells to fill the grid
        total_cells = rows * cols
        for i in range(active_count, total_cells):
            placeholder = QtWidgets.QFrame(grid_widget)
            placeholder.setFrameStyle(QtWidgets.QFrame.Shape.Box | QtWidgets.QFrame.Shadow.Plain)
            placeholder.setLineWidth(1)
            placeholder.setStyleSheet("background-color: #2a2a2a;")
            row = i // cols
            col = i % cols
            grid_layout.addWidget(placeholder, row, col)

        main_layout.addWidget(grid_widget, 1)

        _LOGGER.info(
            "Created RayMultiWorkerFastLaneTab: run_id=%s, workers=%s, grid=%dx%d",
            run_id[:8],
            worker_indices,
            rows,
            cols,
        )

    def cleanup(self) -> None:
        for cell in self._cells:
            cell.cleanup()
        self._cells.clear()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cleanup()
        super().closeEvent(event)
