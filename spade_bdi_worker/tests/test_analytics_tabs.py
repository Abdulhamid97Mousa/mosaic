"""Ensure SPADE-BDI analytics manifests surface TensorBoard & W&B tabs."""

from __future__ import annotations

import os
from typing import Any

from qtpy import QtWidgets

from gym_gui.ui.panels.analytics_tabs import AnalyticsTabManager
from gym_gui.ui.widgets.tensorboard_artifact_tab import TensorboardArtifactTab
from gym_gui.ui.widgets.wandb_artifact_tab import WandbArtifactTab

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _RecordingRenderTabs:
    def __init__(self) -> None:
        self._agent_tabs: dict[str, dict[str, QtWidgets.QWidget]] = {}

    def add_dynamic_tab(self, run_id: str, name: str, widget: QtWidgets.QWidget) -> None:
        self._agent_tabs.setdefault(run_id, {})[name] = widget


class TestSpadeBdiAnalyticsTabs:
    """Validates analytics tabs triggered from SPADE-BDI metadata."""

    @classmethod
    def setup_class(cls) -> None:  # noqa: D401 - pytest-style setup
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setup_method(self) -> None:  # noqa: D401 - pytest-style setup
        self.render_tabs = _RecordingRenderTabs()
        self.parent = QtWidgets.QWidget()
        self.manager = AnalyticsTabManager(self.render_tabs, self.parent)

    def teardown_method(self) -> None:
        self.parent.deleteLater()

    def test_tensorboard_manifest_creates_tab(self) -> None:
        metadata = {
            "artifacts": {
                "tensorboard": {
                    "enabled": True,
                    "log_dir": "/tmp/spade_bdi_tensorboard",
                }
            }
        }

        self.manager.ensure_tensorboard_tab("run-sb-tb", "spade-agent", metadata)

        tab_map = self.render_tabs._agent_tabs.get("run-sb-tb", {})
        assert "TensorBoard-Agent-spade-agent" in tab_map
        widget = tab_map["TensorBoard-Agent-spade-agent"]
        assert isinstance(widget, TensorboardArtifactTab)

    def test_wandb_manifest_creates_tab(self) -> None:
        metadata = {
            "artifacts": {
                "wandb": {
                    "enabled": True,
                    "run_path": "abdulhamid97mousa/MOSAIC/runs/spade-test",
                }
            }
        }

        self.manager.ensure_wandb_tab("run-sb-wab", "spade-agent", metadata)

        tab_map = self.render_tabs._agent_tabs.get("run-sb-wab", {})
        assert "WAB-Agent-spade-agent" in tab_map
        widget = tab_map["WAB-Agent-spade-agent"]
        assert isinstance(widget, WandbArtifactTab)
        widget.append_status_line("wandb login succeeded for SPADE-BDI")
