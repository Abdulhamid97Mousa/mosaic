"""Ensure SPADE-BDI analytics manifests surface TensorBoard & WANDB tabs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GYM_GUI_DISABLE_WANDB_AUTO_EMBED", "1")

import pytest
from qtpy import QtWidgets

from gym_gui.ui.panels import analytics_tabs
from gym_gui.ui.panels.analytics_tabs import AnalyticsTabManager
from gym_gui.ui.widgets.tensorboard_artifact_tab import TensorboardArtifactTab
from gym_gui.ui.widgets.wandb_artifact_tab import WandbArtifactTab


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
        assert "WANDB-Agent-spade-agent" in tab_map
        widget = tab_map["WANDB-Agent-spade-agent"]
        assert isinstance(widget, WandbArtifactTab)
        widget.append_status_line("wandb login succeeded for SPADE-BDI")

    def test_load_and_create_tabs_from_disk(self, monkeypatch: Any, tmp_path: Path) -> None:
        run_id = "spade-manifest-run"
        agent_id = "spade-agent"

        trainer_root = tmp_path / "trainer"
        run_dir = trainer_root / "runs" / run_id
        tensorboard_dir = run_dir / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        analytics_payload = {
            "artifacts": {
                "tensorboard": {
                    "enabled": True,
                    "log_dir": str(tensorboard_dir),
                },
                "wandb": {
                    "enabled": True,
                    "run_path": "entity/project/runs/spade-disk",
                },
            }
        }
        (run_dir / "analytics.json").write_text(json.dumps(analytics_payload), encoding="utf-8")

        monkeypatch.setattr(analytics_tabs, "VAR_TRAINER_DIR", trainer_root, raising=False)

        self.manager.load_and_create_tabs(run_id, agent_id)

        tab_map = self.render_tabs._agent_tabs.get(run_id, {})
        assert "TensorBoard-Agent-spade-agent" in tab_map
        assert "WANDB-Agent-spade-agent" in tab_map
        assert isinstance(tab_map["TensorBoard-Agent-spade-agent"], TensorboardArtifactTab)
        assert isinstance(tab_map["WANDB-Agent-spade-agent"], WandbArtifactTab)

    def test_placeholder_wandb_tab_updates_when_manifest_ready(self, monkeypatch: Any, tmp_path: Path) -> None:
        run_id = "spade-manifest-wait"
        agent_id = "spade-agent"

        trainer_root = tmp_path / "trainer"
        run_dir = trainer_root / "runs" / run_id
        tensorboard_dir = run_dir / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        wandb_manifest = run_dir / "wandb.json"

        analytics_payload = {
            "artifacts": {
                "tensorboard": {
                    "enabled": True,
                    "log_dir": str(tensorboard_dir),
                },
                "wandb": {
                    "enabled": True,
                    "run_path": "",
                    "manifest_file": str(wandb_manifest),
                    "entity": "test-entity",
                    "project": "test-project",
                },
            }
        }
        (run_dir / "analytics.json").write_text(json.dumps(analytics_payload), encoding="utf-8")

        monkeypatch.setattr(analytics_tabs, "VAR_TRAINER_DIR", trainer_root, raising=False)

        monkeypatch.setattr(AnalyticsTabManager, "_schedule_retry", lambda *args, **kwargs: None)

        # First pass: manifest missing run_path, expect runs listing placeholder
        self.manager.load_and_create_tabs(run_id, agent_id)
        tab_map = self.render_tabs._agent_tabs.get(run_id, {})
        wandb_tab = tab_map.get("WANDB-Agent-spade-agent")
        assert isinstance(wandb_tab, WandbArtifactTab)
        assert wandb_tab.has_run_path()
        assert wandb_tab._run_path.endswith("/runs/")

        # Write manifest with run_path and reload
        wandb_manifest.write_text(
            json.dumps(
                {
                    "run_path": "entity/project/runs/spade-manifest",
                    "run_id": run_id,
                    "agent_id": agent_id,
                }
            ),
            encoding="utf-8",
        )

        self.manager.load_and_create_tabs(run_id, agent_id)
        assert wandb_tab.has_run_path()
