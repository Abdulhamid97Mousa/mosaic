"""Tests for CleanRL training form."""

from __future__ import annotations

import os
from typing import Dict, Any

import pytest
from qtpy import QtWidgets

from gym_gui.core.enums import GameId
from gym_gui.ui.widgets.cleanrl_train_form import CleanRlTrainForm

# Ensure Qt renders offscreen in CI environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _base_form(qt_app) -> CleanRlTrainForm:
    form = CleanRlTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    index = form._algo_combo.findText("ppo")
    if index >= 0:
        form._algo_combo.setCurrentIndex(index)
    form._custom_env_checkbox.setChecked(False)
    form._env_combo.setCurrentIndex(0)
    form._timesteps_spin.setValue(4096)
    form._seed_spin.setValue(123)
    form._agent_id_input.setText("agent-cleanrl-test")
    form._worker_id_input.setText("cleanrl-test-worker")
    form._tensorboard_checkbox.setChecked(True)
    form._track_wandb_checkbox.setChecked(False)
    form._notes_edit.setPlainText("integration-test")
    return form


def test_get_config_includes_worker_metadata(qt_app) -> None:
    form = _base_form(qt_app)
    config = form.get_config()

    assert isinstance(config, dict)
    metadata = config.get("metadata", {})
    worker_meta: Dict[str, Any] = metadata.get("worker", {})
    assert worker_meta.get("module") == "cleanrl_worker.cli"
    assert worker_meta.get("worker_id") == "cleanrl-test-worker"
    assert worker_meta.get("config", {}).get("algo") == "ppo"
    assert worker_meta.get("config", {}).get("env_id") == "CartPole-v1"
    assert worker_meta.get("config", {}).get("seed") == 123
    assert worker_meta.get("arguments") == ["--dry-run", "--emit-summary"]

    extras = worker_meta.get("config", {}).get("extras", {})
    assert extras.get("tensorboard_dir") == "tensorboard"
    assert extras.get("notes") == "integration-test"
    assert "algo_params" in extras

    artifacts = metadata.get("artifacts", {})
    tensorboard_meta = artifacts.get("tensorboard", {})
    assert tensorboard_meta.get("enabled") is True
    assert tensorboard_meta.get("relative_path").endswith("tensorboard")
    wandb_meta = artifacts.get("wandb", {})
    assert wandb_meta.get("enabled") is False

    form.deleteLater()


def test_disable_dry_run_removes_arguments(qt_app) -> None:
    form = _base_form(qt_app)
    form._dry_run_checkbox.setChecked(False)

    config = form.get_config()
    worker_meta = config["metadata"]["worker"]
    assert worker_meta.get("arguments") == []

    form.deleteLater()
