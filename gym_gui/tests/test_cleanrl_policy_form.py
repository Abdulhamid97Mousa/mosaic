"""Tests for CleanRL policy form and metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from qtpy import QtWidgets

from gym_gui.core.enums import EnvironmentFamily, GameId
from gym_gui.telemetry.semconv import VideoModes
from gym_gui.workers.cleanrl_policy_metadata import (
    CleanRlCheckpoint,
    discover_policies,
    load_metadata_for_policy,
)


os_env = __import__("os").environ
os_env.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _write_mock_config(root: Path, run_id: str, env_id: str = "Walker2d-v5") -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "worker": {
                "config": {
                    "algo": "ppo_continuous_action",
                    "env_id": env_id,
                    "seed": 7,
                    "extras": {
                        "fastlane_only": True,
                        "fastlane_video_mode": VideoModes.GRID,
                        "fastlane_grid_limit": 4,
                        "algo_params": {"num_envs": 4},
                    },
                }
            }
        }
    }
    cfg_path = config_dir / f"config-{run_id}.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    return cfg_path


def test_metadata_loader_extracts_run(monkeypatch, tmp_path):
    var_dir = tmp_path / "var" / "trainer"
    monkeypatch.setattr("gym_gui.config.paths.VAR_TRAINER_DIR", var_dir)
    monkeypatch.setattr(
        "gym_gui.workers.cleanrl_policy_metadata.VAR_TRAINER_DIR",
        var_dir,
    )
    run_id = "01TESTMETADATA"
    _write_mock_config(var_dir, run_id)
    policy = var_dir / "runs" / run_id / "runs" / "cleanrl-run" / "ppo.cleanrl_model"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text("", encoding="utf-8")

    meta = load_metadata_for_policy(policy)
    assert meta is not None
    assert meta.run_id == run_id
    assert meta.env_id == "Walker2d-v5"
    assert meta.fastlane_video_mode == VideoModes.GRID


def test_policy_form_builds_config(monkeypatch, qt_app):
    from gym_gui.ui.widgets.cleanrl_policy_form import CleanRlPolicyForm

    checkpoint = CleanRlCheckpoint(
        policy_path=Path("/tmp/policy.cleanrl_model"),
        run_id="01FAKE",
        cleanrl_run_name="cleanrl-run",
        env_id="Walker2d-v5",
        algo="ppo_continuous_action",
        seed=11,
        num_envs=4,
        fastlane_only=True,
        fastlane_video_mode=VideoModes.GRID,
        fastlane_grid_limit=4,
        config_path=None,
    )

    monkeypatch.setattr(
        "gym_gui.ui.widgets.cleanrl_policy_form.discover_policies",
        lambda: [checkpoint],
    )

    form = CleanRlPolicyForm(current_game=GameId.WALKER2D)
    form._apply_checkpoint(checkpoint)
    form._eval_video_checkbox.setChecked(True)
    form._grid_spin.setValue(4)
    form._seed_spin.setValue(42)
    form._episode_spin.setValue(75)
    form._repeat_checkbox.setChecked(True)
    form._gamma_spin.setValue(0.95)
    form._max_steps_spin.setValue(123)
    form._max_seconds_spin.setValue(12.5)
    form._fastlane_only_checkbox.setChecked(True)
    form._video_mode_combo.setCurrentIndex(form._video_mode_combo.findData(VideoModes.GRID))
    if form._ok_button is not None:
        form._ok_button.setEnabled(True)
    form._on_accept()
    config = form.get_config()
    assert isinstance(config, dict)
    worker = config["metadata"]["worker"]["config"]
    assert worker["env_id"] == "Walker2d-v5"
    extras = worker["extras"]
    assert extras["mode"] == "policy_eval"
    assert extras["eval_capture_video"] is True
    assert worker["total_timesteps"] == 75
    assert extras["eval_episodes"] == 75
    assert extras["eval_batch_size"] == 75
    assert extras["eval_repeat"] is True
    assert extras["tensorboard_dir"] == "tensorboard_eval"
    assert extras["eval_gamma"] == 0.95
    assert extras["eval_max_episode_steps"] == 123
    assert extras["eval_max_episode_seconds"] == 12.5
    artifacts = config["metadata"].get("artifacts", {})
    tb_meta = artifacts.get("tensorboard")
    assert tb_meta and tb_meta["relative_path"].endswith("tensorboard_eval")
    ui_meta = config["metadata"].get("ui", {})
    assert ui_meta.get("eval_gamma") == 0.95
    assert ui_meta.get("eval_max_episode_steps") == 123
    assert ui_meta.get("eval_max_episode_seconds") == 12.5
    form.deleteLater()
