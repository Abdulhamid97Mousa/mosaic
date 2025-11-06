"""Runtime unit tests for the CleanRL worker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import pytest

from cleanrl_worker.config import WorkerConfig
from cleanrl_worker.runtime import CleanRLWorkerRuntime
from cleanrl_worker.MOSAIC_CLEANRL_WORKER.runtime import REPO_ROOT


def _make_config(
    *,
    extras: Mapping[str, object] | None = None,
) -> WorkerConfig:
    return WorkerConfig(
        run_id="runtime-test-run",
        algo="ppo",
        env_id="CartPole-v1",
        total_timesteps=512,
        seed=42,
        extras=dict(extras or {}),
    )


def test_build_cleanrl_args_includes_common_flags() -> None:
    config = _make_config(
        extras={
            "track_wandb": True,
            "tensorboard_dir": "tensorboard",
            "algo_params": {
                "learning_rate": 0.0003,
                "num_envs": 8,
                "use_sde": True,
            },
        }
    )
    runtime = CleanRLWorkerRuntime(
        config,
        use_grpc=False,
        grpc_target="127.0.0.1:50055",
        dry_run=True,
    )

    args = runtime.build_cleanrl_args()
    assert f"--env-id={config.env_id}" in args
    assert f"--total-timesteps={config.total_timesteps}" in args
    assert "--track" in args
    assert "--learning-rate=0.0003" in args
    assert "--num-envs=8" in args
    assert "--use-sde" in args


def test_run_dry_run_returns_summary() -> None:
    runtime = CleanRLWorkerRuntime(
        _make_config(),
        use_grpc=False,
        grpc_target="127.0.0.1:50055",
        dry_run=True,
    )

    summary = runtime.run()
    assert summary.status == "dry-run"
    assert summary.config["env_id"] == "CartPole-v1"
    assert summary.config["algo"] == "ppo"


def test_build_cleanrl_args_respects_cli_overrides() -> None:
    config = _make_config(
        extras={
            "cuda": False,
            "capture_video": True,
            "wandb_project_name": "test-project",
        }
    )
    runtime = CleanRLWorkerRuntime(
        config,
        use_grpc=False,
        grpc_target="127.0.0.1:50055",
        dry_run=True,
    )

    args = runtime.build_cleanrl_args()
    assert "--cuda" not in args
    assert "--capture-video=true" in args
    assert "--wandb-project-name=test-project" in args


def test_run_uses_launcher_and_writes_logs(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(extras={"tensorboard_dir": "tensorboard", "track_wandb": True})
    run_dir = tmp_path / "trainer"

    runtime = CleanRLWorkerRuntime(
        config,
        use_grpc=False,
        grpc_target="127.0.0.1:50055",
        dry_run=False,
    )

    monkeypatch.setattr(
        CleanRLWorkerRuntime, "_register_with_trainer", lambda self: None
    )
    monkeypatch.setattr(
        "cleanrl_worker.MOSAIC_CLEANRL_WORKER.runtime.VAR_TRAINER_DIR",
        run_dir,
        raising=False,
    )
    monkeypatch.setattr(
        "cleanrl_worker.MOSAIC_CLEANRL_WORKER.runtime.ensure_var_directories",
        lambda: None,
    )

    launched: dict[str, object] = {}

    class DummyProc:
        def __init__(self, cmd, cwd, stdout, stderr, env):
            launched["cmd"] = cmd
            launched["cwd"] = Path(cwd)
            launched["stdout_path"] = Path(stdout.name)
            launched["stderr_path"] = Path(stderr.name)
            launched["env"] = env
            self._polled = False

        def poll(self):
            if not self._polled:
                self._polled = True
                return 0
            return 0

    monkeypatch.setattr(
        "cleanrl_worker.MOSAIC_CLEANRL_WORKER.runtime.subprocess.Popen",
        DummyProc,
    )

    summary = runtime.run()

    assert launched["cmd"][2] == "cleanrl_worker.MOSAIC_CLEANRL_WORKER.launcher"
    assert launched["cmd"][3] in {"cleanrl.ppo", "cleanrl_worker.cleanrl.ppo"}
    assert all("--tensorboard-dir" not in arg for arg in launched["cmd"][4:])
    assert launched["cwd"] == (run_dir / "runs" / config.run_id).resolve()
    assert launched["stdout_path"].exists()
    assert launched["stderr_path"].exists()
    assert summary.status == "completed"
    pythonpath = launched["env"].get("PYTHONPATH", "")
    assert str(REPO_ROOT) in pythonpath.split(os.pathsep)
    tb_dir = (launched["cwd"] / "tensorboard").resolve()
    assert launched["env"]["CLEANRL_TENSORBOARD_DIR"] == str(tb_dir)
    assert tb_dir.exists()
    wandb_dir = launched["env"]["WANDB_DIR"]
    assert Path(wandb_dir).exists()
    assert "127.0.0.1" in launched["env"]["no_proxy"]
    assert "localhost" in launched["env"]["NO_PROXY"]
