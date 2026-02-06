"""Runtime unit tests for the CleanRL worker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from cleanrl_worker.config import WorkerConfig
from cleanrl_worker.runtime import CleanRLWorkerRuntime
from cleanrl_worker.runtime import REPO_ROOT


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


def test_allowed_entry_modules_includes_cleanrl_aliases() -> None:
    runtime = CleanRLWorkerRuntime(
        _make_config(),
        use_grpc=False,
        grpc_target="127.0.0.1:50055",
        dry_run=True,
    )

    modules = runtime._allowed_entry_modules()

    assert "cleanrl_worker.algorithms.ppo_with_save" in modules
    assert (
        "cleanrl_worker.cleanrl_worker.algorithms.ppo_with_save" in modules
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
    assert summary["status"] == "dry-run"
    assert summary["config"]["env_id"] == "CartPole-v1"
    assert summary["config"]["algo"] == "ppo"


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


def test_build_cleanrl_args_ignores_fastlane_extras() -> None:
    config = _make_config(
        extras={
            "fastlane_only": True,
            "fastlane_slot": 3,
            "fastlane_video_mode": "grid",
            "fastlane_grid_limit": 8,
            "wandb_project_name": "fastlane-demo",
        }
    )
    runtime = CleanRLWorkerRuntime(
        config,
        use_grpc=False,
        grpc_target="127.0.0.1:50055",
        dry_run=True,
    )

    args = runtime.build_cleanrl_args()
    assert "--wandb-project-name=fastlane-demo" in args
    assert all(not arg.startswith("--fastlane-only") for arg in args)
    assert all(not arg.startswith("--fastlane-slot") for arg in args)
    assert all(not arg.startswith("--fastlane-video-mode") for arg in args)
    assert all(not arg.startswith("--fastlane-grid-limit") for arg in args)


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
        "cleanrl_worker.runtime.VAR_TRAINER_DIR",
        run_dir,
        raising=False,
    )
    monkeypatch.setattr(
        "cleanrl_worker.runtime.ensure_var_directories",
        lambda: None,
    )

    launched: dict[str, Any] = {}

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
        "cleanrl_worker.runtime.subprocess.Popen",
        DummyProc,
    )

    summary = runtime.run()

    assert launched["cmd"][2] == "cleanrl_worker.launcher"
    assert launched["cmd"][3] in {
        "cleanrl_worker.algorithms.ppo_with_save",
        "cleanrl.ppo",
        "cleanrl_worker.cleanrl.ppo",
    }
    assert all("--tensorboard-dir" not in arg for arg in launched["cmd"][4:])
    assert launched["cwd"] == (run_dir / "runs" / config.run_id).resolve()
    assert launched["stdout_path"].exists()
    assert launched["stderr_path"].exists()
    assert summary["status"] == "completed"
    pythonpath = launched["env"].get("PYTHONPATH", "")
    assert str(REPO_ROOT) in pythonpath.split(os.pathsep)
    tb_dir = (launched["cwd"] / "tensorboard").resolve()
    assert launched["env"]["CLEANRL_TENSORBOARD_DIR"] == str(tb_dir)
    assert tb_dir.exists()
    wandb_dir = launched["env"]["WANDB_DIR"]
    assert Path(wandb_dir).exists()
    assert "127.0.0.1" in launched["env"]["no_proxy"]
    assert "localhost" in launched["env"]["NO_PROXY"]


def test_run_sets_fastlane_env_vars_in_training_mode(monkeypatch, tmp_path: Path) -> None:
    """Test that FastLane environment variables are passed to training subprocess.

    This is critical for grid video mode - the subprocess needs these env vars
    to enable the correct FastLane video mode and grid limit.
    """
    config = _make_config(
        extras={
            "fastlane_only": True,
            "fastlane_slot": 0,
            "fastlane_video_mode": "grid",
            "fastlane_grid_limit": 4,
            "algo_params": {"num_envs": 4},
        }
    )
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
        "cleanrl_worker.runtime.VAR_TRAINER_DIR",
        run_dir,
        raising=False,
    )
    monkeypatch.setattr(
        "cleanrl_worker.runtime.ensure_var_directories",
        lambda: None,
    )

    captured_env: dict[str, Any] = {}

    class DummyProc:
        def __init__(self, cmd, cwd, stdout, stderr, env):
            captured_env.update(env)
            self._polled = False

        def poll(self):
            if not self._polled:
                self._polled = True
                return 0
            return 0

    monkeypatch.setattr(
        "cleanrl_worker.runtime.subprocess.Popen",
        DummyProc,
    )

    runtime.run()

    # Verify FastLane env vars are set correctly
    assert captured_env.get("GYM_GUI_FASTLANE_ONLY") == "1", \
        "FASTLANE_ONLY should be '1' when fastlane_only=True"
    assert captured_env.get("GYM_GUI_FASTLANE_VIDEO_MODE") == "grid", \
        "FASTLANE_VIDEO_MODE should be 'grid'"
    assert captured_env.get("GYM_GUI_FASTLANE_GRID_LIMIT") == "4", \
        "FASTLANE_GRID_LIMIT should be '4'"
    assert captured_env.get("GYM_GUI_FASTLANE_SLOT") == "0", \
        "FASTLANE_SLOT should be '0'"
    assert captured_env.get("CLEANRL_NUM_ENVS") == "4", \
        "CLEANRL_NUM_ENVS should match algo_params.num_envs"
    assert captured_env.get("CLEANRL_RUN_ID") == config.run_id, \
        "CLEANRL_RUN_ID should be set to the run_id"
    assert captured_env.get("CLEANRL_SEED") == "42", \
        "CLEANRL_SEED should be set when seed is provided"


def test_write_eval_tensorboard_emits_scalars(monkeypatch, tmp_path: Path) -> None:
    runtime = CleanRLWorkerRuntime(
        _make_config(),
        use_grpc=False,
        grpc_target="127.0.0.1:50055",
        dry_run=True,
    )

    recorded: list[tuple[str, float, int]] = []

    class DummyWriter:
        def __init__(self, log_dir: str):
            recorded.append(("init", 0.0, Path(log_dir).exists()))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            recorded.append((tag, value, step))

        def flush(self) -> None:
            recorded.append(("flush", 0.0, -1))

    monkeypatch.setattr(
        "cleanrl_worker.runtime._TensorBoardWriter",
        DummyWriter,
    )

    runtime._write_eval_tensorboard(tmp_path / "tb", [1.0, 2.0], 1.5)

    tags = [entry[0] for entry in recorded]
    assert "eval/episode_return" in tags
    assert "eval/avg_return" in tags
