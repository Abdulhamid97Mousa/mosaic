import sys
from pathlib import Path

from gym_gui.validations.validation_cleanrl_worker_form import run_cleanrl_dry_run


def _minimal_config() -> dict:
    run_id = "test-cleanrl"
    metadata = {
        "ui": {
            "worker_id": "cleanrl_worker_test",
            "agent_id": "test-agent",
            "algo": "ppo",
            "env_id": "CartPole-v1",
            "dry_run": True,
        },
        "worker": {
            "worker_id": "cleanrl_worker_test",
            "module": "cleanrl_worker.cli",
            "use_grpc": False,
            "grpc_target": "127.0.0.1:50055",
            "arguments": ["--dry-run", "--emit-summary"],
            "config": {
                "run_id": run_id,
                "algo": "ppo",
                "env_id": "CartPole-v1",
                "total_timesteps": 1024,
                "extras": {},
            },
        },
        "artifacts": {
            "tensorboard": {"enabled": False, "relative_path": None},
            "wandb": {
                "enabled": False,
                "run_path": None,
                "use_vpn_proxy": False,
                "http_proxy": None,
                "https_proxy": None,
            },
            "notes": None,
        },
    }

    return {
        "run_name": run_id,
        "entry_point": sys.executable,
        "arguments": ["-m", "cleanrl_worker.cli"],
        "environment": {
            "CLEANRL_RUN_ID": run_id,
            "CLEANRL_AGENT_ID": "test-agent",
            "TRACK_TENSORBOARD": "0",
            "TRACK_WANDB": "0",
        },
        "resources": {
            "cpus": 1,
            "memory_mb": 512,
            "gpus": {"requested": 0, "mandatory": False},
        },
        "artifacts": {
            "output_prefix": f"runs/{run_id}",
            "persist_logs": True,
            "keep_checkpoints": False,
        },
        "metadata": metadata,
    }


def test_cleanrl_cli_dry_run_succeeds():
    config = _minimal_config()
    success, output = run_cleanrl_dry_run(config)
    assert success, f"Dry-run failed: {output}"
    assert "total_timesteps" in output or output.strip(), "Expected dry-run summary output"
