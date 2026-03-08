"""Unit tests for CleanRL worker config parsing helpers."""

from __future__ import annotations

from cleanrl_worker.config import parse_worker_config


def test_policy_eval_with_zero_timesteps_derives_value_from_eval_episodes() -> None:
    payload = {
        "run_id": "policy-eval",
        "algo": "ppo_continuous_action",
        "env_id": "Walker2d-v5",
        "total_timesteps": 0,
        "extras": {
            "mode": "policy_eval",
            "policy_path": "/tmp/model.cleanrl_model",
            "eval_episodes": 7,
        },
    }

    config = parse_worker_config(payload)

    assert config.total_timesteps == 7
    assert config.extras["eval_episodes"] == 7
