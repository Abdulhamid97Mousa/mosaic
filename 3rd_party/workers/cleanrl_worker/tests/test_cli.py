"""CLI smoke tests for cleanrl_worker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cleanrl_worker import cli


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    payload = {
        "metadata": {
            "worker": {
                "config": {
                    "run_id": "cli-test-run",
                    "algo": "ppo",
                    "env_id": "CartPole-v1",
                    "total_timesteps": 2048,
                }
            }
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_dry_run_outputs_summary(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--dry-run",
            "--emit-summary",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "CartPole-v1" in captured.out
    assert "cli-test-run" in captured.out
