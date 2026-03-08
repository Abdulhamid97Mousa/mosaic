"""Tests for MARLlib worker CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marllib_worker.cli import main


class TestCliDryRun:
    """Verify CLI dry-run mode works end-to-end."""

    def test_direct_args_dry_run(self, capsys):
        rc = main([
            "--run-id", "cli-test",
            "--algo", "mappo",
            "--environment-name", "mpe",
            "--map-name", "simple_spread",
            "--dry-run",
            "--emit-summary",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        summary = json.loads(captured.out)
        assert summary["status"] == "dry-run"
        assert summary["algo"] == "mappo"

    def test_config_file_dry_run(self, tmp_path: Path, capsys):
        config = {
            "run_id": "cli-cfg-test",
            "algo": "qmix",
            "environment_name": "smac",
            "map_name": "3m",
            "seed": 7,
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(config))

        rc = main(["--config", str(cfg_path), "--dry-run", "--emit-summary"])
        assert rc == 0
        captured = capsys.readouterr()
        summary = json.loads(captured.out)
        assert summary["algo"] == "qmix"
        assert summary["algo_type"] == "VD"

    def test_missing_required_arg_returns_error(self):
        # missing --map-name
        rc = main([
            "--run-id", "x",
            "--algo", "mappo",
            "--environment-name", "mpe",
            "--dry-run",
        ])
        assert rc == 1

    def test_bad_algo_returns_error(self):
        rc = main([
            "--run-id", "x",
            "--algo", "nonexistent_algo",
            "--environment-name", "mpe",
            "--map-name", "y",
            "--dry-run",
        ])
        assert rc == 1

    def test_config_file_not_found_returns_error(self):
        rc = main(["--config", "/nonexistent/path.json", "--dry-run"])
        assert rc == 1

    def test_verbose_flag(self, capsys):
        rc = main([
            "--run-id", "v-test",
            "--algo", "ippo",
            "--environment-name", "mpe",
            "--map-name", "simple_spread",
            "--dry-run",
            "--verbose",
        ])
        assert rc == 0

    def test_algo_params_json(self, capsys):
        rc = main([
            "--run-id", "ap-test",
            "--algo", "mappo",
            "--environment-name", "mpe",
            "--map-name", "simple_spread",
            "--algo-params", '{"lr": 0.001}',
            "--dry-run",
            "--emit-summary",
        ])
        assert rc == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["config"]["algo_params"]["lr"] == 0.001

    def test_no_checkpoint_end_flag(self, capsys):
        rc = main([
            "--run-id", "nce-test",
            "--algo", "mappo",
            "--environment-name", "mpe",
            "--map-name", "simple_spread",
            "--no-checkpoint-end",
            "--dry-run",
            "--emit-summary",
        ])
        assert rc == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["config"]["checkpoint_end"] is False
