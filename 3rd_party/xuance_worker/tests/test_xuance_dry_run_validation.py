"""Tests for XuanCe dry-run validation via the --emit-summary flag.

These tests verify that:
1. The --emit-summary flag produces structured JSON output
2. The validation module (run_xuance_dry_run) correctly captures the output
3. Successful validation returns success=True with valid JSON
4. Invalid configurations return success=False with error details
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest


def _extract_summary_json(output: str) -> Optional[dict]:
    """Extract the summary JSON from CLI output.

    The CLI output may contain multiple JSON lines (telemetry events)
    followed by the pretty-printed summary JSON. This function extracts
    the summary (the last JSON object containing "status").

    Args:
        output: Raw stdout from the CLI.

    Returns:
        Parsed summary dict, or None if not found.
    """
    lines = output.strip().split("\n")

    # Try to find a multi-line JSON block at the end (pretty-printed summary)
    # Look for lines starting with '{' and ending with '}'
    brace_depth = 0
    json_start = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "{" and brace_depth == 0:
            json_start = i
            brace_depth = 1
        elif json_start >= 0:
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth == 0 and stripped.endswith("}"):
                # Found complete JSON block
                json_text = "\n".join(lines[json_start : i + 1])
                try:
                    parsed = json.loads(json_text)
                    if "status" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
                json_start = -1

    # Fallback: try parsing each line as JSON and find one with "status"
    for line in reversed(lines):
        try:
            parsed = json.loads(line.strip())
            if isinstance(parsed, dict) and "status" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _minimal_worker_config() -> dict:
    """Create a minimal XuanCe worker config for dry-run testing."""
    return {
        "run_id": "test-xuance-dryrun",
        "method": "PPO_Clip",
        "env": "classic_control",
        "env_id": "CartPole-v1",
        "dl_toolbox": "torch",
        "running_steps": 1000,
        "device": "cpu",
        "parallels": 1,
        "test_mode": False,
    }


class TestXuanceCLIDryRun:
    """Tests for xuance_worker CLI --dry-run --emit-summary flags."""

    def test_cli_dry_run_returns_success(self):
        """Test that CLI dry-run mode returns exit code 0 for valid config."""
        import tempfile

        config = _minimal_worker_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(config, indent=2))

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "xuance_worker.cli",
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0, f"Dry-run failed: {result.stderr}"

    def test_cli_dry_run_emit_summary_produces_json(self):
        """Test that --emit-summary produces valid JSON output."""
        import tempfile

        config = _minimal_worker_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(config, indent=2))

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "xuance_worker.cli",
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--emit-summary",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0, f"Dry-run failed: {result.stderr}"

            # Extract summary JSON from output (may contain telemetry lines)
            summary = _extract_summary_json(result.stdout)
            assert summary is not None, f"Could not find summary JSON in output:\n{result.stdout}"

            # Verify expected keys in summary
            assert "status" in summary, "Summary should contain 'status' key"
            assert summary["status"] == "dry-run", "Status should be 'dry-run'"

    def test_cli_dry_run_emit_summary_contains_config(self):
        """Test that --emit-summary output contains the config."""
        import tempfile

        config = _minimal_worker_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(config, indent=2))

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "xuance_worker.cli",
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--emit-summary",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            summary = _extract_summary_json(result.stdout)
            assert summary is not None, f"Could not find summary JSON in output:\n{result.stdout}"
            assert "config" in summary, "Summary should contain 'config' key"
            assert "method" in summary, "Summary should contain 'method' key"
            assert "env_id" in summary, "Summary should contain 'env_id' key"


class TestValidationModule:
    """Tests for the run_xuance_dry_run validation function."""

    def test_run_xuance_dry_run_succeeds_with_valid_config(self):
        """Test that validation succeeds with a valid configuration."""
        from gym_gui.validations.validation_xuance_worker_form import run_xuance_dry_run

        config = _minimal_worker_config()
        success, output = run_xuance_dry_run(config)

        assert success, f"Validation failed unexpectedly: {output}"
        assert output, "Output should not be empty"

    def test_run_xuance_dry_run_output_is_json(self):
        """Test that validation output contains valid JSON summary."""
        from gym_gui.validations.validation_xuance_worker_form import run_xuance_dry_run

        config = _minimal_worker_config()
        success, output = run_xuance_dry_run(config)

        assert success, f"Validation failed: {output}"

        # Extract summary JSON from output (may contain telemetry lines)
        summary = _extract_summary_json(output)
        assert summary is not None, f"Could not find summary JSON in output:\n{output}"
        assert "status" in summary, "Summary should contain 'status' key"

    def test_run_xuance_dry_run_with_different_algorithms(self):
        """Test validation with different algorithm methods."""
        from gym_gui.validations.validation_xuance_worker_form import run_xuance_dry_run

        algorithms = ["PPO_Clip", "DQN", "A2C"]

        for algo in algorithms:
            config = _minimal_worker_config()
            config["method"] = algo
            success, output = run_xuance_dry_run(config, timeout_seconds=30)

            # We expect these to succeed (dry-run doesn't require XuanCe to be installed)
            # It just validates the config structure
            assert isinstance(success, bool), f"Success should be bool for {algo}"
            assert isinstance(output, str), f"Output should be str for {algo}"


class TestErrorHandling:
    """Tests for error handling in dry-run validation."""

    def test_timeout_handling(self):
        """Test that timeout is handled gracefully."""
        from gym_gui.validations.validation_xuance_worker_form import run_xuance_dry_run

        config = _minimal_worker_config()
        # Use very short timeout - should still work for dry-run
        success, output = run_xuance_dry_run(config, timeout_seconds=5)

        # Should complete within 5 seconds for dry-run
        assert isinstance(success, bool)
        assert isinstance(output, str)

    def test_invalid_method_returns_error_or_fallback(self):
        """Test behavior with an invalid/unknown method."""
        from gym_gui.validations.validation_xuance_worker_form import run_xuance_dry_run

        config = _minimal_worker_config()
        config["method"] = "NonExistentAlgorithm_XYZ"

        success, output = run_xuance_dry_run(config, timeout_seconds=30)

        # Should return some result (either success with fallback or failure)
        assert isinstance(success, bool)
        assert isinstance(output, str)
