"""Test that the worker runs without matplotlib dependencies."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_worker_imports_without_matplotlib():
    """Verify worker can import core modules without matplotlib installed."""
    code = """
import sys
# Block matplotlib import to simulate it not being installed
sys.modules['matplotlib'] = None
sys.modules['matplotlib.pyplot'] = None

# Now try importing the worker components
from spade_bdi_worker.core import RunConfig, HeadlessTrainer, TelemetryEmitter
from spade_bdi_worker.adapters.frozenlake import FrozenLakeAdapter
from spade_bdi_worker.algorithms import QLearningAgent

print("SUCCESS: All imports work without matplotlib")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "SUCCESS" in result.stdout


def test_worker_dry_run():
    """Test worker accepts config via stdin and starts without errors."""
    config = {
        "run_id": "test_run_001",
        "game_id": "FrozenLake-v2",  # Use v2 which is registered
        "agent_id": "qlearning_test",
        "seed": 42,
        "max_episodes": 2,  # Very short run
        "max_steps_per_episode": 10,
        "policy_strategy": "train",
        "policy_path": None,
        "capture_video": False,
        "headless": True,
        "extra": {},
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "telemetry.jsonl"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "spade_bdi_worker.worker",
            ],
            input=json.dumps(config),
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "TELEMETRY_OUTPUT": str(output_file),
            },
        )
        
        # Worker should complete successfully
        assert result.returncode == 0, f"Worker failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        
        # Check telemetry was written
        if output_file.exists():
            lines = output_file.read_text().strip().split("\n")
            assert len(lines) > 0, "No telemetry written"
            
            # Parse first line (should be run_started)
            first_event = json.loads(lines[0])
            assert first_event["event"] == "run_started"
            assert first_event["run_id"] == "test_run_001"


def test_no_matplotlib_in_refactored_package():
    """Ensure no matplotlib imports exist in the refactored package."""
    refactored_dir = Path(__file__).resolve().parents[1]
    
    for py_file in refactored_dir.rglob("*.py"):
        # Skip the test file itself
        if py_file.name == "test_worker_no_matplotlib.py":
            continue
            
        content = py_file.read_text()
        
        # Check for matplotlib imports (not in comments or docstrings)
        for line_no, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Check for actual import statements
            if stripped.startswith("import matplotlib") or stripped.startswith("from matplotlib"):
                pytest.fail(f"Found matplotlib import in {py_file}:{line_no}: {line}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
