"""Tests for Jumanji InteractiveRuntime functionality.

This test verifies that the InteractiveRuntime correctly implements the
IPC protocol for GUI step-by-step policy evaluation.

Note: Jumanji environments are JAX-based and may have different behavior
than Gymnasium environments. Rendering support varies by environment.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


# Test environment configuration
ENV_ID = "Game2048-v1"
AGENT = "a2c"


@pytest.fixture
def interactive_process():
    """Launch an interactive jumanji_worker process."""
    # Create a dummy policy path for testing (worker falls back to random if not found)
    dummy_policy = Path("/tmp/dummy_jumanji_policy.pkl")
    dummy_policy.touch(exist_ok=True)

    cmd = [
        sys.executable,
        "-m", "jumanji_worker.cli",
        "--interactive",
        "--run-id", "test_interactive",
        "--agent", AGENT,
        "--env-id", ENV_ID,
        "--policy-path", str(dummy_policy),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    yield proc

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def read_json_response(proc, timeout=15.0):
    """Read a single JSON line from the process stdout.

    Note: Jumanji/JAX initialization can take longer, so default timeout is higher.
    """
    import select

    start = time.time()
    while time.time() - start < timeout:
        readable, _, _ = select.select([proc.stdout], [], [], 0.1)
        if readable:
            line = proc.stdout.readline()
            if line:
                line = line.strip()
                if line:
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
    return None


class TestJumanjiInteractiveRuntime:
    """Test suite for Jumanji InteractiveRuntime."""

    def test_import_interactive_runtime(self):
        """Test that InteractiveRuntime can be imported."""
        from jumanji_worker.runtime import InteractiveRuntime, InteractiveConfig
        assert InteractiveRuntime is not None
        assert InteractiveConfig is not None

    def test_interactive_config_creation(self):
        """Test InteractiveConfig dataclass creation."""
        from jumanji_worker.runtime import InteractiveConfig

        config = InteractiveConfig(
            run_id="test_run",
            env_id="Game2048-v1",
            agent="a2c",
            policy_path="/path/to/policy.pkl",
            device="cpu",
        )

        assert config.run_id == "test_run"
        assert config.env_id == "Game2048-v1"
        assert config.agent == "a2c"
        assert config.policy_path == "/path/to/policy.pkl"
        assert config.device == "cpu"

    @pytest.mark.skipif(
        not Path("/tmp/dummy_jumanji_policy.pkl").exists(),
        reason="Requires JAX to be installed"
    )
    def test_init_message_emitted(self, interactive_process):
        """Test that init message is emitted on startup."""
        proc = interactive_process

        # Read init message (JAX initialization can be slow)
        response = read_json_response(proc, timeout=15)

        assert response is not None, "No init response received"
        assert response.get("type") == "init", f"Expected 'init' type, got: {response}"
        assert response.get("env_id") == ENV_ID
        assert response.get("agent") == AGENT
        assert "run_id" in response
        assert "version" in response

    @pytest.mark.skipif(
        not Path("/tmp/dummy_jumanji_policy.pkl").exists(),
        reason="Requires JAX to be installed"
    )
    def test_reset_emits_ready(self, interactive_process):
        """Test that reset command emits ready response with stats."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=15)
        assert init is not None
        assert init.get("type") == "init"

        # Send reset command
        reset_cmd = json.dumps({"cmd": "reset", "seed": 42}) + "\n"
        proc.stdin.write(reset_cmd)
        proc.stdin.flush()

        # Read ready response (may take time due to JAX compilation)
        response = read_json_response(proc, timeout=30)

        assert response is not None, "No ready response received"
        assert response.get("type") == "ready", f"Expected 'ready' type, got: {response}"
        assert response.get("seed") == 42

        # Verify stats are included for GUI reset
        assert "step_index" in response, "Missing step_index in ready response"
        assert response.get("step_index") == 0
        assert "episode_index" in response, "Missing episode_index in ready response"
        assert "episode_reward" in response, "Missing episode_reward in ready response"
        assert response.get("episode_reward") == 0.0

    @pytest.mark.skipif(
        not Path("/tmp/dummy_jumanji_policy.pkl").exists(),
        reason="Requires JAX to be installed"
    )
    def test_step_emits_correct_structure(self, interactive_process):
        """Test that step command emits correct response structure."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=15)
        assert init is not None and init.get("type") == "init"

        # Send reset
        proc.stdin.write(json.dumps({"cmd": "reset", "seed": 42}) + "\n")
        proc.stdin.flush()

        ready = read_json_response(proc, timeout=30)
        assert ready is not None and ready.get("type") == "ready"

        # Send step command
        proc.stdin.write(json.dumps({"cmd": "step"}) + "\n")
        proc.stdin.flush()

        # Read step response
        response = read_json_response(proc, timeout=15)

        assert response is not None, "No step response received"
        assert response.get("type") == "step", f"Expected 'step' type, got: {response}"

        # Check required fields
        assert "step_index" in response
        assert "episode_index" in response
        assert "action" in response
        assert "reward" in response
        assert "terminated" in response
        assert "truncated" in response
        assert "episode_reward" in response

        # Note: render_payload may not be available for all Jumanji environments

    @pytest.mark.skipif(
        not Path("/tmp/dummy_jumanji_policy.pkl").exists(),
        reason="Requires JAX to be installed"
    )
    def test_ping_pong(self, interactive_process):
        """Test ping/pong health check."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=15)
        assert init is not None

        # Send ping
        proc.stdin.write(json.dumps({"cmd": "ping"}) + "\n")
        proc.stdin.flush()

        response = read_json_response(proc, timeout=10)

        assert response is not None
        assert response.get("type") == "pong"

    @pytest.mark.skipif(
        not Path("/tmp/dummy_jumanji_policy.pkl").exists(),
        reason="Requires JAX to be installed"
    )
    def test_stop_command(self, interactive_process):
        """Test stop command terminates gracefully."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=15)
        assert init is not None

        # Send stop
        proc.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
        proc.stdin.flush()

        response = read_json_response(proc, timeout=10)

        assert response is not None
        assert response.get("type") == "stopped"

        # Process should exit cleanly
        proc.wait(timeout=5)
        assert proc.returncode == 0

    @pytest.mark.skipif(
        not Path("/tmp/dummy_jumanji_policy.pkl").exists(),
        reason="Requires JAX to be installed"
    )
    def test_invalid_json_error(self, interactive_process):
        """Test that invalid JSON triggers error response."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=15)
        assert init is not None

        # Send invalid JSON
        proc.stdin.write("not valid json\n")
        proc.stdin.flush()

        response = read_json_response(proc, timeout=10)

        assert response is not None
        assert response.get("type") == "error"
        assert "Invalid JSON" in response.get("message", "")

    @pytest.mark.skipif(
        not Path("/tmp/dummy_jumanji_policy.pkl").exists(),
        reason="Requires JAX to be installed"
    )
    def test_unknown_command_error(self, interactive_process):
        """Test that unknown commands trigger error response."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=15)
        assert init is not None

        # Send unknown command
        proc.stdin.write(json.dumps({"cmd": "unknown_command"}) + "\n")
        proc.stdin.flush()

        response = read_json_response(proc, timeout=10)

        assert response is not None
        assert response.get("type") == "error"
        assert "Unknown command" in response.get("message", "")


class TestJumanjiInteractiveRuntimeDirect:
    """Direct unit tests without subprocess."""

    def test_interactive_config_defaults(self):
        """Test InteractiveConfig default values."""
        from jumanji_worker.runtime import InteractiveConfig

        config = InteractiveConfig(
            run_id="test",
            env_id="Game2048-v1",
            agent="a2c",
            policy_path="/path",
        )

        assert config.device == "cpu"

    def test_logic_environments_available(self):
        """Test that LOGIC_ENVIRONMENTS includes expected envs."""
        from jumanji_worker.config import LOGIC_ENVIRONMENTS

        assert "Game2048-v1" in LOGIC_ENVIRONMENTS
        assert "Sudoku-v0" in LOGIC_ENVIRONMENTS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
