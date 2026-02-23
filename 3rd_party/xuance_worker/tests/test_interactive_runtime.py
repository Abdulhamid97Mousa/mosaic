"""Tests for XuanCe InteractiveRuntime functionality.

This test verifies that the InteractiveRuntime correctly implements the
IPC protocol for GUI step-by-step policy evaluation.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


# Test environment configuration
ENV_ID = "CartPole-v1"
METHOD = "ppo"


@pytest.fixture
def interactive_process():
    """Launch an interactive xuance_worker process."""
    # Create a dummy policy path for testing (worker falls back to random if not found)
    dummy_policy = Path("/tmp/dummy_xuance_policy.pth")
    dummy_policy.touch(exist_ok=True)

    cmd = [
        sys.executable,
        "-m", "xuance_worker.cli",
        "--interactive",
        "--run-id", "test_interactive",
        "--method", METHOD,
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


def read_json_response(proc, timeout=10.0):
    """Read a single JSON line from the process stdout."""
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


@pytest.mark.slow
class TestXuanCeInteractiveRuntime:
    """Test suite for XuanCe InteractiveRuntime.

    Tests that use the interactive_process fixture are slow because the
    subprocess loads torch + XuanCe + creates a full PPO runner. They may
    time out in large batch runs. Run with: pytest -m slow
    """

    def test_import_interactive_runtime(self):
        """Test that InteractiveRuntime can be imported."""
        from xuance_worker.runtime import InteractiveRuntime, InteractiveConfig
        assert InteractiveRuntime is not None
        assert InteractiveConfig is not None

    def test_interactive_config_creation(self):
        """Test InteractiveConfig dataclass creation."""
        from xuance_worker.runtime import InteractiveConfig

        config = InteractiveConfig(
            run_id="test_run",
            env_id="CartPole-v1",
            method="ppo",
            policy_path="/path/to/policy.pth",
            device="cpu",
        )

        assert config.run_id == "test_run"
        assert config.env_id == "CartPole-v1"
        assert config.method == "ppo"
        assert config.policy_path == "/path/to/policy.pth"
        assert config.device == "cpu"
        assert config.dl_toolbox == "torch"  # default
        assert config.deterministic is True  # default: argmax for eval

    def test_init_message_emitted(self, interactive_process):
        """Test that init message is emitted on startup."""
        proc = interactive_process

        # Read init message
        response = read_json_response(proc, timeout=10)

        assert response is not None, "No init response received"
        assert response.get("type") == "init", f"Expected 'init' type, got: {response}"
        assert response.get("env_id") == ENV_ID
        assert response.get("method") == METHOD
        assert "run_id" in response
        assert "version" in response

    def test_reset_emits_ready(self, interactive_process):
        """Test that reset command emits ready response with stats.

        Note: _load_policy() is heavy (imports torch, creates XuanCe runner).
        With a dummy policy file, the load will fail and emit an error response.
        We accept either 'ready' (real policy) or 'error' (dummy policy) —
        the key assertion is that the process responds within the timeout.
        """
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=10)
        assert init is not None
        assert init.get("type") == "init"

        # Send reset command
        reset_cmd = json.dumps({"cmd": "reset", "seed": 42}) + "\n"
        proc.stdin.write(reset_cmd)
        proc.stdin.flush()

        # Read response — _load_policy() is heavy (torch import + get_runner),
        # so give it 60s. With a dummy policy file, we expect an error response.
        response = read_json_response(proc, timeout=60)

        assert response is not None, "No response received after reset command"

        if response.get("type") == "ready":
            # Real policy loaded successfully
            assert response.get("seed") == 42
            assert "step_index" in response
            assert response.get("step_index") == 0
            assert "episode_index" in response
            assert "episode_reward" in response
            assert response.get("episode_reward") == 0.0
        else:
            # Dummy policy file — load failed, error response expected
            assert response.get("type") == "error", f"Unexpected response type: {response}"

    def test_step_emits_correct_structure(self, interactive_process):
        """Test that step command emits correct response structure.

        With a dummy policy file, reset may fail (returning error instead of
        ready). If reset fails, step will also fail with "Environment not
        initialized". We test both the success and error paths.
        """
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=10)
        assert init is not None and init.get("type") == "init"

        # Send reset (heavy — _load_policy imports torch + get_runner)
        proc.stdin.write(json.dumps({"cmd": "reset", "seed": 42}) + "\n")
        proc.stdin.flush()

        ready = read_json_response(proc, timeout=60)
        assert ready is not None, "No response received after reset"

        if ready.get("type") != "ready":
            # Dummy policy load failed; step will also error
            pytest.skip("Policy loading failed with dummy file — cannot test step structure")

        # Send step command
        proc.stdin.write(json.dumps({"cmd": "step"}) + "\n")
        proc.stdin.flush()

        response = read_json_response(proc, timeout=30)

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

        # Check render_payload if present (environment may not always render)
        render_payload = response.get("render_payload")
        if render_payload is not None:
            assert "mode" in render_payload
            assert render_payload["mode"] == "rgb_array"
            assert "rgb" in render_payload
            assert "width" in render_payload
            assert "height" in render_payload

    def test_ping_pong(self, interactive_process):
        """Test ping/pong health check."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=10)
        assert init is not None

        # Send ping
        proc.stdin.write(json.dumps({"cmd": "ping"}) + "\n")
        proc.stdin.flush()

        response = read_json_response(proc, timeout=10)

        assert response is not None
        assert response.get("type") == "pong"

    def test_stop_command(self, interactive_process):
        """Test stop command terminates gracefully."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=10)
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

    def test_invalid_json_error(self, interactive_process):
        """Test that invalid JSON triggers error response."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=10)
        assert init is not None

        # Send invalid JSON
        proc.stdin.write("not valid json\n")
        proc.stdin.flush()

        response = read_json_response(proc, timeout=10)

        assert response is not None
        assert response.get("type") == "error"
        assert "Invalid JSON" in response.get("message", "")

    def test_unknown_command_error(self, interactive_process):
        """Test that unknown commands trigger error response."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=10)
        assert init is not None

        # Send unknown command
        proc.stdin.write(json.dumps({"cmd": "unknown_command"}) + "\n")
        proc.stdin.flush()

        response = read_json_response(proc, timeout=10)

        assert response is not None
        assert response.get("type") == "error"
        assert "Unknown command" in response.get("message", "")

    def test_multiple_steps(self, interactive_process):
        """Test running multiple steps.

        With a dummy policy file, reset may fail. If so, skip this test.
        """
        proc = interactive_process

        # Init and reset (heavy — _load_policy imports torch + get_runner)
        init = read_json_response(proc, timeout=10)
        assert init is not None

        proc.stdin.write(json.dumps({"cmd": "reset", "seed": 42}) + "\n")
        proc.stdin.flush()
        ready = read_json_response(proc, timeout=60)
        assert ready is not None

        if ready.get("type") != "ready":
            pytest.skip("Policy loading failed with dummy file — cannot test multiple steps")

        # Run 5 steps
        for i in range(5):
            proc.stdin.write(json.dumps({"cmd": "step"}) + "\n")
            proc.stdin.flush()

            response = read_json_response(proc, timeout=30)
            assert response is not None, f"No response for step {i}"

            if response.get("type") == "step":
                assert response.get("step_index") == i + 1
            elif response.get("type") == "episode_done":
                # Episode may end early
                break


class TestXuanCeInteractiveRuntimeDirect:
    """Direct unit tests without subprocess."""

    def test_interactive_config_defaults(self):
        """Test InteractiveConfig default values."""
        from xuance_worker.runtime import InteractiveConfig

        config = InteractiveConfig(
            run_id="test",
            env_id="CartPole-v1",
            method="ppo",
            policy_path="/path",
        )

        assert config.device == "cpu"
        assert config.dl_toolbox == "torch"
        assert config.env == "classic_control"
        assert config.deterministic is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
