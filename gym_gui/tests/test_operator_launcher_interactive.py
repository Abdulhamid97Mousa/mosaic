"""Tests for OperatorLauncher interactive mode support."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


def _has_pyqt6() -> bool:
    """Check if PyQt6 is available."""
    try:
        from PyQt6 import QtCore  # noqa: F401
        return True
    except ImportError:
        return False


# Skip all tests if PyQt6 is not available
if not _has_pyqt6():
    # Create mock classes for tests that run without PyQt6
    class OperatorConfig:
        pass

    class OperatorLauncher:
        pass

    class OperatorProcessHandle:
        pass

    class OperatorLaunchError(Exception):
        pass
else:
    from gym_gui.services.operator import OperatorConfig
    from gym_gui.services.operator_launcher import (
        OperatorLauncher,
        OperatorProcessHandle,
        OperatorLaunchError,
    )


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestOperatorProcessHandleInteractive(unittest.TestCase):
    """Test OperatorProcessHandle interactive methods."""

    def _make_mock_handle(
        self,
        interactive: bool = True,
        is_running: bool = True,
        has_stdin: bool = True,
    ) -> OperatorProcessHandle:
        """Create a mock OperatorProcessHandle for testing."""
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = None if is_running else 0

        if has_stdin:
            mock_process.stdin = MagicMock()
            mock_process.stdin.write = MagicMock()
            mock_process.stdin.flush = MagicMock()
        else:
            mock_process.stdin = None

        mock_config = MagicMock(spec=OperatorConfig)

        return OperatorProcessHandle(
            operator_id="test_op",
            run_id="test_run_123",
            process=mock_process,
            log_path=Path("/tmp/test.log"),
            config=mock_config,
            interactive=interactive,
        )

    def test_send_command_success(self) -> None:
        """Should send JSON command to stdin."""
        handle = self._make_mock_handle(interactive=True, is_running=True)

        result = handle.send_command({"cmd": "step"})

        self.assertTrue(result)
        handle.process.stdin.write.assert_called_once()
        handle.process.stdin.flush.assert_called_once()

        # Verify JSON format
        written_data = handle.process.stdin.write.call_args[0][0]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "step")

    def test_send_command_not_interactive(self) -> None:
        """Should return False for non-interactive operator."""
        handle = self._make_mock_handle(interactive=False)

        result = handle.send_command({"cmd": "step"})

        self.assertFalse(result)
        handle.process.stdin.write.assert_not_called()

    def test_send_command_not_running(self) -> None:
        """Should return False if process not running."""
        handle = self._make_mock_handle(interactive=True, is_running=False)

        result = handle.send_command({"cmd": "step"})

        self.assertFalse(result)

    def test_send_command_no_stdin(self) -> None:
        """Should return False if no stdin pipe."""
        handle = self._make_mock_handle(interactive=True, has_stdin=False)

        result = handle.send_command({"cmd": "step"})

        self.assertFalse(result)

    def test_send_reset_with_seed(self) -> None:
        """Should send reset command with seed."""
        handle = self._make_mock_handle(interactive=True)

        result = handle.send_reset(seed=42)

        self.assertTrue(result)
        written_data = handle.process.stdin.write.call_args[0][0]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "reset")
        self.assertEqual(parsed["seed"], 42)

    def test_send_reset_without_seed(self) -> None:
        """Should send reset command without seed."""
        handle = self._make_mock_handle(interactive=True)

        result = handle.send_reset()

        self.assertTrue(result)
        written_data = handle.process.stdin.write.call_args[0][0]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "reset")
        self.assertNotIn("seed", parsed)

    def test_send_step(self) -> None:
        """Should send step command."""
        handle = self._make_mock_handle(interactive=True)

        result = handle.send_step()

        self.assertTrue(result)
        written_data = handle.process.stdin.write.call_args[0][0]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "step")

    def test_send_stop(self) -> None:
        """Should send stop command."""
        handle = self._make_mock_handle(interactive=True)

        result = handle.send_stop()

        self.assertTrue(result)
        written_data = handle.process.stdin.write.call_args[0][0]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "stop")

    def test_send_command_write_error(self) -> None:
        """Should handle write errors gracefully."""
        handle = self._make_mock_handle(interactive=True)
        handle.process.stdin.write.side_effect = IOError("Broken pipe")

        result = handle.send_command({"cmd": "step"})

        self.assertFalse(result)


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestOperatorLauncherInteractive(unittest.TestCase):
    """Test OperatorLauncher interactive mode."""

    def _make_config(self) -> OperatorConfig:
        """Create a test operator config."""
        return OperatorConfig(
            operator_id="test_operator",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="Test Operator",
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
            settings={
                "client_name": "vllm",
                "model_id": "test-model",
                "base_url": "http://localhost:8000/v1",
            },
        )

    @patch("gym_gui.services.operator_launcher.validated_popen")
    @patch("gym_gui.services.operator_launcher.VAR_OPERATORS_DIR", Path("/tmp/operators"))
    @patch("gym_gui.services.operator_launcher.VAR_TELEMETRY_DIR", Path("/tmp/telemetry"))
    @patch("gym_gui.services.operator_launcher.ensure_var_directories")
    def test_launch_interactive_uses_pipes(
        self,
        mock_ensure_dirs: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Should use stdin/stdout pipes in interactive mode."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("gym_gui.services.operator_launcher.VAR_OPERATORS_DIR", Path(tmpdir)):
                launcher = OperatorLauncher()
                config = self._make_config()

                handle = launcher.launch_operator(config, interactive=True)

        # Verify popen was called with stdin=PIPE
        call_kwargs = mock_popen.call_args[1]
        self.assertEqual(call_kwargs["stdin"], subprocess.PIPE)
        self.assertEqual(call_kwargs["stdout"], subprocess.PIPE)

        # Verify handle is marked as interactive
        self.assertTrue(handle.interactive)

    @patch("gym_gui.services.operator_launcher.validated_popen")
    @patch("gym_gui.services.operator_launcher.VAR_OPERATORS_DIR", Path("/tmp/operators"))
    @patch("gym_gui.services.operator_launcher.VAR_TELEMETRY_DIR", Path("/tmp/telemetry"))
    @patch("gym_gui.services.operator_launcher.ensure_var_directories")
    def test_launch_non_interactive_uses_log_file(
        self,
        mock_ensure_dirs: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Should use log file for stdout in non-interactive mode."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("gym_gui.services.operator_launcher.VAR_OPERATORS_DIR", Path(tmpdir)):
                launcher = OperatorLauncher()
                config = self._make_config()

                handle = launcher.launch_operator(config, interactive=False)

        # Verify popen was called with file for stdout (not PIPE)
        call_kwargs = mock_popen.call_args[1]
        self.assertNotEqual(call_kwargs.get("stdin"), subprocess.PIPE)
        self.assertNotEqual(call_kwargs["stdout"], subprocess.PIPE)

        # Verify handle is not marked as interactive
        self.assertFalse(handle.interactive)

    def test_build_llm_command_interactive_flag(self) -> None:
        """Should include --interactive flag when interactive=True."""
        launcher = OperatorLauncher()
        config = self._make_config()

        cmd_interactive = launcher._build_llm_command(config, "run123", interactive=True)
        cmd_normal = launcher._build_llm_command(config, "run123", interactive=False)

        self.assertIn("--interactive", cmd_interactive)
        self.assertNotIn("--interactive", cmd_normal)

    def test_build_llm_command_has_required_args(self) -> None:
        """Should include all required CLI arguments."""
        launcher = OperatorLauncher()
        config = self._make_config()

        cmd = launcher._build_llm_command(config, "run123", interactive=True)

        # Check required args
        self.assertIn("--run-id", cmd)
        self.assertIn("run123", cmd)
        self.assertIn("--env", cmd)
        self.assertIn("babyai", cmd)
        self.assertIn("--task", cmd)
        self.assertIn("BabyAI-GoToRedBall-v0", cmd)
        self.assertIn("--client", cmd)
        self.assertIn("vllm", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("test-model", cmd)
        self.assertIn("--base-url", cmd)
        self.assertIn("http://localhost:8000/v1", cmd)


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestOperatorProcessHandleDataclass(unittest.TestCase):
    """Test OperatorProcessHandle dataclass fields."""

    def test_interactive_field_default(self) -> None:
        """Interactive field should default to False."""
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_config = MagicMock(spec=OperatorConfig)

        handle = OperatorProcessHandle(
            operator_id="test",
            run_id="run",
            process=mock_process,
            log_path=Path("/tmp/test.log"),
            config=mock_config,
        )

        self.assertFalse(handle.interactive)

    def test_interactive_field_explicit(self) -> None:
        """Interactive field should be settable."""
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_config = MagicMock(spec=OperatorConfig)

        handle = OperatorProcessHandle(
            operator_id="test",
            run_id="run",
            process=mock_process,
            log_path=Path("/tmp/test.log"),
            config=mock_config,
            interactive=True,
        )

        self.assertTrue(handle.interactive)


if __name__ == "__main__":
    unittest.main()
