"""Tests for barlog_worker runtime module, including InteractiveRuntime."""

import io
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from barlog_worker.config import BarlogWorkerConfig


def _has_runtime_deps() -> bool:
    """Check if runtime dependencies are available."""
    try:
        from omegaconf import OmegaConf  # noqa: F401
        return True
    except ImportError:
        return False


class TestStepTelemetry(unittest.TestCase):
    """Test StepTelemetry dataclass."""

    def test_step_telemetry_creation(self) -> None:
        """Should create StepTelemetry with all fields."""
        from barlog_worker.runtime import StepTelemetry

        step = StepTelemetry(
            run_id="test_run",
            episode_id="test_run-ep000000",
            step_index=0,
            observation="You see a red ball",
            action="go forward",
            reward=0.5,
            terminated=False,
            truncated=False,
        )

        self.assertEqual(step.run_id, "test_run")
        self.assertEqual(step.step_index, 0)
        self.assertEqual(step.reward, 0.5)
        self.assertFalse(step.terminated)

    def test_step_telemetry_auto_timestamp(self) -> None:
        """Should auto-generate timestamp if not provided."""
        from barlog_worker.runtime import StepTelemetry

        step = StepTelemetry(
            run_id="test",
            episode_id="test-ep0",
            step_index=0,
            observation="obs",
            action="act",
            reward=0.0,
            terminated=False,
            truncated=False,
        )

        self.assertTrue(len(step.timestamp) > 0)
        # Should be ISO format
        datetime.fromisoformat(step.timestamp)


class TestEpisodeTelemetry(unittest.TestCase):
    """Test EpisodeTelemetry dataclass."""

    def test_episode_telemetry_creation(self) -> None:
        """Should create EpisodeTelemetry with all fields."""
        from barlog_worker.runtime import EpisodeTelemetry

        episode = EpisodeTelemetry(
            run_id="test_run",
            episode_id="test_run-ep000000",
            episode_index=0,
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
            total_reward=1.5,
            num_steps=10,
            terminated=True,
            truncated=False,
            success=True,
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:00:05",
            duration_seconds=5.0,
        )

        self.assertEqual(episode.run_id, "test_run")
        self.assertEqual(episode.total_reward, 1.5)
        self.assertTrue(episode.success)
        self.assertEqual(episode.duration_seconds, 5.0)


class TestTelemetryEmitter(unittest.TestCase):
    """Test TelemetryEmitter class."""

    def test_emitter_creates_files(self) -> None:
        """Should create telemetry files when emit_jsonl=True."""
        from barlog_worker.runtime import TelemetryEmitter

        with tempfile.TemporaryDirectory() as tmpdir:
            emitter = TelemetryEmitter(
                run_id="test_emitter",
                telemetry_dir=tmpdir,
                emit_jsonl=True,
            )

            step_path = Path(tmpdir) / "test_emitter_steps.jsonl"
            episode_path = Path(tmpdir) / "test_emitter_episodes.jsonl"

            self.assertTrue(step_path.exists())
            self.assertTrue(episode_path.exists())

            emitter.close()

    def test_emitter_no_files_when_disabled(self) -> None:
        """Should not create files when emit_jsonl=False."""
        from barlog_worker.runtime import TelemetryEmitter

        with tempfile.TemporaryDirectory() as tmpdir:
            emitter = TelemetryEmitter(
                run_id="test_no_jsonl",
                telemetry_dir=tmpdir,
                emit_jsonl=False,
            )

            step_path = Path(tmpdir) / "test_no_jsonl_steps.jsonl"
            self.assertFalse(step_path.exists())

            emitter.close()


class TestInteractiveRuntimeProtocol(unittest.TestCase):
    """Test InteractiveRuntime command protocol (without actual BALROG deps)."""

    def _make_config(self) -> BarlogWorkerConfig:
        """Create a test config."""
        return BarlogWorkerConfig(
            run_id="test_interactive",
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
            client_name="openai",
            model_id="gpt-4o-mini",
            telemetry_dir="/tmp/test_telemetry",
            emit_jsonl=False,
        )

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_interactive_runtime_init_message(self) -> None:
        """Should emit init message on start."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()

        # Capture stdout
        captured_stdout = io.StringIO()

        with patch.object(sys, 'stdin', io.StringIO('{"cmd": "stop"}\n')):
            with patch.object(sys, 'stdout', captured_stdout):
                runtime = InteractiveRuntime(config)
                runtime.run()

        output = captured_stdout.getvalue()
        lines = [l for l in output.strip().split('\n') if l]

        # Should have init message
        self.assertTrue(len(lines) >= 1)
        init_msg = json.loads(lines[0])
        self.assertEqual(init_msg["type"], "init")
        self.assertEqual(init_msg["run_id"], "test_interactive")

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_interactive_runtime_stop_command(self) -> None:
        """Should handle stop command and exit cleanly."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()
        captured_stdout = io.StringIO()

        with patch.object(sys, 'stdin', io.StringIO('{"cmd": "stop"}\n')):
            with patch.object(sys, 'stdout', captured_stdout):
                runtime = InteractiveRuntime(config)
                runtime.run()

        output = captured_stdout.getvalue()
        lines = [l for l in output.strip().split('\n') if l]

        # Should have stopped message
        stopped_msgs = [json.loads(l) for l in lines if "stopped" in l]
        self.assertTrue(len(stopped_msgs) >= 1)
        self.assertEqual(stopped_msgs[0]["type"], "stopped")

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_interactive_runtime_ping_pong(self) -> None:
        """Should respond to ping with pong."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()
        captured_stdout = io.StringIO()

        commands = '{"cmd": "ping"}\n{"cmd": "stop"}\n'
        with patch.object(sys, 'stdin', io.StringIO(commands)):
            with patch.object(sys, 'stdout', captured_stdout):
                runtime = InteractiveRuntime(config)
                runtime.run()

        output = captured_stdout.getvalue()
        lines = [l for l in output.strip().split('\n') if l]

        # Should have pong response
        pong_msgs = [json.loads(l) for l in lines if '"type": "pong"' in l or '"pong"' in l]
        self.assertTrue(len(pong_msgs) >= 1)

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_interactive_runtime_invalid_json(self) -> None:
        """Should handle invalid JSON gracefully."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()
        captured_stdout = io.StringIO()

        commands = 'not valid json\n{"cmd": "stop"}\n'
        with patch.object(sys, 'stdin', io.StringIO(commands)):
            with patch.object(sys, 'stdout', captured_stdout):
                runtime = InteractiveRuntime(config)
                runtime.run()

        output = captured_stdout.getvalue()
        lines = [l for l in output.strip().split('\n') if l]

        # Should have error message for invalid JSON
        error_msgs = [json.loads(l) for l in lines if '"error"' in l.lower()]
        self.assertTrue(len(error_msgs) >= 1)

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_interactive_runtime_unknown_command(self) -> None:
        """Should handle unknown command gracefully."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()
        captured_stdout = io.StringIO()

        commands = '{"cmd": "unknown_cmd"}\n{"cmd": "stop"}\n'
        with patch.object(sys, 'stdin', io.StringIO(commands)):
            with patch.object(sys, 'stdout', captured_stdout):
                runtime = InteractiveRuntime(config)
                runtime.run()

        output = captured_stdout.getvalue()
        lines = [l for l in output.strip().split('\n') if l]

        # Should have error message for unknown command
        error_msgs = [json.loads(l) for l in lines if '"error"' in l.lower()]
        self.assertTrue(len(error_msgs) >= 1)

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_interactive_runtime_step_without_reset(self) -> None:
        """Should error on step command before reset."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()
        captured_stdout = io.StringIO()

        commands = '{"cmd": "step"}\n{"cmd": "stop"}\n'
        with patch.object(sys, 'stdin', io.StringIO(commands)):
            with patch.object(sys, 'stdout', captured_stdout):
                runtime = InteractiveRuntime(config)
                runtime.run()

        output = captured_stdout.getvalue()
        lines = [l for l in output.strip().split('\n') if l]

        # Should have error message about not initialized
        error_found = False
        for line in lines:
            try:
                msg = json.loads(line)
                if msg.get("type") == "error" and "not initialized" in msg.get("message", "").lower():
                    error_found = True
                    break
            except json.JSONDecodeError:
                pass
        self.assertTrue(error_found)


class TestCLIInteractiveFlag(unittest.TestCase):
    """Test CLI --interactive flag."""

    def test_cli_parses_interactive_flag(self) -> None:
        """CLI should parse --interactive flag."""
        from barlog_worker.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--run-id", "test",
            "--interactive",
        ])

        self.assertTrue(args.interactive)

    def test_cli_interactive_default_false(self) -> None:
        """CLI should default interactive to False."""
        from barlog_worker.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--run-id", "test"])

        self.assertFalse(args.interactive)

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_main_uses_interactive_runtime(self) -> None:
        """Main should use InteractiveRuntime when --interactive is set."""
        from barlog_worker.cli import main
        import barlog_worker.runtime as runtime_module

        mock_interactive_runtime = MagicMock()

        with patch.object(runtime_module, "InteractiveRuntime", return_value=mock_interactive_runtime):
            with patch.object(sys, 'stdin', io.StringIO('{"cmd": "stop"}\n')):
                result = main(["--run-id", "test_interactive", "--interactive"])

        mock_interactive_runtime.run.assert_called_once()

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_main_uses_batch_runtime_without_flag(self) -> None:
        """Main should use BarlogWorkerRuntime when --interactive is not set."""
        from barlog_worker.cli import main
        import barlog_worker.runtime as runtime_module

        mock_batch_runtime = MagicMock()

        with patch.object(runtime_module, "BarlogWorkerRuntime", return_value=mock_batch_runtime):
            result = main(["--run-id", "test_batch"])

        mock_batch_runtime.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
