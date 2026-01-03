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


class TestMultiAgentCommands(unittest.TestCase):
    """Test multi-agent commands for PettingZoo game support.

    These commands allow the worker to act as an action-selector where
    the GUI owns the environment and workers just provide actions.
    """

    def _make_config(self) -> BarlogWorkerConfig:
        """Create a test config."""
        return BarlogWorkerConfig(
            run_id="test_multiagent",
            env_name="pettingzoo",
            task="chess_v6",
            client_name="openai",
            model_id="gpt-4o-mini",
            telemetry_dir="/tmp/test_telemetry",
            emit_jsonl=False,
        )

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_init_agent_command(self) -> None:
        """Should handle init_agent command and respond with agent_ready."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()
        captured_stdout = io.StringIO()

        # Mock agent creation to avoid actual LLM initialization
        with patch("barlog_worker.runtime.InteractiveRuntime._create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.prompt_builder = MagicMock()
            mock_create.return_value = mock_agent

            commands = (
                '{"cmd": "init_agent", "game_name": "chess_v6", "player_id": "player_0"}\n'
                '{"cmd": "stop"}\n'
            )
            with patch.object(sys, 'stdin', io.StringIO(commands)):
                with patch.object(sys, 'stdout', captured_stdout):
                    runtime = InteractiveRuntime(config)
                    runtime.run()

        output = captured_stdout.getvalue()
        lines = [l for l in output.strip().split('\n') if l]

        # Should have agent_ready message
        agent_ready_found = False
        for line in lines:
            try:
                msg = json.loads(line)
                if msg.get("type") == "agent_ready":
                    agent_ready_found = True
                    self.assertEqual(msg["mode"], "action_selector")
                    self.assertEqual(msg["game_name"], "chess_v6")
                    self.assertEqual(msg["player_id"], "player_0")
                    break
            except json.JSONDecodeError:
                pass
        self.assertTrue(agent_ready_found, "Expected agent_ready message")

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_select_action_without_init_returns_error(self) -> None:
        """Should return error if select_action called without init_agent."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()
        captured_stdout = io.StringIO()

        commands = (
            '{"cmd": "select_action", "observation": "test", "player_id": "player_0"}\n'
            '{"cmd": "stop"}\n'
        )
        with patch.object(sys, 'stdin', io.StringIO(commands)):
            with patch.object(sys, 'stdout', captured_stdout):
                runtime = InteractiveRuntime(config)
                runtime.run()

        output = captured_stdout.getvalue()
        lines = [l for l in output.strip().split('\n') if l]

        # Should have error message
        error_found = False
        for line in lines:
            try:
                msg = json.loads(line)
                if msg.get("type") == "error" and "not initialized" in msg.get("message", "").lower():
                    error_found = True
                    break
            except json.JSONDecodeError:
                pass
        self.assertTrue(error_found, "Expected error about agent not initialized")

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_select_action_returns_action(self) -> None:
        """Should return action_selected after init_agent and select_action."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()
        captured_stdout = io.StringIO()

        # Mock agent and its act method
        mock_response = MagicMock()
        mock_response.completion = "e2e4"
        mock_response.input_tokens = 100
        mock_response.output_tokens = 5

        with patch("barlog_worker.runtime.InteractiveRuntime._create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.prompt_builder = MagicMock()
            mock_agent.act.return_value = mock_response
            mock_create.return_value = mock_agent

            commands = (
                '{"cmd": "init_agent", "game_name": "chess_v6", "player_id": "player_0"}\n'
                '{"cmd": "select_action", "observation": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", '
                '"info": {"legal_moves": ["e2e4", "d2d4", "g1f3"]}, "player_id": "player_0"}\n'
                '{"cmd": "stop"}\n'
            )
            with patch.object(sys, 'stdin', io.StringIO(commands)):
                with patch.object(sys, 'stdout', captured_stdout):
                    runtime = InteractiveRuntime(config)
                    runtime.run()

        output = captured_stdout.getvalue()
        lines = [l for l in output.strip().split('\n') if l]

        # Should have action_selected message
        action_found = False
        for line in lines:
            try:
                msg = json.loads(line)
                if msg.get("type") == "action_selected":
                    action_found = True
                    self.assertEqual(msg["action_str"], "e2e4")
                    self.assertEqual(msg["player_id"], "player_0")
                    self.assertEqual(msg["input_tokens"], 100)
                    self.assertEqual(msg["output_tokens"], 5)
                    break
            except json.JSONDecodeError:
                pass
        self.assertTrue(action_found, "Expected action_selected message")

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_game_instruction_prompts(self) -> None:
        """Should have game-specific instruction prompts."""
        from barlog_worker.runtime import InteractiveRuntime

        config = self._make_config()
        runtime = InteractiveRuntime(config)

        # Chess prompt
        chess_prompt = runtime._get_game_instruction_prompt("chess_v6", "player_0")
        self.assertIn("chess", chess_prompt.lower())
        self.assertIn("uci", chess_prompt.lower())

        # Connect Four prompt
        c4_prompt = runtime._get_game_instruction_prompt("connect_four_v3", "player_1")
        self.assertIn("connect four", c4_prompt.lower())
        self.assertIn("column", c4_prompt.lower())

        # Go prompt
        go_prompt = runtime._get_game_instruction_prompt("go_v5", "black_0")
        self.assertIn("go", go_prompt.lower())

        # Unknown game gets generic prompt
        unknown_prompt = runtime._get_game_instruction_prompt("unknown_game", "player_x")
        self.assertIn("unknown_game", unknown_prompt)
        self.assertIn("player_x", unknown_prompt)


if __name__ == "__main__":
    unittest.main()
