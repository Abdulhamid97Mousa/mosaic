# gym_gui/services/operator_launcher.py

from __future__ import annotations

"""Utilities for launching and supervising operator subprocess workers.

This module provides subprocess management for LLM and RL
operator workers, allowing the GUI to run multiple operators in parallel
for side-by-side comparison.

Architecture Overview:
    OperatorLauncher creates subprocess workers based on OperatorConfig:

    LLM Operators -> LLM_worker subprocess -> BALROG agent -> LLM API
    RL Operators  -> RL_worker subprocess -> Policy inference

    Each subprocess:
    - Writes logs to VAR_OPERATORS_LOGS_DIR
    - Emits telemetry to VAR_OPERATORS_TELEMETRY_DIR (JSONL + stdout)
    - Can be started/stopped independently
"""

import logging
import os
import select
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from typing import IO, Any, Dict, List, Optional

from gym_gui.config.paths import VAR_OPERATORS_LOGS_DIR, VAR_OPERATORS_TELEMETRY_DIR, ensure_var_directories
from gym_gui.core.subprocess_validation import validated_popen
from gym_gui.services.operator import OperatorConfig
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_OPERATOR_INTERACTIVE_LAUNCHED,
    LOG_OPERATOR_RESET_COMMAND_SENT,
    LOG_OPERATOR_STEP_COMMAND_SENT,
    LOG_OPERATOR_STOP_COMMAND_SENT,
    LOG_OPERATOR_COMMAND_FAILED,
    LOG_OPERATOR_INIT_AGENT_SENT,
    LOG_OPERATOR_SELECT_ACTION_SENT,
    LOG_OPERATOR_MULTIAGENT_LAUNCHED,
    LOG_OPERATOR_MULTIAGENT_INIT_FAILED,
    LOG_OPERATOR_MULTIAGENT_ACTION_FAILED,
)


LOGGER = logging.getLogger("gym_gui.services.operator_launcher")


class OperatorLaunchError(RuntimeError):
    """Raised when an operator worker subprocess fails to launch."""


@dataclass
class OperatorProcessHandle:
    """Tracks a launched operator subprocess.

    Attributes:
        operator_id: The operator's unique ID from the GUI.
        run_id: The run ID for telemetry routing.
        process: The subprocess.Popen handle.
        log_path: Path to the operator's log file.
        config: The operator configuration used to launch.
        interactive: Whether this operator is in interactive mode.
    """

    operator_id: str
    run_id: str
    process: subprocess.Popen[str]
    log_path: Path
    config: OperatorConfig
    interactive: bool = False
    _log_file: Optional[IO[str]] = field(default=None, repr=False)

    @property
    def pid(self) -> int:
        """Return the process ID."""
        return self.process.pid

    @property
    def is_running(self) -> bool:
        """Check if the process is still running."""
        return self.process.poll() is None

    @property
    def return_code(self) -> Optional[int]:
        """Return the exit code if terminated, None if still running."""
        return self.process.poll()

    def send_command(self, cmd: Dict[str, Any]) -> bool:
        """Send a JSON command to the interactive subprocess.

        Args:
            cmd: Command dictionary, e.g. {"cmd": "step"} or {"cmd": "reset", "seed": 42}

        Returns:
            True if command was sent, False if process not running or not interactive.
        """
        if not self.interactive:
            log_constant(
                LOGGER, LOG_OPERATOR_COMMAND_FAILED,
                message="Cannot send command to non-interactive operator",
                extra={"operator_id": self.operator_id, "cmd": cmd.get("cmd")},
            )
            return False

        if not self.is_running:
            log_constant(
                LOGGER, LOG_OPERATOR_COMMAND_FAILED,
                message="Cannot send command to terminated operator",
                extra={"operator_id": self.operator_id, "cmd": cmd.get("cmd")},
            )
            return False

        if self.process.stdin is None:
            log_constant(
                LOGGER, LOG_OPERATOR_COMMAND_FAILED,
                message="Operator has no stdin pipe",
                extra={"operator_id": self.operator_id, "cmd": cmd.get("cmd")},
            )
            return False

        try:
            line = json.dumps(cmd) + "\n"
            self.process.stdin.write(line)
            self.process.stdin.flush()
            LOGGER.debug("Sent command to operator %s: %s", self.operator_id, cmd)
            return True
        except Exception as exc:
            log_constant(
                LOGGER, LOG_OPERATOR_COMMAND_FAILED,
                message=f"Failed to send command: {exc}",
                extra={"operator_id": self.operator_id, "cmd": cmd.get("cmd")},
                exc_info=exc,
            )
            return False

    def send_reset(self, seed: Optional[int] = None, max_steps: Optional[int] = None) -> bool:
        """Send reset command to initialize environment with seed and max_steps.

        Args:
            seed: Random seed for environment reset
            max_steps: Maximum steps per episode before truncation (optional)

        Returns:
            True if command was sent successfully
        """
        cmd: Dict[str, Any] = {"cmd": "reset"}
        if seed is not None:
            cmd["seed"] = seed

        # Get max_steps from config if not provided as parameter
        if max_steps is None and hasattr(self.config, 'max_steps'):
            max_steps = self.config.max_steps

        if max_steps is not None:
            cmd["max_steps"] = max_steps

        result = self.send_command(cmd)
        if result:
            log_constant(
                LOGGER, LOG_OPERATOR_RESET_COMMAND_SENT,
                message=f"Reset command sent with seed={seed}, max_steps={max_steps}",
                extra={"operator_id": self.operator_id, "seed": seed, "max_steps": max_steps},
            )
        return result

    def send_step(self) -> bool:
        """Send step command to execute one environment step."""
        result = self.send_command({"cmd": "step"})
        if result:
            log_constant(
                LOGGER, LOG_OPERATOR_STEP_COMMAND_SENT,
                message="Step command sent",
                extra={"operator_id": self.operator_id},
            )
        return result

    def send_step_with_action(self, action: int) -> bool:
        """Send step command with action index for human operators.

        Args:
            action: The action index to execute.

        Returns:
            True if command was sent, False otherwise.
        """
        result = self.send_command({"cmd": "step", "action": action})
        if result:
            log_constant(
                LOGGER, LOG_OPERATOR_STEP_COMMAND_SENT,
                message=f"Step command sent with action={action}",
                extra={"operator_id": self.operator_id, "action": action},
            )
        return result

    def send_init_agent(
        self,
        game_name: str,
        player_id: str,
        instruction_prompt: Optional[str] = None,
    ) -> bool:
        """Send init_agent command for multi-agent action-selector mode.

        In action-selector mode, the worker doesn't own the environment.
        It just provides actions when given observations.

        Args:
            game_name: Name of the game (e.g., "chess_v6").
            player_id: Which player this worker controls (e.g., "player_0").
            instruction_prompt: Optional custom instruction for the LLM.

        Returns:
            True if command was sent successfully.
        """
        cmd: Dict[str, Any] = {
            "cmd": "init_agent",
            "game_name": game_name,
            "player_id": player_id,
        }
        if instruction_prompt:
            cmd["instruction_prompt"] = instruction_prompt

        result = self.send_command(cmd)
        if result:
            log_constant(
                LOGGER, LOG_OPERATOR_INIT_AGENT_SENT,
                message=f"Init agent command sent to {self.operator_id} for {game_name} as {player_id}",
                extra={
                    "operator_id": self.operator_id,
                    "game_name": game_name,
                    "player_id": player_id,
                },
            )
        return result

    def send_select_action(
        self,
        observation: Any,
        player_id: str,
        info: Optional[Dict[str, Any]] = None,
        action_mask: Optional[List[bool]] = None,
    ) -> bool:
        """Send select_action command for multi-agent games.

        The worker will use its LLM to select an action based on the
        observation and return it. The GUI then executes the action
        on the shared environment.

        Args:
            observation: Current game state (string or dict).
            player_id: Which player is acting.
            info: Additional info (e.g., legal_moves).
            action_mask: Optional boolean mask of valid actions.

        Returns:
            True if command was sent successfully.
        """
        cmd: Dict[str, Any] = {
            "cmd": "select_action",
            "observation": observation,
            "player_id": player_id,
        }
        if info:
            cmd["info"] = info
        if action_mask:
            cmd["action_mask"] = action_mask

        result = self.send_command(cmd)
        if result:
            log_constant(
                LOGGER, LOG_OPERATOR_SELECT_ACTION_SENT,
                message=f"Select action command sent to {self.operator_id} for {player_id}",
                extra={
                    "operator_id": self.operator_id,
                    "player_id": player_id,
                },
            )
        return result

    def send_stop(self) -> bool:
        """Send stop command to terminate gracefully."""
        result = self.send_command({"cmd": "stop"})
        if result:
            log_constant(
                LOGGER, LOG_OPERATOR_STOP_COMMAND_SENT,
                message="Stop command sent",
                extra={"operator_id": self.operator_id},
            )
        return result

    def try_read_response(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Try to read one JSON response from stdout (non-blocking).

        Args:
            timeout: Seconds to wait for data (0 = no wait, instant check).

        Returns:
            Parsed JSON dict if available, None if no data or not interactive.
        """
        if not self.interactive:
            return None

        if not self.is_running and self.process.stdout is None:
            return None

        if self.process.stdout is None:
            return None

        try:
            # Use select for non-blocking check (Unix only)
            readable, _, _ = select.select([self.process.stdout], [], [], timeout)
            if not readable:
                return None

            line = self.process.stdout.readline()
            if not line:
                return None

            return json.loads(line.strip())
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            LOGGER.debug(
                "Failed to read response from operator %s: %s",
                self.operator_id, exc
            )
            return None

    def poll_responses(self, max_responses: int = 100) -> list[Dict[str, Any]]:
        """Read all available responses from stdout (non-blocking).

        Args:
            max_responses: Maximum number of responses to read in one call.

        Returns:
            List of parsed JSON responses (may be empty).
        """
        responses = []
        for _ in range(max_responses):
            response = self.try_read_response(timeout=0.0)
            if response is None:
                break
            responses.append(response)
        return responses

    def read_response(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Read one JSON response from stdout (blocking with timeout).

        Args:
            timeout: Seconds to wait for response.

        Returns:
            Parsed JSON dict, or None on timeout/error.
        """
        return self.try_read_response(timeout=timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the operator subprocess.

        Args:
            timeout: Seconds to wait before force-killing.
        """
        if self.process.poll() is not None:
            self._close_log()
            return

        LOGGER.debug("Stopping operator %s (pid=%s)", self.operator_id, self.process.pid)
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            LOGGER.warning(
                "Operator %s did not exit on SIGTERM - forcing kill",
                self.operator_id,
                extra={"pid": self.process.pid}
            )
            self.process.kill()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                LOGGER.error(
                    "Operator %s failed to terminate after SIGKILL",
                    self.operator_id,
                    extra={"pid": self.process.pid}
                )
        finally:
            self._close_log()

    def _close_log(self) -> None:
        """Close the log file handle."""
        if self._log_file and not self._log_file.closed:
            try:
                self._log_file.flush()
            except Exception as exc:
                LOGGER.warning(
                    "Failed to flush operator log file: %s",
                    exc,
                    extra={"log_path": str(self.log_path)}
                )
            self._log_file.close()


@dataclass
class MultiAgentOperatorHandle:
    """Manages multiple worker processes for a multi-agent operator.

    In multi-agent mode (e.g., chess, Go), one operator controls multiple
    players, each with its own worker subprocess. The GUI owns the shared
    environment and routes turns to the appropriate worker.

    Attributes:
        operator_id: The operator's unique ID.
        config: The multi-agent operator configuration.
        player_handles: Dict mapping player_id to OperatorProcessHandle.
        game_name: The PettingZoo game being played.
    """

    operator_id: str
    config: OperatorConfig
    player_handles: Dict[str, OperatorProcessHandle] = field(default_factory=dict)
    game_name: str = ""

    @property
    def is_running(self) -> bool:
        """Check if all player workers are running."""
        if not self.player_handles:
            return False
        return all(h.is_running for h in self.player_handles.values())

    @property
    def player_ids(self) -> List[str]:
        """Get list of player IDs."""
        return list(self.player_handles.keys())

    def get_handle(self, player_id: str) -> Optional[OperatorProcessHandle]:
        """Get the handle for a specific player."""
        return self.player_handles.get(player_id)

    def send_init_agents(self) -> bool:
        """Send init_agent command to all player workers.

        Returns:
            True if all commands were sent successfully.
        """
        success = True
        for player_id, handle in self.player_handles.items():
            if not handle.send_init_agent(self.game_name, player_id):
                log_constant(
                    LOGGER, LOG_OPERATOR_MULTIAGENT_INIT_FAILED,
                    message=f"Failed to init agent for {player_id} in operator {self.operator_id}",
                    extra={
                        "operator_id": self.operator_id,
                        "player_id": player_id,
                        "game_name": self.game_name,
                    },
                )
                success = False
        return success

    def send_select_action(
        self,
        player_id: str,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
        action_mask: Optional[List[bool]] = None,
    ) -> bool:
        """Send select_action command to a specific player's worker.

        Args:
            player_id: Which player should select an action.
            observation: Current game state.
            info: Additional info (e.g., legal_moves).
            action_mask: Optional boolean mask of valid actions.

        Returns:
            True if command was sent successfully.
        """
        handle = self.player_handles.get(player_id)
        if handle is None:
            log_constant(
                LOGGER, LOG_OPERATOR_MULTIAGENT_ACTION_FAILED,
                message=f"No handle for player {player_id} in operator {self.operator_id}",
                extra={
                    "operator_id": self.operator_id,
                    "player_id": player_id,
                },
            )
            return False
        return handle.send_select_action(observation, player_id, info, action_mask)

    def read_response(
        self,
        player_id: str,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Read response from a specific player's worker.

        Args:
            player_id: Which player to read from.
            timeout: Seconds to wait for response.

        Returns:
            Parsed JSON response, or None on timeout/error.
        """
        handle = self.player_handles.get(player_id)
        if handle is None:
            return None
        return handle.read_response(timeout=timeout)

    def poll_all_responses(self, max_per_player: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Poll responses from all player workers.

        Returns:
            Dict mapping player_id to list of responses.
        """
        results = {}
        for player_id, handle in self.player_handles.items():
            results[player_id] = handle.poll_responses(max_responses=max_per_player)
        return results

    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all player workers."""
        for player_id, handle in self.player_handles.items():
            LOGGER.debug("Stopping worker for %s", player_id)
            handle.send_stop()
            handle.stop(timeout=timeout)


class OperatorLauncher:
    """
    Launches and manages operator subprocess workers.

    This class handles spawning subprocess workers for both LLM
    and RL operator types, managing their lifecycle and log files.

    Example::

        launcher = OperatorLauncher()

        # Launch an LLM operator
        config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="balrog_worker",
            display_name="GPT-4 Agent",
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
            settings={
                "client_name": "vllm",
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "base_url": "http://localhost:8000/v1",
            }
        )
        handle = launcher.launch_operator(config)

        # Check status
        if handle.is_running:
            print(f"Operator running with PID {handle.pid}")

        # Stop operator
        launcher.stop_operator("op_0")
    """

    def __init__(self, python_executable: Optional[str] = None) -> None:
        """Initialize the launcher.

        Args:
            python_executable: Path to Python executable (defaults to current).
        """
        self._python_executable = python_executable or sys.executable
        self._handles: Dict[str, OperatorProcessHandle] = {}

    def launch_operator(
        self,
        config: OperatorConfig,
        *,
        run_id: Optional[str] = None,
        interactive: bool = False,
    ) -> OperatorProcessHandle:
        """Launch an operator subprocess based on its configuration.

        Args:
            config: The operator configuration.
            run_id: Optional run ID (auto-generated if not provided).
            interactive: If True, launch in interactive mode for step-by-step control.
                        The subprocess will read commands from stdin and emit telemetry
                        to stdout, enabling scientific comparison with lock-step execution.

        Returns:
            A handle for managing the launched subprocess.

        Raises:
            OperatorLaunchError: If the subprocess fails to start.
        """
        run_id = run_id or f"{config.operator_id}_{uuid.uuid4().hex[:8]}"

        ensure_var_directories()
        VAR_OPERATORS_LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Create log file for this operator
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = VAR_OPERATORS_LOGS_DIR / f"{config.operator_id}_{timestamp}.log"
        log_file = log_path.open("a", encoding="utf-8")

        # Build command based on operator type
        if config.operator_type == "llm":
            cmd = self._build_llm_command(config, run_id, interactive=interactive)
        elif config.operator_type == "rl":
            cmd = self._build_rl_command(config, run_id, interactive=interactive)
        elif config.operator_type == "human":
            cmd = self._build_human_command(config, run_id)
        elif config.operator_type == "baseline":
            cmd = self._build_baseline_command(config, run_id, interactive=interactive)
        else:
            log_file.close()
            raise OperatorLaunchError(f"Unknown operator type: {config.operator_type}")

        # Set up environment
        env = os.environ.copy()
        env.setdefault("QT_DEBUG_PLUGINS", "0")

        # Clear proxy settings for vLLM (local inference on localhost)
        # SOCKS proxies can block httpx calls even for localhost
        # Keep proxy for remote providers like OpenRouter
        settings = config.settings or {}
        if settings.get("client_name") == "vllm":
            for proxy_var in ["ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                              "HTTPS_PROXY", "https_proxy"]:
                env.pop(proxy_var, None)

        # Pass telemetry directory
        env["TELEMETRY_DIR"] = str(VAR_OPERATORS_TELEMETRY_DIR)
        env["OPERATOR_RUN_ID"] = run_id
        env["OPERATOR_ID"] = config.operator_id

        LOGGER.info(
            "Launching operator %s (%s) with run_id=%s interactive=%s",
            config.operator_id,
            config.operator_type,
            run_id,
            interactive
        )
        LOGGER.debug("Command: %s", " ".join(cmd))

        try:
            # In interactive mode, we need stdin pipe for sending commands
            # and stdout pipe for receiving telemetry
            if interactive:
                process = validated_popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            else:
                process = validated_popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
        except Exception as exc:
            log_file.close()
            raise OperatorLaunchError(
                f"Failed to launch operator {config.operator_id}: {exc}"
            ) from exc

        handle = OperatorProcessHandle(
            operator_id=config.operator_id,
            run_id=run_id,
            process=process,
            log_path=log_path,
            interactive=interactive,
            config=config,
            _log_file=log_file,
        )
        self._handles[config.operator_id] = handle

        if interactive:
            log_constant(
                LOGGER, LOG_OPERATOR_INTERACTIVE_LAUNCHED,
                message=f"Operator launched in interactive mode",
                extra={
                    "operator_id": config.operator_id,
                    "run_id": run_id,
                    "pid": process.pid,
                    "log_path": str(log_path),
                    "operator_type": config.operator_type,
                    "env_name": config.env_name,
                    "task": config.task,
                },
            )
        else:
            LOGGER.info(
                "Operator %s launched successfully (pid=%s, log=%s)",
                config.operator_id,
                process.pid,
                log_path
            )
        return handle

    def launch_multiagent_operator(
        self,
        config: OperatorConfig,
        *,
        run_id: Optional[str] = None,
    ) -> MultiAgentOperatorHandle:
        """Launch a multi-agent operator with one worker per player.

        For multi-agent games (chess, Go, connect-four), the GUI owns the
        shared environment. This method launches a worker subprocess for
        each player, where each worker acts as an action-selector.

        Args:
            config: Multi-agent operator configuration with workers dict.
            run_id: Optional base run ID (players get suffixed IDs).

        Returns:
            MultiAgentOperatorHandle for managing all player workers.

        Raises:
            OperatorLaunchError: If any worker fails to launch.
            ValueError: If config is not a multi-agent configuration.
        """
        if not config.is_multiagent:
            raise ValueError(
                f"Operator {config.operator_id} is not multi-agent. "
                f"Use launch_operator for single-agent operators."
            )

        base_run_id = run_id or f"{config.operator_id}_{uuid.uuid4().hex[:8]}"
        game_name = config.task  # e.g., "chess_v6"

        player_handles: Dict[str, OperatorProcessHandle] = {}

        for player_id, worker_assignment in config.workers.items():
            # Create a single-agent config for this player's worker
            player_run_id = f"{base_run_id}_{player_id}"
            player_config = OperatorConfig.single_agent(
                operator_id=f"{config.operator_id}_{player_id}",
                display_name=f"{config.display_name} - {player_id}",
                worker_id=worker_assignment.worker_id,
                worker_type=worker_assignment.worker_type,
                env_name="pettingzoo",  # Worker knows it's action-selector mode
                task=game_name,
                settings=worker_assignment.settings,
            )

            try:
                # Launch in interactive mode for IPC
                handle = self.launch_operator(
                    player_config,
                    run_id=player_run_id,
                    interactive=True,
                )
                player_handles[player_id] = handle
            except OperatorLaunchError as e:
                # Clean up already-launched workers on failure
                log_constant(
                    LOGGER, LOG_OPERATOR_MULTIAGENT_INIT_FAILED,
                    message=f"Failed to launch worker for {player_id}: {e} - cleaning up",
                    extra={
                        "operator_id": config.operator_id,
                        "player_id": player_id,
                        "error": str(e),
                    },
                )
                for launched_handle in player_handles.values():
                    try:
                        launched_handle.stop(timeout=2.0)
                    except Exception:
                        pass
                raise

        multi_handle = MultiAgentOperatorHandle(
            operator_id=config.operator_id,
            config=config,
            player_handles=player_handles,
            game_name=game_name,
        )

        log_constant(
            LOGGER, LOG_OPERATOR_MULTIAGENT_LAUNCHED,
            message=f"Multi-agent operator {config.operator_id} launched with {len(player_handles)} players",
            extra={
                "operator_id": config.operator_id,
                "player_count": len(player_handles),
                "player_ids": list(player_handles.keys()),
                "game_name": game_name,
            },
        )

        return multi_handle

    def _build_llm_command(
        self,
        config: OperatorConfig,
        run_id: str,
        *,
        interactive: bool = False,
    ) -> list[str]:
        """Build command line for LLM operator.

        Dispatches to the appropriate worker based on config.worker_id:
        - chess_worker: For chess games (PettingZoo chess_v6)
        - mosaic_llm_worker: MOSAIC multi-agent LLM with Theory of Mind
        - balrog_worker: BALROG benchmark environments (default)

        Args:
            config: Operator configuration with LLM settings.
            run_id: Run ID for telemetry.
            interactive: If True, add --interactive flag for step-by-step control.

        Returns:
            Command arguments list.
        """
        settings = config.settings or {}

        # Get LLM-specific settings
        client_name = settings.get("client_name", "openai")
        model_id = settings.get("model_id", "gpt-4o-mini")
        base_url = settings.get("base_url")
        api_key = settings.get("api_key")
        agent_type = settings.get("agent_type", "naive")
        num_episodes = settings.get("num_episodes", 5)
        max_steps = settings.get("max_steps", 100)
        temperature = settings.get("temperature", 0.7)

        # Dispatch based on worker_id
        worker_id = config.worker_id or "balrog_worker"
        is_chess = config.task and "chess" in config.task.lower()

        if is_chess:
            # Chess worker with multi-turn conversation and retry logic
            cmd = [
                self._python_executable,
                "-m", "chess_worker.cli",
                "--run-id", run_id,
                "--client-name", client_name,
                "--model-id", model_id,
                "--temperature", str(temperature),
                "--telemetry-dir", str(VAR_OPERATORS_TELEMETRY_DIR),
            ]

            # Add optional base URL (for vLLM)
            if base_url:
                cmd.extend(["--base-url", base_url])

            # Add API key if provided
            if api_key:
                cmd.extend(["--api-key", api_key])

            # Chess-specific settings
            max_retries = settings.get("max_retries", 3)
            max_dialog_turns = settings.get("max_dialog_turns", 10)
            cmd.extend(["--max-retries", str(max_retries)])
            cmd.extend(["--max-dialog-turns", str(max_dialog_turns)])

        elif worker_id == "mosaic_llm_worker":
            # MOSAIC LLM Worker - uses llm_worker.cli with BALROG-compatible arguments
            cmd = [
                self._python_executable,
                "-m", "llm_worker.cli",
                "--run-id", run_id,
                "--env", config.env_name,  # CLI uses --env not --env-name
                "--task", config.task,
                "--client", client_name,
                "--model", model_id,
                "--agent-type", agent_type,
                "--num-episodes", str(num_episodes),
                "--max-steps", str(max_steps),
                "--temperature", str(temperature),
                "--telemetry-dir", str(VAR_OPERATORS_TELEMETRY_DIR),
                "--render-mode", "rgb_array",  # Required for GUI display
                "-v",  # Verbose logging
            ]

            # Add interactive mode flag for step-by-step GUI control
            if interactive:
                cmd.append("--interactive")

            # Add optional base URL (for vLLM)
            if base_url:
                cmd.extend(["--base-url", base_url])  # CLI uses --base-url not --api-base-url

            # Add API key if provided
            if api_key:
                cmd.extend(["--api-key", api_key])

        else:
            # BALROG worker for other environments (default)
            cmd = [
                self._python_executable,
                "-m", "balrog_worker.cli",
                "--run-id", run_id,
                "--env", config.env_name,
                "--task", config.task,
                "--client", client_name,
                "--model", model_id,
                "--agent-type", agent_type,
                "--num-episodes", str(num_episodes),
                "--max-steps", str(max_steps),
                "--temperature", str(temperature),
                "--telemetry-dir", str(VAR_OPERATORS_TELEMETRY_DIR),
                "-v",  # Verbose logging
            ]

            # Add interactive mode flag for step-by-step GUI control
            if interactive:
                cmd.append("--interactive")

            # Add optional base URL (for vLLM)
            if base_url:
                cmd.extend(["--base-url", base_url])

            # Add API key if provided
            if api_key:
                cmd.extend(["--api-key", api_key])

            # Add max_image_history (VLM mode: 0=text-only, >=1=vision)
            max_image_history = settings.get("max_image_history", 0)
            cmd.extend(["--max-image-history", str(max_image_history)])

            # Add render mode for GUI display
            cmd.extend(["--render-mode", "rgb_array"])

        return cmd

    def _build_rl_command(
        self,
        config: OperatorConfig,
        run_id: str,
        *,
        interactive: bool = False,
    ) -> list[str]:
        """Build command line for RL operator.

        Uses cleanrl_worker for policy evaluation. In interactive mode,
        the worker reads commands from stdin and emits telemetry to stdout,
        following the same IPC protocol as balrog_worker.

        Args:
            config: Operator configuration with RL settings.
            run_id: Run ID for telemetry.
            interactive: If True, add --interactive flag for step-by-step control.

        Returns:
            Command arguments list.

        Raises:
            OperatorLaunchError: If required settings are missing.
        """
        settings = config.settings or {}

        # Get RL-specific settings
        policy_path = settings.get("policy_path")
        algorithm = settings.get("algorithm", "ppo")

        # Validate required settings
        if not policy_path:
            raise OperatorLaunchError(
                f"RL operator {config.operator_id} requires 'policy_path' in settings"
            )

        # Determine environment ID (task field contains the full env ID)
        env_id = config.task
        if not env_id:
            raise OperatorLaunchError(
                f"RL operator {config.operator_id} requires 'task' (environment ID)"
            )

        # Dispatch based on worker_id
        if config.worker_id == "xuance_worker":
            # XuanCe MARL worker (IPPO, MAPPO, etc.)
            # InteractiveRuntime reads commands from stdin/stdout.
            cmd = [
                self._python_executable,
                "-m", "xuance_worker.cli",
                "--interactive",
                "--env-id", env_id,
                "--method", algorithm,
                "--policy-path", str(policy_path),
            ]

            seed = settings.get("seed")
            if seed is not None:
                cmd.extend(["--seed", str(seed)])

        else:
            # Default: CleanRL worker
            cmd = [
                self._python_executable,
                "-m", "cleanrl_worker.cli",
                "--interactive",  # Always use interactive mode for operators
                "--run-id", run_id,
                "--algo", algorithm,
                "--env-id", env_id,
                "--policy-path", str(policy_path),
            ]

            seed = settings.get("seed")
            if seed is not None:
                cmd.extend(["--seed", str(seed)])

            if settings.get("verbose"):
                cmd.append("--verbose")

        LOGGER.info(
            "Built RL command for operator %s | worker=%s algo=%s env=%s policy=%s",
            config.operator_id,
            config.worker_id,
            algorithm,
            env_id,
            policy_path,
        )

        return cmd

    def _build_human_command(
        self,
        config: OperatorConfig,
        run_id: str,
    ) -> list[str]:
        """Build command line for Human operator.

        In interactive mode (default for single-agent envs), the human_worker
        owns the gymnasium environment. The GUI sends action commands, and
        the worker returns rendered frames.

        In board-game mode (for PettingZoo games like chess), the GUI owns
        the environment and the worker just provides move selection.

        Args:
            config: Operator configuration with human settings.
            run_id: Run ID for telemetry.

        Returns:
            Command arguments list.
        """
        settings = config.settings or {}

        # Get human-specific settings
        player_name = settings.get("player_name", "Human")
        show_legal_moves = settings.get("show_legal_moves", True)
        confirm_moves = settings.get("confirm_moves", False)
        timeout_seconds = settings.get("timeout_seconds", 0.0)

        # Determine mode: interactive for gymnasium, board-game for PettingZoo
        is_pettingzoo = config.env_name == "pettingzoo"
        mode = "board-game" if is_pettingzoo else "interactive"

        cmd = [
            self._python_executable,
            "-m", "human_worker.cli",
            "--mode", mode,
            "--run-id", run_id,
            "--player-name", player_name,
            "--timeout", str(timeout_seconds),
            "--telemetry-dir", str(VAR_OPERATORS_TELEMETRY_DIR),
        ]

        # Add environment configuration for interactive mode
        if mode == "interactive":
            cmd.extend(["--env-name", config.env_name or ""])
            cmd.extend(["--task", config.task or ""])
            seed = settings.get("seed", 42)
            cmd.extend(["--seed", str(seed)])

            # Add game resolution for Crafter (controls native render size)
            game_resolution = settings.get("game_resolution")
            if game_resolution and isinstance(game_resolution, (list, tuple)) and len(game_resolution) == 2:
                res_str = f"{game_resolution[0]}x{game_resolution[1]}"
                cmd.extend(["--game-resolution", res_str])

        if show_legal_moves:
            cmd.append("--show-legal-moves")
        else:
            cmd.append("--no-show-legal-moves")

        if confirm_moves:
            cmd.append("--confirm-moves")

        LOGGER.info(
            "Building human operator command for %s (mode=%s, player=%s, env=%s/%s)",
            config.operator_id,
            mode,
            player_name,
            config.env_name,
            config.task,
        )

        return cmd

    def _build_baseline_command(
        self,
        config: OperatorConfig,
        run_id: str,
        *,
        interactive: bool = False,
    ) -> list[str]:
        """Build command line for baseline operator.

        Uses random_worker for simple baseline behaviors (random, noop, cycling).
        Baseline operators are used for ablation studies and credit assignment research.

        Args:
            config: Operator configuration with baseline settings.
            run_id: Run ID for telemetry.
            interactive: If True, add --interactive flag for step-by-step control.

        Returns:
            Command arguments list.

        Raises:
            OperatorLaunchError: If required settings are missing.
        """
        settings = config.settings or {}

        # Get baseline-specific settings
        behavior = settings.get("behavior", "random")
        seed = settings.get("seed")

        # Validate behavior
        valid_behaviors = ("random", "noop", "cycling")
        if behavior not in valid_behaviors:
            raise OperatorLaunchError(
                f"Invalid baseline behavior '{behavior}'. Must be one of {valid_behaviors}"
            )

        # Determine environment ID (task field contains the full env ID)
        env_id = config.task
        if not env_id:
            raise OperatorLaunchError(
                f"Baseline operator {config.operator_id} requires 'task' (environment ID)"
            )

        # Determine environment name
        env_name = config.env_name or "babyai"

        # Build operators-worker command
        cmd = [
            self._python_executable,
            "-m", "random_worker",
            "--run-id", run_id,
            "--behavior", behavior,
            "--env-name", env_name,
            "--task", env_id,
            "--telemetry-dir", str(VAR_OPERATORS_TELEMETRY_DIR),
        ]

        # Add interactive mode flag (required for GUI control)
        if interactive:
            cmd.append("--interactive")

        # Add seed if provided
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        LOGGER.info(
            "Built baseline command for operator %s | behavior=%s env=%s task=%s",
            config.operator_id,
            behavior,
            env_name,
            env_id,
        )

        return cmd

    def stop_operator(self, operator_id: str, timeout: float = 5.0) -> bool:
        """Stop a running operator subprocess.

        Args:
            operator_id: The operator ID to stop.
            timeout: Seconds to wait before force-killing.

        Returns:
            True if the operator was stopped, False if not found.
        """
        handle = self._handles.get(operator_id)
        if handle is None:
            return False

        handle.stop(timeout=timeout)
        del self._handles[operator_id]
        return True

    def stop_all(self, timeout: float = 5.0) -> list[str]:
        """Stop all running operator subprocesses.

        Args:
            timeout: Seconds to wait for each operator before force-killing.

        Returns:
            List of operator IDs that were stopped.
        """
        stopped = []
        for operator_id in list(self._handles.keys()):
            if self.stop_operator(operator_id, timeout=timeout):
                stopped.append(operator_id)
        return stopped

    def get_handle(self, operator_id: str) -> Optional[OperatorProcessHandle]:
        """Get the process handle for an operator."""
        return self._handles.get(operator_id)

    def get_all_handles(self) -> Dict[str, OperatorProcessHandle]:
        """Get all active operator handles."""
        return dict(self._handles)

    def check_operator_status(self, operator_id: str) -> Optional[str]:
        """Check the status of an operator.

        Args:
            operator_id: The operator ID to check.

        Returns:
            "running", "completed", "error", or None if not found.
        """
        handle = self._handles.get(operator_id)
        if handle is None:
            return None

        return_code = handle.return_code
        if return_code is None:
            return "running"
        elif return_code == 0:
            return "completed"
        else:
            return "error"

    def cleanup_finished(self) -> list[str]:
        """Remove handles for operators that have finished.

        Returns:
            List of operator IDs that were cleaned up.
        """
        cleaned = []
        for operator_id in list(self._handles.keys()):
            handle = self._handles[operator_id]
            if handle.return_code is not None:
                handle._close_log()
                del self._handles[operator_id]
                cleaned.append(operator_id)
        return cleaned


__all__ = [
    "OperatorLauncher",
    "OperatorLaunchError",
    "OperatorProcessHandle",
]
