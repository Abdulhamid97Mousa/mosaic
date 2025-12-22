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
    - Writes logs to VAR_OPERATORS_DIR
    - Emits telemetry to VAR_TELEMETRY_DIR (JSONL + stdout)
    - Can be started/stopped independently
"""

import logging
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Dict, Optional

from gym_gui.config.paths import VAR_OPERATORS_DIR, VAR_TELEMETRY_DIR, ensure_var_directories
from gym_gui.core.subprocess_validation import validated_popen
from gym_gui.services.operator import OperatorConfig


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
    """

    operator_id: str
    run_id: str
    process: subprocess.Popen[str]
    log_path: Path
    config: OperatorConfig
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


class OperatorLauncher:
    """Launches and manages operator subprocess workers.

    This class handles spawning subprocess workers for both LLM
    and RL operator types, managing their lifecycle and log files.

    Example:
        launcher = OperatorLauncher()

        # Launch an LLM operator
        config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
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
    ) -> OperatorProcessHandle:
        """Launch an operator subprocess based on its configuration.

        Args:
            config: The operator configuration.
            run_id: Optional run ID (auto-generated if not provided).

        Returns:
            A handle for managing the launched subprocess.

        Raises:
            OperatorLaunchError: If the subprocess fails to start.
        """
        run_id = run_id or f"{config.operator_id}_{uuid.uuid4().hex[:8]}"

        ensure_var_directories()
        VAR_OPERATORS_DIR.mkdir(parents=True, exist_ok=True)

        # Create log file for this operator
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = VAR_OPERATORS_DIR / f"{config.operator_id}_{timestamp}.log"
        log_file = log_path.open("a", encoding="utf-8")

        # Build command based on operator type
        if config.operator_type == "llm":
            cmd = self._build_llm_command(config, run_id)
        elif config.operator_type == "rl":
            cmd = self._build_rl_command(config, run_id)
        else:
            log_file.close()
            raise OperatorLaunchError(f"Unknown operator type: {config.operator_type}")

        # Set up environment
        env = os.environ.copy()
        env.setdefault("QT_DEBUG_PLUGINS", "0")

        # Pass telemetry directory
        env["TELEMETRY_DIR"] = str(VAR_TELEMETRY_DIR)
        env["OPERATOR_RUN_ID"] = run_id
        env["OPERATOR_ID"] = config.operator_id

        LOGGER.info(
            "Launching operator %s (%s) with run_id=%s",
            config.operator_id,
            config.operator_type,
            run_id
        )
        LOGGER.debug("Command: %s", " ".join(cmd))

        try:
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
            config=config,
            _log_file=log_file,
        )
        self._handles[config.operator_id] = handle

        LOGGER.info(
            "Operator %s launched successfully (pid=%s, log=%s)",
            config.operator_id,
            process.pid,
            log_path
        )
        return handle

    def _build_llm_command(self, config: OperatorConfig, run_id: str) -> list[str]:
        """Build command line for LLM operator.

        Args:
            config: Operator configuration with LLM settings.
            run_id: Run ID for telemetry.

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

        cmd = [
            self._python_executable,
            "-m", "barlog_worker.cli",
            "--run-id", run_id,
            "--env", config.env_name,
            "--task", config.task,
            "--client", client_name,
            "--model", model_id,
            "--agent-type", agent_type,
            "--num-episodes", str(num_episodes),
            "--max-steps", str(max_steps),
            "--temperature", str(temperature),
            "--telemetry-dir", str(VAR_TELEMETRY_DIR),
            "-v",  # Verbose logging
        ]

        # Add optional base URL (for vLLM)
        if base_url:
            cmd.extend(["--base-url", base_url])

        # Add API key if provided
        if api_key:
            cmd.extend(["--api-key", api_key])

        return cmd

    def _build_rl_command(self, config: OperatorConfig, run_id: str) -> list[str]:
        """Build command line for RL operator.

        Args:
            config: Operator configuration with RL settings.
            run_id: Run ID for telemetry.

        Returns:
            Command arguments list.
        """
        settings = config.settings or {}

        # Get RL-specific settings
        policy_path = settings.get("policy_path")
        algorithm = settings.get("algorithm", "ppo")
        num_episodes = settings.get("num_episodes", 5)

        # TODO: Integrate with actual RL worker (cleanrl_worker, xuance_worker, etc.)
        # For now, RL uses a placeholder that logs its intent
        cmd = [
            self._python_executable,
            "-c",
            f"import time; print('RL operator {config.operator_id} started. "
            f"Policy: {policy_path}, Algorithm: {algorithm}, "
            f"Env: {config.env_name}, Task: {config.task}'); "
            f"time.sleep(2); print('RL operator placeholder complete')"
        ]

        LOGGER.warning(
            "RL operator launching not yet fully implemented. "
            "Operator %s will run a placeholder.",
            config.operator_id
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
