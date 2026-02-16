"""Runtime for operators worker subprocess.

Handles:
- Stdin/stdout JSON protocol for GUI communication
- Gymnasium environment lifecycle
- Operator action selection
- Telemetry emission to JSONL files

Communication Protocol:
    Commands (stdin JSON):
        {"cmd": "reset", "seed": 42, "env_name": "babyai", "task": "BabyAI-GoToRedBall-v0"}
        {"cmd": "step"}
        {"cmd": "stop"}

    Responses (stdout JSON):
        {"type": "init"}
        {"type": "ready", "render_payload": {...}}
        {"type": "step", "reward": 0.0, "terminated": false, "render_payload": {...}}
        {"type": "episode_end", "return": 10.0, "steps": 42}
        {"type": "error", "message": "..."}
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, TextIO
import gymnasium as gym
import numpy as np

# Import environment packages to register them with gymnasium
try:
    import minigrid  # Registers MiniGrid-* environments
except ImportError:
    pass  # MiniGrid not installed

try:
    import mosaic_multigrid  # Registers MultiGrid-* environments
except ImportError:
    pass  # MultiGrid not installed

from operators_worker.config import OperatorsWorkerConfig
from operators_worker.operators import (
    RandomOperator,
    NoopOperator,
    CyclingOperator,
    create_baseline_operator,
)


class TelemetryEmitter:
    """Emits telemetry to JSONL files for post-hoc analysis.

    Writes two files:
        - {run_id}_steps.jsonl: Per-step data (action, reward, obs, etc.)
        - {run_id}_episodes.jsonl: Per-episode summary (return, steps, seed)

    Attributes:
        telemetry_dir: Directory to write JSONL files
        run_id: Unique run identifier
        steps_file: File handle for steps JSONL
        episodes_file: File handle for episodes JSONL
    """

    def __init__(self, telemetry_dir: str, run_id: str):
        """Initialize telemetry emitter.

        Args:
            telemetry_dir: Directory path for telemetry files
            run_id: Unique run identifier
        """
        self.telemetry_dir = Path(telemetry_dir)
        self.run_id = run_id

        # Create directory if it doesn't exist
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        # Open JSONL files
        self.steps_path = self.telemetry_dir / f"{run_id}_steps.jsonl"
        self.episodes_path = self.telemetry_dir / f"{run_id}_episodes.jsonl"

        self.steps_file = open(self.steps_path, "a")
        self.episodes_file = open(self.episodes_path, "a")

        # Episode tracking
        self.current_episode_steps = []
        self.episode_count = 0

    def emit_step(
        self,
        step: int,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Emit step data to steps JSONL.

        Args:
            step: Step number in episode
            action: Action taken
            reward: Reward received
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dict from environment
        """
        record = {
            "run_id": self.run_id,
            "episode": self.episode_count,
            "step": step,
            "action": int(action) if isinstance(action, (int, np.integer)) else action,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
        }

        # Add info fields if present
        if info:
            record["info"] = {
                k: v for k, v in info.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }

        # Write to JSONL
        self.steps_file.write(json.dumps(record) + "\n")
        self.steps_file.flush()

        # Track for episode summary
        self.current_episode_steps.append(record)

    def emit_episode(
        self,
        seed: Optional[int],
        episode_return: float,
        num_steps: int,
        success: bool = False,
    ) -> None:
        """Emit episode summary to episodes JSONL.

        Args:
            seed: Random seed used for episode
            episode_return: Total cumulative reward
            num_steps: Number of steps in episode
            success: Whether episode was successful (task-specific)
        """
        record = {
            "run_id": self.run_id,
            "episode": self.episode_count,
            "seed": seed,
            "return": float(episode_return),
            "steps": num_steps,
            "success": success,
        }

        self.episodes_file.write(json.dumps(record) + "\n")
        self.episodes_file.flush()

        # Increment episode counter
        self.episode_count += 1

        # Clear episode buffer
        self.current_episode_steps = []

    def close(self) -> None:
        """Close telemetry files."""
        if hasattr(self, "steps_file"):
            self.steps_file.close()
        if hasattr(self, "episodes_file"):
            self.episodes_file.close()


class OperatorsWorkerRuntime:
    """Main runtime for operators worker subprocess.

    Manages:
        - Stdin/stdout JSON communication with GUI
        - Gymnasium environment lifecycle
        - Operator action selection
        - Telemetry emission

    Attributes:
        config: Worker configuration
        env: Gymnasium environment (created on reset)
        operator: Baseline operator (created on reset)
        telemetry: Telemetry emitter (if emit_jsonl=True)
        current_obs: Current observation
        current_info: Current info dict
    """

    def __init__(
        self,
        config: OperatorsWorkerConfig,
        stdin: TextIO = sys.stdin,
        stdout: TextIO = sys.stdout,
    ):
        """Initialize runtime.

        Args:
            config: Worker configuration
            stdin: Input stream for commands (default: sys.stdin)
            stdout: Output stream for responses (default: sys.stdout)
        """
        self.config = config
        self.stdin = stdin
        self.stdout = stdout

        # Environment and operator (created on reset)
        self.env: Optional[gym.Env] = None
        self.operator: Optional[Any] = None
        self._current_seed: Optional[int] = None  # Track actual seed for telemetry

        # Telemetry
        self.telemetry: Optional[TelemetryEmitter] = None
        if self.config.emit_jsonl:
            self.telemetry = TelemetryEmitter(
                self.config.telemetry_dir,
                self.config.run_id,
            )

        # Episode state
        self.current_obs: Optional[Any] = None
        self.current_info: Optional[Dict] = None
        self.episode_step = 0
        self.episode_return = 0.0

    def emit_response(self, response: Dict[str, Any]) -> None:
        """Emit JSON response to stdout.

        Args:
            response: Response dictionary
        """
        # Convert numpy types to native Python types
        response = self._serialize_response(response)

        self.stdout.write(json.dumps(response) + "\n")
        self.stdout.flush()

    def _serialize_response(self, obj: Any) -> Any:
        """Recursively serialize response for JSON output.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._serialize_response(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_response(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Fallback: convert to string
            return str(obj)

    def _create_environment(self, env_name: str, task: str, max_steps: Optional[int] = None) -> gym.Env:
        """Create gymnasium environment.

        Args:
            env_name: Environment family ("babyai", "minigrid", etc.)
            task: Specific task (e.g., "BabyAI-GoToRedBall-v0")
            max_steps: Maximum steps per episode before truncation (optional)

        Returns:
            Gymnasium environment

        Raises:
            ValueError: If environment creation fails
        """
        try:
            # Attempt to create environment
            env = gym.make(task, render_mode="rgb_array")

            # Apply max_steps truncation if specified
            if max_steps is not None:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

            return env
        except Exception as e:
            raise ValueError(f"Failed to create environment {task}: {e}")

    def _create_operator(self, behavior: str) -> Any:
        """Create baseline operator.

        Args:
            behavior: Operator behavior ("random", "noop", "cycling")

        Returns:
            Configured baseline operator

        Raises:
            ValueError: If behavior is invalid
        """
        operator = create_baseline_operator(
            behavior=behavior,
            operator_id=self.config.run_id,
        )

        # Configure action space from environment
        if self.env is None:
            raise RuntimeError("Environment must be created before operator")

        operator.set_action_space(self.env.action_space)

        return operator

    def _get_render_payload(self) -> Dict[str, Any]:
        """Get render payload from environment.

        Returns:
            Dictionary with RGB array in GUI-expected format:
                - mode: "rgb"
                - rgb: flattened list of RGB values
                - width: image width
                - height: image height
        """
        if self.env is None:
            return {}

        try:
            # Get RGB array from environment
            rgb_array = self.env.render()

            if rgb_array is not None and isinstance(rgb_array, np.ndarray):
                # Convert to GUI-expected format (same pattern as human_worker)
                height, width = int(rgb_array.shape[0]), int(rgb_array.shape[1])

                return {
                    "mode": "rgb",
                    "rgb": rgb_array.tolist(),  # Keep 3D structure, don't flatten!
                    "width": width,
                    "height": height,
                }
        except Exception as e:
            # Render failed - return empty payload
            return {"error": f"Render failed: {e}"}

        return {}

    def handle_reset(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reset command.

        Args:
            cmd: Command dict with optional "seed", "env_name", "task", "max_steps"

        Returns:
            Response dict with "type": "ready" and render_payload
        """
        try:
            # Extract parameters
            seed = cmd.get("seed", self.config.seed)
            env_name = cmd.get("env_name", self.config.env_name)
            task = cmd.get("task", self.config.task)
            max_steps = cmd.get("max_steps", self.config.max_steps)

            # Create environment if needed (or if max_steps changed)
            if (self.env is None or
                env_name != self.config.env_name or
                task != self.config.task or
                max_steps != self.config.max_steps):
                if self.env is not None:
                    self.env.close()

                self.env = self._create_environment(env_name, task, max_steps)
                self.config.env_name = env_name
                self.config.task = task
                self.config.max_steps = max_steps

            # Create operator if needed
            if self.operator is None:
                self.operator = self._create_operator(self.config.behavior)

            # Reset environment
            self.current_obs, self.current_info = self.env.reset(seed=seed)

            # Reset operator
            self.operator.reset(seed=seed)

            # Store actual seed for telemetry (not config default)
            self._current_seed = seed

            # Reset episode state
            self.episode_step = 0
            self.episode_return = 0.0

            # Get render payload
            render_payload = self._get_render_payload()

            return {
                "type": "ready",
                "render_payload": render_payload,
                "seed": seed,
                "step_index": 0,
                "episode_index": self.telemetry.episode_count if self.telemetry else 0,
            }

        except Exception as e:
            return {
                "type": "error",
                "message": f"Reset failed: {e}",
            }

    def handle_step(self) -> Dict[str, Any]:
        """Handle step command.

        Returns:
            Response dict with "type": "step" and step results
        """
        try:
            if self.env is None or self.operator is None:
                return {
                    "type": "error",
                    "message": "Environment not initialized. Send reset first.",
                }

            # Select action
            action = self.operator.select_action(
                self.current_obs,
                self.current_info,
            )

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Update operator with result
            self.operator.on_step_result(obs, reward, terminated, truncated, info)

            # Update episode state
            self.episode_step += 1
            self.episode_return += reward

            # Emit telemetry
            if self.telemetry:
                self.telemetry.emit_step(
                    self.episode_step - 1,  # 0-indexed
                    action,
                    reward,
                    terminated,
                    truncated,
                    info,
                )

            # Update current state
            self.current_obs = obs
            self.current_info = info

            # Get render payload
            render_payload = self._get_render_payload()

            # Check for episode end
            if terminated or truncated:
                # Emit episode summary
                if self.telemetry:
                    success = info.get("success", False)
                    self.telemetry.emit_episode(
                        seed=self._current_seed,
                        episode_return=self.episode_return,
                        num_steps=self.episode_step,
                        success=success,
                    )

                return {
                    "type": "episode_end",
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "render_payload": render_payload,
                    "episode_return": self.episode_return,
                    "episode_steps": self.episode_step,
                    "step_index": self.episode_step,
                    "episode_index": self.telemetry.episode_count if self.telemetry else 0,
                }

            return {
                "type": "step",
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "render_payload": render_payload,
                "step_index": self.episode_step,
                "episode_index": self.telemetry.episode_count if self.telemetry else 0,
                "total_reward": self.episode_return,
            }

        except Exception as e:
            return {
                "type": "error",
                "message": f"Step failed: {e}",
            }

    def run_interactive(self) -> None:
        """Run interactive mode: read commands from stdin, emit responses to stdout.

        This is the main loop for subprocess mode. It blocks waiting for JSON
        commands from the GUI and responds with JSON messages.

        Protocol:
            Commands:
                {"cmd": "reset", "seed": 42, ...}
                {"cmd": "step"}
                {"cmd": "stop"}

            Responses:
                {"type": "init"}
                {"type": "ready", "render_payload": {...}}
                {"type": "step", "reward": 0.0, ...}
                {"type": "episode_end", ...}
                {"type": "error", "message": "..."}
        """
        # Emit init message
        self.emit_response({"type": "init", "run_id": self.config.run_id})

        try:
            # Read commands from stdin
            for line in self.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError as e:
                    self.emit_response({
                        "type": "error",
                        "message": f"Invalid JSON: {e}",
                    })
                    continue

                # Handle command
                cmd_type = cmd.get("cmd")

                if cmd_type == "reset":
                    response = self.handle_reset(cmd)
                    self.emit_response(response)

                elif cmd_type == "step":
                    response = self.handle_step()
                    self.emit_response(response)

                elif cmd_type == "stop":
                    self.emit_response({"type": "stopped"})
                    break

                else:
                    self.emit_response({
                        "type": "error",
                        "message": f"Unknown command: {cmd_type}",
                    })

        except KeyboardInterrupt:
            self.emit_response({"type": "interrupted"})

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.env is not None:
            self.env.close()

        if self.telemetry is not None:
            self.telemetry.close()
