"""Runtime for BARLOG Worker.

This module provides the main episode loop that:
1. Creates BALROG environments and agents
2. Runs episodes with LLM-based action selection
3. Emits telemetry for the GUI
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from barlog_worker.config import BarlogWorkerConfig

logger = logging.getLogger(__name__)


@dataclass
class StepTelemetry:
    """Telemetry for a single step."""

    run_id: str
    episode_id: str
    step_index: int
    observation: str  # Text observation
    action: str
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)
    llm_response: Optional[str] = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class EpisodeTelemetry:
    """Telemetry for a complete episode."""

    run_id: str
    episode_id: str
    episode_index: int
    env_name: str
    task: str
    total_reward: float
    num_steps: int
    terminated: bool
    truncated: bool
    success: bool
    start_time: str
    end_time: str
    duration_seconds: float
    llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class TelemetryEmitter:
    """Emits telemetry to JSONL files and stdout."""

    def __init__(
        self,
        run_id: str,
        telemetry_dir: str,
        emit_jsonl: bool = True,
    ) -> None:
        self.run_id = run_id
        self.telemetry_dir = Path(telemetry_dir)
        self.emit_jsonl = emit_jsonl
        self._step_file: Optional[Any] = None
        self._episode_file: Optional[Any] = None

        if self.emit_jsonl:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            step_path = self.telemetry_dir / f"{run_id}_steps.jsonl"
            episode_path = self.telemetry_dir / f"{run_id}_episodes.jsonl"
            self._step_file = open(step_path, "a")
            self._episode_file = open(episode_path, "a")
            logger.info(f"Telemetry files: {step_path}, {episode_path}")

    def emit_step(self, step: StepTelemetry) -> None:
        """Emit step telemetry."""
        data = asdict(step)
        # Also print to stdout for GUI consumption
        print(json.dumps({"type": "step", **data}), flush=True)
        if self._step_file:
            self._step_file.write(json.dumps(data) + "\n")
            self._step_file.flush()

    def emit_episode(self, episode: EpisodeTelemetry) -> None:
        """Emit episode telemetry."""
        data = asdict(episode)
        # Also print to stdout for GUI consumption
        print(json.dumps({"type": "episode", **data}), flush=True)
        if self._episode_file:
            self._episode_file.write(json.dumps(data) + "\n")
            self._episode_file.flush()

    def close(self) -> None:
        """Close telemetry files."""
        if self._step_file:
            self._step_file.close()
        if self._episode_file:
            self._episode_file.close()


class BarlogWorkerRuntime:
    """Main runtime for running LLM agents on BALROG environments."""

    def __init__(self, config: BarlogWorkerConfig) -> None:
        self.config = config
        self.telemetry = TelemetryEmitter(
            run_id=config.run_id,
            telemetry_dir=config.telemetry_dir,
            emit_jsonl=config.emit_jsonl,
        )

        # Add BALROG to Python path
        balrog_path = Path(__file__).parent.parent / "BALROG"
        if str(balrog_path) not in sys.path:
            sys.path.insert(0, str(balrog_path))
            logger.debug(f"Added BALROG to path: {balrog_path}")

    def _create_agent(self) -> Any:
        """Create BALROG agent based on config."""
        # Import BALROG components
        from balrog.agents import AgentFactory
        from omegaconf import OmegaConf

        balrog_config = OmegaConf.create(self.config.to_balrog_config())
        factory = AgentFactory(balrog_config)
        return factory.create_agent()

    def _create_env(self) -> Any:
        """Create BALROG environment based on config.

        Uses our own environment wrapper (barlog_worker.environments)
        which fixes compatibility issues with standard Gymnasium environments.
        """
        from barlog_worker.environments import make_env
        from omegaconf import OmegaConf

        balrog_config = OmegaConf.create(self.config.to_balrog_config())
        return make_env(
            self.config.env_name,
            self.config.task,
            balrog_config,
            render_mode=self.config.render_mode,
        )

    def run(self) -> None:
        """Run all episodes."""
        logger.info(f"Starting {self.config.num_episodes} episodes")

        for episode_idx in range(self.config.num_episodes):
            episode_id = f"{self.config.run_id}-ep{episode_idx:06d}"
            logger.info(f"Starting episode {episode_idx + 1}/{self.config.num_episodes}: {episode_id}")

            try:
                self._run_episode(episode_idx, episode_id)
            except Exception as e:
                logger.exception(f"Episode {episode_id} failed: {e}")
                # Emit error telemetry
                print(json.dumps({
                    "type": "error",
                    "run_id": self.config.run_id,
                    "episode_id": episode_id,
                    "error": str(e),
                }), flush=True)

        self.telemetry.close()
        logger.info("All episodes completed")

    def _run_episode(self, episode_idx: int, episode_id: str) -> None:
        """Run a single episode."""
        start_time = datetime.utcnow()

        # Create fresh agent and environment for each episode
        agent = self._create_agent()
        env = self._create_env()

        # Reset environment
        obs, info = env.reset(seed=self.config.seed)

        # Episode tracking
        total_reward = 0.0
        llm_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        prev_action = None

        for step_idx in range(self.config.max_steps):
            # Get action from LLM agent
            try:
                response = agent.act(obs, prev_action=prev_action)
                llm_calls += 1
                action_str = response.completion.strip()
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            except Exception as e:
                logger.warning(f"Agent act failed at step {step_idx}: {e}")
                action_str = ""
                input_tokens = 0
                output_tokens = 0

            # Validate and execute action
            action = env.check_action_validity(action_str)
            obs_new, reward, terminated, truncated, info = env.step(action)

            # Emit step telemetry
            step_telemetry = StepTelemetry(
                run_id=self.config.run_id,
                episode_id=episode_id,
                step_index=step_idx,
                observation=self._obs_to_str(obs),
                action=action_str,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=self._sanitize_info(info),
                llm_response=action_str,
                llm_input_tokens=input_tokens,
                llm_output_tokens=output_tokens,
            )
            self.telemetry.emit_step(step_telemetry)

            total_reward += reward
            prev_action = action_str
            obs = obs_new

            if terminated or truncated:
                logger.debug(f"Episode ended at step {step_idx}: terminated={terminated}, truncated={truncated}")
                break

        # Episode complete
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        # Determine success (environment-specific)
        success = info.get("success", terminated and reward > 0)

        episode_telemetry = EpisodeTelemetry(
            run_id=self.config.run_id,
            episode_id=episode_id,
            episode_index=episode_idx,
            env_name=self.config.env_name,
            task=self.config.task,
            total_reward=total_reward,
            num_steps=step_idx + 1,
            terminated=terminated,
            truncated=truncated,
            success=bool(success),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            llm_calls=llm_calls,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )
        self.telemetry.emit_episode(episode_telemetry)

        logger.info(
            f"Episode {episode_idx + 1} complete: "
            f"reward={total_reward:.2f}, steps={step_idx + 1}, "
            f"success={success}, duration={duration:.1f}s"
        )

        # Close environment
        env.close()

    def _obs_to_str(self, obs: Any) -> str:
        """Convert observation to string for telemetry."""
        if isinstance(obs, str):
            return obs
        if isinstance(obs, dict):
            # BALROG environments often have 'text' or 'message' keys
            if "text" in obs:
                return str(obs["text"])
            if "message" in obs:
                return str(obs["message"])
            return json.dumps({k: str(v)[:100] for k, v in obs.items()})
        return str(obs)[:500]

    def _sanitize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize info dict for JSON serialization."""
        result = {}
        for key, value in info.items():
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                result[key] = str(value)[:100]
        return result


class InteractiveRuntime:
    """Interactive runtime for step-by-step control from GUI.

    Reads JSON commands from stdin, executes one step at a time,
    and emits telemetry to stdout. This enables scientific comparison
    with synchronized lock-step execution across multiple operators.

    Protocol:
        Input (stdin):
            {"cmd": "reset", "seed": 42}  - Reset environment with seed
            {"cmd": "step"}               - Execute one step
            {"cmd": "stop"}               - Terminate

        Output (stdout):
            {"type": "ready", "run_id": "...", "env": "...", "task": "..."}
            {"type": "step", "step_index": 0, "observation": "...", ...}
            {"type": "episode_done", "total_reward": 0.5, ...}
            {"type": "error", "message": "..."}
    """

    def __init__(self, config: BarlogWorkerConfig) -> None:
        self.config = config
        self.telemetry = TelemetryEmitter(
            run_id=config.run_id,
            telemetry_dir=config.telemetry_dir,
            emit_jsonl=config.emit_jsonl,
        )

        # Add BALROG to Python path
        balrog_path = Path(__file__).parent.parent / "BALROG"
        if str(balrog_path) not in sys.path:
            sys.path.insert(0, str(balrog_path))
            logger.debug(f"Added BALROG to path: {balrog_path}")

        # State
        self._agent = None
        self._env = None
        self._episode_idx = 0
        self._step_idx = 0
        self._total_reward = 0.0
        self._prev_action = None
        self._obs = None
        self._episode_start_time = None
        self._llm_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _create_agent(self) -> Any:
        """Create BALROG agent based on config."""
        from balrog.agents import AgentFactory
        from omegaconf import OmegaConf

        balrog_config = OmegaConf.create(self.config.to_balrog_config())
        factory = AgentFactory(balrog_config)
        return factory.create_agent()

    def _create_env(self) -> Any:
        """Create BALROG environment based on config."""
        from barlog_worker.environments import make_env
        from omegaconf import OmegaConf

        balrog_config = OmegaConf.create(self.config.to_balrog_config())
        return make_env(
            self.config.env_name,
            self.config.task,
            balrog_config,
            render_mode=self.config.render_mode,
        )

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit JSON to stdout for GUI consumption."""
        print(json.dumps(data), flush=True)

    def _handle_reset(self, seed: Optional[int] = None) -> None:
        """Handle reset command - initialize environment with seed."""
        try:
            # Close existing env if any
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass

            # Create fresh agent and environment
            self._agent = self._create_agent()
            self._env = self._create_env()

            # Reset with seed
            effective_seed = seed if seed is not None else self.config.seed
            self._obs, info = self._env.reset(seed=effective_seed)

            # Reset episode state
            self._episode_idx = 0
            self._step_idx = 0
            self._total_reward = 0.0
            self._prev_action = None
            self._episode_start_time = datetime.utcnow()
            self._llm_calls = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0

            self._emit({
                "type": "ready",
                "run_id": self.config.run_id,
                "env": self.config.env_name,
                "task": self.config.task,
                "seed": effective_seed,
                "observation": self._obs_to_str(self._obs),
            })
            logger.info(f"Environment reset with seed={effective_seed}")

        except Exception as e:
            logger.exception(f"Reset failed: {e}")
            self._emit({"type": "error", "message": str(e)})

    def _handle_step(self) -> None:
        """Handle step command - execute one environment step."""
        if self._env is None or self._agent is None:
            self._emit({"type": "error", "message": "Environment not initialized. Send reset first."})
            return

        try:
            # Get action from LLM agent
            input_tokens = 0
            output_tokens = 0
            try:
                response = self._agent.act(self._obs, prev_action=self._prev_action)
                self._llm_calls += 1
                action_str = response.completion.strip()
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
            except Exception as e:
                logger.warning(f"Agent act failed at step {self._step_idx}: {e}")
                action_str = ""

            # Validate and execute action
            action = self._env.check_action_validity(action_str)
            obs_new, reward, terminated, truncated, info = self._env.step(action)

            # Emit step telemetry
            episode_id = f"{self.config.run_id}-ep{self._episode_idx:06d}"
            step_data = {
                "type": "step",
                "run_id": self.config.run_id,
                "episode_id": episode_id,
                "step_index": self._step_idx,
                "observation": self._obs_to_str(self._obs),
                "action": action_str,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "info": self._sanitize_info(info),
                "llm_response": action_str,
                "llm_input_tokens": input_tokens,
                "llm_output_tokens": output_tokens,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._emit(step_data)

            # Also write to telemetry files
            step_telemetry = StepTelemetry(
                run_id=self.config.run_id,
                episode_id=episode_id,
                step_index=self._step_idx,
                observation=self._obs_to_str(self._obs),
                action=action_str,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=self._sanitize_info(info),
                llm_response=action_str,
                llm_input_tokens=input_tokens,
                llm_output_tokens=output_tokens,
            )
            if self.telemetry._step_file:
                self.telemetry._step_file.write(json.dumps(asdict(step_telemetry)) + "\n")
                self.telemetry._step_file.flush()

            # Update state
            self._total_reward += reward
            self._prev_action = action_str
            self._obs = obs_new
            self._step_idx += 1

            # Check for episode end
            if terminated or truncated:
                self._emit_episode_done(terminated, truncated, info)

        except Exception as e:
            logger.exception(f"Step failed: {e}")
            self._emit({"type": "error", "message": str(e)})

    def _emit_episode_done(self, terminated: bool, truncated: bool, info: Dict[str, Any]) -> None:
        """Emit episode completion telemetry."""
        end_time = datetime.utcnow()
        duration = (end_time - self._episode_start_time).total_seconds() if self._episode_start_time else 0.0
        success = info.get("success", terminated and self._total_reward > 0)

        episode_id = f"{self.config.run_id}-ep{self._episode_idx:06d}"
        episode_data = {
            "type": "episode_done",
            "run_id": self.config.run_id,
            "episode_id": episode_id,
            "episode_index": self._episode_idx,
            "env_name": self.config.env_name,
            "task": self.config.task,
            "total_reward": self._total_reward,
            "num_steps": self._step_idx,
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(success),
            "duration_seconds": duration,
            "llm_calls": self._llm_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
        self._emit(episode_data)

        logger.info(
            f"Episode {self._episode_idx + 1} complete: "
            f"reward={self._total_reward:.2f}, steps={self._step_idx}, "
            f"success={success}"
        )

    def _obs_to_str(self, obs: Any) -> str:
        """Convert observation to string for telemetry."""
        if isinstance(obs, str):
            return obs
        if isinstance(obs, dict):
            if "text" in obs:
                return str(obs["text"])
            if "message" in obs:
                return str(obs["message"])
            return json.dumps({k: str(v)[:100] for k, v in obs.items()})
        return str(obs)[:500]

    def _sanitize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize info dict for JSON serialization."""
        result = {}
        for key, value in info.items():
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                result[key] = str(value)[:100]
        return result

    def run(self) -> None:
        """Main loop - read commands from stdin, execute, respond."""
        logger.info("Interactive mode started. Waiting for commands on stdin...")
        self._emit({
            "type": "init",
            "run_id": self.config.run_id,
            "env": self.config.env_name,
            "task": self.config.task,
            "version": "1.0",
        })

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError as e:
                    self._emit({"type": "error", "message": f"Invalid JSON: {e}"})
                    continue

                cmd_type = cmd.get("cmd", "").lower()

                if cmd_type == "reset":
                    seed = cmd.get("seed")
                    self._handle_reset(seed)
                elif cmd_type == "step":
                    self._handle_step()
                elif cmd_type == "stop":
                    logger.info("Stop command received")
                    self._emit({"type": "stopped"})
                    break
                elif cmd_type == "ping":
                    self._emit({"type": "pong"})
                else:
                    self._emit({"type": "error", "message": f"Unknown command: {cmd_type}"})

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass
            self.telemetry.close()
            logger.info("Interactive runtime stopped")


__all__ = [
    "BarlogWorkerRuntime",
    "InteractiveRuntime",
    "StepTelemetry",
    "EpisodeTelemetry",
    "TelemetryEmitter",
]
