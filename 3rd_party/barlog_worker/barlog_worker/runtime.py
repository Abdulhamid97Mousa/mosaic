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

from omegaconf import OmegaConf

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

        balrog_config = OmegaConf.create(self.config.to_balrog_config())
        factory = AgentFactory(balrog_config)
        return factory.create_agent()

    def _create_env(self) -> Any:
        """Create BALROG environment based on config."""
        from balrog.environments import make_env

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


__all__ = [
    "BarlogWorkerRuntime",
    "StepTelemetry",
    "EpisodeTelemetry",
    "TelemetryEmitter",
]
