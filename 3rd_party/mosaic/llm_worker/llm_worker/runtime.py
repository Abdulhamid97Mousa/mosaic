"""Runtime for BALROG Worker.

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

from llm_worker.config import LLMWorkerConfig
from llm_worker.analytics import write_analytics_manifest

# Import standardized telemetry from gym_gui
try:
    from gym_gui.core.worker import TelemetryEmitter as StandardTelemetryEmitter
    from gym_gui.logging_config.helpers import log_constant
    from gym_gui.logging_config.log_constants import (
        LOG_WORKER_BALROG_RUNTIME_STARTED,
        LOG_WORKER_BALROG_RUNTIME_STOPPED,
        LOG_WORKER_BALROG_RUNTIME_ERROR,
        LOG_WORKER_BALROG_EPISODE_STARTED,
        LOG_WORKER_BALROG_EPISODE_COMPLETED,
        LOG_WORKER_BALROG_LLM_REQUEST,
        LOG_WORKER_BALROG_LLM_RESPONSE,
        LOG_WORKER_BALROG_LLM_ERROR,
        LOG_WORKER_BALROG_ACTION_SELECTED,
        LOG_WORKER_BALROG_STEP_TELEMETRY,
        LOG_WORKER_BALROG_EPISODE_TELEMETRY,
        LOG_WORKER_BALROG_CONFIG_LOADED,
        LOG_WORKER_BALROG_ENV_CREATED,
        LOG_WORKER_BALROG_AGENT_CREATED,
        LOG_WORKER_BALROG_DEBUG,
        LOG_WORKER_MOSAIC_RUNTIME_INTEGRATION,
        LOG_WORKER_MOSAIC_ACTION_PARSED,
        LOG_WORKER_MOSAIC_LLM_EPISODE_AUTO_RESET,
        LOG_WORKER_MOSAIC_LLM_ACTION_DEFAULTED,
    )
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    StandardTelemetryEmitter = None
    log_constant = None

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


class LLMTelemetryEmitter:
    """Emits BALROG-specific telemetry to JSONL files and stdout.

    This is kept separate from the standardized TelemetryEmitter for backwards
    compatibility with existing BALROG telemetry format.
    """

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


def _setup_api_key_env(config: LLMWorkerConfig) -> None:
    """Set up API key environment variable from config.

    BALROG clients read API keys from environment variables, not from config.
    This function bridges the gap by setting the appropriate env var based on
    the client_name and api_key in the config.

    Environment variable mapping:
        - openrouter -> OPENROUTER_API_KEY
        - openai -> OPENAI_API_KEY
        - anthropic -> ANTHROPIC_API_KEY
        - google -> GOOGLE_API_KEY
    """
    if not config.api_key:
        logger.debug("No API key in config, relying on environment variables")
        return

    client_name = config.client_name.lower()
    env_var_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "vllm": None,  # vLLM doesn't need API key
    }

    env_var = env_var_map.get(client_name)
    if env_var:
        os.environ[env_var] = config.api_key
        logger.info(f"Set {env_var} from config (client: {client_name})")
    else:
        logger.debug(f"No env var mapping for client: {client_name}")


class LLMWorkerRuntime:
    """Main runtime for running LLM agents on BALROG environments."""

    def __init__(self, config: LLMWorkerConfig) -> None:
        self.config = config
        self.telemetry = LLMTelemetryEmitter(
            run_id=config.run_id,
            telemetry_dir=config.telemetry_dir,
            emit_jsonl=config.emit_jsonl,
        )

        # Create standardized telemetry emitter for lifecycle events
        if _HAS_GYM_GUI:
            self._lifecycle_emitter = StandardTelemetryEmitter(run_id=config.run_id)
        else:
            self._lifecycle_emitter = None

        # Set API key environment variable from config (LLM client reads from env)
        _setup_api_key_env(config)

    def _create_agent(self) -> Any:
        """Create LLM agent based on config."""
        from llm_worker.agents import AgentFactory
        from omegaconf import OmegaConf

        balrog_config = OmegaConf.create(self.config.to_llm_config())
        factory = AgentFactory(balrog_config)
        return factory.create_agent()

    def _create_env(self) -> Any:
        """Create BALROG environment based on config.

        Uses our own environment wrapper (llm_worker.environments)
        which fixes compatibility issues with standard Gymnasium environments.
        """
        from llm_worker.environments import make_env
        from omegaconf import OmegaConf

        balrog_config = OmegaConf.create(self.config.to_llm_config())
        return make_env(
            self.config.env_name,
            self.config.task,
            balrog_config,
            render_mode=self.config.render_mode,
        )

    def run(self) -> Dict[str, Any]:
        """Run all episodes.

        Returns:
            Dict[str, Any]: Summary of the run with total episodes, successes, etc.
        """
        logger.info(f"Starting {self.config.num_episodes} episodes")

        # Log config loaded
        if _HAS_GYM_GUI and log_constant:
            log_constant(
                logger,
                LOG_WORKER_BALROG_CONFIG_LOADED,
                extra={
                    "run_id": self.config.run_id,
                    "env_name": self.config.env_name,
                    "task": self.config.task,
                    "agent_type": self.config.agent_type,
                },
            )

        # Emit run_started lifecycle event
        if self._lifecycle_emitter:
            self._lifecycle_emitter.run_started(
                {
                    "worker_type": "balrog",
                    "env_name": self.config.env_name,
                    "task": self.config.task,
                    "client_name": self.config.client_name,
                    "model_id": self.config.model_id,
                    "agent_type": self.config.agent_type,
                    "num_episodes": self.config.num_episodes,
                    "max_steps": self.config.max_steps,
                },
                constant=LOG_WORKER_BALROG_RUNTIME_STARTED,
            )

        # Track run statistics
        total_episodes = 0
        successful_episodes = 0
        total_reward = 0.0
        total_steps = 0
        errors = []

        try:
            for episode_idx in range(self.config.num_episodes):
                episode_id = f"{self.config.run_id}-ep{episode_idx:06d}"
                logger.info(f"Starting episode {episode_idx + 1}/{self.config.num_episodes}: {episode_id}")

                try:
                    episode_result = self._run_episode(episode_idx, episode_id)
                    total_episodes += 1
                    if episode_result.get("success"):
                        successful_episodes += 1
                    total_reward += episode_result.get("total_reward", 0.0)
                    total_steps += episode_result.get("num_steps", 0)

                    # Emit heartbeat periodically
                    if self._lifecycle_emitter and (episode_idx + 1) % 5 == 0:
                        self._lifecycle_emitter.heartbeat(
                            {
                                "episodes_completed": total_episodes,
                                "success_rate": successful_episodes / total_episodes if total_episodes > 0 else 0.0,
                                "average_reward": total_reward / total_episodes if total_episodes > 0 else 0.0,
                            },
                            constant=LOG_WORKER_BALROG_RUNTIME_STARTED,  # Using RUNTIME_STARTED for heartbeat
                        )

                except Exception as e:
                    logger.exception(f"Episode {episode_id} failed: {e}")
                    errors.append({"episode_id": episode_id, "error": str(e)})
                    # Emit error telemetry
                    print(json.dumps({
                        "type": "error",
                        "run_id": self.config.run_id,
                        "episode_id": episode_id,
                        "error": str(e),
                    }), flush=True)

            # Generate analytics manifest
            try:
                manifest_path = write_analytics_manifest(
                    self.config,
                    notes=f"BALROG run with {total_episodes} episodes completed",
                )
                logger.info(f"Analytics manifest written to: {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to write analytics manifest: {e}")

            # Build run summary
            summary = {
                "run_id": self.config.run_id,
                "worker_type": "balrog",
                "total_episodes": total_episodes,
                "successful_episodes": successful_episodes,
                "success_rate": successful_episodes / total_episodes if total_episodes > 0 else 0.0,
                "total_reward": total_reward,
                "average_reward": total_reward / total_episodes if total_episodes > 0 else 0.0,
                "total_steps": total_steps,
                "average_steps": total_steps / total_episodes if total_episodes > 0 else 0.0,
                "errors": errors,
            }

            # Emit run_completed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(
                    summary,
                    constant=LOG_WORKER_BALROG_RUNTIME_STOPPED,
                )

            self.telemetry.close()
            logger.info("All episodes completed")
            return summary

        except Exception as e:
            # Emit run_failed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_failed(
                    {"error": str(e)},
                    constant=LOG_WORKER_BALROG_RUNTIME_ERROR,
                )
            self.telemetry.close()
            raise

    def _run_episode(self, episode_idx: int, episode_id: str) -> Dict[str, Any]:
        """Run a single episode.

        Returns:
            Dict[str, Any]: Episode summary with success, total_reward, num_steps, etc.
        """
        start_time = datetime.utcnow()

        # Log episode started
        if _HAS_GYM_GUI and log_constant:
            log_constant(
                logger,
                LOG_WORKER_BALROG_EPISODE_STARTED,
                extra={
                    "run_id": self.config.run_id,
                    "episode_id": episode_id,
                    "episode_index": episode_idx,
                },
            )

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
                action=action,  # Use validated action, not raw LLM output
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=self._sanitize_info(info),
                llm_response=action_str,  # Keep raw LLM response for debugging
                llm_input_tokens=input_tokens,
                llm_output_tokens=output_tokens,
            )
            self.telemetry.emit_step(step_telemetry)

            total_reward += reward
            prev_action = action  # Use validated action so agent knows what was executed
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

        # Log episode completed
        if _HAS_GYM_GUI and log_constant:
            log_constant(
                logger,
                LOG_WORKER_BALROG_EPISODE_COMPLETED,
                extra={
                    "run_id": self.config.run_id,
                    "episode_id": episode_id,
                    "episode_index": episode_idx,
                    "success": bool(success),
                    "total_reward": total_reward,
                    "num_steps": step_idx + 1,
                    "duration_seconds": duration,
                },
            )

        logger.info(
            f"Episode {episode_idx + 1} complete: "
            f"reward={total_reward:.2f}, steps={step_idx + 1}, "
            f"success={success}, duration={duration:.1f}s"
        )

        # Close environment
        env.close()

        # Return episode summary for run statistics
        return {
            "success": bool(success),
            "total_reward": total_reward,
            "num_steps": step_idx + 1,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _obs_to_str(self, obs: Any) -> str:
        """Convert observation to string for telemetry."""
        if isinstance(obs, str):
            return obs

        # MOSAIC MultiGrid extension
        if self.config.env_name in ("multigrid", "mosaic_multigrid") and hasattr(obs, "shape"):
            from llm_worker.environments import generate_multigrid_description
            import numpy as np

            if isinstance(obs, np.ndarray):
                agent_id = getattr(self.config, "agent_id", 0)
                observation_mode = getattr(self.config, "observation_mode", "egocentric")

                # Note: In real environment, agent state would come from env
                # For now, use defaults for direction and carrying
                description = generate_multigrid_description(
                    obs,
                    agent_id=agent_id,
                    env=self._env if hasattr(self, "_env") else None,
                    observation_mode=observation_mode,
                    agent_direction=0,  # TODO: Get from env state
                    carrying=None,  # TODO: Get from env state
                )
                return description

        if isinstance(obs, dict):
            # BALROG environments often have 'text' key with nested context
            if "text" in obs:
                text_data = obs["text"]
                if isinstance(text_data, dict):
                    # BabyAI: extract long_term_context (spatial descriptions)
                    ltc = text_data.get("long_term_context", "")
                    stc = text_data.get("short_term_context", "")
                    # Include mission if available
                    mission = obs.get("mission", "")
                    parts = []
                    if mission:
                        parts.append(f"Mission: {mission}")
                    if ltc:
                        parts.append(f"Environment:\n{ltc}")
                    if stc:
                        parts.append(f"Status: {stc}")
                    return "\n\n".join(parts) if parts else str(text_data)
                return str(text_data)
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


class InteractiveLLMRuntime:
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

    def __init__(self, config: LLMWorkerConfig) -> None:
        self.config = config
        self.telemetry = LLMTelemetryEmitter(
            run_id=config.run_id,
            telemetry_dir=config.telemetry_dir,
            emit_jsonl=config.emit_jsonl,
        )

        # Create standardized telemetry emitter for lifecycle events
        if _HAS_GYM_GUI:
            self._lifecycle_emitter = StandardTelemetryEmitter(run_id=config.run_id)
        else:
            self._lifecycle_emitter = None

        # Set API key environment variable from config (BALROG client reads from env)
        _setup_api_key_env(config)

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
        self._reset_seed = None

    def _create_agent(self) -> Any:
        """Create BALROG agent based on config."""
        from llm_worker.agents import AgentFactory
        from omegaconf import OmegaConf

        balrog_config = OmegaConf.create(self.config.to_llm_config())
        factory = AgentFactory(balrog_config)
        return factory.create_agent()

    def _create_env(self) -> Any:
        """Create BALROG environment based on config."""
        from llm_worker.environments import make_env
        from omegaconf import OmegaConf

        balrog_config = OmegaConf.create(self.config.to_llm_config())
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
            self._reset_seed = effective_seed  # Store for deterministic auto-reset

            # Set instruction prompt with valid actions (critical for LLM to know valid actions)
            instructions = None
            if self.config.env_name in ["babyai", "minigrid"]:
                # Extract mission from observation for BabyAI/MiniGrid environments
                instructions = self._obs.get("mission") if isinstance(self._obs, dict) else None
                # Set the instruction prompt with valid actions list
                self._agent.prompt_builder.update_instruction_prompt(
                    self._env.get_instruction_prompt(instructions=instructions)
                )
            elif self.config.env_name in ("multigrid", "mosaic_multigrid"):
                # MOSAIC MultiGrid extension
                from llm_worker.environments import get_multigrid_instruction_prompt

                agent_id = getattr(self.config, "agent_id", 0)
                coordination_level = getattr(self.config, "coordination_level", 1)
                role = getattr(self.config, "role", "forward")

                instructions = get_multigrid_instruction_prompt(
                    agent_id=agent_id,
                    env_id=self.config.task,
                    coordination_level=coordination_level,
                    role=role,
                )
                self._agent.prompt_builder.update_instruction_prompt(instructions)

                if _HAS_GYM_GUI and log_constant:
                    log_constant(
                        logger,
                        LOG_WORKER_MOSAIC_RUNTIME_INTEGRATION,
                        extra={
                            "agent_id": agent_id,
                            "coordination_level": coordination_level,
                            "role": role,
                            "env_id": self.config.task,
                        },
                    )
                else:
                    logger.info(f"MOSAIC: Set MultiGrid prompt | agent={agent_id} level={coordination_level} role={role}")
            else:
                self._agent.prompt_builder.update_instruction_prompt(
                    self._env.get_instruction_prompt(instructions=instructions)
                )

            # Reset episode state
            self._episode_idx = 0
            self._step_idx = 0
            self._total_reward = 0.0
            self._prev_action = None
            self._episode_start_time = datetime.utcnow()
            self._llm_calls = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0

            # Get system prompt from agent's prompt builder (env-family specific)
            system_prompt = ""
            if hasattr(self._agent, "prompt_builder") and hasattr(self._agent.prompt_builder, "system_prompt"):
                system_prompt = self._agent.prompt_builder.system_prompt or ""

            # Build ready response with render payload for GUI display
            ready_data = {
                "type": "ready",
                "run_id": self.config.run_id,
                "env": self.config.env_name,
                "task": self.config.task,
                "seed": effective_seed,
                "episode_index": self._episode_idx,
                "step_index": self._step_idx,
                "observation": self._obs_to_str(self._obs),
                "system_prompt": system_prompt,  # Env-family specific instruction
            }

            # Add render payload for GUI display if render_mode is rgb_array
            if self.config.render_mode == "rgb_array":
                try:
                    rgb_frame = self._env.render()
                    if rgb_frame is not None:
                        import numpy as np
                        if isinstance(rgb_frame, np.ndarray) and rgb_frame.ndim >= 2:
                            h, w = rgb_frame.shape[0], rgb_frame.shape[1]
                            ready_data["render_payload"] = {
                                "mode": "rgb",
                                "rgb": rgb_frame.tolist(),
                                "width": w,
                                "height": h,
                            }
                except Exception as render_err:
                    logger.debug(f"Failed to render frame on reset: {render_err}")

            self._emit(ready_data)
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
                action_str = "go forward"  # Default to safe action on LLM failure
                if _HAS_GYM_GUI and log_constant:
                    log_constant(logger, LOG_WORKER_MOSAIC_LLM_ACTION_DEFAULTED, extra={
                        "step_index": self._step_idx,
                        "error": str(e),
                        "default_action": action_str,
                    })

            # Validate and execute action
            if self.config.env_name in ("multigrid", "mosaic_multigrid"):
                # MOSAIC MultiGrid extension - parse action from LLM text
                from mosaic_extension.multigrid import parse_action
                action = parse_action(action_str)

                if _HAS_GYM_GUI and log_constant:
                    log_constant(
                        logger,
                        LOG_WORKER_MOSAIC_ACTION_PARSED,
                        extra={
                            "action_index": action,
                            "llm_output": action_str[:50],
                        },
                    )
                else:
                    logger.debug(f"MOSAIC: Parsed action {action} from LLM: {action_str[:50]}")
            else:
                try:
                    action = self._env.check_action_validity(action_str)
                except ValueError:
                    # LLM returned invalid action text — default to "go forward"
                    bad_action = action_str
                    action_str = "go forward"
                    action = self._env.check_action_validity(action_str)
                    logger.warning(
                        f"Invalid LLM action '{bad_action}' at step {self._step_idx}, "
                        f"defaulting to '{action_str}'"
                    )
                    if _HAS_GYM_GUI and log_constant:
                        log_constant(logger, LOG_WORKER_MOSAIC_LLM_ACTION_DEFAULTED, extra={
                            "step_index": self._step_idx,
                            "error": f"Invalid LLM output: '{bad_action}'",
                            "default_action": action_str,
                        })

            obs_new, reward, terminated, truncated, info = self._env.step(action)

            # Emit step telemetry
            episode_id = f"{self.config.run_id}-ep{self._episode_idx:06d}"
            step_data = {
                "type": "step",
                "run_id": self.config.run_id,
                "episode_id": episode_id,
                "episode_index": self._episode_idx,  # For GUI display
                "step_index": self._step_idx,
                # LLM/Provider metadata for verification (BALROG best practice)
                "client_name": self.config.client_name,
                "model_id": self.config.model_id,
                "base_url": self.config.base_url,
                "agent_type": self.config.agent_type,
                # Environment state
                "observation": self._obs_to_str(self._obs),
                "action": action_str,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "info": self._sanitize_info(info),
                # LLM response details
                "llm_response": action_str,
                "llm_input_tokens": input_tokens,
                "llm_output_tokens": output_tokens,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add render payload for GUI display if render_mode is rgb_array
            if self.config.render_mode == "rgb_array":
                try:
                    rgb_frame = self._env.render()
                    if rgb_frame is not None:
                        import numpy as np
                        if isinstance(rgb_frame, np.ndarray) and rgb_frame.ndim >= 2:
                            h, w = rgb_frame.shape[0], rgb_frame.shape[1]
                            step_data["render_payload"] = {
                                "mode": "rgb",
                                "rgb": rgb_frame.tolist(),
                                "width": w,
                                "height": h,
                            }
                except Exception as render_err:
                    logger.debug(f"Failed to render frame: {render_err}")

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
                # Auto-reset for next episode
                self._episode_idx += 1
                # Re-seed for MiniGrid/BabyAI to preserve deterministic layout
                if self.config.env_name in ("babyai", "minigrid") and self._reset_seed is not None:
                    self._obs, _ = self._env.reset(seed=self._reset_seed)
                    if _HAS_GYM_GUI and log_constant:
                        log_constant(logger, LOG_WORKER_MOSAIC_LLM_EPISODE_AUTO_RESET, extra={
                            "episode_index": self._episode_idx,
                            "seed": self._reset_seed,
                            "env_name": self.config.env_name,
                        })
                else:
                    self._obs, _ = self._env.reset()
                self._step_idx = 0
                self._total_reward = 0.0
                self._prev_action = None
                self._episode_start_time = datetime.utcnow()
                self._llm_calls = 0
                self._total_input_tokens = 0
                self._total_output_tokens = 0
                logger.info(f"Auto-reset for episode {self._episode_idx + 1}")

        except Exception as e:
            logger.exception(f"Step failed: {e}")
            self._emit({"type": "error", "message": str(e)})

    def _handle_init_agent(self, cmd: Dict[str, Any]) -> None:
        """Handle init_agent command - initialize agent for action-selector mode.

        In action-selector mode, the worker doesn't own the environment.
        The GUI owns the PettingZoo environment and sends observations to
        multiple workers for action selection.

        Args:
            cmd: Command with optional fields:
                - instruction_prompt: Game rules and valid move format
                - game_name: Name of the game (e.g., "chess_v6")
                - player_id: Which player this agent is (e.g., "player_0")
        """
        try:
            # Create agent without environment
            self._agent = self._create_agent()

            # Get instruction prompt from command or use default
            instruction_prompt = cmd.get("instruction_prompt")
            game_name = cmd.get("game_name", "unknown")
            player_id = cmd.get("player_id", "player")

            if instruction_prompt:
                # Update agent's prompt builder with custom instruction
                self._agent.prompt_builder.update_instruction_prompt(instruction_prompt)
            else:
                # Use default game-specific prompt
                default_prompt = self._get_game_instruction_prompt(game_name, player_id)
                self._agent.prompt_builder.update_instruction_prompt(default_prompt)

            # Reset state (no env in action-selector mode)
            self._env = None
            self._episode_idx = 0
            self._step_idx = 0
            self._total_reward = 0.0
            self._prev_action = None
            self._episode_start_time = datetime.utcnow()
            self._llm_calls = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0

            self._emit({
                "type": "agent_ready",
                "run_id": self.config.run_id,
                "game_name": game_name,
                "player_id": player_id,
                "mode": "action_selector",
            })
            logger.info(f"Agent initialized for {game_name} as {player_id} (action-selector mode)")

        except Exception as e:
            logger.exception(f"Agent initialization failed: {e}")
            self._emit({"type": "error", "message": str(e)})

    def _get_game_instruction_prompt(self, game_name: str, player_id: str) -> str:
        """Get default instruction prompt for a PettingZoo game.

        Args:
            game_name: Name of the game (e.g., "chess_v6", "connect_four_v3")
            player_id: Which player (e.g., "player_0", "black_0")

        Returns:
            Instruction prompt string for the LLM.
        """
        # Game-specific prompts
        prompts = {
            "chess_v6": (
                f"You are playing chess as {player_id}. "
                "Analyze the board position and select your next move. "
                "Output ONLY the move in UCI format (e.g., 'e2e4', 'g1f3', 'e1g1' for castling). "
                "The legal moves will be provided in the observation."
            ),
            "connect_four_v3": (
                f"You are playing Connect Four as {player_id}. "
                "Select which column (0-6) to drop your piece. "
                "Output ONLY a single number representing the column index."
            ),
            "go_v5": (
                f"You are playing Go as {player_id}. "
                "Output your move as coordinates (row, col) or 'pass'. "
                "The legal moves will be provided."
            ),
            "tictactoe_v3": (
                f"You are playing Tic-Tac-Toe as {player_id}. "
                "Select a position (0-8) on the 3x3 grid. "
                "Output ONLY a single number."
            ),
        }

        # Get game-specific prompt or use generic
        if game_name in prompts:
            return prompts[game_name]

        return (
            f"You are playing {game_name} as {player_id}. "
            "Analyze the game state and select your action. "
            "Output ONLY the action, no explanation."
        )

    def _handle_select_action(self, cmd: Dict[str, Any]) -> None:
        """Handle select_action command - select action for multi-agent game.

        In action-selector mode, the GUI owns the environment and sends
        observations to workers. Each worker uses its LLM to select an action
        and returns it. The GUI then executes the action on the environment.

        Args:
            cmd: Command with fields:
                - observation: Game state as string or dict
                - info: Additional info (e.g., legal_moves)
                - action_mask: Optional boolean mask of valid actions
                - player_id: Which player is acting
        """
        if self._agent is None:
            self._emit({
                "type": "error",
                "message": "Agent not initialized. Send init_agent or reset first."
            })
            return

        try:
            # Extract observation from command
            observation = cmd.get("observation", "")
            info = cmd.get("info", {})
            action_mask = cmd.get("action_mask")
            player_id = cmd.get("player_id", "unknown")

            # Build observation string for LLM
            if isinstance(observation, dict):
                obs_str = str(observation)
            else:
                obs_str = str(observation)

            # Add legal moves to observation if provided
            legal_moves = info.get("legal_moves", [])
            if legal_moves:
                obs_str += f"\nLegal moves: {legal_moves}"

            # Create observation object for agent
            # The BALROG agent expects observation as a dict with nested structure:
            # obs["text"]["long_term_context"] and obs["text"]["short_term_context"]
            # (see HistoryPromptBuilder.update_observation in BALROG)
            obs_for_agent = {
                "text": {
                    "long_term_context": "",  # No long-term context for action-selector mode
                    "short_term_context": obs_str,  # Current observation
                }
            }

            # Get action from LLM agent
            input_tokens = 0
            output_tokens = 0
            action_str = ""
            reasoning = ""

            try:
                response = self._agent.act(obs_for_agent, prev_action=self._prev_action)
                self._llm_calls += 1
                action_str = response.completion.strip()
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

                # Extract reasoning if the response contains it
                if "\n" in action_str:
                    lines = action_str.split("\n")
                    action_str = lines[-1].strip()  # Last line is the action
                    reasoning = "\n".join(lines[:-1])  # Everything else is reasoning

            except Exception as e:
                logger.warning(f"Agent act failed: {e}")
                # Try to return a random valid action if available
                if legal_moves:
                    import random
                    action_str = str(random.choice(legal_moves))
                else:
                    # No legal_moves available — fall back to NOOP (action 0)
                    # to avoid sending an empty action string to the GUI.
                    action_str = "0"
                    logger.warning("No legal_moves available, falling back to NOOP (0)")

            # Convert action string to action index if action_mask provided
            action_index = None
            if action_mask is not None and legal_moves:
                try:
                    # Find the index of the selected action in legal moves
                    if action_str in legal_moves:
                        # Find the corresponding index in the action space
                        for i, valid in enumerate(action_mask):
                            if valid and legal_moves[list(action_mask[:i+1]).count(True) - 1] == action_str:
                                action_index = i
                                break
                except Exception:
                    pass

            # Ensure action is a valid integer for environments that expect
            # numeric actions (e.g., MultiGrid, MeltingPot).  If the LLM
            # returned non-numeric text, try to extract a digit; otherwise
            # fall back to NOOP (0).
            if action_index is None and action_str:
                # Try to parse as int directly
                try:
                    action_index = int(action_str)
                except ValueError:
                    # Try to extract first digit from the response
                    import re
                    digits = re.findall(r'\d+', action_str)
                    if digits:
                        action_index = int(digits[0])
                        logger.debug(
                            "Extracted action %d from LLM response: %r",
                            action_index, action_str,
                        )
                    else:
                        logger.warning(
                            "Could not parse action from LLM response: %r, "
                            "falling back to NOOP (0)", action_str,
                        )
                        action_index = 0

            # Emit action selection result
            self._emit({
                "type": "action_selected",
                "run_id": self.config.run_id,
                "player_id": player_id,
                "action": action_index if action_index is not None else 0,
                "action_str": action_str,
                "reasoning": reasoning,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "timestamp": datetime.utcnow().isoformat(),
            })

            # Update state for next turn
            self._prev_action = action_str
            self._step_idx += 1

        except Exception as e:
            logger.exception(f"Action selection failed: {e}")
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
            # BALROG environments often have 'text' key with nested context
            if "text" in obs:
                text_data = obs["text"]
                if isinstance(text_data, dict):
                    # BabyAI: extract long_term_context (spatial descriptions)
                    ltc = text_data.get("long_term_context", "")
                    stc = text_data.get("short_term_context", "")
                    # Include mission if available
                    mission = obs.get("mission", "")
                    parts = []
                    if mission:
                        parts.append(f"Mission: {mission}")
                    if ltc:
                        parts.append(f"Environment:\n{ltc}")
                    if stc:
                        parts.append(f"Status: {stc}")
                    return "\n\n".join(parts) if parts else str(text_data)
                return str(text_data)
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
                elif cmd_type == "init_agent":
                    # Multi-agent mode: initialize agent without environment
                    self._handle_init_agent(cmd)
                elif cmd_type == "select_action":
                    # Multi-agent mode: select action given external observation
                    self._handle_select_action(cmd)
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
    "LLMWorkerRuntime",
    "InteractiveLLMRuntime",
    "StepTelemetry",
    "EpisodeTelemetry",
    "LLMTelemetryEmitter",
]
