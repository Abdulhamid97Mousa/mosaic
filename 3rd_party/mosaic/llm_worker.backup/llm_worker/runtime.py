"""MOSAIC LLM Worker Runtime - Execution Engines.

This module provides two runtime modes:
- LLMWorkerRuntime: Autonomous multi-episode execution
- InteractiveLLMRuntime: Step-by-step execution for GUI integration

This module is FULLY INDEPENDENT - no imports from balrog_worker or BALROG.
All functionality is self-contained.
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
from typing import Any, Dict, List, Optional, Union

from .config import LLMWorkerConfig
from .clients import create_client, BaseLLMClient
from .env_utils import (
    make_env,
    get_pettingzoo_instruction_prompt,
    sanitize_info,
    obs_to_str,
)
from .environments.multigrid import (
    generate_multigrid_description,
    get_instruction_prompt as get_multigrid_instruction_prompt,
    MultiGridPromptGenerator,
)
from .environments.babyai_text import (
    BabyAIPromptGenerator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Try to import gym_gui components
# =============================================================================

try:
    from gym_gui.core.worker import TelemetryEmitter as StandardTelemetryEmitter
    from gym_gui.logging_config.helpers import log_constant
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    StandardTelemetryEmitter = None
    log_constant = None

# Try to import log constants
try:
    from gym_gui.logging_config.log_constants import (
        LOG_WORKER_MOSAIC_PROMPT_GENERATED,
        LOG_WORKER_MOSAIC_OBSERVATION_EGOCENTRIC,
        LOG_WORKER_MOSAIC_OBSERVATION_TEAMMATES,
        LOG_WORKER_MOSAIC_ACTION_PARSED,
        LOG_WORKER_MOSAIC_ACTION_PARSE_FAILED,
        LOG_WORKER_MOSAIC_RUNTIME_INTEGRATION,
    )
    _HAS_LOG_CONSTANTS = True
except ImportError:
    _HAS_LOG_CONSTANTS = False
    LOG_WORKER_MOSAIC_PROMPT_GENERATED = None
    LOG_WORKER_MOSAIC_OBSERVATION_EGOCENTRIC = None
    LOG_WORKER_MOSAIC_OBSERVATION_TEAMMATES = None
    LOG_WORKER_MOSAIC_ACTION_PARSED = None
    LOG_WORKER_MOSAIC_ACTION_PARSE_FAILED = None
    LOG_WORKER_MOSAIC_RUNTIME_INTEGRATION = None


# =============================================================================
# Telemetry Dataclasses
# =============================================================================

@dataclass
class StepTelemetry:
    """Telemetry for a single step."""
    run_id: str
    episode_id: str
    step_index: int
    observation: str
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


# =============================================================================
# Telemetry Emitter
# =============================================================================

class LLMTelemetryEmitter:
    """Emits LLM worker telemetry to JSONL files and stdout."""

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
        print(json.dumps({"type": "step", **data}), flush=True)
        if self._step_file:
            self._step_file.write(json.dumps(data) + "\n")
            self._step_file.flush()

    def emit_episode(self, episode: EpisodeTelemetry) -> None:
        """Emit episode telemetry."""
        data = asdict(episode)
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


# =============================================================================
# API Key Setup
# =============================================================================

def _setup_api_key_env(config: LLMWorkerConfig) -> None:
    """Set up API key environment variable from config.

    LLM clients read API keys from environment variables.
    This function bridges the gap by setting the appropriate env var.

    Environment variable mapping:
        - openrouter -> OPENROUTER_API_KEY
        - openai -> OPENAI_API_KEY
        - anthropic -> ANTHROPIC_API_KEY
        - google -> GOOGLE_API_KEY
        - vllm -> None (doesn't need API key)
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
        "vllm": None,
    }

    env_var = env_var_map.get(client_name)
    if env_var:
        os.environ[env_var] = config.api_key
        logger.info(f"Set {env_var} from config (client: {client_name})")
    else:
        logger.debug(f"No env var mapping for client: {client_name}")


# =============================================================================
# Main Runtime
# =============================================================================

class LLMWorkerRuntime:
    """Autonomous multi-episode runtime for MOSAIC LLM Worker.

    Runs multiple episodes autonomously, stepping through multi-agent
    environments and generating LLM actions for each agent.
    """

    def __init__(self, config: LLMWorkerConfig, *, dry_run: bool = False):
        self.config = config
        self._dry_run = dry_run
        self._env = None
        self._client: Optional[BaseLLMClient] = None
        self._prompt_generator: Optional[Union[MultiGridPromptGenerator, BabyAIPromptGenerator]] = None

        # Set up API key from config
        _setup_api_key_env(config)

        # Telemetry
        telemetry_dir = config.telemetry_dir or "var/operators/telemetry"
        self._telemetry = LLMTelemetryEmitter(
            run_id=config.run_id,
            telemetry_dir=telemetry_dir,
            emit_jsonl=config.emit_jsonl,
        )

        # Lifecycle telemetry emitter
        if _HAS_GYM_GUI and StandardTelemetryEmitter is not None:
            self._lifecycle_emitter = StandardTelemetryEmitter(run_id=config.run_id)
        else:
            self._lifecycle_emitter = None

    def _setup(self) -> None:
        """Initialize environment, LLM client, and prompt generator."""
        # Create LLM client
        self._client = create_client(
            client_name=self.config.client_name,
            model_id=self.config.model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            base_url=self.config.api_base_url,
        )

        # Create prompt generator based on environment
        if "MultiGrid" in self.config.task or self.config.env_name == "multigrid":
            self._prompt_generator = MultiGridPromptGenerator(
                env_id=self.config.task,
                num_agents=self.config.num_agents,
            )
        elif self.config.env_name in ("minigrid", "babyai") or "MiniGrid" in self.config.task or "BabyAI" in self.config.task:
            # Use BabyAI prompt generator for MiniGrid/BabyAI single-agent environments
            self._prompt_generator = BabyAIPromptGenerator(
                task=self.config.task,
            )
        else:
            # Default to MultiGrid for unknown environments
            self._prompt_generator = MultiGridPromptGenerator(
                env_id=self.config.task,
                num_agents=self.config.num_agents,
            )

        # Create environment
        self._env = make_env(
            env_name=self.config.env_name,
            task=self.config.task,
            render_mode=self.config.render_mode,
        )

    def run(self) -> Dict[str, Any]:
        """Execute autonomous training run."""
        logger.info(f"Starting {self.config.num_episodes} episodes")

        # Emit run_started lifecycle event
        if self._lifecycle_emitter:
            self._lifecycle_emitter.run_started({
                "worker_type": "llm",
                "env_name": self.config.env_name,
                "task": self.config.task,
                "client_name": self.config.client_name,
                "model_id": self.config.model_id,
                "coordination_level": self.config.coordination_level,
                "observation_mode": self.config.observation_mode,
                "num_episodes": self.config.num_episodes,
                "max_steps": self.config.max_steps_per_episode,
            })

        if self._dry_run:
            logger.info("Dry-run mode | task=%s", self.config.task)
            summary = {
                "status": "dry-run",
                "task": self.config.task,
                "config": self.config.to_dict(),
            }
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(summary)
            return summary

        # Track run statistics
        total_episodes = 0
        successful_episodes = 0
        total_reward = 0.0
        total_steps = 0
        errors: List[Dict[str, Any]] = []

        try:
            self._setup()

            if self._env is None:
                raise RuntimeError(f"Failed to create environment: {self.config.task}")

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
                        self._lifecycle_emitter.heartbeat({
                            "episodes_completed": total_episodes,
                            "success_rate": successful_episodes / total_episodes if total_episodes > 0 else 0.0,
                            "average_reward": total_reward / total_episodes if total_episodes > 0 else 0.0,
                        })

                except Exception as e:
                    logger.exception(f"Episode {episode_id} failed: {e}")
                    errors.append({"episode_id": episode_id, "error": str(e)})
                    print(json.dumps({
                        "type": "error",
                        "run_id": self.config.run_id,
                        "episode_id": episode_id,
                        "error": str(e),
                    }), flush=True)
                    # Continue with next episode (graceful degradation)

            # Generate analytics manifest
            try:
                from .analytics import write_analytics_manifest
                manifest_path = write_analytics_manifest(
                    self.config,
                    notes=f"LLM worker run with {total_episodes} episodes completed",
                )
                logger.info(f"Analytics manifest written to: {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to write analytics manifest: {e}")

            # Build run summary
            summary = {
                "status": "completed",
                "run_id": self.config.run_id,
                "worker_type": "llm",
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
                self._lifecycle_emitter.run_completed(summary)

            self._telemetry.close()
            logger.info("All episodes completed")
            return summary

        except Exception as e:
            # Emit run_failed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_failed({"error": str(e)})
            self._telemetry.close()
            raise

        finally:
            if self._env:
                self._env.close()

    def _run_episode(self, episode_idx: int, episode_id: str) -> Dict[str, Any]:
        """Run a single episode."""
        start_time = datetime.utcnow()

        # Reset environment
        obs, info = self._env.reset(seed=self.config.seed)

        # Episode tracking
        total_reward = 0.0
        llm_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        terminated = False
        truncated = False

        # Generate system prompts for each agent
        system_prompts = {}
        for agent_id in range(self.config.num_agents):
            role = self.config.get_agent_role(agent_id)
            system_prompts[agent_id] = self._prompt_generator.get_system_prompt(
                agent_id=agent_id,
                coordination_level=self.config.coordination_level,
                role=role,
            )

        step_idx = 0
        for step_idx in range(self.config.max_steps_per_episode):
            # Step each agent
            actions = {}
            for agent_id in range(self.config.num_agents):
                agent_key = f"agent_{agent_id}"

                # Get observation for this agent
                agent_obs = obs.get(agent_key) if isinstance(obs, dict) else obs

                # Format observation as text
                obs_text = self._prompt_generator.format_observation(
                    agent_obs,
                    agent_id,
                    observation_mode=self.config.observation_mode,
                )

                # Generate action via LLM
                input_tokens = 0
                output_tokens = 0
                try:
                    response = self._client.generate_with_retry(
                        prompt=obs_text,
                        system=system_prompts[agent_id],
                    )
                    llm_calls += 1
                    action = self._prompt_generator.parse_action(response.content)
                    llm_response = response.content
                    input_tokens = getattr(response, 'input_tokens', 0)
                    output_tokens = getattr(response, 'output_tokens', 0)
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                except Exception as e:
                    logger.warning(f"LLM error for agent {agent_id}: {e}")
                    action = 0  # Default to "still"
                    llm_response = f"ERROR: {e}"

                # Validate action
                action = self._validate_action(action)
                actions[agent_key] = action

                # Log step telemetry
                action_name = self._prompt_generator.action_space[action]
                step_telemetry = StepTelemetry(
                    run_id=self.config.run_id,
                    episode_id=episode_id,
                    step_index=step_idx,
                    observation=obs_text[:500],
                    action=action_name,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info={},
                    llm_response=llm_response[:200] if llm_response else None,
                    llm_input_tokens=input_tokens,
                    llm_output_tokens=output_tokens,
                )
                self._telemetry.emit_step(step_telemetry)

            # Step environment - handle single-agent vs multi-agent
            is_single_agent = self.config.num_agents == 1
            if is_single_agent:
                # Single-agent: pass int action, get scalar returns
                single_action = actions.get("agent_0", 0)
                obs, reward, terminated, truncated, info = self._env.step(single_action)
                total_reward += float(reward)
            else:
                # Multi-agent: pass dict actions, get dict returns
                obs, rewards, terminations, truncations, info = self._env.step(actions)
                for agent_key, reward in rewards.items():
                    total_reward += reward
                terminated = all(terminations.values())
                truncated = all(truncations.values())

            if terminated or truncated:
                logger.debug(f"Episode ended at step {step_idx}: terminated={terminated}, truncated={truncated}")
                break

        # Episode complete
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        success = info.get("success", terminated and total_reward > 0)

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
        self._telemetry.emit_episode(episode_telemetry)

        logger.info(
            f"Episode {episode_idx + 1} complete: "
            f"reward={total_reward:.2f}, steps={step_idx + 1}, "
            f"success={success}, duration={duration:.1f}s"
        )

        return {
            "success": bool(success),
            "total_reward": total_reward,
            "num_steps": step_idx + 1,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _validate_action(self, action: int) -> int:
        """Validate action against action space."""
        if self._prompt_generator is None:
            return max(0, action)
        if action < 0 or action >= len(self._prompt_generator.action_space):
            logger.warning(f"Invalid action {action}, defaulting to 0")
            return 0
        return action

    def _obs_to_str(self, obs: Any) -> str:
        """Convert observation to string for telemetry."""
        return obs_to_str(obs, self.config.env_name)

    def _sanitize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize info dict for JSON serialization."""
        return sanitize_info(info)


# =============================================================================
# Interactive Runtime
# =============================================================================

class InteractiveLLMRuntime:
    """Interactive step-by-step runtime for GUI integration.

    Reads JSON commands from stdin, executes single steps,
    and emits telemetry to stdout.

    Protocol:
        Input (stdin):
            {"cmd": "reset", "seed": 42}
            {"cmd": "step"}
            {"cmd": "init_agent", "game_name": "chess_v6", "player_id": "player_0"}
            {"cmd": "select_action", "observation": "...", "info": {...}}
            {"cmd": "stop"}

        Output (stdout):
            {"type": "ready", "run_id": "...", "env": "...", "task": "..."}
            {"type": "step", "step_index": 0, "observation": "...", ...}
            {"type": "episode_done", "total_reward": 0.5, ...}
            {"type": "action_selected", "action": "e2e4", ...}
            {"type": "error", "message": "..."}
    """

    def __init__(self, config: LLMWorkerConfig):
        self.config = config

        # Set up API key from config
        _setup_api_key_env(config)

        # Telemetry
        telemetry_dir = config.telemetry_dir or "var/operators/telemetry"
        self._telemetry = LLMTelemetryEmitter(
            run_id=config.run_id,
            telemetry_dir=telemetry_dir,
            emit_jsonl=config.emit_jsonl,
        )

        # Lifecycle telemetry emitter
        if _HAS_GYM_GUI and StandardTelemetryEmitter is not None:
            self._lifecycle_emitter = StandardTelemetryEmitter(run_id=config.run_id)
        else:
            self._lifecycle_emitter = None

        # State
        self._client: Optional[BaseLLMClient] = None
        self._prompt_generator: Optional[Union[MultiGridPromptGenerator, BabyAIPromptGenerator]] = None
        self._env = None
        self._episode_idx = 0
        self._step_idx = 0
        self._total_reward = 0.0
        self._prev_action = None
        self._obs = None
        self._info: Dict[str, Any] = {}  # Store env info for format_observation
        self._episode_start_time = None
        self._llm_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._system_prompts: Dict[int, str] = {}

    def _setup(self) -> None:
        """Initialize LLM client and prompt generator."""
        self._client = create_client(
            client_name=self.config.client_name,
            model_id=self.config.model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            base_url=self.config.api_base_url,
        )

        # Create prompt generator based on environment type
        if "MultiGrid" in self.config.task or self.config.env_name == "multigrid":
            self._prompt_generator = MultiGridPromptGenerator(
                env_id=self.config.task,
                num_agents=self.config.num_agents,
            )
        elif self.config.env_name in ("minigrid", "babyai") or "MiniGrid" in self.config.task or "BabyAI" in self.config.task:
            # Use BabyAI prompt generator for MiniGrid/BabyAI single-agent environments
            self._prompt_generator = BabyAIPromptGenerator(
                task=self.config.task,
            )
        else:
            # Default to MultiGrid for unknown environments
            self._prompt_generator = MultiGridPromptGenerator(
                env_id=self.config.task,
                num_agents=self.config.num_agents,
            )

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit JSON to stdout for GUI consumption."""
        print(json.dumps(data), flush=True)

    def run(self) -> None:
        """Main loop - read commands from stdin, execute, respond."""
        logger.info("Interactive mode started. Waiting for commands on stdin...")
        self._setup()

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
                    self._handle_reset(cmd)
                elif cmd_type == "step":
                    self._handle_step(cmd)
                elif cmd_type == "init_agent":
                    self._handle_init_agent(cmd)
                elif cmd_type == "select_action":
                    self._handle_select_action(cmd)
                elif cmd_type == "get_action":
                    self._handle_get_action(cmd)
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
            self._telemetry.close()
            logger.info("Interactive runtime stopped")

    def _handle_reset(self, cmd: Dict[str, Any]) -> None:
        """Handle reset command - initialize environment with seed."""
        try:
            # Close existing env if any
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass

            # Create environment
            self._env = make_env(
                env_name=self.config.env_name,
                task=self.config.task,
                render_mode=self.config.render_mode,
            )

            # Reset with seed
            seed = cmd.get("seed", self.config.seed)
            self._obs, info = self._env.reset(seed=seed)
            self._info = info  # Store info for format_observation (contains descriptions)

            # Debug: Log info dict from reset
            logger.debug(f"LOG1040 _handle_reset info keys: {list(info.keys()) if info else 'None'}")
            if info and "descriptions" in info:
                logger.debug(f"LOG1041 reset descriptions: {info['descriptions']}")

            # Generate system prompts
            self._system_prompts = {}
            for agent_id in range(self.config.num_agents):
                role = self.config.get_agent_role(agent_id)
                self._system_prompts[agent_id] = self._prompt_generator.get_system_prompt(
                    agent_id=agent_id,
                    coordination_level=self.config.coordination_level,
                    role=role,
                )
                logger.debug(f"LOG1045 system_prompt for agent {agent_id}: {self._system_prompts[agent_id][:150]}...")

            # Reset state
            self._episode_idx = 0
            self._step_idx = 0
            self._total_reward = 0.0
            self._prev_action = None
            self._episode_start_time = datetime.utcnow()
            self._llm_calls = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0

            # Build response
            response_data = {
                "type": "ready",
                "run_id": self.config.run_id,
                "env": self.config.env_name,
                "task": self.config.task,
                "seed": seed,
                "observation": obs_to_str(self._obs, self.config.env_name),
                # Include system prompts for GUI to display
                "system_prompts": {str(k): v for k, v in self._system_prompts.items()},
            }
            # For single-agent, add singular system_prompt for GUI compatibility
            if self.config.num_agents == 1:
                response_data["system_prompt"] = self._system_prompts.get(0, "")

            # Add render payload if available
            if self.config.render_mode == "rgb_array":
                try:
                    import numpy as np
                    rgb_frame = self._env.render()
                    if rgb_frame is not None and isinstance(rgb_frame, np.ndarray):
                        h, w = rgb_frame.shape[0], rgb_frame.shape[1]
                        response_data["render_payload"] = {
                            "mode": "rgb",
                            "rgb": rgb_frame.tolist(),
                            "width": w,
                            "height": h,
                        }
                except Exception as e:
                    logger.debug(f"Failed to render: {e}")

            self._emit(response_data)
            logger.info(f"Environment reset with seed={seed}")

        except Exception as e:
            logger.exception(f"Reset failed: {e}")
            self._emit({"type": "error", "message": str(e)})

    def _handle_step(self, cmd: Dict[str, Any]) -> None:
        """Handle step command - execute one environment step.

        Supports both single-agent (MiniGrid, BabyAI) and multi-agent (MultiGrid) environments.
        - Single-agent: env.step(int) -> (obs, reward, terminated, truncated, info)
        - Multi-agent: env.step(dict) -> (obs, rewards_dict, terms_dict, truncs_dict, info)
        """
        if self._env is None:
            self._emit({"type": "error", "message": "Environment not initialized. Send reset first."})
            return

        try:
            # Detect single-agent vs multi-agent mode
            is_single_agent = self.config.num_agents == 1

            # Get actions for all agents
            actions = {}
            llm_responses = {}  # Capture LLM responses for telemetry
            for agent_id in range(self.config.num_agents):
                agent_key = f"agent_{agent_id}"
                # Extract agent observation - handle both multi-agent (keyed by agent) and
                # single-agent (dict with image/mission/direction) observations
                if isinstance(self._obs, dict) and agent_key in self._obs:
                    # Multi-agent: obs is keyed by agent_id
                    agent_obs = self._obs[agent_key]
                else:
                    # Single-agent (MiniGrid/BabyAI): obs is the observation directly
                    agent_obs = self._obs

                # Format observation - pass info dict for descriptions (BabyAI/MiniGrid)
                # Debug: Log agent_obs and info dict
                logger.debug(f"LOG1041b agent_obs type: {type(agent_obs).__name__}, keys: {list(agent_obs.keys()) if isinstance(agent_obs, dict) else 'N/A'}")
                logger.debug(f"LOG1042 _handle_step info keys: {list(self._info.keys()) if self._info else 'None'}")
                if self._info and "descriptions" in self._info:
                    logger.debug(f"LOG1043 descriptions: {self._info['descriptions']}")

                obs_text = self._prompt_generator.format_observation(
                    agent_obs,
                    agent_id,
                    info=self._info,
                    observation_mode=self.config.observation_mode,
                )
                logger.debug(f"LOG1044 formatted obs_text: {obs_text[:200]}...")

                # Get LLM action
                input_tokens = 0
                output_tokens = 0
                llm_response = ""
                try:
                    response = self._client.generate_with_retry(
                        prompt=obs_text,
                        system=self._system_prompts.get(agent_id, ""),
                    )
                    self._llm_calls += 1
                    action = self._prompt_generator.parse_action(response.content)
                    llm_response = response.content
                    input_tokens = getattr(response, 'input_tokens', 0)
                    output_tokens = getattr(response, 'output_tokens', 0)
                    self._total_input_tokens += input_tokens
                    self._total_output_tokens += output_tokens
                except Exception as e:
                    logger.warning(f"LLM error for agent {agent_id}: {e}")
                    action = 0
                    llm_response = f"ERROR: {e}"

                # Validate action
                if action < 0 or action >= len(self._prompt_generator.action_space):
                    action = 0
                actions[agent_key] = action
                llm_responses[agent_key] = llm_response

            # Step environment - handle single-agent vs multi-agent
            if is_single_agent:
                # Single-agent: pass int action, get scalar returns
                single_action = actions.get("agent_0", 0)
                self._obs, reward, terminated, truncated, info = self._env.step(single_action)
                self._info = info  # Store updated info for next format_observation
                step_reward = float(reward)
            else:
                # Multi-agent: pass dict actions, get dict returns
                self._obs, rewards, terminations, truncations, info = self._env.step(actions)
                self._info = info  # Store updated info for next format_observation
                step_reward = sum(rewards.values())
                terminated = all(terminations.values())
                truncated = all(truncations.values())

            self._step_idx += 1
            self._total_reward += step_reward

            # Build step response
            action_names = {k: self._prompt_generator.action_space[v] for k, v in actions.items()}
            step_data = {
                "type": "step",
                "run_id": self.config.run_id,
                "episode_index": self._episode_idx,
                "step_index": self._step_idx,
                "client_name": self.config.client_name,
                "model_id": self.config.model_id,
                "observation": obs_to_str(self._obs, self.config.env_name),
                "actions": action_names,
                "reward": step_reward,
                "total_reward": self._total_reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": sanitize_info(info),
                "timestamp": datetime.utcnow().isoformat(),
                # LLM responses for GUI Chat panel
                "llm_responses": llm_responses,
            }
            # For single-agent, add singular fields for GUI compatibility
            if is_single_agent:
                step_data["action"] = action_names.get("agent_0", "unknown")
                step_data["llm_response"] = llm_responses.get("agent_0", "")

            # Add render payload
            if self.config.render_mode == "rgb_array":
                try:
                    import numpy as np
                    rgb_frame = self._env.render()
                    if rgb_frame is not None and isinstance(rgb_frame, np.ndarray):
                        h, w = rgb_frame.shape[0], rgb_frame.shape[1]
                        step_data["render_payload"] = {
                            "mode": "rgb",
                            "rgb": rgb_frame.tolist(),
                            "width": w,
                            "height": h,
                        }
                except Exception as e:
                    logger.debug(f"Failed to render: {e}")

            self._emit(step_data)

            # Check for episode end
            if terminated or truncated:
                self._emit_episode_done(terminated, truncated, info)
                # Auto-reset
                self._episode_idx += 1
                self._obs, reset_info = self._env.reset()
                self._info = reset_info  # Store reset info for next format_observation
                self._step_idx = 0
                self._total_reward = 0.0
                self._prev_action = None
                self._episode_start_time = datetime.utcnow()
                self._llm_calls = 0
                self._total_input_tokens = 0
                self._total_output_tokens = 0

        except Exception as e:
            logger.exception(f"Step failed: {e}")
            self._emit({"type": "error", "message": str(e)})

    def _handle_init_agent(self, cmd: Dict[str, Any]) -> None:
        """Handle init_agent command - initialize agent for action-selector mode.

        In action-selector mode, the worker doesn't own the environment.
        The GUI owns the environment and sends observations to workers.
        """
        try:
            instruction_prompt = cmd.get("instruction_prompt")
            game_name = cmd.get("game_name", "unknown")
            player_id = cmd.get("player_id", "player")

            # Store instruction prompt for action selection
            if instruction_prompt:
                self._instruction_prompt = instruction_prompt
            else:
                self._instruction_prompt = get_pettingzoo_instruction_prompt(game_name, player_id)

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

    def _handle_select_action(self, cmd: Dict[str, Any]) -> None:
        """Handle select_action command - select action for multi-agent game.

        In action-selector mode, the GUI owns the environment and sends
        observations to workers. Each worker uses its LLM to select an action.
        """
        try:
            observation = cmd.get("observation", "")
            info = cmd.get("info", {})
            action_mask = cmd.get("action_mask")
            player_id = cmd.get("player_id", "unknown")

            # Build observation string
            if isinstance(observation, dict):
                obs_str = json.dumps(observation)
            else:
                obs_str = str(observation)

            # Add legal moves if provided
            legal_moves = info.get("legal_moves", [])
            if legal_moves:
                obs_str += f"\nLegal moves: {legal_moves}"

            # Get action from LLM
            input_tokens = 0
            output_tokens = 0
            action_str = ""
            reasoning = ""

            try:
                response = self._client.generate_with_retry(
                    prompt=obs_str,
                    system=getattr(self, '_instruction_prompt', ''),
                )
                self._llm_calls += 1
                action_str = response.content.strip()
                input_tokens = getattr(response, 'input_tokens', 0)
                output_tokens = getattr(response, 'output_tokens', 0)
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

                # Extract reasoning if present
                if "\n" in action_str:
                    lines = action_str.split("\n")
                    action_str = lines[-1].strip()
                    reasoning = "\n".join(lines[:-1])

            except Exception as e:
                logger.warning(f"Agent act failed: {e}")
                if legal_moves:
                    import random
                    action_str = str(random.choice(legal_moves))

            self._emit({
                "type": "action_selected",
                "run_id": self.config.run_id,
                "player_id": player_id,
                "action": action_str,
                "action_str": action_str,
                "reasoning": reasoning,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "timestamp": datetime.utcnow().isoformat(),
            })

            self._prev_action = action_str
            self._step_idx += 1

        except Exception as e:
            logger.exception(f"Action selection failed: {e}")
            self._emit({"type": "error", "message": str(e)})

    def _handle_get_action(self, cmd: Dict[str, Any]) -> None:
        """Handle get_action command - generate LLM action for an agent."""
        agent_id = cmd.get("agent_id", 0)
        agent_key = f"agent_{agent_id}"

        if self._obs is None:
            self._emit({"type": "error", "message": "No observation available. Call reset first."})
            return

        try:
            # Get observation
            agent_obs = self._obs.get(agent_key) if isinstance(self._obs, dict) else self._obs

            # Format observation
            obs_text = self._prompt_generator.format_observation(
                agent_obs,
                agent_id,
                observation_mode=self.config.observation_mode,
            )

            # Generate action
            response = self._client.generate_with_retry(
                prompt=obs_text,
                system=self._system_prompts.get(agent_id, ""),
            )
            action = self._prompt_generator.parse_action(response.content)

            # Validate
            if action < 0 or action >= len(self._prompt_generator.action_space):
                action = 0
            action_name = self._prompt_generator.action_space[action]

            self._emit({
                "type": "action_result",
                "agent_id": agent_id,
                "action": action,
                "action_name": action_name,
                "llm_response": response.content,
                "llm_latency_ms": getattr(response, 'latency_ms', None),
                "observation_text": obs_text[:500],
            })

        except Exception as e:
            self._emit({
                "type": "action_result",
                "agent_id": agent_id,
                "action": 0,
                "action_name": "still",
                "error": str(e),
            })

    def _emit_episode_done(self, terminated: bool, truncated: bool, info: Dict[str, Any]) -> None:
        """Emit episode completion telemetry."""
        end_time = datetime.utcnow()
        duration = (end_time - self._episode_start_time).total_seconds() if self._episode_start_time else 0.0
        success = info.get("success", terminated and self._total_reward > 0)

        episode_id = f"{self.config.run_id}-ep{self._episode_idx:06d}"
        self._emit({
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
        })

        logger.info(
            f"Episode {self._episode_idx + 1} complete: "
            f"reward={self._total_reward:.2f}, steps={self._step_idx}, "
            f"success={success}"
        )


__all__ = [
    "LLMWorkerRuntime",
    "InteractiveLLMRuntime",
    "LLMTelemetryEmitter",
    "StepTelemetry",
    "EpisodeTelemetry",
]
