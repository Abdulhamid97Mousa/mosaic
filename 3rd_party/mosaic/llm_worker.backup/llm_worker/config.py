"""Configuration for MOSAIC LLM Worker.

This module defines the configuration dataclass for multi-agent LLM coordination.
It follows the MOSAIC WorkerConfig protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# Valid environment names supported by MOSAIC LLM Worker
ENV_NAMES = (
    "multigrid",      # gym-multigrid multi-agent environments
    "pettingzoo",     # PettingZoo multi-agent games
    "gymnasium",      # Standard Gymnasium environments
    "minigrid",       # Single-agent MiniGrid
    "babyai",         # BabyAI language grounding
)

# Valid LLM client names
CLIENT_NAMES = ("openrouter", "openai", "anthropic", "google", "vllm")

# Valid agent types
AGENT_TYPES = ("naive", "cot", "robust_naive", "robust_cot", "few_shot", "dummy")


@dataclass
class LLMWorkerConfig:
    """Configuration for MOSAIC LLM Worker.

    This dataclass implements the WorkerConfig protocol and provides
    configuration for multi-agent LLM coordination experiments.

    Attributes:
        run_id: Unique run identifier (REQUIRED by protocol).
        seed: Random seed (REQUIRED by protocol, can be None).

        # Environment Config
        env_name: Environment family (multigrid, pettingzoo, etc.).
        task: Specific environment task (MultiGrid-Soccer-v0, etc.).
        num_agents: Number of agents in the environment.

        # LLM Config
        client_name: LLM provider (openrouter, openai, anthropic, vllm).
        model_id: Model identifier (e.g., anthropic/claude-3.5-sonnet).
        temperature: Generation temperature (0.0-2.0).
        max_tokens: Maximum response tokens.
        timeout: API request timeout in seconds.
        api_base_url: Custom API base URL (for vLLM local servers).
        api_key: API key for the LLM client (can also use env vars).

        # Retry settings
        max_retries: Maximum retry attempts for LLM calls.
        retry_delay: Delay between retries in seconds.

        # Multi-Agent Coordination Config
        coordination_level: Coordination strategy level (1, 2, or 3).
            - 1 = Emergent: Minimal guidance, test emergent coordination.
            - 2 = Basic Hints: Add cooperation tips.
            - 3 = Role-Based: Explicit roles with detailed strategies.
        observation_mode: Observation text generation mode.
            - "egocentric": Agent sees only its own view.
            - "visible_teammates": Include visible teammates (Theory of Mind).
        agent_roles: Optional role assignments for Level 3 coordination.
        agent_id: Agent index for multi-agent environments.
        role: Agent role for Level 3 (e.g., "forward", "defender").

        # Execution Config
        num_episodes: Number of episodes to run.
        max_steps_per_episode: Maximum steps per episode.
        alternate_roles: Whether to alternate roles between episodes.

        # Rendering
        render_mode: Rendering mode (None, "human", "rgb_array").

        # VLM (Vision-Language Model) settings
        max_image_history: Max images to keep in history (0 for text-only).

        # Telemetry Config
        telemetry_dir: Directory for telemetry output.
        emit_jsonl: Write telemetry to JSONL files.
        emit_stdout: Write telemetry to stdout.
    """

    # Protocol required fields
    run_id: str = ""
    seed: Optional[int] = None

    # Environment configuration
    env_name: Literal["multigrid", "pettingzoo", "gymnasium", "minigrid", "babyai"] = "multigrid"
    task: str = "MultiGrid-Soccer-v0"
    num_agents: int = 4

    # LLM configuration
    client_name: Literal["openrouter", "openai", "anthropic", "google", "vllm"] = "openrouter"
    model_id: str = "anthropic/claude-3.5-sonnet"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 60.0
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Multi-agent coordination
    coordination_level: int = 1
    observation_mode: str = "egocentric"
    agent_roles: Optional[List[str]] = None
    agent_id: int = 0
    role: Optional[str] = None

    # Execution
    num_episodes: int = 1
    max_steps_per_episode: int = 100
    alternate_roles: bool = True

    # Rendering
    render_mode: Optional[str] = None

    # VLM settings
    max_image_history: int = 0

    # Telemetry
    telemetry_dir: Optional[str] = None  # Resolved in __post_init__
    emit_jsonl: bool = True
    emit_stdout: bool = True

    def __post_init__(self) -> None:
        """Validate configuration and assert protocol compliance."""
        # Validate env_name
        if self.env_name not in ENV_NAMES:
            raise ValueError(
                f"Invalid env_name '{self.env_name}'. "
                f"Must be one of: {ENV_NAMES}"
            )

        # Validate client_name
        if self.client_name not in CLIENT_NAMES:
            raise ValueError(
                f"Invalid client_name '{self.client_name}'. "
                f"Must be one of: {CLIENT_NAMES}"
            )

        # Validate coordination level
        if self.coordination_level not in (1, 2, 3):
            raise ValueError(f"coordination_level must be 1, 2, or 3, got {self.coordination_level}")

        # Validate observation mode
        valid_modes = ("egocentric", "visible_teammates")
        if self.observation_mode not in valid_modes:
            raise ValueError(f"observation_mode must be one of {valid_modes}, got {self.observation_mode}")

        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")

        # Validate num_episodes
        if self.num_episodes < 1:
            raise ValueError("num_episodes must be >= 1")

        # Validate max_steps_per_episode
        if self.max_steps_per_episode < 1:
            raise ValueError("max_steps_per_episode must be >= 1")

        # Validate agent_roles length if provided
        if self.agent_roles is not None and len(self.agent_roles) != self.num_agents:
            raise ValueError(
                f"agent_roles length ({len(self.agent_roles)}) must match num_agents ({self.num_agents})"
            )

        # Resolve telemetry_dir to VAR_OPERATORS_TELEMETRY_DIR if not specified
        if self.telemetry_dir is None:
            try:
                from gym_gui.config.paths import VAR_OPERATORS_TELEMETRY_DIR
                object.__setattr__(self, "telemetry_dir", str(VAR_OPERATORS_TELEMETRY_DIR))
            except ImportError:
                object.__setattr__(self, "telemetry_dir", "var/operators/telemetry")

        # Protocol compliance check (graceful fallback if gym_gui unavailable)
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
            assert isinstance(self, WorkerConfigProtocol), (
                "LLMWorkerConfig must implement WorkerConfig protocol"
            )
        except ImportError:
            pass  # gym_gui not available, skip protocol check

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (REQUIRED by protocol)."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "env_name": self.env_name,
            "task": self.task,
            "num_agents": self.num_agents,
            "client_name": self.client_name,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "api_base_url": self.api_base_url,
            "api_key": "***" if self.api_key else None,  # Mask for safety
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "coordination_level": self.coordination_level,
            "observation_mode": self.observation_mode,
            "agent_roles": self.agent_roles,
            "agent_id": self.agent_id,
            "role": self.role,
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "alternate_roles": self.alternate_roles,
            "render_mode": self.render_mode,
            "max_image_history": self.max_image_history,
            "telemetry_dir": self.telemetry_dir,
            "emit_jsonl": self.emit_jsonl,
            "emit_stdout": self.emit_stdout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMWorkerConfig":
        """Create config from dictionary (REQUIRED by protocol)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "LLMWorkerConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def get_agent_role(self, agent_id: int) -> str:
        """Get role for a specific agent.

        Args:
            agent_id: Agent index.

        Returns:
            Role string or default based on agent position.
        """
        if self.agent_roles and agent_id < len(self.agent_roles):
            return self.agent_roles[agent_id]

        # Use explicit role if set
        if self.role:
            return self.role

        # Default roles for Soccer (4 agents, 2 teams)
        if "Soccer" in self.task:
            # Team 0: agents 0, 1; Team 1: agents 2, 3
            # First agent in each team is forward, second is defender
            is_forward = (agent_id % 2) == 0
            return "forward" if is_forward else "defender"

        return "agent"


def load_worker_config(config_path: str) -> LLMWorkerConfig:
    """Load worker configuration from JSON file.

    Handles both direct config format and nested metadata.worker.config structure.

    Args:
        config_path: Path to JSON configuration file.

    Returns:
        Parsed LLMWorkerConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw_data = json.load(f)

    # Handle nested metadata.worker.config structure from GUI
    if "metadata" in raw_data and "worker" in raw_data["metadata"]:
        worker_data = raw_data["metadata"]["worker"]
        config_data = worker_data.get("config", {})
    else:
        # Direct config format
        config_data = raw_data

    return LLMWorkerConfig.from_dict(config_data)


__all__ = [
    "LLMWorkerConfig",
    "load_worker_config",
    "ENV_NAMES",
    "CLIENT_NAMES",
    "AGENT_TYPES",
]
