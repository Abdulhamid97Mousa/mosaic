"""Configuration dataclass for BARLOG Worker.

Defines all configuration options for running LLM agents on BALROG environments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional


# Valid environment names supported by BALROG
# Note: "minigrid" uses the same wrapper as "babyai" (BabyAI is built on MiniGrid)
ENV_NAMES = ("babyai", "minigrid", "minihack", "crafter", "nle", "textworld")

# Valid LLM client names
CLIENT_NAMES = ("openai", "anthropic", "google", "vllm")

# Valid agent types
AGENT_TYPES = ("naive", "cot", "robust_naive", "robust_cot", "few_shot", "dummy")


@dataclass
class BarlogWorkerConfig:
    """Configuration for BARLOG Worker subprocess.

    This config controls which environment, LLM client, and agent type to use,
    along with episode limits and telemetry output settings.

    Attributes:
        run_id: Unique identifier for this run (from GUI).
        env_name: Environment to use (babyai, minihack, crafter, etc.).
        task: Specific task/level within the environment.
        client_name: LLM client to use (openai, anthropic, google, vllm).
        model_id: Model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet").
        agent_type: Agent strategy (naive, cot, robust_naive, etc.).
        num_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode before truncation.
        temperature: LLM sampling temperature.
        api_key: API key for the LLM client (can also use env vars).
        base_url: Custom base URL for LLM API (for vLLM or proxies).
        telemetry_dir: Directory to write telemetry output.
        emit_jsonl: Whether to emit JSONL telemetry files.
        seed: Random seed for reproducibility.
        render_mode: Rendering mode for environments (None, "human", "rgb_array").
    """

    run_id: str
    env_name: Literal["babyai", "minigrid", "minihack", "crafter", "nle", "textworld"] = "babyai"
    task: str = "BabyAI-GoToRedBall-v0"
    client_name: Literal["openai", "anthropic", "google", "vllm"] = "openai"
    model_id: str = "gpt-4o-mini"
    agent_type: Literal["naive", "cot", "robust_naive", "robust_cot", "few_shot", "dummy"] = "naive"
    num_episodes: int = 5
    max_steps: int = 100
    temperature: float = 0.7
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    telemetry_dir: str = "./telemetry"
    emit_jsonl: bool = True
    seed: Optional[int] = None
    render_mode: Optional[str] = None
    # Advanced options
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    alternate_roles: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.env_name not in ENV_NAMES:
            raise ValueError(
                f"Invalid env_name '{self.env_name}'. "
                f"Must be one of: {ENV_NAMES}"
            )
        if self.client_name not in CLIENT_NAMES:
            raise ValueError(
                f"Invalid client_name '{self.client_name}'. "
                f"Must be one of: {CLIENT_NAMES}"
            )
        if self.agent_type not in AGENT_TYPES:
            raise ValueError(
                f"Invalid agent_type '{self.agent_type}'. "
                f"Must be one of: {AGENT_TYPES}"
            )
        if self.num_episodes < 1:
            raise ValueError("num_episodes must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "run_id": self.run_id,
            "env_name": self.env_name,
            "task": self.task,
            "client_name": self.client_name,
            "model_id": self.model_id,
            "agent_type": self.agent_type,
            "num_episodes": self.num_episodes,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "api_key": "***" if self.api_key else None,  # Mask for safety
            "base_url": self.base_url,
            "telemetry_dir": self.telemetry_dir,
            "emit_jsonl": self.emit_jsonl,
            "seed": self.seed,
            "render_mode": self.render_mode,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "alternate_roles": self.alternate_roles,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarlogWorkerConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BarlogWorkerConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_balrog_config(self) -> Dict[str, Any]:
        """Convert to BALROG-compatible OmegaConf structure.

        Returns a dict that can be converted to OmegaConf for use with
        BALROG's AgentFactory and make_env functions.
        """
        return {
            "client": {
                "client_name": self.client_name,
                "model_id": self.model_id,
                "base_url": self.base_url,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "delay": self.retry_delay,
                "alternate_roles": self.alternate_roles,
                "generate_kwargs": {
                    "temperature": self.temperature,
                },
            },
            "agent": {
                "type": self.agent_type,
                "max_icl_history": 5,  # For few_shot agent
            },
            "envs": {
                "names": self.env_name,
                "max_steps": self.max_steps,
            },
            "eval": {
                "num_episodes": self.num_episodes,
                "seed": self.seed,
            },
        }


__all__ = [
    "BarlogWorkerConfig",
    "ENV_NAMES",
    "CLIENT_NAMES",
    "AGENT_TYPES",
]
