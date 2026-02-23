"""Configuration dataclasses for PettingZoo environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PettingZooConfig:
    """Configuration for PettingZoo multi-agent environments.

    Attributes:
        env_id: The PettingZoo environment identifier (e.g., "chess_v6")
        family: The environment family (classic, mpe, sisl, butterfly, atari)
        render_mode: Rendering mode ("rgb_array", "human", "ansi")
        max_cycles: Maximum number of cycles before truncation
        seed: Random seed for reproducibility
        env_kwargs: Additional keyword arguments passed to environment constructor
        human_player: Name of the agent controlled by human (for hybrid modes)
        agent_configs: Per-agent configuration (for training, policies, etc.)
    """

    env_id: str
    family: str = "classic"
    render_mode: str = "rgb_array"
    max_cycles: int = 500
    seed: Optional[int] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    human_player: Optional[str] = None
    agent_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_full_env_name(self) -> str:
        """Get the full import path for the environment.

        Returns:
            Full environment path like "pettingzoo.classic.chess_v6"
        """
        return f"pettingzoo.{self.family}.{self.env_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "env_id": self.env_id,
            "family": self.family,
            "render_mode": self.render_mode,
            "max_cycles": self.max_cycles,
            "seed": self.seed,
            "env_kwargs": self.env_kwargs,
            "human_player": self.human_player,
            "agent_configs": self.agent_configs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PettingZooConfig":
        """Create config from dictionary."""
        return cls(
            env_id=data["env_id"],
            family=data.get("family", "classic"),
            render_mode=data.get("render_mode", "rgb_array"),
            max_cycles=data.get("max_cycles", 500),
            seed=data.get("seed"),
            env_kwargs=data.get("env_kwargs", {}),
            human_player=data.get("human_player"),
            agent_configs=data.get("agent_configs", {}),
        )


@dataclass
class AgentConfig:
    """Configuration for a single agent in a multi-agent environment.

    Attributes:
        name: Agent identifier (e.g., "player_0", "player_1")
        controller: Controller type ("human", "random", "policy", "llm")
        policy_path: Path to trained policy (if controller is "policy")
        policy_kwargs: Additional kwargs for policy loading
    """

    name: str
    controller: str = "random"
    policy_path: Optional[str] = None
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "controller": self.controller,
            "policy_path": self.policy_path,
            "policy_kwargs": self.policy_kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            controller=data.get("controller", "random"),
            policy_path=data.get("policy_path"),
            policy_kwargs=data.get("policy_kwargs", {}),
        )
