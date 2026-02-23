"""Configuration for Human Worker."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Action labels for common environments
ACTION_LABELS: Dict[str, List[str]] = {
    # MiniGrid/BabyAI: 7 actions
    "minigrid": ["Turn Left", "Turn Right", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    "babyai": ["Turn Left", "Turn Right", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    # MultiGrid: 8 actions (includes "Still")
    "multigrid": ["Still", "Turn Left", "Turn Right", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    # Classic control
    "FrozenLake-v1": ["Left", "Down", "Right", "Up"],
    "FrozenLake-v2": ["Left", "Down", "Right", "Up"],
    "Taxi-v3": ["South", "North", "East", "West", "Pickup", "Dropoff"],
    "CliffWalking-v0": ["Up", "Right", "Down", "Left"],
    "Blackjack-v1": ["Stand", "Hit"],
    "CartPole-v1": ["Push Left", "Push Right"],
    "MountainCar-v0": ["Push Left", "No Push", "Push Right"],
    "Acrobot-v1": ["Torque -1", "Torque 0", "Torque +1"],
    "LunarLander-v3": ["Noop", "Fire Left", "Fire Main", "Fire Right"],
    # Crafter: 17 actions
    "crafter": [
        "Noop", "Move Left", "Move Right", "Move Up", "Move Down",
        "Do", "Sleep", "Place Stone", "Place Table", "Place Furnace",
        "Place Plant", "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe",
        "Make Wood Sword", "Make Stone Sword", "Make Iron Sword"
    ],
    # NetHack/NLE
    "nle": [
        "North", "East", "South", "West", "NE", "SE", "SW", "NW",
        "Wait", "Kick", "Open", "Search", "Look", "Pray", "Pickup", "Drop",
        "Inventory", "Fire", "Apply", "Eat", "Quaff", "Read", "Wear", "Wield"
    ],
}


def get_action_labels(env_name: str, task: str, action_space_n: int) -> List[str]:
    """Get action labels for an environment.

    Args:
        env_name: Environment family (minigrid, babyai, etc.)
        task: Specific task (MiniGrid-Empty-8x8-v0, etc.)
        action_space_n: Number of actions in action space

    Returns:
        List of action label strings
    """
    # Check task first (more specific)
    if task in ACTION_LABELS:
        labels = ACTION_LABELS[task]
        return labels[:action_space_n]

    # Check task prefix patterns
    if task.startswith("MiniGrid") or task.startswith("BabyAI"):
        labels = ACTION_LABELS.get("minigrid", [])
        return labels[:action_space_n]
    if task.startswith("MultiGrid"):
        labels = ACTION_LABELS.get("multigrid", [])
        return labels[:action_space_n]
    if task.startswith("NetHack") or task.startswith("MiniHack"):
        labels = ACTION_LABELS.get("nle", [])
        return labels[:action_space_n]

    # Check env_name family
    if env_name in ACTION_LABELS:
        labels = ACTION_LABELS[env_name]
        return labels[:action_space_n]

    # Default: generic action labels
    return [f"Action {i}" for i in range(action_space_n)]


@dataclass
class HumanWorkerConfig:
    """Configuration for Human Worker.

    Attributes:
        run_id: Unique run identifier.
        player_name: Human player's display name.
        env_name: Environment family (minigrid, babyai, frozenlake, etc.)
        task: Specific environment task (MiniGrid-Empty-8x8-v0, etc.)
        render_mode: Render mode for environment (default: rgb_array)
        seed: Random seed for environment
        timeout_seconds: Timeout for human input (0 = no timeout).
        show_legal_moves: Whether to highlight legal moves in UI.
        confirm_moves: Whether to require move confirmation.
    """

    run_id: str = ""
    player_name: str = "Human"

    # Environment configuration
    env_name: str = ""  # Environment family
    task: str = ""  # Specific task/environment
    render_mode: str = "rgb_array"
    seed: int = 42
    game_resolution: Tuple[int, int] = (512, 512)  # Render resolution for Crafter

    # Timeout settings
    timeout_seconds: float = 0.0  # 0 = no timeout (wait forever)

    # UI hints
    show_legal_moves: bool = True
    confirm_moves: bool = False

    # Telemetry
    telemetry_dir: str = "var/telemetry"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "player_name": self.player_name,
            "env_name": self.env_name,
            "task": self.task,
            "render_mode": self.render_mode,
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
            "show_legal_moves": self.show_legal_moves,
            "confirm_moves": self.confirm_moves,
            "telemetry_dir": self.telemetry_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanWorkerConfig":
        """Create from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            player_name=data.get("player_name", "Human"),
            env_name=data.get("env_name", ""),
            task=data.get("task", ""),
            render_mode=data.get("render_mode", "rgb_array"),
            seed=data.get("seed", 42),
            timeout_seconds=data.get("timeout_seconds", 0.0),
            show_legal_moves=data.get("show_legal_moves", True),
            confirm_moves=data.get("confirm_moves", False),
            telemetry_dir=data.get("telemetry_dir", "var/telemetry"),
        )
