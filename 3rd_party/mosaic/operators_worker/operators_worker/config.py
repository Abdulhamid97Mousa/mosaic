"""Configuration for operators worker."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OperatorsWorkerConfig:
    """Configuration for baseline operator worker.

    Attributes:
        run_id: Unique identifier for this worker run (e.g., operator_0_abc123)
        behavior: Operator behavior - "random", "noop", or "cycling"
        env_name: Environment family (e.g., "babyai", "minigrid", "multigrid")
        task: Specific environment task (e.g., "BabyAI-GoToRedBall-v0")
        telemetry_dir: Directory to write JSONL telemetry files (auto-resolved to VAR_OPERATORS_TELEMETRY_DIR)
        emit_jsonl: Whether to write JSONL telemetry files (default: True)
        seed: Random seed for reproducibility (optional)
        interactive: Whether to run in interactive mode (read commands from stdin)
        max_steps: Maximum steps per episode before truncation (optional, defaults to env's max_episode_steps)

    Interactive Mode:
        When interactive=True, the worker waits for JSON commands from stdin:
            {"cmd": "reset", "seed": 42, "env_name": "babyai", "task": "BabyAI-GoToRedBall-v0", "max_steps": 100}
            {"cmd": "step"}
            {"cmd": "stop"}

        Responses are emitted to stdout as JSON:
            {"type": "init"}
            {"type": "ready", "render_payload": {...}}
            {"type": "step", "reward": 0.0, "terminated": false, "render_payload": {...}}
    """

    run_id: str
    behavior: str = "random"  # "random", "noop", "cycling"
    env_name: str = "babyai"
    task: str = "BabyAI-GoToRedBall-v0"
    telemetry_dir: Optional[str] = None
    emit_jsonl: bool = True
    seed: Optional[int] = None
    interactive: bool = False  # Set to True for GUI-controlled execution
    max_steps: Optional[int] = None  # Maximum steps per episode (None = use env default)

    def __post_init__(self):
        """Resolve telemetry_dir to VAR_OPERATORS_TELEMETRY_DIR if not specified."""
        if self.telemetry_dir is None:
            try:
                from gym_gui.config.paths import VAR_OPERATORS_TELEMETRY_DIR
                object.__setattr__(self, "telemetry_dir", str(VAR_OPERATORS_TELEMETRY_DIR))
            except ImportError:
                # Fallback if gym_gui not available (e.g., standalone testing)
                object.__setattr__(self, "telemetry_dir", "var/operators/telemetry")

        # Validate behavior
        valid_behaviors = ("random", "noop", "cycling")
        if self.behavior not in valid_behaviors:
            raise ValueError(
                f"Invalid behavior '{self.behavior}'. Must be one of {valid_behaviors}"
            )
