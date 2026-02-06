"""Baseline operators for credit assignment experiments.

These operators implement simple action selection strategies for ablation studies:
- RandomOperator: Uniformly random actions
- NoopOperator: Always returns action 0 (typically no-op/stay)
- CyclingOperator: Deterministic cycling through action space

Design Philosophy:
    - Dynamic action spaces (configured from environment at runtime)
    - Trajectory tracking for credit assignment analysis
    - Reproducible via seed control
    - Compatible with Operator protocol (select_action, reset, on_step_result)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import random
from gymnasium import spaces


@dataclass
class RandomOperator:
    """Operator that selects uniformly random actions.

    Useful for ablation experiments to measure contribution of trained agents
    by replacing them with random baselines.

    Attributes:
        id: Unique identifier for this operator instance
        name: Display name for UI
        action_space: Gymnasium action space (set via set_action_space())

    Example:
        >>> env = gym.make("BabyAI-GoToRedBall-v0")
        >>> operator = RandomOperator()
        >>> operator.set_action_space(env.action_space)
        >>> operator.reset(seed=42)
        >>> action = operator.select_action(obs)
    """

    id: str = "random_baseline"
    name: str = "Random Baseline"

    # Dynamic action space - configured from environment
    action_space: Optional[spaces.Space] = None

    # Internal state
    _rng: random.Random = field(default_factory=random.Random)
    _trajectory: List[Dict[str, Any]] = field(default_factory=list)
    _seed: Optional[int] = None

    def set_action_space(self, space: spaces.Space) -> None:
        """Configure action space from environment.

        Args:
            space: Gymnasium action space (Discrete, Box, MultiDiscrete, etc.)

        Raises:
            TypeError: If space is not a valid gymnasium space
        """
        if not isinstance(space, spaces.Space):
            raise TypeError(
                f"Expected gymnasium.spaces.Space, got {type(space).__name__}"
            )
        self.action_space = space

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Sample uniformly random action from configured action space.

        Args:
            observation: Environment observation (unused for random baseline)
            info: Optional info dict from environment

        Returns:
            Random action sampled from action_space

        Raises:
            RuntimeError: If action_space not configured
        """
        if self.action_space is None:
            raise RuntimeError(
                "action_space not configured. "
                "Call set_action_space(env.action_space) first."
            )

        # Use gymnasium's native sampling (handles all space types)
        return self.action_space.sample()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset operator state for new episode.

        Args:
            seed: Random seed for reproducibility
        """
        self._seed = seed
        if seed is not None:
            self._rng.seed(seed)
            # Also seed the action space for reproducible sampling
            if self.action_space is not None:
                self.action_space.seed(seed)

        # Clear trajectory for new episode
        self._trajectory = []

    def on_step_result(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Track step result for credit assignment analysis.

        Args:
            observation: Next observation from environment
            reward: Reward received for this step
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dict from environment
        """
        self._trajectory.append({
            "step": len(self._trajectory),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

    # Credit assignment analysis helpers

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Return trajectory for post-hoc analysis.

        Returns:
            List of step dictionaries with reward, terminated, truncated
        """
        return self._trajectory.copy()

    def get_episode_return(self) -> float:
        """Return total cumulative reward for current episode.

        Returns:
            Sum of all rewards in trajectory
        """
        return sum(step["reward"] for step in self._trajectory)


@dataclass
class NoopOperator:
    """Operator that always returns the same action (typically no-op).

    Useful for extreme ablation: pair with trained agent to measure whether
    the trained agent can compensate for a completely passive partner.

    Attributes:
        id: Unique identifier for this operator instance
        name: Display name for UI
        action_index: Action to always return (default 0)
        action_space: Gymnasium action space (set via set_action_space())

    Example:
        >>> operator = NoopOperator(action_index=0)  # Always "stay"
        >>> operator.set_action_space(env.action_space)
        >>> operator.reset()
        >>> action = operator.select_action(obs)  # Always returns 0
    """

    id: str = "noop_baseline"
    name: str = "No-Op Baseline"
    action_index: int = 0  # Configurable "do nothing" action

    # Dynamic action space - for validation
    action_space: Optional[spaces.Space] = None

    # Internal state
    _trajectory: List[Dict[str, Any]] = field(default_factory=list)

    def set_action_space(self, space: spaces.Space) -> None:
        """Configure action space from environment.

        Args:
            space: Gymnasium action space

        Raises:
            TypeError: If space is not a valid gymnasium space
            ValueError: If action_index is invalid for the space
        """
        if not isinstance(space, spaces.Space):
            raise TypeError(
                f"Expected gymnasium.spaces.Space, got {type(space).__name__}"
            )

        # Validate action_index for Discrete spaces
        if isinstance(space, spaces.Discrete):
            if not (0 <= self.action_index < space.n):
                raise ValueError(
                    f"action_index {self.action_index} invalid for "
                    f"Discrete({space.n}) space"
                )

        self.action_space = space

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Return the configured no-op action.

        Args:
            observation: Environment observation (unused)
            info: Optional info dict from environment

        Returns:
            The configured action_index
        """
        return self.action_index

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset operator state for new episode.

        Args:
            seed: Random seed (unused for no-op, but accepted for interface)
        """
        self._trajectory = []

    def on_step_result(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Track step result for credit assignment analysis.

        Args:
            observation: Next observation from environment
            reward: Reward received for this step
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dict from environment
        """
        self._trajectory.append({
            "step": len(self._trajectory),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Return trajectory for post-hoc analysis."""
        return self._trajectory.copy()

    def get_episode_return(self) -> float:
        """Return total cumulative reward for current episode."""
        return sum(step["reward"] for step in self._trajectory)


@dataclass
class CyclingOperator:
    """Operator that cycles through actions deterministically.

    Action sequence: 0, 1, 2, ..., n-1, 0, 1, 2, ...

    Useful for systematic exploration and testing environment response
    to deterministic action patterns.

    Attributes:
        id: Unique identifier for this operator instance
        name: Display name for UI
        action_space: Gymnasium action space (set via set_action_space())

    Example:
        >>> operator = CyclingOperator()
        >>> operator.set_action_space(env.action_space)
        >>> operator.reset()
        >>> actions = [operator.select_action(obs) for _ in range(10)]
        >>> # actions = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2] for 7-action space
    """

    id: str = "cycling_baseline"
    name: str = "Cycling Baseline"

    # Dynamic action space
    action_space: Optional[spaces.Space] = None

    # Internal state
    _action_index: int = 0
    _trajectory: List[Dict[str, Any]] = field(default_factory=list)

    def set_action_space(self, space: spaces.Space) -> None:
        """Configure action space from environment.

        Args:
            space: Gymnasium action space

        Raises:
            TypeError: If space is not a valid gymnasium space
            ValueError: If space is not Discrete (cycling only works with discrete)
        """
        if not isinstance(space, spaces.Space):
            raise TypeError(
                f"Expected gymnasium.spaces.Space, got {type(space).__name__}"
            )

        # Only Discrete spaces supported for cycling
        if not isinstance(space, spaces.Discrete):
            raise ValueError(
                f"CyclingOperator only supports Discrete spaces, got {type(space).__name__}"
            )

        self.action_space = space

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Return next action in cycle.

        Args:
            observation: Environment observation (unused)
            info: Optional info dict from environment

        Returns:
            Next action in sequence (0, 1, 2, ..., n-1, repeat)

        Raises:
            RuntimeError: If action_space not configured
            ValueError: If action_space is not Discrete
        """
        if self.action_space is None:
            raise RuntimeError(
                "action_space not configured. "
                "Call set_action_space(env.action_space) first."
            )

        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError(
                "CyclingOperator requires Discrete action space"
            )

        # Get next action in cycle
        action = self._action_index % self.action_space.n
        self._action_index += 1

        return action

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset cycle and trajectory for new episode.

        Args:
            seed: Random seed (unused for cycling, but accepted for interface)
        """
        self._action_index = 0
        self._trajectory = []

    def on_step_result(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Track step result for credit assignment analysis.

        Args:
            observation: Next observation from environment
            reward: Reward received for this step
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dict from environment
        """
        self._trajectory.append({
            "step": len(self._trajectory),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Return trajectory for post-hoc analysis."""
        return self._trajectory.copy()

    def get_episode_return(self) -> float:
        """Return total cumulative reward for current episode."""
        return sum(step["reward"] for step in self._trajectory)


# Factory function for easy creation
def create_baseline_operator(
    behavior: str = "random",
    operator_id: Optional[str] = None,
    operator_name: Optional[str] = None,
    action_index: int = 0,  # For NoopOperator
) -> Any:
    """Factory function to create baseline operators.

    Args:
        behavior: "random", "noop", or "cycling"
        operator_id: Optional custom ID
        operator_name: Optional custom display name
        action_index: Action index for NoopOperator (default 0)

    Returns:
        Configured baseline operator (action_space still needs to be set)

    Raises:
        ValueError: If behavior is not recognized

    Example:
        >>> env = gym.make("BabyAI-GoToRedBall-v0")
        >>> operator = create_baseline_operator("random", operator_id="op_001")
        >>> operator.set_action_space(env.action_space)
        >>> operator.reset(seed=42)
    """
    if behavior == "random":
        op = RandomOperator()
    elif behavior == "noop":
        op = NoopOperator(action_index=action_index)
    elif behavior == "cycling":
        op = CyclingOperator()
    else:
        raise ValueError(
            f"Unknown baseline behavior: {behavior}. "
            f"Must be one of: 'random', 'noop', 'cycling'"
        )

    # Set custom ID and name if provided
    if operator_id:
        op.id = operator_id
    if operator_name:
        op.name = operator_name

    return op
