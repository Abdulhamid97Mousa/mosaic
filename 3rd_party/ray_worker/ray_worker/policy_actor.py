# 3rd_party/ray_worker/ray_worker/policy_actor.py

"""RayPolicyActor for inference using trained Ray RLlib policies.

This module provides:
- RayPolicyActor: Actor implementation for Ray RLlib policy inference
- RayPolicyController: PolicyController for multi-agent paradigm support
- Checkpoint loading and policy restoration utilities

The RayPolicyActor enables:
1. Loading trained policies from Ray RLlib checkpoints
2. Single-agent action selection (implements Actor protocol)
3. Multi-agent action selection (implements PolicyController protocol)
4. Integration with MOSAIC's ActorService and PolicyMappingService

Example:
    >>> from ray_worker.policy_actor import RayPolicyActor
    >>>
    >>> # Load a trained checkpoint
    >>> actor = RayPolicyActor.from_checkpoint(
    ...     checkpoint_path="/path/to/checkpoint",
    ...     policy_id="shared",
    ... )
    >>>
    >>> # Use in ActorService
    >>> actor_service.register_actor(actor, display_name="Ray RLlib Policy")

See Also:
    - docs/Development_Progress/1.0_DAY_42/TASK_2/03_multi_agent_tab_architecture.md
    - gym_gui/services/actor.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

_LOGGER = logging.getLogger(__name__)


# Lazy imports for Ray to avoid import errors when Ray is not installed
def _get_ray_imports():
    """Lazy import Ray modules."""
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.policy.policy import Policy
    return ray, Algorithm, Policy


@dataclass
class RayPolicyConfig:
    """Configuration for RayPolicyActor.

    Attributes:
        checkpoint_path: Path to Ray RLlib checkpoint directory.
        policy_id: ID of the policy to use (e.g., "shared", "main", agent_id).
        env_name: Optional environment name for Ray registration.
        device: Device for inference ("cpu" or "cuda").
        deterministic: Whether to use deterministic actions.
    """
    checkpoint_path: str
    policy_id: str = "shared"
    env_name: Optional[str] = None
    device: str = "cpu"
    deterministic: bool = False


@dataclass
class RayPolicyActor:
    """Actor implementation for Ray RLlib policy inference.

    Implements the Actor protocol for integration with MOSAIC's ActorService.
    Loads trained policies from Ray RLlib checkpoints and performs inference.

    Attributes:
        id: Unique identifier for this actor.
        config: Configuration specifying checkpoint and policy settings.

    Example:
        >>> actor = RayPolicyActor.from_checkpoint(
        ...     "/path/to/checkpoint",
        ...     policy_id="shared",
        ... )
        >>> action = actor.select_action(step_snapshot)
    """

    id: str = "ray_policy"
    config: Optional[RayPolicyConfig] = None

    # Internal state (not part of dataclass fields for serialization)
    _algorithm: Any = field(default=None, repr=False, init=False)
    _policy: Any = field(default=None, repr=False, init=False)
    _policy_mapping: Dict[str, str] = field(default_factory=dict, repr=False, init=False)
    _initialized: bool = field(default=False, repr=False, init=False)

    def __post_init__(self):
        """Initialize internal state."""
        self._algorithm = None
        self._policy = None
        self._policy_mapping = {}
        self._initialized = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        policy_id: str = "shared",
        actor_id: str = "ray_policy",
        env_name: Optional[str] = None,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> "RayPolicyActor":
        """Create a RayPolicyActor from a checkpoint.

        Args:
            checkpoint_path: Path to Ray RLlib checkpoint directory.
            policy_id: ID of the policy to use for inference.
            actor_id: Unique identifier for this actor.
            env_name: Optional environment name.
            device: Device for inference.
            deterministic: Whether to use deterministic actions.

        Returns:
            Initialized RayPolicyActor ready for inference.

        Raises:
            FileNotFoundError: If checkpoint path doesn't exist.
            ValueError: If policy_id not found in checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        config = RayPolicyConfig(
            checkpoint_path=str(checkpoint_path),
            policy_id=policy_id,
            env_name=env_name,
            device=device,
            deterministic=deterministic,
        )

        actor = cls(id=actor_id, config=config)
        actor._load_checkpoint()
        return actor

    def _load_checkpoint(self) -> None:
        """Load the algorithm and policy from checkpoint."""
        if self.config is None:
            raise ValueError("Config must be set before loading checkpoint")

        ray, Algorithm, Policy = _get_ray_imports()

        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        checkpoint_path = Path(self.config.checkpoint_path)

        # Try to load algorithm from checkpoint
        try:
            self._algorithm = Algorithm.from_checkpoint(str(checkpoint_path))
            _LOGGER.info(f"Loaded algorithm from checkpoint: {checkpoint_path}")

            # Get available policies
            available_policies = list(self._algorithm.workers.local_worker().policy_map.keys())
            _LOGGER.info(f"Available policies: {available_policies}")

            # Validate requested policy exists
            if self.config.policy_id not in available_policies:
                # Try to find a matching policy
                if len(available_policies) == 1:
                    actual_policy = available_policies[0]
                    _LOGGER.warning(
                        f"Policy '{self.config.policy_id}' not found, "
                        f"using '{actual_policy}'"
                    )
                    self.config.policy_id = actual_policy
                else:
                    raise ValueError(
                        f"Policy '{self.config.policy_id}' not found. "
                        f"Available: {available_policies}"
                    )

            # Get the policy
            self._policy = self._algorithm.get_policy(self.config.policy_id)
            self._initialized = True

        except Exception as e:
            _LOGGER.error(f"Failed to load checkpoint: {e}")
            raise

    def _preprocess_observation(self, observation: Any) -> np.ndarray:
        """Preprocess observation for policy inference.

        Args:
            observation: Raw observation from environment.

        Returns:
            Preprocessed observation as numpy array.
        """
        if isinstance(observation, np.ndarray):
            return observation
        elif isinstance(observation, (list, tuple)):
            return np.array(observation)
        elif isinstance(observation, dict):
            # Handle dict observations (e.g., for complex spaces)
            return observation
        else:
            return np.array(observation)

    def select_action(self, step: "StepSnapshot") -> Optional[int]:
        """Select action based on current observation.

        Implements the Actor protocol for single-agent environments.

        Args:
            step: Current step snapshot with observation.

        Returns:
            Action to take, or None if not ready.
        """
        if not self._initialized or self._policy is None:
            _LOGGER.warning("Policy not initialized, returning None")
            return None

        try:
            obs = self._preprocess_observation(step.observation)

            # Compute action using the policy
            action = self._policy.compute_single_action(
                obs,
                explore=not (self.config and self.config.deterministic),
            )

            # Handle action format
            if isinstance(action, tuple):
                # Policy returns (action, state, info)
                action = action[0]

            if isinstance(action, np.ndarray):
                if action.size == 1:
                    action = int(action.item())
                else:
                    # Multi-dimensional action - return as-is for continuous
                    return action

            return int(action)

        except Exception as e:
            _LOGGER.error(f"Action selection failed: {e}")
            return None

    def select_action_for_agent(
        self,
        agent_id: str,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Select action for a specific agent (multi-agent mode).

        Args:
            agent_id: The agent needing an action.
            observation: Agent's current observation.
            info: Optional environment info.

        Returns:
            Action for the agent.
        """
        if not self._initialized:
            return None

        try:
            # Determine which policy to use for this agent
            policy_id = self._policy_mapping.get(agent_id, self.config.policy_id)

            # Get the policy (may be different per agent)
            if self._algorithm is not None:
                policy = self._algorithm.get_policy(policy_id)
            else:
                policy = self._policy

            if policy is None:
                _LOGGER.warning(f"No policy found for agent {agent_id}")
                return None

            obs = self._preprocess_observation(observation)

            action = policy.compute_single_action(
                obs,
                explore=not (self.config and self.config.deterministic),
            )

            if isinstance(action, tuple):
                action = action[0]

            return action

        except Exception as e:
            _LOGGER.error(f"Action selection for {agent_id} failed: {e}")
            return None

    def select_actions(
        self,
        observations: Dict[str, Any],
        infos: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Select actions for all agents simultaneously.

        Args:
            observations: Dict mapping agent_id to observation.
            infos: Optional dict mapping agent_id to info.

        Returns:
            Dict mapping agent_id to action.
        """
        actions = {}
        for agent_id, obs in observations.items():
            info = infos.get(agent_id) if infos else None
            action = self.select_action_for_agent(agent_id, obs, info)
            if action is not None:
                actions[agent_id] = action
        return actions

    def set_policy_mapping(self, mapping: Dict[str, str]) -> None:
        """Set agent-to-policy mapping for multi-agent inference.

        Args:
            mapping: Dict mapping agent_id to policy_id.
        """
        self._policy_mapping = mapping
        _LOGGER.info(f"Policy mapping updated: {mapping}")

    def on_step(self, step: "StepSnapshot") -> None:
        """Receive feedback after action (no-op for inference-only actor).

        Args:
            step: Step result.
        """
        pass

    def on_episode_end(self, summary: "EpisodeSummary") -> None:
        """Episode lifecycle hook (no-op for inference-only actor).

        Args:
            summary: Episode summary.
        """
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state for new episode.

        Args:
            seed: Optional seed (not used for inference).
        """
        # Reset any internal RNN states if applicable
        if self._policy is not None and hasattr(self._policy, 'get_initial_state'):
            pass  # RNN state handling would go here

    def cleanup(self) -> None:
        """Release resources."""
        if self._algorithm is not None:
            try:
                self._algorithm.stop()
            except Exception as e:
                _LOGGER.warning(f"Error stopping algorithm: {e}")
            self._algorithm = None
        self._policy = None
        self._initialized = False

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    # --- Properties ---

    @property
    def is_ready(self) -> bool:
        """Check if actor is ready for inference."""
        return self._initialized and self._policy is not None

    @property
    def available_policies(self) -> List[str]:
        """Get list of available policy IDs."""
        if self._algorithm is None:
            return [self.config.policy_id] if self.config else []
        try:
            return list(self._algorithm.workers.local_worker().policy_map.keys())
        except Exception:
            return []


@dataclass
class RayPolicyController:
    """PolicyController implementation for multi-agent paradigm support.

    Wraps RayPolicyActor with full PolicyController protocol support
    for integration with PolicyMappingService.

    Attributes:
        actor: Underlying RayPolicyActor.
        paradigm: The stepping paradigm this controller supports.
    """

    actor: RayPolicyActor
    _paradigm: "SteppingParadigm" = None

    @property
    def id(self) -> str:
        """Unique identifier."""
        return self.actor.id

    @property
    def paradigm(self) -> "SteppingParadigm":
        """The stepping paradigm."""
        if self._paradigm is None:
            from gym_gui.core.enums import SteppingParadigm
            return SteppingParadigm.SINGLE_AGENT
        return self._paradigm

    def select_action(
        self,
        agent_id: str,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Select action for a specific agent."""
        return self.actor.select_action_for_agent(agent_id, observation, info)

    def select_actions(
        self,
        observations: Dict[str, Any],
        infos: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Select actions for all agents."""
        return self.actor.select_actions(observations, infos)

    def on_step_result(
        self,
        agent_id: str,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Receive step feedback (no-op for inference)."""
        pass

    def on_episode_end(
        self,
        agent_id: str,
        summary: "EpisodeSummary",
    ) -> None:
        """Episode end callback."""
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset for new episode."""
        self.actor.reset(seed)


# --- Factory functions ---

def create_ray_actor(
    checkpoint_path: Union[str, Path],
    *,
    policy_id: str = "shared",
    actor_id: str = "ray_policy",
    deterministic: bool = False,
) -> RayPolicyActor:
    """Convenience function to create a RayPolicyActor.

    Args:
        checkpoint_path: Path to checkpoint.
        policy_id: Policy to use.
        actor_id: Actor identifier.
        deterministic: Use deterministic actions.

    Returns:
        Initialized RayPolicyActor.
    """
    return RayPolicyActor.from_checkpoint(
        checkpoint_path,
        policy_id=policy_id,
        actor_id=actor_id,
        deterministic=deterministic,
    )


def list_checkpoint_policies(checkpoint_path: Union[str, Path]) -> List[str]:
    """List available policies in a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint.

    Returns:
        List of policy IDs.
    """
    ray, Algorithm, _ = _get_ray_imports()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    try:
        algo = Algorithm.from_checkpoint(str(checkpoint_path))
        policies = list(algo.workers.local_worker().policy_map.keys())
        algo.stop()
        return policies
    except Exception as e:
        _LOGGER.error(f"Failed to list policies: {e}")
        return []


__all__ = [
    "RayPolicyConfig",
    "RayPolicyActor",
    "RayPolicyController",
    "create_ray_actor",
    "list_checkpoint_policies",
]
