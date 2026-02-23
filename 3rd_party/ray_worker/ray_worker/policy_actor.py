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
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

import numpy as np

# Type checking imports for forward references
if TYPE_CHECKING:
    from gym_gui.core.enums import SteppingParadigm
    from gym_gui.services.actor import StepSnapshot, EpisodeSummary

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
        env_family: Environment family (sisl, mpe, butterfly, classic).
        env_id: Environment ID (e.g., "pursuit_v4", "waterworld_v4").
        device: Device for inference ("cpu" or "cuda").
        deterministic: Whether to use deterministic actions.
    """
    checkpoint_path: str
    policy_id: str = "shared"
    env_name: Optional[str] = None
    env_family: Optional[str] = None
    env_id: Optional[str] = None
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
        env_family: Optional[str] = None,
        env_id: Optional[str] = None,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> "RayPolicyActor":
        """Create a RayPolicyActor from a checkpoint.

        Args:
            checkpoint_path: Path to Ray RLlib checkpoint directory.
            policy_id: ID of the policy to use for inference.
            actor_id: Unique identifier for this actor.
            env_name: Optional environment name (format: "{family}_{env_id}").
            env_family: Environment family (sisl, mpe, butterfly, classic).
            env_id: Environment ID (e.g., "pursuit_v4", "waterworld_v4").
            device: Device for inference.
            deterministic: Whether to use deterministic actions.

        Returns:
            Initialized RayPolicyActor ready for inference.

        Raises:
            FileNotFoundError: If checkpoint path doesn't exist.
            ValueError: If policy_id not found in checkpoint.

        Note:
            Either env_name OR (env_family + env_id) must be provided for
            environment registration with Ray. The checkpoint was trained
            with a specific environment name that must be re-registered.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Build env_name from family and env_id if not provided
        if env_name is None and env_family and env_id:
            env_name = f"{env_family}_{env_id}"

        config = RayPolicyConfig(
            checkpoint_path=str(checkpoint_path),
            policy_id=policy_id,
            env_name=env_name,
            env_family=env_family,
            env_id=env_id,
            device=device,
            deterministic=deterministic,
        )

        actor = cls(id=actor_id, config=config)
        actor._load_checkpoint()
        return actor

    def _load_checkpoint(self) -> None:
        """Load the algorithm and policy from checkpoint.

        This method:
        1. Initializes Ray if needed (with fresh state for evaluation)
        2. Registers the environment with Ray using the same name as training
        3. Loads the algorithm from checkpoint
        4. Extracts the policy for inference
        """
        if self.config is None:
            raise ValueError("Config must be set before loading checkpoint")

        ray, Algorithm, Policy = _get_ray_imports()
        from ray.tune.registry import register_env

        # IMPORTANT: Do NOT shutdown Ray if it's already running
        # The training process may have started Ray, and shutting it down
        # causes background threads to fail with GCS connection errors,
        # which can terminate the entire application after 60 seconds.
        if not ray.is_initialized():
            _LOGGER.info("Initializing Ray for policy evaluation")
            ray.init(ignore_reinit_error=True, log_to_driver=False)
        else:
            _LOGGER.info("Reusing existing Ray instance for evaluation")

        checkpoint_path = Path(self.config.checkpoint_path)

        # Register the environment before loading checkpoint
        # The checkpoint expects the environment to be registered with the same name
        if self.config.env_name and self.config.env_family and self.config.env_id:
            self._register_environment(register_env)
        else:
            _LOGGER.warning(
                "Environment info not provided - checkpoint may fail to load if "
                "it requires a registered environment. Provide env_family and env_id."
            )

        # Try to load algorithm from checkpoint
        try:
            self._algorithm = Algorithm.from_checkpoint(str(checkpoint_path))
            _LOGGER.info(f"Loaded algorithm from checkpoint: {checkpoint_path}")

            # Get available policies - handle different Ray RLlib versions
            available_policies = self._get_available_policies()
            _LOGGER.info(f"Available policies: {available_policies}")

            # Validate requested policy exists
            if available_policies and self.config.policy_id not in available_policies:
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
            if self._policy is None:
                # Try common fallback policy names
                for fallback in ["shared", "default_policy", "main"]:
                    self._policy = self._algorithm.get_policy(fallback)
                    if self._policy is not None:
                        _LOGGER.warning(f"Using fallback policy: {fallback}")
                        self.config.policy_id = fallback
                        break

            if self._policy is None:
                raise ValueError(f"Could not load any policy from checkpoint")

            self._initialized = True

        except Exception as e:
            _LOGGER.error(f"Failed to load checkpoint: {e}")
            raise

    def _get_available_policies(self) -> List[str]:
        """Get list of available policy IDs from algorithm.

        Handles different Ray RLlib versions and worker configurations.
        """
        # Try different methods to get policies
        try:
            # Method 1: Through workers (Ray RLlib < 2.x or with workers enabled)
            if hasattr(self._algorithm, 'workers'):
                workers = self._algorithm.workers
                if hasattr(workers, 'local_worker') and callable(workers.local_worker):
                    local_worker = workers.local_worker()
                    if hasattr(local_worker, 'policy_map'):
                        return list(local_worker.policy_map.keys())  # type: ignore[union-attr]
        except Exception as e:
            _LOGGER.debug(f"Could not get policies from workers: {e}")

        try:
            # Method 2: Through algorithm config (Ray RLlib 2.x)
            if hasattr(self._algorithm, 'config'):
                config = self._algorithm.config
                if hasattr(config, 'policies') and config.policies:
                    if isinstance(config.policies, dict):
                        return list(config.policies.keys())
                    elif isinstance(config.policies, set):
                        return list(config.policies)
        except Exception as e:
            _LOGGER.debug(f"Could not get policies from config: {e}")

        try:
            # Method 3: Through get_policy with common names
            common_names = ["shared", "default_policy", "main"]
            found = []
            for name in common_names:
                policy = self._algorithm.get_policy(name)
                if policy is not None:
                    found.append(name)
            if found:
                return found
        except Exception as e:
            _LOGGER.debug(f"Could not probe for common policy names: {e}")

        # Return requested policy as fallback (will be validated later)
        if self.config is None:
            return []
        return [self.config.policy_id] if self.config.policy_id else []

    def _register_environment(self, register_env) -> None:
        """Register the environment with Ray for checkpoint loading.

        Creates an environment factory and registers it with the same name
        that was used during training, so Algorithm.from_checkpoint() can
        find and recreate the environment.
        """
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv, PettingZooEnv

        if self.config is None:
            raise ValueError("Config not set, cannot register environment")
        env_name = self.config.env_name
        env_family = self.config.env_family.lower() if self.config.env_family else ""
        env_id = self.config.env_id

        _LOGGER.info(f"Registering environment: {env_name} (family={env_family}, env_id={env_id})")

        def env_creator(_config: dict):
            """Create the PettingZoo environment for Ray."""
            # Create environment based on family
            if env_family == "sisl":
                if env_id == "waterworld_v4":
                    from pettingzoo.sisl import waterworld_v4
                    env = waterworld_v4.parallel_env(render_mode="rgb_array")
                elif env_id == "multiwalker_v9":
                    from pettingzoo.sisl import multiwalker_v9
                    env = multiwalker_v9.parallel_env(render_mode="rgb_array")
                elif env_id == "pursuit_v4":
                    from pettingzoo.sisl import pursuit_v4
                    env = pursuit_v4.parallel_env(render_mode="rgb_array")
                else:
                    raise ValueError(f"Unknown SISL environment: {env_id}")
            elif env_family == "mpe":
                if env_id == "simple_spread_v3":
                    from pettingzoo.mpe import simple_spread_v3
                    env = simple_spread_v3.parallel_env(render_mode="rgb_array")
                elif env_id == "simple_adversary_v3":
                    from pettingzoo.mpe import simple_adversary_v3
                    env = simple_adversary_v3.parallel_env(render_mode="rgb_array")
                elif env_id == "simple_tag_v3":
                    from pettingzoo.mpe import simple_tag_v3
                    env = simple_tag_v3.parallel_env(render_mode="rgb_array")
                else:
                    raise ValueError(f"Unknown MPE environment: {env_id}")
            elif env_family == "butterfly":
                if env_id == "knights_archers_zombies_v10":
                    from pettingzoo.butterfly import knights_archers_zombies_v10
                    env = knights_archers_zombies_v10.parallel_env(render_mode="rgb_array")
                elif env_id == "cooperative_pong_v5":
                    from pettingzoo.butterfly import cooperative_pong_v5
                    env = cooperative_pong_v5.parallel_env(render_mode="rgb_array")
                elif env_id == "pistonball_v6":
                    from pettingzoo.butterfly import pistonball_v6
                    env = pistonball_v6.parallel_env(render_mode="rgb_array")
                else:
                    raise ValueError(f"Unknown Butterfly environment: {env_id}")
            elif env_family == "classic":
                # Classic games use AEC API
                if env_id == "chess_v6":
                    from pettingzoo.classic import chess_v6
                    env = chess_v6.env()
                elif env_id == "go_v5":
                    from pettingzoo.classic import go_v5
                    env = go_v5.env()
                elif env_id == "connect_four_v3":
                    from pettingzoo.classic import connect_four_v3
                    env = connect_four_v3.env()
                elif env_id == "tictactoe_v3":
                    from pettingzoo.classic import tictactoe_v3
                    env = tictactoe_v3.env()
                else:
                    raise ValueError(f"Unknown Classic environment: {env_id}")
            else:
                raise ValueError(f"Unknown environment family: {env_family}")

            # Wrap for Ray RLlib
            # Classic games are AEC, others are Parallel
            if env_family == "classic":
                return PettingZooEnv(env)
            else:
                return ParallelPettingZooEnv(env)

        # Register the environment
        register_env(env_name, env_creator)
        _LOGGER.info(f"Environment registered: {env_name}")

    def _preprocess_observation(self, observation: Any) -> Union[np.ndarray, Dict[str, Any]]:
        """Preprocess observation for policy inference.

        Args:
            observation: Raw observation from environment.

        Returns:
            Preprocessed observation as numpy array or dict for complex spaces.
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

    def select_action(self, step: "StepSnapshot") -> Union[int, np.ndarray, None]:
        """Select action based on current observation.

        Implements the Actor protocol for single-agent environments.

        Args:
            step: Current step snapshot with observation.

        Returns:
            Action to take (int for discrete, array for continuous), or None if not ready.
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
            default_policy = self.config.policy_id if self.config else "shared"
            policy_id = self._policy_mapping.get(agent_id, default_policy)

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
        return self._get_available_policies()


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
    _paradigm: Optional["SteppingParadigm"] = None

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
        # Access workers and policy_map with proper checks for Ray internals
        workers = getattr(algo, 'workers', None)
        if workers is not None:
            local_worker = workers.local_worker() if callable(getattr(workers, 'local_worker', None)) else None
            if local_worker is not None:
                policy_map = getattr(local_worker, 'policy_map', {})
                policies = list(policy_map.keys())
            else:
                policies = []
        else:
            policies = []
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
