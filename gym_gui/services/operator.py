# gym_gui/services/operator.py

from __future__ import annotations

"""Operator abstractions and registry for action selection.

This module provides:
- Operator: Protocol for single-agent action selection
- OperatorController: Paradigm-aware protocol for multi-agent/multi-paradigm support
- OperatorService: Registry for managing active operators

The Operator abstraction replaces the legacy Actor concept with clearer semantics:
an Operator operates on observations to produce actions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol

from gym_gui.core.enums import SteppingParadigm
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import LOG_SERVICE_ACTOR_SEED_ERROR


class Operator(Protocol):
    """Protocol for action selection operators.

    An Operator receives observations and returns actions.
    This is the fundamental abstraction for any decision-making entity:
    - Human keyboard input
    - Trained RL policy
    - LLM agent
    - BDI reasoning system
    """

    @property
    def id(self) -> str:
        """Unique identifier for this operator."""
        ...

    @property
    def name(self) -> str:
        """Human-readable display name for UI."""
        ...

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Select an action given the current observation.

        Args:
            observation: The current environment observation.
            info: Optional environment info dict.

        Returns:
            The action to take, or None to abstain.
        """
        ...

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state for a new episode.

        Args:
            seed: Optional deterministic seed.
        """
        ...

    def on_step_result(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Receive feedback after action execution (optional learning hook).

        Args:
            observation: New observation after the step.
            reward: Reward received.
            terminated: Whether episode ended naturally.
            truncated: Whether episode was truncated.
            info: Environment info dict.
        """
        ...


class OperatorController(Protocol):
    """Paradigm-aware protocol for multi-agent and multi-paradigm operator control.

    This protocol extends the Operator concept with:
    1. Agent-specific action selection (for multi-agent environments)
    2. Batch action selection (for SIMULTANEOUS/POSG paradigms)
    3. Explicit paradigm declaration

    OperatorController is designed to work with the WorkerOrchestrator and
    PolicyMappingService for paradigm-agnostic training coordination.

    Example (Sequential/AEC):
        >>> controller.select_action("player_0", observation, info)

    Example (Simultaneous/POSG):
        >>> controller.select_actions({"player_0": obs0, "player_1": obs1})
    """

    @property
    def id(self) -> str:
        """Unique identifier for this operator controller."""
        ...

    @property
    def name(self) -> str:
        """Human-readable display name for UI."""
        ...

    @property
    def paradigm(self) -> SteppingParadigm:
        """The stepping paradigm this controller is designed for."""
        ...

    def select_action(
        self,
        agent_id: str,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Select action for a specific agent (Sequential/AEC mode).

        Args:
            agent_id: The identifier of the agent needing an action.
            observation: The agent's current observation.
            info: Optional environment info dict.

        Returns:
            The action to take, or None to abstain.
        """
        ...

    def select_actions(
        self,
        observations: Dict[str, Any],
        infos: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Select actions for all agents at once (Simultaneous/POSG mode).

        Args:
            observations: Dict mapping agent_id to observation.
            infos: Optional dict mapping agent_id to info dict.

        Returns:
            Dict mapping agent_id to action.
        """
        ...

    def on_step_result(
        self,
        agent_id: str,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Receive feedback after a step (for learning updates).

        Args:
            agent_id: The agent that took the action.
            observation: New observation after the step.
            reward: Reward received.
            terminated: Whether episode ended naturally.
            truncated: Whether episode was truncated.
            info: Environment info dict.
        """
        ...

    def on_episode_end(
        self,
        agent_id: str,
        episode_return: float,
        episode_length: int,
    ) -> None:
        """Called when an episode ends for a specific agent.

        Args:
            agent_id: The agent whose episode ended.
            episode_return: Total reward for the episode.
            episode_length: Number of steps in the episode.
        """
        ...

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state for a new episode.

        Args:
            seed: Optional deterministic seed.
        """
        ...


@dataclass(frozen=True)
class OperatorDescriptor:
    """Metadata describing a registered operator for UI presentation."""

    operator_id: str
    display_name: str
    description: str | None = None
    category: str = "default"  # "human", "llm", "rl", "worker"
    supports_training: bool = False
    requires_api_key: bool = False


@dataclass
class WorkerAssignment:
    """Configuration for a single worker assigned to a player in an operator.

    This dataclass holds worker-specific settings for one player/agent slot
    within an operator. For single-agent environments, there's one WorkerAssignment.
    For multi-agent environments (e.g., chess), there's one per player.

    Attributes:
        worker_id: References WorkerDefinition (e.g., "barlog_worker", "cleanrl_worker").
        worker_type: Type of worker - "llm", "vlm", "rl", or "human".
        settings: Worker-specific settings (client_name, model_id, api_key, etc.).
    """

    worker_id: str  # References WorkerDefinition in worker catalog
    worker_type: str  # "llm", "vlm", "rl", "human"
    settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate worker assignment."""
        valid_types = ("llm", "vlm", "rl", "human")
        if self.worker_type not in valid_types:
            raise ValueError(f"worker_type must be one of {valid_types}, got '{self.worker_type}'")


@dataclass
class OperatorConfig:
    """Configuration for a single operator instance in multi-operator mode.

    This dataclass holds all the information needed to configure and run
    an operator (LLM or RL worker) in the multi-operator comparison view.

    An Operator binds one or more workers to a single environment:
    - Single-agent envs (babyai, minigrid): 1 worker → 1 environment
    - Multi-agent envs (chess, connect4): N workers → 1 environment

    Attributes:
        operator_id: Unique ID for this operator instance (e.g., "operator_0").
        display_name: User-visible name (e.g., "GPT-4 LLM", "Chess Match").
        env_name: Environment family (e.g., "babyai", "minigrid", "pettingzoo").
        task: Task/level within the environment (e.g., "BabyAI-GoToRedBall-v0", "chess_v6").
        workers: Dict mapping player_id to WorkerAssignment.
                 Single-agent: {"agent": WorkerAssignment(...)}
                 Multi-agent: {"player_0": WorkerAssignment(...), "player_1": WorkerAssignment(...)}
        run_id: Assigned run ID when operator is started (for telemetry routing).

    Factory Methods:
        Use OperatorConfig.single_agent() for single-agent environments.
        Use OperatorConfig.multi_agent() for multi-agent environments (chess, Go, etc.).

    Backwards Compatibility:
        Properties operator_type, worker_id, settings read from workers["agent"]
        for compatibility with existing code that expects single-worker operators.
    """

    operator_id: str
    display_name: str
    env_name: str = "babyai"
    task: str = "BabyAI-GoToRedBall-v0"
    workers: Dict[str, WorkerAssignment] = field(default_factory=dict)
    run_id: str | None = None  # Assigned when operator starts

    def __post_init__(self) -> None:
        """Validate configuration and ensure workers dict is populated."""
        # Ensure workers dict is not empty
        if not self.workers:
            # Default single-agent worker
            self.workers = {
                "agent": WorkerAssignment(
                    worker_id="barlog_worker",
                    worker_type="llm",
                    settings={},
                )
            }

    # -------------------------------------------------------------------------
    # Backwards Compatibility Properties
    # -------------------------------------------------------------------------

    @property
    def operator_type(self) -> str:
        """Get operator type (for backwards compatibility).

        Returns the worker_type of the first worker (single-agent mode)
        or 'multiagent' if multiple workers are assigned.
        """
        if len(self.workers) > 1:
            return "multiagent"
        if self.workers:
            return next(iter(self.workers.values())).worker_type
        return "llm"

    @property
    def worker_id(self) -> str:
        """Get worker ID (for backwards compatibility).

        Returns the worker_id of the first worker.
        """
        if self.workers:
            return next(iter(self.workers.values())).worker_id
        return "barlog_worker"

    @property
    def settings(self) -> Dict[str, Any]:
        """Get worker settings (for backwards compatibility).

        Returns the settings of the first worker.
        """
        if self.workers:
            return next(iter(self.workers.values())).settings
        return {}

    # -------------------------------------------------------------------------
    # Multi-Agent Properties
    # -------------------------------------------------------------------------

    @property
    def is_multiagent(self) -> bool:
        """Check if this operator has multiple workers (multi-agent environment)."""
        return len(self.workers) > 1

    @property
    def player_ids(self) -> list[str]:
        """Get list of player IDs this operator manages."""
        return list(self.workers.keys())

    def get_worker_for_player(self, player_id: str) -> WorkerAssignment | None:
        """Get the worker assignment for a specific player.

        Args:
            player_id: The player ID (e.g., "player_0", "agent").

        Returns:
            WorkerAssignment for that player, or None if not found.
        """
        return self.workers.get(player_id)

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def single_agent(
        cls,
        operator_id: str,
        display_name: str,
        worker_id: str,
        worker_type: str,
        env_name: str = "babyai",
        task: str = "BabyAI-GoToRedBall-v0",
        settings: Dict[str, Any] | None = None,
    ) -> "OperatorConfig":
        """Create a single-agent operator config.

        Args:
            operator_id: Unique operator ID.
            display_name: Display name for UI.
            worker_id: Worker to use (e.g., "barlog_worker").
            worker_type: Type of worker ("llm", "vlm", "rl", "human").
            env_name: Environment family.
            task: Specific task/level.
            settings: Worker-specific settings.

        Returns:
            OperatorConfig with single worker assigned to "agent".
        """
        return cls(
            operator_id=operator_id,
            display_name=display_name,
            env_name=env_name,
            task=task,
            workers={
                "agent": WorkerAssignment(
                    worker_id=worker_id,
                    worker_type=worker_type,
                    settings=settings or {},
                )
            },
        )

    @classmethod
    def multi_agent(
        cls,
        operator_id: str,
        display_name: str,
        env_name: str,
        task: str,
        player_workers: Dict[str, WorkerAssignment],
    ) -> "OperatorConfig":
        """Create a multi-agent operator config.

        Args:
            operator_id: Unique operator ID.
            display_name: Display name for UI.
            env_name: Environment family (e.g., "pettingzoo").
            task: Specific task (e.g., "chess_v6").
            player_workers: Dict mapping player_id to WorkerAssignment.

        Returns:
            OperatorConfig with multiple workers for multi-agent env.
        """
        return cls(
            operator_id=operator_id,
            display_name=display_name,
            env_name=env_name,
            task=task,
            workers=player_workers,
        )

    def with_run_id(self, run_id: str) -> "OperatorConfig":
        """Return a copy of this config with the run_id set."""
        # Deep copy workers
        workers_copy = {
            player_id: WorkerAssignment(
                worker_id=wa.worker_id,
                worker_type=wa.worker_type,
                settings=wa.settings.copy(),
            )
            for player_id, wa in self.workers.items()
        }
        return OperatorConfig(
            operator_id=self.operator_id,
            display_name=self.display_name,
            env_name=self.env_name,
            task=self.task,
            workers=workers_copy,
            run_id=run_id,
        )


class OperatorService(LogConstantMixin):
    """Registry that manages operators for action selection.

    This service manages:
    - Registration of operators with metadata
    - Active operator selection
    - Action delegation to the currently active operator
    - Seeding propagation to all registered operators
    """

    def __init__(self) -> None:
        self._operators: Dict[str, Operator] = {}
        self._descriptors: Dict[str, OperatorDescriptor] = {}
        self._active_operator_id: Optional[str] = None
        self._last_seed: int | None = None
        self._logger = logging.getLogger("gym_gui.services.operator")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_operator(
        self,
        operator: Operator,
        *,
        display_name: str | None = None,
        description: str | None = None,
        category: str = "default",
        supports_training: bool = False,
        requires_api_key: bool = False,
        activate: bool = False,
    ) -> None:
        """Register an operator with the service.

        Args:
            operator: The operator instance to register.
            display_name: Human-readable name for UI (defaults to operator.name).
            description: Optional description for UI.
            category: Category for grouping ("human", "llm", "rl", "worker").
            supports_training: Whether this operator supports training mode.
            requires_api_key: Whether this operator needs an API key.
            activate: Whether to make this the active operator.
        """
        operator_id = operator.id
        label = display_name or getattr(operator, "name", operator_id.replace("_", " ").title())
        self._operators[operator_id] = operator
        self._descriptors[operator_id] = OperatorDescriptor(
            operator_id=operator_id,
            display_name=label,
            description=description,
            category=category,
            supports_training=supports_training,
            requires_api_key=requires_api_key,
        )
        if activate or self._active_operator_id is None:
            self._active_operator_id = operator_id

    def available_operator_ids(self) -> Iterable[str]:
        """Return all registered operator IDs."""
        return self._operators.keys()

    def describe_operators(self) -> tuple[OperatorDescriptor, ...]:
        """Return metadata for all registered operators in registration order."""
        return tuple(self._descriptors[operator_id] for operator_id in self._operators.keys())

    def get_operator(self, operator_id: str) -> Optional[Operator]:
        """Get a specific operator by ID."""
        return self._operators.get(operator_id)

    def get_operator_descriptor(self, operator_id: str) -> Optional[OperatorDescriptor]:
        """Get metadata for a specific operator."""
        return self._descriptors.get(operator_id)

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------
    def set_active_operator(self, operator_id: str) -> None:
        """Set the active operator by ID.

        Args:
            operator_id: ID of the operator to activate.

        Raises:
            KeyError: If the operator ID is not registered.
        """
        if operator_id not in self._operators:
            raise KeyError(f"Unknown operator '{operator_id}'")
        self._active_operator_id = operator_id

    def get_active_operator(self) -> Optional[Operator]:
        """Get the currently active operator, or None if none is active."""
        if self._active_operator_id is None:
            return None
        return self._operators.get(self._active_operator_id)

    def get_active_operator_id(self) -> Optional[str]:
        """Get the ID of the currently active operator."""
        return self._active_operator_id

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------
    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Delegate action selection to the active operator.

        Args:
            observation: The current environment observation.
            info: Optional environment info dict.

        Returns:
            The action selected by the active operator, or None if no operator.
        """
        operator = self.get_active_operator()
        if operator is None:
            return None
        return operator.select_action(observation, info)

    def notify_step_result(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Notify the active operator of a step result.

        Args:
            observation: New observation after the step.
            reward: Reward received.
            terminated: Whether episode ended naturally.
            truncated: Whether episode was truncated.
            info: Environment info dict.
        """
        operator = self.get_active_operator()
        if operator is None:
            return
        callback = getattr(operator, "on_step_result", None)
        if callable(callback):
            callback(observation, reward, terminated, truncated, info)

    def reset_active_operator(self, seed: Optional[int] = None) -> None:
        """Reset the active operator for a new episode.

        Args:
            seed: Optional deterministic seed.
        """
        operator = self.get_active_operator()
        if operator is None:
            return
        callback = getattr(operator, "reset", None)
        if callable(callback):
            callback(seed)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def seed(self, seed: int) -> None:
        """Propagate a deterministic seed to all registered operators."""
        self._last_seed = seed
        for operator_id, operator in self._operators.items():
            callback = getattr(operator, "reset", None)
            if not callable(callback):
                continue
            try:
                callback(seed)
            except Exception as exc:  # pragma: no cover - defensive guard
                self.log_constant(
                    LOG_SERVICE_ACTOR_SEED_ERROR,  # Reuse existing log constant
                    message="operator_seed_failed",
                    extra={"operator_id": operator_id},
                    exc_info=exc,
                )

    @property
    def last_seed(self) -> Optional[int]:
        """Return the last seed that was propagated."""
        return self._last_seed


# -----------------------------------------------------------------------------
# Built-in Operator Implementations
# -----------------------------------------------------------------------------


@dataclass
class HumanOperator:
    """Operator for human keyboard input.

    The actual action is provided by the UI via HumanInputController.
    This operator returns None to signal that the action comes from elsewhere.
    """

    id: str = "human_keyboard"
    name: str = "Human (Keyboard)"

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Human action is injected via UI, not selected here."""
        return None

    def reset(self, seed: Optional[int] = None) -> None:
        """No state to reset for human input."""
        pass

    def on_step_result(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """No feedback processing for human input."""
        pass


@dataclass
class WorkerOperator:
    """Placeholder operator for worker subprocess backends.

    Workers manage their own action selection and training.
    This operator signals that a worker is handling decisions.
    """

    id: str
    name: str
    worker_id: str  # References WorkerDefinition in worker catalog

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Worker manages its own action selection."""
        return None

    def reset(self, seed: Optional[int] = None) -> None:
        """Worker handles its own reset."""
        pass

    def on_step_result(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Worker processes its own feedback."""
        pass


# -----------------------------------------------------------------------------
# Multi-Operator Service for Parallel Operator Management
# -----------------------------------------------------------------------------


class MultiOperatorService(LogConstantMixin):
    """Extended service for managing multiple active operators in parallel.

    This service enables side-by-side comparison of LLM vs RL agents,
    or multiple algorithms running on the same or different environments.

    Features:
    - Manage N active operator configurations
    - Each operator gets a unique run_id for telemetry routing
    - Start/stop individual or all operators
    - Track operator state (pending, running, stopped)
    """

    def __init__(self) -> None:
        self._active_operators: Dict[str, OperatorConfig] = {}  # operator_id -> config
        self._operator_runs: Dict[str, str] = {}  # operator_id -> run_id
        self._operator_states: Dict[str, str] = {}  # operator_id -> "pending"|"running"|"stopped"
        self._next_operator_index: int = 0
        self._freed_indices: set[int] = set()  # Track freed indices for reuse
        self._logger = logging.getLogger("gym_gui.services.multi_operator")

    # ------------------------------------------------------------------
    # Operator Configuration Management
    # ------------------------------------------------------------------
    def add_operator(self, config: OperatorConfig) -> None:
        """Add a new active operator configuration.

        Args:
            config: The operator configuration to add.
        """
        self._active_operators[config.operator_id] = config
        self._operator_states[config.operator_id] = "pending"
        self._logger.info(f"Added operator: {config.operator_id} ({config.display_name})")

    def remove_operator(self, operator_id: str) -> None:
        """Remove an active operator.

        Args:
            operator_id: ID of the operator to remove.
        """
        if operator_id in self._active_operators:
            del self._active_operators[operator_id]
        if operator_id in self._operator_runs:
            del self._operator_runs[operator_id]
        if operator_id in self._operator_states:
            del self._operator_states[operator_id]

        # Track freed index for reuse (extract number from "operator_N")
        if operator_id.startswith("operator_"):
            try:
                freed_idx = int(operator_id.split("_")[1])
                self._freed_indices.add(freed_idx)
                self._logger.info(f"Freed operator index {freed_idx} for reuse")
            except (IndexError, ValueError):
                pass

        self._logger.info(f"Removed operator: {operator_id}")

        # Reset everything when all operators are removed
        if len(self._active_operators) == 0:
            self._next_operator_index = 0
            self._freed_indices.clear()
            self._logger.info("All operators removed, reset operator index")

    def update_operator(self, config: OperatorConfig) -> None:
        """Update an existing operator configuration.

        Args:
            config: The updated operator configuration.
        """
        if config.operator_id in self._active_operators:
            self._active_operators[config.operator_id] = config
            self._logger.info(f"Updated operator: {config.operator_id}")

    def get_operator(self, operator_id: str) -> Optional[OperatorConfig]:
        """Get a specific operator configuration by ID."""
        return self._active_operators.get(operator_id)

    def get_active_operators(self) -> Dict[str, OperatorConfig]:
        """Get all active operator configurations."""
        return dict(self._active_operators)

    def get_operator_ids(self) -> list[str]:
        """Get list of all operator IDs in order."""
        return list(self._active_operators.keys())

    def clear_operators(self) -> None:
        """Remove all operators and reset the operator index counter."""
        self._active_operators.clear()
        self._operator_runs.clear()
        self._operator_states.clear()
        self._next_operator_index = 0  # Reset counter so next operator starts at 0
        self._freed_indices.clear()  # Clear freed indices
        self._logger.info("Cleared all operators")

    # ------------------------------------------------------------------
    # Operator ID Generation
    # ------------------------------------------------------------------
    def generate_operator_id(self) -> str:
        """Generate a unique operator ID, reusing freed indices when available."""
        if self._freed_indices:
            # Reuse the smallest freed index
            reused_idx = min(self._freed_indices)
            self._freed_indices.remove(reused_idx)
            operator_id = f"operator_{reused_idx}"
            self._logger.info(f"Reusing freed operator index {reused_idx}")
        else:
            # Use the next new index
            operator_id = f"operator_{self._next_operator_index}"
            self._next_operator_index += 1
        return operator_id

    # ------------------------------------------------------------------
    # Run ID Management
    # ------------------------------------------------------------------
    def assign_run_id(self, operator_id: str, run_id: str) -> None:
        """Assign a run ID to an operator for telemetry routing.

        Args:
            operator_id: The operator to assign the run ID to.
            run_id: The run ID to assign.
        """
        self._operator_runs[operator_id] = run_id
        # Update the config with the run_id
        if operator_id in self._active_operators:
            config = self._active_operators[operator_id]
            self._active_operators[operator_id] = config.with_run_id(run_id)

    def get_run_id(self, operator_id: str) -> Optional[str]:
        """Get the run ID for an operator."""
        return self._operator_runs.get(operator_id)

    def get_operator_by_run_id(self, run_id: str) -> Optional[OperatorConfig]:
        """Get the operator config associated with a run ID."""
        for operator_id, config_run_id in self._operator_runs.items():
            if config_run_id == run_id:
                return self._active_operators.get(operator_id)
        return None

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------
    def set_operator_state(self, operator_id: str, state: str) -> None:
        """Set the state of an operator.

        Args:
            operator_id: The operator ID.
            state: One of "pending", "running", "stopped", "error".
        """
        if state not in ("pending", "running", "stopped", "error"):
            raise ValueError(f"Invalid state: {state}")
        self._operator_states[operator_id] = state

    def get_operator_state(self, operator_id: str) -> Optional[str]:
        """Get the current state of an operator."""
        return self._operator_states.get(operator_id)

    def get_running_operators(self) -> list[str]:
        """Get list of operator IDs that are currently running."""
        return [
            op_id for op_id, state in self._operator_states.items()
            if state == "running"
        ]

    # ------------------------------------------------------------------
    # Lifecycle Helpers
    # ------------------------------------------------------------------
    def start_all(self) -> list[str]:
        """Mark all pending operators as ready to start.

        Returns:
            List of operator IDs that are ready to start.
        """
        to_start = []
        for operator_id, state in self._operator_states.items():
            if state == "pending":
                to_start.append(operator_id)
        return to_start

    def stop_all(self) -> list[str]:
        """Mark all running operators as stopped.

        Returns:
            List of operator IDs that were stopped.
        """
        stopped = []
        for operator_id, state in list(self._operator_states.items()):
            if state == "running":
                self._operator_states[operator_id] = "stopped"
                stopped.append(operator_id)
        return stopped

    @property
    def operator_count(self) -> int:
        """Get the number of active operators."""
        return len(self._active_operators)


__all__ = [
    "Operator",
    "OperatorController",
    "OperatorService",
    "OperatorDescriptor",
    "OperatorConfig",
    "WorkerAssignment",
    "MultiOperatorService",
    "HumanOperator",
    "WorkerOperator",
]
