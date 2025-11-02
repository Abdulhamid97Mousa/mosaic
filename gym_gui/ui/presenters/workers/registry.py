"""Registry for worker-specific presenters.

This module provides a registry pattern for managing multiple worker presenter implementations.
Each worker (SPADE-BDI, HuggingFace, etc.) registers its own presenter, allowing the UI
to remain agnostic about the specific worker orchestration logic.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class WorkerPresenter(Protocol):
    """Protocol for worker-specific UI presenters.

    A presenter encapsulates the logic for:
    1. Composing training requests from form data
    2. Creating worker-specific UI tabs
    3. Extracting metadata for DTO/API contracts

    Example:
        SpadeBdiWorkerPresenter handles SPADE-BDI specific configuration,
        tab creation, and metadata composition.
    """

    @property
    def id(self) -> str:
        """Unique identifier for this worker presenter."""
        ...

    def build_train_request(self, policy_path: Any, current_game: Optional[Any]) -> dict:
        """Build a training request from form data.

        Args:
            policy_path: Path to the policy file
            current_game: Currently selected game (GameId enum or None)

        Returns:
            dict: Configuration dictionary suitable for TrainerClient submission
        """
        ...

    def create_tabs(self, run_id: str, agent_id: str, first_payload: dict, parent: Any) -> List[Any]:
        """Create worker-specific UI tabs for a running agent.

        Args:
            run_id: Unique run identifier
            agent_id: Agent identifier within the run
            first_payload: First telemetry step payload containing metadata
            parent: Parent Qt widget

        Returns:
            list: List of QWidget tab instances
        """
        ...


class WorkerPresenterRegistry:
    """Registry for available worker presenters.

    Manages registration and lookup of worker presenter implementations.
    Typically populated at application startup via bootstrap_default_services().
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._presenters: Dict[str, WorkerPresenter] = {}

    def register(self, worker_id: str, presenter: WorkerPresenter) -> None:
        """Register a presenter for a worker.

        Args:
            worker_id: Unique identifier for the worker (e.g., 'spade_bdi_worker')
            presenter: WorkerPresenter instance

        Raises:
            ValueError: If worker_id is already registered
        """
        if worker_id in self._presenters:
            raise ValueError(f"Worker '{worker_id}' already registered")
        self._presenters[worker_id] = presenter

    def get(self, worker_id: str) -> Optional[WorkerPresenter]:
        """Retrieve a presenter by worker ID.

        Args:
            worker_id: Unique identifier for the worker

        Returns:
            WorkerPresenter instance or None if not found
        """
        return self._presenters.get(worker_id)

    def available_workers(self) -> list[str]:
        """List all registered worker IDs.

        Returns:
            list: Sorted list of worker identifiers
        """
        return sorted(self._presenters.keys())

    def __contains__(self, worker_id: str) -> bool:
        """Check if a worker is registered."""
        return worker_id in self._presenters
