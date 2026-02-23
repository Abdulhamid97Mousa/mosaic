"""Registry for worker-specific presenters.

This module provides a registry pattern for managing multiple worker presenter implementations.
Each worker (CleanRL, Ray, etc.) registers its own presenter, allowing the UI
to remain agnostic about the specific worker orchestration logic.

Workers are discovered automatically via setuptools entry points, and can also
be registered manually for backwards compatibility.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import logging

from gym_gui.core.worker import WorkerDiscovery, DiscoveredWorker

logger = logging.getLogger(__name__)


@runtime_checkable
class WorkerPresenter(Protocol):
    """Protocol for worker-specific UI presenters.

    A presenter encapsulates the logic for:
    1. Composing training requests from form data
    2. Creating worker-specific UI tabs
    3. Extracting metadata for DTO/API contracts

    Example:
        RayWorkerPresenter handles Ray RLlib specific configuration,
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
    Workers are discovered automatically via setuptools entry points, and can
    also be registered manually for backwards compatibility.

    Typically populated at application startup via bootstrap_default_services().
    """

    def __init__(self):
        """Initialize registry with worker discovery."""
        self._presenters: Dict[str, WorkerPresenter] = {}
        self._discovery = WorkerDiscovery()
        self._discovered_workers: Dict[str, DiscoveredWorker] = {}

    def register(self, worker_id: str, presenter: WorkerPresenter) -> None:
        """Register a presenter for a worker.

        Args:
            worker_id: Unique identifier for the worker (e.g., 'cleanrl_worker', 'ray_worker')
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
        """List all registered worker IDs (both manual and discovered).

        Returns:
            list: Sorted list of worker identifiers
        """
        # Combine manually registered and discovered workers
        all_workers = set(self._presenters.keys()) | set(self._discovered_workers.keys())
        return sorted(all_workers)

    def discover_workers(self) -> None:
        """Discover workers via setuptools entry points.

        This should be called during application startup to automatically
        discover and cache all registered MOSAIC workers.

        Example:
            registry = WorkerPresenterRegistry()
            registry.discover_workers()  # Discover via entry points
            registry.register("custom_worker", CustomPresenter())  # Manual registration
        """
        logger.info("Discovering MOSAIC workers via entry points...")

        try:
            workers = self._discovery.discover_all()

            for worker in workers:
                self._discovered_workers[worker.worker_id] = worker
                logger.info(
                    f"Discovered worker: {worker.worker_id} "
                    f"({worker.metadata.name} v{worker.metadata.version})"
                )

            logger.info(f"Worker discovery complete: {len(workers)} workers found")

        except Exception as e:
            logger.error(f"Failed to discover workers: {e}", exc_info=True)

    def get_worker_metadata(self, worker_id: str) -> Optional[tuple]:
        """Get metadata and capabilities for a discovered worker.

        Args:
            worker_id: Worker identifier

        Returns:
            tuple of (WorkerMetadata, WorkerCapabilities) if worker is discovered,
            None otherwise

        Example:
            metadata, capabilities = registry.get_worker_metadata("cleanrl")
            if metadata:
                print(f"CleanRL version: {metadata.version}")
                print(f"Supports GPU: {capabilities.requires_gpu}")
        """
        worker = self._discovered_workers.get(worker_id)
        if worker:
            return (worker.metadata, worker.capabilities)
        return None

    def has_discovered_worker(self, worker_id: str) -> bool:
        """Check if a worker was discovered via entry points.

        Args:
            worker_id: Worker identifier

        Returns:
            True if worker was discovered, False otherwise
        """
        return worker_id in self._discovered_workers

    def list_discovered_workers(self) -> list[str]:
        """Get list of workers discovered via entry points.

        Returns:
            Sorted list of discovered worker IDs
        """
        return sorted(self._discovered_workers.keys())

    def __contains__(self, worker_id: str) -> bool:
        """Check if a worker is registered (manual or discovered)."""
        return worker_id in self._presenters or worker_id in self._discovered_workers
