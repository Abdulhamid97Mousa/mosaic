"""Worker discovery via setuptools entry points.

This module provides automatic discovery of MOSAIC workers registered via
setuptools entry points, following the standard plugin architecture pattern
used by pytest, Flask, and other major Python projects.

Workers register themselves by adding an entry point in their pyproject.toml:

    [project.entry-points."mosaic.workers"]
    cleanrl = "cleanrl_worker:get_worker_metadata"

The entry point should reference a callable that returns:
    tuple[WorkerMetadata, WorkerCapabilities]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from .protocol import WorkerMetadata, WorkerCapabilities

# Entry point group for MOSAIC workers
WORKER_ENTRY_POINT_GROUP = "mosaic.workers"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredWorker:
    """Represents a discovered MOSAIC worker.

    Attributes:
        worker_id: Unique worker identifier (from entry point name)
        metadata: Worker metadata (name, version, description, etc.)
        capabilities: Worker capabilities (paradigms, environments, resources)
        entry_point_name: Original entry point name
        module_path: Module path to metadata factory function

    Example:
        worker = DiscoveredWorker(
            worker_id="cleanrl",
            metadata=WorkerMetadata(...),
            capabilities=WorkerCapabilities(...),
            entry_point_name="cleanrl",
            module_path="cleanrl_worker:get_worker_metadata"
        )
    """

    worker_id: str
    metadata: WorkerMetadata
    capabilities: WorkerCapabilities
    entry_point_name: str
    module_path: str


@dataclass
class WorkerDiscovery:
    """Discovers and manages MOSAIC workers via entry points.

    This class scans for workers registered in the 'mosaic.workers' entry point
    group and loads their metadata and capabilities.

    Usage:
        discovery = WorkerDiscovery()
        workers = discovery.discover_all()

        for worker in workers:
            print(f"Found: {worker.worker_id} - {worker.metadata.name}")

        # Get specific worker
        cleanrl = discovery.get_worker("cleanrl")
        if cleanrl:
            print(f"CleanRL version: {cleanrl.metadata.version}")

    Attributes:
        _cache: Internal cache of discovered workers
        _loaded: Whether discovery has been performed
    """

    _cache: dict[str, DiscoveredWorker] = field(default_factory=dict, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def discover_all(self, *, force_reload: bool = False) -> list[DiscoveredWorker]:
        """Discover all registered MOSAIC workers.

        Scans the 'mosaic.workers' entry point group and loads metadata for
        each registered worker.

        Args:
            force_reload: If True, clear cache and rediscover workers

        Returns:
            List of discovered workers, sorted by worker_id

        Example:
            discovery = WorkerDiscovery()
            workers = discovery.discover_all()

            for worker in workers:
                print(f"{worker.worker_id}: {worker.metadata.name} v{worker.metadata.version}")
        """
        if force_reload:
            self._cache.clear()
            self._loaded = False

        if self._loaded:
            return sorted(self._cache.values(), key=lambda w: w.worker_id)

        logger.info(f"Discovering workers via entry point group '{WORKER_ENTRY_POINT_GROUP}'")

        # Import entry_points - handle both Python 3.10+ and 3.9
        try:
            from importlib.metadata import entry_points
        except ImportError:
            # Python 3.9 fallback
            from importlib_metadata import entry_points  # type: ignore

        # Get all entry points in our group
        try:
            # Python 3.10+ API
            eps = entry_points(group=WORKER_ENTRY_POINT_GROUP)
        except TypeError:
            # Python 3.9 fallback - entry_points() returns dict-like
            all_eps = entry_points()
            eps = all_eps.get(WORKER_ENTRY_POINT_GROUP, [])

        discovered_count = 0
        failed_count = 0

        for ep in eps:
            worker_id = ep.name
            module_path = f"{ep.value}"

            try:
                # Load the entry point (returns the callable)
                factory = ep.load()

                # Call the factory to get metadata and capabilities
                result = factory()

                # Validate return type
                if not isinstance(result, tuple) or len(result) != 2:
                    logger.error(
                        f"Worker '{worker_id}' entry point must return "
                        f"tuple[WorkerMetadata, WorkerCapabilities], got {type(result)}"
                    )
                    failed_count += 1
                    continue

                metadata, capabilities = result

                # Validate types
                if not isinstance(metadata, WorkerMetadata):
                    logger.error(
                        f"Worker '{worker_id}' returned invalid metadata type: {type(metadata)}"
                    )
                    failed_count += 1
                    continue

                if not isinstance(capabilities, WorkerCapabilities):
                    logger.error(
                        f"Worker '{worker_id}' returned invalid capabilities type: {type(capabilities)}"
                    )
                    failed_count += 1
                    continue

                # Ensure worker_type matches entry point name
                if capabilities.worker_type != worker_id:
                    logger.warning(
                        f"Worker '{worker_id}' has mismatched worker_type in capabilities: "
                        f"'{capabilities.worker_type}' (using entry point name)"
                    )

                # Create discovered worker
                discovered = DiscoveredWorker(
                    worker_id=worker_id,
                    metadata=metadata,
                    capabilities=capabilities,
                    entry_point_name=ep.name,
                    module_path=module_path,
                )

                self._cache[worker_id] = discovered
                discovered_count += 1

                logger.debug(
                    f"Discovered worker: {worker_id} "
                    f"({metadata.name} v{metadata.version})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to load worker '{worker_id}' from entry point: {e}",
                    exc_info=True,
                )
                failed_count += 1

        self._loaded = True

        logger.info(
            f"Worker discovery complete: {discovered_count} workers found, "
            f"{failed_count} failed"
        )

        return sorted(self._cache.values(), key=lambda w: w.worker_id)

    def get_worker(self, worker_id: str) -> Optional[DiscoveredWorker]:
        """Get a specific discovered worker by ID.

        Args:
            worker_id: Worker identifier (e.g., "cleanrl", "ray")

        Returns:
            DiscoveredWorker instance if found, None otherwise

        Example:
            discovery = WorkerDiscovery()
            discovery.discover_all()

            cleanrl = discovery.get_worker("cleanrl")
            if cleanrl:
                print(f"Found CleanRL: {cleanrl.metadata.description}")
        """
        if not self._loaded:
            self.discover_all()

        return self._cache.get(worker_id)

    def list_worker_ids(self) -> list[str]:
        """Get list of all discovered worker IDs.

        Returns:
            Sorted list of worker IDs

        Example:
            discovery = WorkerDiscovery()
            discovery.discover_all()

            worker_ids = discovery.list_worker_ids()
            # Returns: ["balrog", "cleanrl", "ray"]
        """
        if not self._loaded:
            self.discover_all()

        return sorted(self._cache.keys())

    def has_worker(self, worker_id: str) -> bool:
        """Check if a worker is registered.

        Args:
            worker_id: Worker identifier to check

        Returns:
            True if worker is discovered, False otherwise

        Example:
            discovery = WorkerDiscovery()
            if discovery.has_worker("cleanrl"):
                print("CleanRL worker is available")
        """
        if not self._loaded:
            self.discover_all()

        return worker_id in self._cache

    def get_capabilities(self, worker_id: str) -> Optional[WorkerCapabilities]:
        """Get capabilities for a specific worker.

        Args:
            worker_id: Worker identifier

        Returns:
            WorkerCapabilities if worker found, None otherwise

        Example:
            discovery = WorkerDiscovery()
            caps = discovery.get_capabilities("cleanrl")

            if caps and caps.requires_gpu:
                print("CleanRL requires GPU")
        """
        worker = self.get_worker(worker_id)
        return worker.capabilities if worker else None

    def filter_by_paradigm(self, paradigm: str) -> list[DiscoveredWorker]:
        """Find workers supporting a specific paradigm.

        Args:
            paradigm: Paradigm name (e.g., "sequential", "self_play", "parameter_sharing")

        Returns:
            List of workers supporting the paradigm

        Example:
            discovery = WorkerDiscovery()
            discovery.discover_all()

            sequential_workers = discovery.filter_by_paradigm("sequential")
            for worker in sequential_workers:
                print(f"{worker.worker_id} supports sequential paradigm")
        """
        if not self._loaded:
            self.discover_all()

        return [
            worker
            for worker in self._cache.values()
            if worker.capabilities.supports_paradigm(paradigm)
        ]

    def filter_by_env_family(self, family: str) -> list[DiscoveredWorker]:
        """Find workers supporting a specific environment family.

        Args:
            family: Environment family (e.g., "gymnasium", "pettingzoo")

        Returns:
            List of workers supporting the environment family

        Example:
            discovery = WorkerDiscovery()
            pz_workers = discovery.filter_by_env_family("pettingzoo")

            for worker in pz_workers:
                print(f"{worker.worker_id} supports PettingZoo")
        """
        if not self._loaded:
            self.discover_all()

        return [
            worker
            for worker in self._cache.values()
            if worker.capabilities.supports_env_family(family)
        ]

    def clear_cache(self) -> None:
        """Clear the discovery cache.

        Forces rediscovery on next discover_all() call.

        Example:
            discovery = WorkerDiscovery()
            discovery.discover_all()

            # Later, after installing new workers
            discovery.clear_cache()
            discovery.discover_all()  # Rediscover
        """
        self._cache.clear()
        self._loaded = False
        logger.debug("Worker discovery cache cleared")


def discover_workers() -> list[DiscoveredWorker]:
    """Convenience function to discover all workers.

    Creates a WorkerDiscovery instance and returns all discovered workers.

    Returns:
        List of discovered workers

    Example:
        from gym_gui.core.worker import discover_workers

        workers = discover_workers()
        for worker in workers:
            print(f"Found: {worker.metadata.name}")
    """
    discovery = WorkerDiscovery()
    return discovery.discover_all()
