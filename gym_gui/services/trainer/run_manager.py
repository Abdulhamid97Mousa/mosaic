from __future__ import annotations

"""High-level management service for training runs.

This module provides a unified API for managing training runs, including
stopping running jobs, deleting run artifacts, and querying run state.
It coordinates between the registry (database), telemetry service, and
filesystem to ensure complete cleanup when runs are deleted.
"""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.services.trainer.registry import RunRecord, RunRegistry, RunStatus

if TYPE_CHECKING:
    from gym_gui.services.trainer.client_runner import TrainerClientRunner
    from gym_gui.services.telemetry import TelemetryService


_LOGGER = logging.getLogger(__name__)


class TrainingRunManager:
    """High-level management service for training runs.

    This service provides a clean API for managing training runs, including:
    - Stopping/canceling running jobs
    - Deleting runs and all associated artifacts
    - Querying run state and disk usage

    It coordinates cleanup across multiple systems:
    - trainer.sqlite (run registry)
    - telemetry.sqlite (episode/step data)
    - var/trainer/runs/{run_id}/ (logs, tensorboard, checkpoints)
    - var/trainer/configs/ (configuration files)
    """

    def __init__(
        self,
        registry: RunRegistry,
        client_runner: Optional[TrainerClientRunner] = None,
        telemetry_service: Optional[TelemetryService] = None,
    ) -> None:
        """Initialize the training run manager.

        Args:
            registry: The run registry for database operations.
            client_runner: Optional trainer client for stopping running jobs.
            telemetry_service: Optional telemetry service for deleting telemetry data.
        """
        self._registry = registry
        self._client_runner = client_runner
        self._telemetry = telemetry_service

    def cancel_run(self, run_id: str, *, deadline: float = 5.0) -> bool:
        """Cancel a running training job.

        Sends a cancel request to the trainer daemon which will gracefully
        terminate the worker process.

        Args:
            run_id: The unique identifier of the run to cancel.
            deadline: Timeout in seconds for the gRPC call.

        Returns:
            True if the cancel request was sent successfully, False otherwise.
        """
        if self._client_runner is None:
            _LOGGER.warning(
                "Cannot cancel run: no client runner available",
                extra={"run_id": run_id},
            )
            return False

        try:
            self._client_runner.cancel_run(run_id, deadline=deadline)
            _LOGGER.info("Sent cancel request for run", extra={"run_id": run_id})
            return True
        except Exception as exc:
            _LOGGER.error(
                "Failed to cancel run",
                extra={"run_id": run_id, "error": str(exc)},
                exc_info=exc,
            )
            return False

    def delete_run_completely(self, run_id: str) -> bool:
        """Delete a run and all associated artifacts.

        This method performs a complete cleanup:
        1. Deletes the run from the trainer registry (trainer.sqlite)
        2. Deletes telemetry data (episodes/steps from telemetry.sqlite)
        3. Deletes the run directory (var/trainer/runs/{run_id}/)
        4. Deletes configuration files (var/trainer/configs/*)

        Args:
            run_id: The unique identifier of the run to delete.

        Returns:
            True if the run was deleted, False if it was not found.
        """
        _LOGGER.info("Starting complete deletion of run", extra={"run_id": run_id})

        # 1. Delete from trainer registry
        deleted_from_registry = self._registry.delete_run(run_id)

        # 2. Delete telemetry data
        if self._telemetry is not None:
            try:
                self._telemetry.delete_run(run_id, wait=True)
                _LOGGER.debug(
                    "Deleted telemetry data for run", extra={"run_id": run_id}
                )
            except Exception as exc:
                _LOGGER.warning(
                    "Failed to delete telemetry data for run",
                    extra={"run_id": run_id, "error": str(exc)},
                )

        # 3. Delete run directory (logs, tensorboard, checkpoints)
        run_dir = VAR_TRAINER_DIR / "runs" / run_id
        if run_dir.exists():
            try:
                shutil.rmtree(run_dir)
                _LOGGER.debug("Deleted run directory", extra={"path": str(run_dir)})
            except Exception as exc:
                _LOGGER.warning(
                    "Failed to delete run directory",
                    extra={"path": str(run_dir), "error": str(exc)},
                )

        # 4. Delete configuration files
        configs_dir = VAR_TRAINER_DIR / "configs"
        if configs_dir.exists():
            # Main config file
            config_file = configs_dir / f"config-{run_id}.json"
            if config_file.exists():
                try:
                    config_file.unlink()
                    _LOGGER.debug(
                        "Deleted config file", extra={"path": str(config_file)}
                    )
                except Exception as exc:
                    _LOGGER.warning(
                        "Failed to delete config file",
                        extra={"path": str(config_file), "error": str(exc)},
                    )

            # Worker config files (worker-{run_id}-*.json)
            for worker_config in configs_dir.glob(f"worker-{run_id}-*.json"):
                try:
                    worker_config.unlink()
                    _LOGGER.debug(
                        "Deleted worker config", extra={"path": str(worker_config)}
                    )
                except Exception as exc:
                    _LOGGER.warning(
                        "Failed to delete worker config",
                        extra={"path": str(worker_config), "error": str(exc)},
                    )

        _LOGGER.info(
            "Completed deletion of run",
            extra={"run_id": run_id, "found_in_registry": deleted_from_registry},
        )
        return deleted_from_registry

    def list_runs(
        self, statuses: Optional[list[RunStatus]] = None
    ) -> list[RunRecord]:
        """List all runs from the registry.

        Args:
            statuses: Optional filter by run statuses. If None, returns all runs.

        Returns:
            List of RunRecord objects.
        """
        return self._registry.load_runs(statuses=statuses)

    def get_run_disk_size(self, run_id: str) -> int:
        """Calculate the disk usage for a run in bytes.

        This includes all files in var/trainer/runs/{run_id}/ such as:
        - Logs (stdout, stderr)
        - TensorBoard event files
        - Checkpoints
        - Analytics manifests

        Args:
            run_id: The unique identifier of the run.

        Returns:
            Total size in bytes, or 0 if the directory doesn't exist.
        """
        run_dir = VAR_TRAINER_DIR / "runs" / run_id
        if not run_dir.exists():
            return 0

        try:
            total_size = sum(
                f.stat().st_size for f in run_dir.rglob("*") if f.is_file()
            )
            return total_size
        except Exception as exc:
            _LOGGER.warning(
                "Failed to calculate disk size for run",
                extra={"run_id": run_id, "error": str(exc)},
            )
            return 0

    def get_run_config_json(self, run_id: str) -> Optional[str]:
        """Get the raw config JSON for a run.

        Args:
            run_id: The unique identifier of the run.

        Returns:
            The config JSON string, or None if not found.
        """
        return self._registry.get_run_config_json(run_id)

    def get_terminated_runs(self) -> list[RunRecord]:
        """Get all runs with TERMINATED status.

        Returns:
            List of terminated RunRecord objects.
        """
        return self._registry.load_runs(statuses=[RunStatus.TERMINATED])

    def get_executing_runs(self) -> list[RunRecord]:
        """Get all runs with EXECUTING status.

        Returns:
            List of executing RunRecord objects.
        """
        return self._registry.load_runs(statuses=[RunStatus.EXECUTING])


__all__ = ["TrainingRunManager"]
