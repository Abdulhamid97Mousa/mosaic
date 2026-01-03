"""Runtime orchestration for XuanCe training.

This module provides the XuanCeWorkerRuntime class which wraps XuanCe's
get_runner() API to execute training and benchmark runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict

from .config import XuanCeWorkerConfig

# Import standardized telemetry from gym_gui
try:
    from gym_gui.core.worker import TelemetryEmitter as StandardTelemetryEmitter
    from gym_gui.logging_config.helpers import log_constant
    from gym_gui.logging_config.log_constants import (
        LOG_WORKER_XUANCE_RUNTIME_STARTED,
        LOG_WORKER_XUANCE_RUNTIME_STOPPED,
        LOG_WORKER_XUANCE_RUNTIME_ERROR,
        LOG_WORKER_XUANCE_TRAINING_STARTED,
        LOG_WORKER_XUANCE_TRAINING_COMPLETED,
        LOG_WORKER_XUANCE_CONFIG_LOADED,
        LOG_WORKER_XUANCE_AGENT_CREATED,
        LOG_WORKER_XUANCE_DEBUG,
    )
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    StandardTelemetryEmitter = None
    log_constant = None

# Import analytics manifest writer
try:
    from .analytics import write_analytics_manifest
    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False
    write_analytics_manifest = None

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class XuanCeRuntimeSummary:
    """Summary returned from XuanCe training runs.

    Attributes:
        status: Run status ("completed", "dry-run", "error").
        method: Algorithm that was executed.
        env_id: Environment that was used.
        runner_type: XuanCe runner class name (RunnerDRL, RunnerMARL, etc.).
        config: Dictionary representation of the run configuration.
    """

    status: str
    method: str
    env_id: str
    runner_type: str
    config: Dict[str, Any]


class XuanCeWorkerRuntime:
    """Orchestrate XuanCe algorithm execution.

    This class provides a clean interface to XuanCe's training infrastructure,
    supporting both single-agent (RunnerDRL) and multi-agent (RunnerMARL,
    RunnerPettingzoo) runners.

    Example:
        >>> config = XuanCeWorkerConfig(
        ...     run_id="test_run",
        ...     method="ppo",
        ...     env="classic_control",
        ...     env_id="CartPole-v1",
        ...     running_steps=10000,
        ... )
        >>> runtime = XuanCeWorkerRuntime(config)
        >>> summary = runtime.run()
        >>> print(summary.status)
        'completed'
    """

    def __init__(
        self,
        config: XuanCeWorkerConfig,
        *,
        dry_run: bool = False,
    ) -> None:
        """Initialize the runtime.

        Args:
            config: XuanCe worker configuration.
            dry_run: If True, validate configuration without executing.
        """
        self._config = config
        self._dry_run = dry_run

        # Log config loaded
        if _HAS_GYM_GUI and log_constant:
            log_constant(
                LOGGER,
                LOG_WORKER_XUANCE_CONFIG_LOADED,
                extra={
                    "run_id": config.run_id,
                    "method": config.method,
                    "env": config.env,
                    "env_id": config.env_id,
                },
            )

        # Create standardized telemetry emitter for lifecycle events
        if _HAS_GYM_GUI:
            self._lifecycle_emitter = StandardTelemetryEmitter(run_id=config.run_id)
        else:
            self._lifecycle_emitter = None

    @property
    def config(self) -> XuanCeWorkerConfig:
        """Return the current configuration."""
        return self._config

    def _build_parser_args(self) -> SimpleNamespace:
        """Build argument namespace for XuanCe.

        Converts our config to a SimpleNamespace that XuanCe's
        get_runner() function expects for parser_args.

        Returns:
            SimpleNamespace with XuanCe-compatible arguments.
        """
        args = SimpleNamespace()

        # Deep learning backend
        args.dl_toolbox = self._config.dl_toolbox

        # Core parameters
        args.device = self._config.device
        args.parallels = self._config.parallels
        args.running_steps = self._config.running_steps

        # Seed configuration
        if self._config.seed is not None:
            args.seed = self._config.seed
            args.env_seed = self._config.seed

        # Apply extras
        for key, value in self._config.extras.items():
            setattr(args, key, value)

        return args

    def run(self) -> Dict[str, Any]:
        """Execute the configured XuanCe algorithm.

        This method creates a XuanCe runner and calls its run() method,
        which handles both training and testing based on the test_mode flag.

        Returns:
            Dictionary with execution results (compatible with standardized interface).

        Raises:
            RuntimeError: If XuanCe is not installed.
            Exception: Propagated from XuanCe runner on failure.
        """
        # Emit run_started lifecycle event
        if self._lifecycle_emitter:
            self._lifecycle_emitter.run_started(
                {
                    "method": self._config.method,
                    "env": self._config.env,
                    "env_id": self._config.env_id,
                    "dl_toolbox": self._config.dl_toolbox,
                    "running_steps": self._config.running_steps,
                    "device": self._config.device,
                    "parallels": self._config.parallels,
                },
                constant=LOG_WORKER_XUANCE_RUNTIME_STARTED,
            )

        if self._dry_run:
            LOGGER.info(
                "Dry-run mode | method=%s env=%s env_id=%s",
                self._config.method,
                self._config.env,
                self._config.env_id,
            )
            summary = {
                "status": "dry-run",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "runner_type": "unknown",
                "config": self._config.to_dict(),
            }
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(summary)
            return summary

        try:
            # Import XuanCe here to avoid import issues when not installed
            try:
                from xuance import get_runner
            except ImportError as e:
                raise RuntimeError(
                    "XuanCe is not installed. Install with: pip install -e 3rd_party/xuance_worker"
                ) from e

            LOGGER.info(
                "Starting XuanCe training | method=%s env=%s env_id=%s backend=%s steps=%d",
                self._config.method,
                self._config.env,
                self._config.env_id,
                self._config.dl_toolbox,
                self._config.running_steps,
            )

            parser_args = self._build_parser_args()

            runner = get_runner(
                method=self._config.method,
                env=self._config.env,
                env_id=self._config.env_id,
                config_path=self._config.config_path,
                parser_args=parser_args,
                is_test=self._config.test_mode,
            )

            runner_type = type(runner).__name__
            LOGGER.info("Created XuanCe runner: %s", runner_type)

            # Log agent/runner created
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_AGENT_CREATED,
                    extra={
                        "run_id": self._config.run_id,
                        "runner_type": runner_type,
                        "method": self._config.method,
                    },
                )

            # Log training started
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_TRAINING_STARTED,
                    extra={
                        "run_id": self._config.run_id,
                        "method": self._config.method,
                        "env_id": self._config.env_id,
                        "running_steps": self._config.running_steps,
                    },
                )

            # Execute training
            runner.run()

            LOGGER.info(
                "XuanCe training completed | method=%s runner=%s",
                self._config.method,
                runner_type,
            )

            # Log training completed
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_TRAINING_COMPLETED,
                    extra={
                        "run_id": self._config.run_id,
                        "method": self._config.method,
                        "env_id": self._config.env_id,
                        "runner_type": runner_type,
                    },
                )

            # Generate analytics manifest
            manifest_path = None
            if _HAS_ANALYTICS:
                try:
                    manifest_path = write_analytics_manifest(
                        self._config,
                        notes=f"XuanCe {self._config.method} training on {self._config.env_id}",
                    )
                    LOGGER.info("Analytics manifest written to: %s", manifest_path)
                except Exception as e:
                    LOGGER.warning("Failed to write analytics manifest: %s", e)

            summary = {
                "status": "completed",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "runner_type": runner_type,
                "config": self._config.to_dict(),
                "analytics_manifest": str(manifest_path) if manifest_path else None,
            }

            # Emit run_completed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(
                    summary,
                    constant=LOG_WORKER_XUANCE_RUNTIME_STOPPED,
                )

            return summary

        except Exception as e:
            LOGGER.error("XuanCe training failed: %s", e, exc_info=True)

            error_summary = {
                "status": "failed",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "error": str(e),
                "config": self._config.to_dict(),
            }

            # Emit run_failed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_failed(
                    error_summary,
                    constant=LOG_WORKER_XUANCE_RUNTIME_ERROR,
                )

            raise

    def benchmark(self) -> Dict[str, Any]:
        """Execute benchmark mode (training with periodic evaluation).

        This method uses XuanCe's benchmark() function which trains
        for eval_interval steps, evaluates, and saves the best model.

        Returns:
            Dictionary with execution results (compatible with standardized interface).

        Raises:
            RuntimeError: If XuanCe is not installed.
            Exception: Propagated from XuanCe runner on failure.
        """
        # Emit run_started lifecycle event
        if self._lifecycle_emitter:
            self._lifecycle_emitter.run_started(
                {
                    "mode": "benchmark",
                    "method": self._config.method,
                    "env": self._config.env,
                    "env_id": self._config.env_id,
                    "dl_toolbox": self._config.dl_toolbox,
                    "running_steps": self._config.running_steps,
                },
                constant=LOG_WORKER_XUANCE_RUNTIME_STARTED,
            )

        if self._dry_run:
            LOGGER.info(
                "Dry-run benchmark mode | method=%s env=%s env_id=%s",
                self._config.method,
                self._config.env,
                self._config.env_id,
            )
            summary = {
                "status": "dry-run",
                "mode": "benchmark",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "runner_type": "unknown",
                "config": self._config.to_dict(),
            }
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(summary)
            return summary

        try:
            try:
                from xuance import get_runner
            except ImportError as e:
                raise RuntimeError(
                    "XuanCe is not installed. Install with: pip install -e 3rd_party/xuance_worker"
                ) from e

            LOGGER.info(
                "Starting XuanCe benchmark | method=%s env=%s env_id=%s backend=%s steps=%d",
                self._config.method,
                self._config.env,
                self._config.env_id,
                self._config.dl_toolbox,
                self._config.running_steps,
            )

            parser_args = self._build_parser_args()

            runner = get_runner(
                method=self._config.method,
                env=self._config.env,
                env_id=self._config.env_id,
                config_path=self._config.config_path,
                parser_args=parser_args,
                is_test=self._config.test_mode,
            )

            runner_type = type(runner).__name__
            LOGGER.info("Created XuanCe runner for benchmark: %s", runner_type)

            # Log agent/runner created
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_AGENT_CREATED,
                    extra={
                        "run_id": self._config.run_id,
                        "runner_type": runner_type,
                        "method": self._config.method,
                        "mode": "benchmark",
                    },
                )

            # Log training started
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_TRAINING_STARTED,
                    extra={
                        "run_id": self._config.run_id,
                        "method": self._config.method,
                        "env_id": self._config.env_id,
                        "mode": "benchmark",
                    },
                )

            # Execute benchmark
            runner.benchmark()

            LOGGER.info(
                "XuanCe benchmark completed | method=%s runner=%s",
                self._config.method,
                runner_type,
            )

            # Log training completed
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_TRAINING_COMPLETED,
                    extra={
                        "run_id": self._config.run_id,
                        "method": self._config.method,
                        "env_id": self._config.env_id,
                        "runner_type": runner_type,
                        "mode": "benchmark",
                    },
                )

            # Generate analytics manifest
            manifest_path = None
            if _HAS_ANALYTICS:
                try:
                    manifest_path = write_analytics_manifest(
                        self._config,
                        notes=f"XuanCe {self._config.method} benchmark on {self._config.env_id}",
                    )
                    LOGGER.info("Analytics manifest written to: %s", manifest_path)
                except Exception as e:
                    LOGGER.warning("Failed to write analytics manifest: %s", e)

            summary = {
                "status": "completed",
                "mode": "benchmark",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "runner_type": runner_type,
                "config": self._config.to_dict(),
                "analytics_manifest": str(manifest_path) if manifest_path else None,
            }

            # Emit run_completed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(
                    summary,
                    constant=LOG_WORKER_XUANCE_RUNTIME_STOPPED,
                )

            return summary

        except Exception as e:
            LOGGER.error("XuanCe benchmark failed: %s", e, exc_info=True)

            error_summary = {
                "status": "failed",
                "mode": "benchmark",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "error": str(e),
                "config": self._config.to_dict(),
            }

            # Emit run_failed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_failed(
                    error_summary,
                    constant=LOG_WORKER_XUANCE_RUNTIME_ERROR,
                )

            raise


__all__ = ["XuanCeWorkerRuntime", "XuanCeRuntimeSummary"]
