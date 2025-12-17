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

    def run(self) -> XuanCeRuntimeSummary:
        """Execute the configured XuanCe algorithm.

        This method creates a XuanCe runner and calls its run() method,
        which handles both training and testing based on the test_mode flag.

        Returns:
            XuanCeRuntimeSummary with execution results.

        Raises:
            RuntimeError: If XuanCe is not installed.
            Exception: Propagated from XuanCe runner on failure.
        """
        if self._dry_run:
            LOGGER.info(
                "Dry-run mode | method=%s env=%s env_id=%s",
                self._config.method,
                self._config.env,
                self._config.env_id,
            )
            return XuanCeRuntimeSummary(
                status="dry-run",
                method=self._config.method,
                env_id=self._config.env_id,
                runner_type="unknown",
                config=self._config.to_dict(),
            )

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

        # Execute training
        runner.run()

        LOGGER.info(
            "XuanCe training completed | method=%s runner=%s",
            self._config.method,
            runner_type,
        )

        return XuanCeRuntimeSummary(
            status="completed",
            method=self._config.method,
            env_id=self._config.env_id,
            runner_type=runner_type,
            config=self._config.to_dict(),
        )

    def benchmark(self) -> XuanCeRuntimeSummary:
        """Execute benchmark mode (training with periodic evaluation).

        This method uses XuanCe's benchmark() function which trains
        for eval_interval steps, evaluates, and saves the best model.

        Returns:
            XuanCeRuntimeSummary with execution results.

        Raises:
            RuntimeError: If XuanCe is not installed.
            Exception: Propagated from XuanCe runner on failure.
        """
        if self._dry_run:
            LOGGER.info(
                "Dry-run benchmark mode | method=%s env=%s env_id=%s",
                self._config.method,
                self._config.env,
                self._config.env_id,
            )
            return XuanCeRuntimeSummary(
                status="dry-run",
                method=self._config.method,
                env_id=self._config.env_id,
                runner_type="unknown",
                config=self._config.to_dict(),
            )

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

        # Execute benchmark
        runner.benchmark()

        LOGGER.info(
            "XuanCe benchmark completed | method=%s runner=%s",
            self._config.method,
            runner_type,
        )

        return XuanCeRuntimeSummary(
            status="completed",
            method=self._config.method,
            env_id=self._config.env_id,
            runner_type=runner_type,
            config=self._config.to_dict(),
        )


__all__ = ["XuanCeWorkerRuntime", "XuanCeRuntimeSummary"]
