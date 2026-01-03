"""Base CLI utilities for MOSAIC workers.

Provides standard argument parsing and utilities that all workers can inherit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Type, TypeVar

from gym_gui.logging_config import configure_logging

T = TypeVar("T")


class WorkerCLI:
    """Base CLI utilities for all MOSAIC workers.

    Provides standard command-line argument parsing and logging setup
    that workers can extend.

    Usage:
        # In worker's cli.py:
        from gym_gui.core.worker import WorkerCLI

        def main(argv=None):
            parser = WorkerCLI.create_base_parser(
                prog="cleanrl-worker",
                description="CleanRL worker for MOSAIC"
            )

            # Add worker-specific arguments
            parser.add_argument("--algo", help="Algorithm to run")

            args = parser.parse_args(argv)

            # Setup logging
            WorkerCLI.setup_logging(args.verbose)

            # Load config
            config = WorkerCLI.load_and_validate_config(
                args.config,
                MyWorkerConfig
            )

            # Execute
            runtime = MyWorkerRuntime(config)
            return 0 if runtime.run() else 1
    """

    @staticmethod
    def create_base_parser(
        prog: str,
        description: str,
    ) -> argparse.ArgumentParser:
        """Create argument parser with standard worker arguments.

        Args:
            prog: Program name (e.g., "cleanrl-worker")
            description: Program description

        Returns:
            ArgumentParser with standard arguments:
            - --config (required): Path to worker config JSON
            - --verbose: Enable debug logging
            - --dry-run: Validate config without executing
            - --worker-id: Worker identifier override
            - --grpc: Enable gRPC telemetry handshake (reserved)
            - --grpc-target: gRPC server address (reserved)

        Example:
            parser = WorkerCLI.create_base_parser(
                prog="my-worker",
                description="My custom RL worker"
            )
        """
        parser = argparse.ArgumentParser(
            prog=prog,
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Required arguments
        parser.add_argument(
            "--config",
            required=True,
            type=Path,
            help="Path to trainer-issued worker config JSON file",
        )

        # Optional flags
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable debug logging (default: INFO level)",
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate configuration without executing training",
        )

        parser.add_argument(
            "--worker-id",
            help="Override worker identifier from config",
        )

        # Reserved for future gRPC integration
        parser.add_argument(
            "--grpc",
            action="store_true",
            help="Enable gRPC telemetry handshake (reserved for future use)",
        )

        parser.add_argument(
            "--grpc-target",
            default="127.0.0.1:50055",
            help="gRPC server address (default: 127.0.0.1:50055)",
        )

        return parser

    @staticmethod
    def setup_logging(verbose: bool = False) -> None:
        """Configure standard logging for worker.

        Sets up logging with appropriate level and format for worker processes.

        Args:
            verbose: If True, set log level to DEBUG, else INFO

        Example:
            WorkerCLI.setup_logging(verbose=True)
        """
        level = logging.DEBUG if verbose else logging.INFO

        # Use gym_gui's logging configuration if available
        try:
            configure_logging(
                log_level=level,
                log_file=None,  # Workers log to stdout
                enable_file_logging=False,
            )
        except Exception:
            # Fallback to basic configuration
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                stream=sys.stderr,
            )

    @staticmethod
    def load_and_validate_config(
        path: Path,
        config_class: Type[T],
    ) -> T:
        """Load and validate worker configuration from file.

        Handles both direct config format and nested GUI format, with
        comprehensive error reporting.

        Args:
            path: Path to config JSON file
            config_class: Configuration class with from_dict() method

        Returns:
            Validated configuration instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
            TypeError: If config_class doesn't have from_dict method

        Example:
            config = WorkerCLI.load_and_validate_config(
                Path("worker-config.json"),
                MyWorkerConfig
            )
        """
        from .config_loader import load_worker_config_from_file

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if not hasattr(config_class, "from_dict"):
            raise TypeError(
                f"{config_class.__name__} must implement from_dict() class method"
            )

        try:
            config = load_worker_config_from_file(path, config_class)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {path}: {e}") from e
