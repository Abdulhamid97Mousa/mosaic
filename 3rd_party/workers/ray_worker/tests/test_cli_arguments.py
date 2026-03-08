"""Tests for Ray worker CLI argument parsing.

Ensures the CLI accepts all arguments passed by the trainer dispatcher:
- --config (required)
- --grpc (flag)
- --grpc-target (string)
- --worker-id (string)
- --verbose (flag)
- --dry-run (flag)
- --output-dir (string)
"""

import pytest
from ray_worker.cli import create_parser


class TestCLIArgumentParsing:
    """Test CLI argument parsing matches dispatcher expectations."""

    def test_required_config_argument(self):
        """Config argument is required."""
        parser = create_parser()

        # Should fail without --config
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_config_argument_accepted(self):
        """Config argument is accepted."""
        parser = create_parser()
        args = parser.parse_args(["--config", "/path/to/config.json"])

        assert args.config == "/path/to/config.json"

    def test_grpc_flag_accepted(self):
        """--grpc flag is accepted (dispatcher compatibility)."""
        parser = create_parser()
        args = parser.parse_args(["--config", "config.json", "--grpc"])

        assert args.grpc is True

    def test_grpc_flag_default_false(self):
        """--grpc flag defaults to False."""
        parser = create_parser()
        args = parser.parse_args(["--config", "config.json"])

        assert args.grpc is False

    def test_grpc_target_accepted(self):
        """--grpc-target argument is accepted."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "config.json",
            "--grpc-target", "127.0.0.1:50055"
        ])

        assert args.grpc_target == "127.0.0.1:50055"

    def test_grpc_target_default(self):
        """--grpc-target has correct default."""
        parser = create_parser()
        args = parser.parse_args(["--config", "config.json"])

        assert args.grpc_target == "127.0.0.1:50055"

    def test_worker_id_accepted(self):
        """--worker-id argument is accepted."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "config.json",
            "--worker-id", "ray_worker"
        ])

        assert args.worker_id == "ray_worker"

    def test_worker_id_default(self):
        """--worker-id has correct default."""
        parser = create_parser()
        args = parser.parse_args(["--config", "config.json"])

        assert args.worker_id == "ray_worker"

    def test_verbose_flag_accepted(self):
        """--verbose flag is accepted."""
        parser = create_parser()
        args = parser.parse_args(["--config", "config.json", "--verbose"])

        assert args.verbose is True

    def test_verbose_short_flag_accepted(self):
        """-v short flag is accepted."""
        parser = create_parser()
        args = parser.parse_args(["--config", "config.json", "-v"])

        assert args.verbose is True

    def test_dry_run_flag_accepted(self):
        """--dry-run flag is accepted."""
        parser = create_parser()
        args = parser.parse_args(["--config", "config.json", "--dry-run"])

        assert args.dry_run is True

    def test_output_dir_accepted(self):
        """--output-dir argument is accepted."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "config.json",
            "--output-dir", "/custom/output"
        ])

        assert args.output_dir == "/custom/output"

    def test_full_dispatcher_command(self):
        """Full command as sent by dispatcher is accepted.

        The dispatcher sends:
        python -m ray_worker.cli --config <path> --grpc --grpc-target 127.0.0.1:50055 --worker-id ray_worker
        """
        parser = create_parser()
        args = parser.parse_args([
            "--config", "/var/trainer/configs/worker-01ABC.json",
            "--grpc",
            "--grpc-target", "127.0.0.1:50055",
            "--worker-id", "ray_worker",
        ])

        assert args.config == "/var/trainer/configs/worker-01ABC.json"
        assert args.grpc is True
        assert args.grpc_target == "127.0.0.1:50055"
        assert args.worker_id == "ray_worker"
        assert args.verbose is False
        assert args.dry_run is False
        assert args.output_dir is None

    def test_all_arguments_combined(self):
        """All arguments can be combined."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "config.json",
            "--grpc",
            "--grpc-target", "localhost:9999",
            "--worker-id", "my_worker",
            "--verbose",
            "--dry-run",
            "--output-dir", "/tmp/output",
        ])

        assert args.config == "config.json"
        assert args.grpc is True
        assert args.grpc_target == "localhost:9999"
        assert args.worker_id == "my_worker"
        assert args.verbose is True
        assert args.dry_run is True
        assert args.output_dir == "/tmp/output"


class TestCLIRejectsInvalidArguments:
    """Test CLI rejects unknown arguments."""

    def test_rejects_unknown_argument(self):
        """Unknown arguments are rejected."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "config.json", "--unknown-arg"])

    def test_rejects_train_subcommand(self):
        """'train' subcommand is not accepted (no subcommands in this CLI)."""
        parser = create_parser()

        # This should fail because 'train' is treated as an unrecognized argument
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "config.json", "train"])
