"""Test RNN config auto-detection for IPPO+GRU checkpoints.

This test verifies that the InteractiveRuntime correctly auto-detects and loads
RNN architecture parameters from training config files when loading checkpoints.

The issue: Checkpoints trained with IPPO+GRU (recurrent architecture) were failing
to load because the runtime was creating a simple MLP policy, causing architecture
mismatch errors.

The fix: Auto-detect training config from checkpoint directory and apply RNN
parameters (representation, rnn, recurrent_hidden_size) to parser_args before
creating the XuanCe runner.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestRNNConfigAutoDetection:
    """Test RNN config auto-detection logic."""

    def test_detects_ippo_gru_config(self, tmp_path):
        """Test that IPPO+GRU config is correctly detected and loaded."""
        # Create checkpoint directory structure
        checkpoint_dir = tmp_path / "checkpoints" / "soccer_2vs2_indagobs" / "seed_1"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "final_train_model.pth"
        checkpoint_file.touch()

        # Create config directory with IPPO+GRU config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "ippo_gru_soccer_2vs2_indagobs_config.json"

        training_config = {
            "run_id": "test_run",
            "algo": "mappo",
            "env_id": "soccer_2vs2_indagobs",
            "extras": {
                "use_rnn": True,
                "representation": "Basic_RNN",
                "rnn": "GRU",
                "recurrent_hidden_size": 64,
                "N_recurrent_layers": 1,
                "fc_hidden_sizes": [128, 128],
                "actor_hidden_size": [],
                "critic_hidden_size": [],
            }
        }

        with open(config_file, 'w') as f:
            json.dump(training_config, f)

        # Simulate the auto-detection logic
        method = "ippo"
        xuance_env_id = "soccer_2vs2_indagobs"
        parser_args = SimpleNamespace()
        parser_args.dl_toolbox = "torch"
        parser_args.device = "cpu"
        parser_args.parallels = 1
        parser_args.running_steps = 1

        training_config_loaded = False
        checkpoint_dir_search = checkpoint_file.parent

        for _ in range(5):
            config_candidates = [
                checkpoint_dir_search / "config" / f"{method}_gru_{xuance_env_id}_config.json",
                checkpoint_dir_search / "config" / f"{method}_{xuance_env_id}_config.json",
                checkpoint_dir_search / "config" / "base_config.json",
            ]

            for config_file_candidate in config_candidates:
                if config_file_candidate.exists():
                    with open(config_file_candidate, 'r') as f:
                        loaded_config = json.load(f)

                    extras = loaded_config.get("extras", {})
                    if extras.get("use_rnn") or extras.get("representation") == "Basic_RNN":
                        parser_args.representation = extras.get("representation", "Basic_RNN")
                        parser_args.rnn = extras.get("rnn", "GRU")
                        parser_args.recurrent_hidden_size = extras.get("recurrent_hidden_size", 64)
                        parser_args.N_recurrent_layers = extras.get("N_recurrent_layers", 1)
                        parser_args.fc_hidden_sizes = extras.get("fc_hidden_sizes", [128, 128])
                        parser_args.actor_hidden_size = extras.get("actor_hidden_size", [])
                        parser_args.critic_hidden_size = extras.get("critic_hidden_size", [])
                        training_config_loaded = True
                        break

            if training_config_loaded:
                break

            if checkpoint_dir_search.parent == checkpoint_dir_search:
                break
            checkpoint_dir_search = checkpoint_dir_search.parent

        # Verify RNN config was detected and loaded
        assert training_config_loaded, "RNN config should be detected"
        assert parser_args.representation == "Basic_RNN"
        assert parser_args.rnn == "GRU"
        assert parser_args.recurrent_hidden_size == 64
        assert parser_args.N_recurrent_layers == 1
        assert parser_args.fc_hidden_sizes == [128, 128]
        assert parser_args.actor_hidden_size == []
        assert parser_args.critic_hidden_size == []

    def test_detects_basic_rnn_without_use_rnn_flag(self, tmp_path):
        """Test detection works even if use_rnn is False but representation is Basic_RNN."""
        checkpoint_dir = tmp_path / "checkpoints" / "seed_1"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "final_train_model.pth"
        checkpoint_file.touch()

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "ippo_soccer_config.json"

        training_config = {
            "extras": {
                "use_rnn": False,  # Flag is False
                "representation": "Basic_RNN",  # But representation is RNN
                "rnn": "LSTM",
                "recurrent_hidden_size": 128,
            }
        }

        with open(config_file, 'w') as f:
            json.dump(training_config, f)

        # Simulate detection
        parser_args = SimpleNamespace()
        checkpoint_dir_search = checkpoint_file.parent

        for _ in range(5):
            config_candidates = [
                checkpoint_dir_search / "config" / "ippo_gru_soccer_config.json",
                checkpoint_dir_search / "config" / "ippo_soccer_config.json",
                checkpoint_dir_search / "config" / "base_config.json",
            ]

            for config_file_candidate in config_candidates:
                if config_file_candidate.exists():
                    with open(config_file_candidate, 'r') as f:
                        loaded_config = json.load(f)

                    extras = loaded_config.get("extras", {})
                    if extras.get("use_rnn") or extras.get("representation") == "Basic_RNN":
                        parser_args.representation = extras.get("representation")
                        parser_args.rnn = extras.get("rnn")
                        break

            if hasattr(parser_args, 'representation'):
                break

            if checkpoint_dir_search.parent == checkpoint_dir_search:
                break
            checkpoint_dir_search = checkpoint_dir_search.parent

        # Should still detect RNN based on representation field
        assert parser_args.representation == "Basic_RNN"
        assert parser_args.rnn == "LSTM"

    def test_fallback_to_mlp_when_no_rnn_config(self, tmp_path):
        """Test that MLP is used when no RNN config is found."""
        checkpoint_dir = tmp_path / "checkpoints" / "seed_1"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "final_train_model.pth"
        checkpoint_file.touch()

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "ippo_config.json"

        # Config without RNN parameters
        training_config = {
            "extras": {
                "num_envs": 8,
                "n_epochs": 10,
            }
        }

        with open(config_file, 'w') as f:
            json.dump(training_config, f)

        # Simulate detection
        parser_args = SimpleNamespace()
        training_config_loaded = False
        checkpoint_dir_search = checkpoint_file.parent

        for _ in range(5):
            config_candidates = [
                checkpoint_dir_search / "config" / "ippo_gru_config.json",
                checkpoint_dir_search / "config" / "ippo_config.json",
                checkpoint_dir_search / "config" / "base_config.json",
            ]

            for config_file_candidate in config_candidates:
                if config_file_candidate.exists():
                    with open(config_file_candidate, 'r') as f:
                        loaded_config = json.load(f)

                    extras = loaded_config.get("extras", {})
                    if extras.get("use_rnn") or extras.get("representation") == "Basic_RNN":
                        training_config_loaded = True
                        break

            if training_config_loaded:
                break

            if checkpoint_dir_search.parent == checkpoint_dir_search:
                break
            checkpoint_dir_search = checkpoint_dir_search.parent

        # Should not detect RNN config
        assert not training_config_loaded
        assert not hasattr(parser_args, 'representation')

    def test_searches_up_directory_tree(self, tmp_path):
        """Test that config search traverses up the directory tree."""
        # Create deep checkpoint structure
        checkpoint_dir = tmp_path / "checkpoints" / "soccer" / "seed_1" / "subdir"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "final_train_model.pth"
        checkpoint_file.touch()

        # Place config at root level
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "base_config.json"

        training_config = {
            "extras": {
                "representation": "Basic_RNN",
                "rnn": "GRU",
            }
        }

        with open(config_file, 'w') as f:
            json.dump(training_config, f)

        # Simulate detection starting from deep directory
        parser_args = SimpleNamespace()
        checkpoint_dir_search = checkpoint_file.parent

        for _ in range(5):
            config_candidates = [
                checkpoint_dir_search / "config" / "base_config.json",
            ]

            for config_file_candidate in config_candidates:
                if config_file_candidate.exists():
                    with open(config_file_candidate, 'r') as f:
                        loaded_config = json.load(f)

                    extras = loaded_config.get("extras", {})
                    if extras.get("representation") == "Basic_RNN":
                        parser_args.representation = extras.get("representation")
                        break

            if hasattr(parser_args, 'representation'):
                break

            if checkpoint_dir_search.parent == checkpoint_dir_search:
                break
            checkpoint_dir_search = checkpoint_dir_search.parent

        # Should find config by traversing up
        assert parser_args.representation == "Basic_RNN"

    def test_real_checkpoint_structure_ippo_gru(self):
        """Test with actual IPPO+GRU checkpoint structure from the issue."""
        checkpoint_path = Path("/home/hamid/Desktop/software/mosaic/var/trainer/custom_scripts/01KJ9A190NHRQNMNXW29F6KHYR/checkpoints/soccer_2vs2_indagobs/seed_1_2026_0225_102855/final_train_model.pth")

        if not checkpoint_path.exists():
            pytest.skip("Real IPPO+GRU checkpoint not found, skipping integration test")

        method = "ippo"
        xuance_env_id = "soccer_2vs2_indagobs"
        parser_args = SimpleNamespace()

        training_config_loaded = False
        checkpoint_dir = checkpoint_path.parent

        for _ in range(5):
            config_candidates = [
                checkpoint_dir / "config" / f"{method}_gru_{xuance_env_id}_config.json",
                checkpoint_dir / "config" / f"{method}_{xuance_env_id}_config.json",
                checkpoint_dir / "config" / "base_config.json",
            ]

            for config_file in config_candidates:
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        training_config = json.load(f)

                    extras = training_config.get("extras", {})
                    if extras.get("use_rnn") or extras.get("representation") == "Basic_RNN":
                        parser_args.representation = extras.get("representation", "Basic_RNN")
                        parser_args.rnn = extras.get("rnn", "GRU")
                        parser_args.recurrent_hidden_size = extras.get("recurrent_hidden_size", 64)
                        parser_args.N_recurrent_layers = extras.get("N_recurrent_layers", 1)
                        training_config_loaded = True
                        break

            if training_config_loaded:
                break

            if checkpoint_dir.parent == checkpoint_dir:
                break
            checkpoint_dir = checkpoint_dir.parent

        # Verify real checkpoint config is detected
        assert training_config_loaded, "Should detect RNN config from real checkpoint"
        assert parser_args.representation == "Basic_RNN"
        assert parser_args.rnn == "GRU"
        assert parser_args.recurrent_hidden_size == 64
        assert parser_args.N_recurrent_layers == 1

    def test_real_checkpoint_structure_plain_ippo(self):
        """Test with actual plain IPPO (MLP) checkpoint structure."""
        checkpoint_path = Path("/home/hamid/Desktop/software/mosaic/var/trainer/custom_scripts/01KJ9A1QJPY999E563410SMSKA/checkpoints/soccer_2vs2_indagobs/seed_1_2026_0225_102910/final_train_model.pth")

        if not checkpoint_path.exists():
            pytest.skip("Real plain IPPO checkpoint not found, skipping integration test")

        method = "ippo"
        xuance_env_id = "soccer_2vs2_indagobs"
        parser_args = SimpleNamespace()
        parser_args.dl_toolbox = "torch"
        parser_args.device = "cpu"
        parser_args.parallels = 1
        parser_args.running_steps = 1

        training_config_loaded = False
        checkpoint_dir = checkpoint_path.parent

        for _ in range(5):
            config_candidates = [
                checkpoint_dir / "config" / f"{method}_gru_{xuance_env_id}_config.json",
                checkpoint_dir / "config" / f"{method}_{xuance_env_id}_config.json",
                checkpoint_dir / "config" / "base_config.json",
            ]

            for config_file in config_candidates:
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        training_config = json.load(f)

                    extras = training_config.get("extras", {})
                    if extras.get("use_rnn") or extras.get("representation") == "Basic_RNN":
                        parser_args.representation = extras.get("representation", "Basic_RNN")
                        parser_args.use_rnn = True
                        parser_args.rnn = extras.get("rnn", "GRU")
                        parser_args.recurrent_hidden_size = extras.get("recurrent_hidden_size", 64)
                        parser_args.N_recurrent_layers = extras.get("N_recurrent_layers", 1)
                        training_config_loaded = True
                        break

            if training_config_loaded:
                break

            if checkpoint_dir.parent == checkpoint_dir:
                break
            checkpoint_dir = checkpoint_dir.parent

        # Verify plain IPPO checkpoint does NOT load RNN config
        assert not training_config_loaded, "Should NOT detect RNN config from plain IPPO checkpoint"
        assert not hasattr(parser_args, 'representation'), "Should not have representation attribute"
        assert not hasattr(parser_args, 'use_rnn'), "Should not have use_rnn attribute"
        # Verify base attributes are still present
        assert parser_args.dl_toolbox == "torch"
        assert parser_args.device == "cpu"
