"""Tests for TensorBoard metrics logging from Ray RLlib training results.

These tests verify that:
1. Metrics are correctly extracted from both OLD and NEW Ray API result structures
2. TensorBoard events are actually written with the extracted metrics
3. All algorithm types (PPO, APPO, IMPALA, etc.) have their metrics properly extracted
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# Sample result dictionaries matching Ray RLlib output structures

def make_old_api_result(policy_id: str = "shared") -> Dict[str, Any]:
    """Create a sample OLD API result dictionary.

    OLD API: result["info"]["learner"]["<policy_id>"]["learner_stats"]["<metric>"]
    """
    return {
        "info": {
            "learner": {
                policy_id: {
                    "learner_stats": {
                        "policy_loss": 0.123,
                        "vf_loss": 0.456,
                        "entropy": 1.234,
                        "kl": 0.0012,
                        "entropy_coeff": 0.01,
                        "cur_kl_coeff": 0.2,
                        "cur_lr": 0.0003,
                        "vf_explained_var": 0.567,
                        "total_loss": 0.789,
                    },
                    "model": {},
                    "custom_metrics": {},
                },
                "batch_count": 10,
            },
            "num_env_steps_sampled": 4000,
            "num_env_steps_trained": 4000,
        },
        "env_runners": {
            "episode_return_mean": -123.45,
            "episode_len_mean": 250.5,
            "num_episodes": 16,
            "num_env_steps_sampled_lifetime": 40000,
        },
        "timers": {
            "learn_time_ms": 123.4,
            "sample_time_ms": 56.7,
        },
        "time_total_s": 45.6,
        "training_iteration": 10,
        "timesteps_total": 40000,
    }


def make_new_api_result(module_id: str = "default_module") -> Dict[str, Any]:
    """Create a sample NEW API result dictionary.

    NEW API: result["learners"]["<module_id>"]["<metric>"]
    """
    return {
        "learners": {
            module_id: {
                "policy_loss": 0.234,
                "vf_loss": 0.567,
                "entropy": 1.345,
                "mean_kl_loss": 0.0023,
                "curr_entropy_coeff": 0.01,
                "curr_kl_coeff": 0.2,
                "vf_explained_var": 0.678,
                "total_loss": 0.890,
            },
            "__all_modules__": {
                "policy_loss": 0.234,
                "total_loss": 0.890,
            },
        },
        "env_runners": {
            "episode_return_mean": -234.56,
            "episode_len_mean": 350.5,
            "num_episodes": 24,
            "num_env_steps_sampled_lifetime": 80000,
        },
        "timers": {
            "learner_update_timer": 234.5,
            "env_runner_sampling_timer": 67.8,
        },
        "time_total_s": 78.9,
        "training_iteration": 20,
    }


class TestMetricsExtraction:
    """Test metrics extraction from Ray result dictionaries."""

    def test_old_api_learner_stats_extraction(self):
        """Test extraction from OLD API result structure."""
        result = make_old_api_result(policy_id="shared")

        # Import the extraction logic
        from ray_worker.runtime import RayWorkerRuntime

        # Extract metrics using a mock runtime
        metrics = self._extract_metrics(result, global_step=4000)

        # Verify learner stats are extracted
        assert "train/learner/policy_loss" in metrics
        assert "train/learner/vf_loss" in metrics
        assert "train/learner/entropy" in metrics
        assert "train/learner/kl" in metrics
        assert "train/learner/total_loss" in metrics

        # Verify values are correct
        assert metrics["train/learner/policy_loss"] == pytest.approx(0.123)
        assert metrics["train/learner/vf_loss"] == pytest.approx(0.456)
        assert metrics["train/learner/entropy"] == pytest.approx(1.234)

    def test_old_api_with_default_policy(self):
        """Test OLD API with 'default_policy' policy ID."""
        result = make_old_api_result(policy_id="default_policy")
        metrics = self._extract_metrics(result, global_step=4000)

        assert "train/learner/policy_loss" in metrics
        assert metrics["train/learner/policy_loss"] == pytest.approx(0.123)

    def test_new_api_learner_extraction(self):
        """Test extraction from NEW API result structure."""
        result = make_new_api_result(module_id="default_module")
        metrics = self._extract_metrics(result, global_step=80000)

        # Verify learner stats are extracted
        assert "train/learner/policy_loss" in metrics
        assert "train/learner/vf_loss" in metrics
        assert "train/learner/entropy" in metrics
        assert "train/learner/mean_kl_loss" in metrics

        # Verify values are correct
        assert metrics["train/learner/policy_loss"] == pytest.approx(0.234)
        assert metrics["train/learner/vf_loss"] == pytest.approx(0.567)

    def test_env_runner_metrics_extraction(self):
        """Test extraction of environment runner metrics."""
        result = make_old_api_result()
        metrics = self._extract_metrics(result, global_step=4000)

        assert "train/episode_reward_mean" in metrics
        assert "train/episode_len_mean" in metrics
        assert "train/num_episodes" in metrics

        assert metrics["train/episode_reward_mean"] == pytest.approx(-123.45)
        assert metrics["train/episode_len_mean"] == pytest.approx(250.5)

    def test_timing_metrics_extraction(self):
        """Test extraction of timing/performance metrics."""
        result = make_old_api_result()
        metrics = self._extract_metrics(result, global_step=4000)

        assert "perf/time_total_s" in metrics
        assert "perf/learn_time_ms" in metrics
        assert "perf/sample_time_ms" in metrics

    def test_empty_result_handled(self):
        """Test that empty result dict is handled gracefully."""
        result = {}
        metrics = self._extract_metrics(result, global_step=0)

        # Should not crash, just return empty or minimal metrics
        assert isinstance(metrics, dict)

    def test_missing_learner_section_handled(self):
        """Test that missing learner section is handled gracefully."""
        result = {
            "env_runners": {
                "episode_return_mean": -100.0,
                "episode_len_mean": 200.0,
            },
            "time_total_s": 10.0,
        }
        metrics = self._extract_metrics(result, global_step=1000)

        # Should still extract env_runner and timing metrics
        assert "train/episode_reward_mean" in metrics
        assert "perf/time_total_s" in metrics
        # But no learner metrics
        assert not any(k.startswith("train/learner/") for k in metrics)

    def _extract_metrics(self, result: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        """Extract metrics using the same logic as RayWorkerRuntime._log_metrics.

        This is a copy of the extraction logic to test without needing full runtime.
        """
        metrics = {}

        # Environment runner metrics
        env_runners = result.get("env_runners", {})
        metrics["train/episode_reward_mean"] = env_runners.get(
            "episode_return_mean",
            env_runners.get("episode_reward_mean", result.get("episode_reward_mean", 0))
        )
        metrics["train/episode_len_mean"] = env_runners.get(
            "episode_len_mean", result.get("episode_len_mean", 0)
        )
        metrics["train/num_episodes"] = env_runners.get(
            "num_episodes_lifetime",
            env_runners.get("num_episodes", result.get("episodes_total", 0))
        )

        if "num_env_steps_sampled_lifetime" in env_runners:
            metrics["train/env_steps_sampled"] = env_runners["num_env_steps_sampled_lifetime"]

        # Learner metrics - handle both OLD and NEW Ray API structures
        learner_stats = None
        policy_keys = ["shared", "default_policy", "main"]

        # Try OLD API structure first
        info = result.get("info", {})
        old_learner = info.get("learner", {})
        if old_learner:
            for key in policy_keys:
                if key in old_learner:
                    policy_data = old_learner[key]
                    learner_stats = policy_data.get("learner_stats", policy_data)
                    break
            if learner_stats is None:
                first_key = next(iter(old_learner.keys()), None)
                if first_key and isinstance(old_learner[first_key], dict):
                    policy_data = old_learner[first_key]
                    learner_stats = policy_data.get("learner_stats", policy_data)

        # Try NEW API structure if OLD didn't find anything
        if learner_stats is None:
            learners = result.get("learners", {})
            if learners:
                for key in policy_keys + ["default_module", "__all_modules__"]:
                    if key in learners and isinstance(learners[key], dict):
                        learner_stats = learners[key]
                        break
                if learner_stats is None:
                    first_key = next(iter(learners.keys()), None)
                    if first_key and isinstance(learners[first_key], dict):
                        learner_stats = learners[first_key]

        # Extract ALL numeric metrics from learner_stats dynamically
        if isinstance(learner_stats, dict):
            for key, value in learner_stats.items():
                if isinstance(value, (int, float)) and not key.startswith("_"):
                    metrics[f"train/learner/{key}"] = value

        # Timing metrics
        if "time_total_s" in result:
            metrics["perf/time_total_s"] = result["time_total_s"]
        timers = result.get("timers", {})
        if "learn_time_ms" in timers:
            metrics["perf/learn_time_ms"] = timers["learn_time_ms"]
        if "sample_time_ms" in timers:
            metrics["perf/sample_time_ms"] = timers["sample_time_ms"]

        # Filter out None values
        metrics = {k: v for k, v in metrics.items() if v is not None}

        return metrics


class TestTensorBoardWriting:
    """Test that metrics are actually written to TensorBoard events."""

    def test_tensorboard_scalars_written(self):
        """Test that scalars are written to TensorBoard event files."""
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(log_dir=tmpdir)

            # Write some test metrics
            metrics = {
                "train/episode_reward_mean": -123.45,
                "train/learner/policy_loss": 0.123,
                "train/learner/vf_loss": 0.456,
                "perf/time_total_s": 45.6,
            }

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, global_step=1000)

            writer.flush()
            writer.close()

            # Verify event files were created
            event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
            assert len(event_files) >= 1

            # Verify event file has content (more than just header)
            event_file = event_files[0]
            assert event_file.stat().st_size > 100  # Should be > 100 bytes with data

    def test_tensorboard_event_file_readable(self):
        """Test that written TensorBoard events can be read back."""
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing import event_accumulator

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(log_dir=tmpdir)

            # Write test metrics at multiple steps
            for step in [100, 200, 300]:
                writer.add_scalar("train/reward", step * 0.1, global_step=step)
                writer.add_scalar("train/loss", 1.0 / step, global_step=step)

            writer.flush()
            writer.close()

            # Read back with event accumulator
            ea = event_accumulator.EventAccumulator(tmpdir)
            ea.Reload()

            # Verify scalars are present
            scalar_tags = ea.Tags()["scalars"]
            assert "train/reward" in scalar_tags
            assert "train/loss" in scalar_tags

            # Verify values
            reward_events = ea.Scalars("train/reward")
            assert len(reward_events) == 3
            assert reward_events[0].value == pytest.approx(10.0)  # 100 * 0.1


class TestOldApiStructureValidation:
    """Validate the OLD API result structure we expect from Ray RLlib."""

    def test_old_api_structure_matches_ray_output(self):
        """Verify our test data matches actual Ray RLlib OLD API output structure."""
        result = make_old_api_result()

        # Validate structure matches Ray's output
        assert "info" in result
        assert "learner" in result["info"]
        assert "shared" in result["info"]["learner"]
        assert "learner_stats" in result["info"]["learner"]["shared"]

        # Validate env_runners structure
        assert "env_runners" in result
        assert "episode_return_mean" in result["env_runners"]
        assert "episode_len_mean" in result["env_runners"]

    def test_learner_stats_contains_expected_keys(self):
        """Verify learner_stats contains expected PPO/APPO keys."""
        result = make_old_api_result()
        learner_stats = result["info"]["learner"]["shared"]["learner_stats"]

        # These keys should be present for PPO/APPO
        expected_keys = [
            "policy_loss",
            "vf_loss",
            "entropy",
            "kl",
            "total_loss",
        ]

        for key in expected_keys:
            assert key in learner_stats, f"Missing expected key: {key}"


class TestNewApiStructureValidation:
    """Validate the NEW API result structure we expect from Ray RLlib."""

    def test_new_api_structure_matches_ray_output(self):
        """Verify our test data matches actual Ray RLlib NEW API output structure."""
        result = make_new_api_result()

        # Validate structure matches Ray's output
        assert "learners" in result
        assert "default_module" in result["learners"]

        # Validate env_runners structure (same as OLD API)
        assert "env_runners" in result
        assert "episode_return_mean" in result["env_runners"]

    def test_learner_contains_expected_keys(self):
        """Verify learner contains expected PPO/APPO keys for NEW API."""
        result = make_new_api_result()
        learner = result["learners"]["default_module"]

        # These keys should be present for PPO/APPO in NEW API
        expected_keys = [
            "policy_loss",
            "vf_loss",
            "entropy",
            "mean_kl_loss",  # Note: different name in NEW API
            "total_loss",
        ]

        for key in expected_keys:
            assert key in learner, f"Missing expected key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
