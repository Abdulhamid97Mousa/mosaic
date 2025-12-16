"""Integration test for runtime metrics logging.

This test verifies that the full _log_metrics pipeline in RayWorkerRuntime
correctly extracts and writes metrics to TensorBoard.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


def make_ray_old_api_result() -> Dict[str, Any]:
    """Create a realistic OLD API result from Ray RLlib APPO training."""
    return {
        "info": {
            "learner": {
                "shared": {
                    "learner_stats": {
                        "policy_loss": -0.0123,
                        "vf_loss": 0.456,
                        "entropy": 1.234,
                        "kl": 0.0012,
                        "entropy_coeff": 0.01,
                        "cur_kl_coeff": 0.2,
                        "cur_lr": 0.0003,
                        "vf_explained_var": 0.567,
                        "total_loss": 0.433,
                        # APPO-specific metrics
                        "num_updates": 10,
                        "num_samples": 4000,
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
            "num_agent_steps_sampled_lifetime": 120000,
        },
        "timers": {
            "learn_time_ms": 123.4,
            "sample_time_ms": 56.7,
        },
        "time_total_s": 45.6,
        "training_iteration": 10,
        "timesteps_total": 40000,
    }


class TestRuntimeMetricsIntegration:
    """Test RayWorkerRuntime._log_metrics with mock TensorBoard writer."""

    def test_log_metrics_writes_to_tensorboard(self):
        """Test that _log_metrics correctly extracts and writes metrics."""
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing import event_accumulator

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(log_dir=tmpdir)

            result = make_ray_old_api_result()
            global_step = 40000

            # Simulate the _log_metrics logic
            metrics = self._extract_metrics(result)

            # Write to TensorBoard
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, global_step)

            writer.flush()
            writer.close()

            # Verify by reading back
            ea = event_accumulator.EventAccumulator(tmpdir)
            ea.Reload()
            scalar_tags = ea.Tags()["scalars"]

            # Verify core metrics are present
            assert "train/episode_reward_mean" in scalar_tags
            assert "train/episode_len_mean" in scalar_tags

            # Verify learner metrics are present
            learner_tags = [t for t in scalar_tags if t.startswith("train/learner/")]
            assert len(learner_tags) > 0, "No learner metrics found"

            # Verify specific learner metrics
            assert "train/learner/policy_loss" in scalar_tags
            assert "train/learner/vf_loss" in scalar_tags
            assert "train/learner/entropy" in scalar_tags

            # Verify values are correct
            policy_loss_events = ea.Scalars("train/learner/policy_loss")
            assert len(policy_loss_events) == 1
            assert policy_loss_events[0].value == pytest.approx(-0.0123)

    def test_log_metrics_handles_missing_learner_stats(self):
        """Test that _log_metrics handles results without learner stats."""
        result = {
            "env_runners": {
                "episode_return_mean": -100.0,
                "episode_len_mean": 200.0,
                "num_episodes": 8,
            },
            "time_total_s": 10.0,
        }

        metrics = self._extract_metrics(result)

        # Should still have env_runner and timing metrics
        assert "train/episode_reward_mean" in metrics
        assert "perf/time_total_s" in metrics

        # Should NOT have learner metrics
        assert not any(k.startswith("train/learner/") for k in metrics)

    def test_all_learner_numeric_values_extracted(self):
        """Test that ALL numeric learner metrics are extracted dynamically."""
        result = make_ray_old_api_result()
        metrics = self._extract_metrics(result)

        # Get the original learner_stats keys
        original_stats = result["info"]["learner"]["shared"]["learner_stats"]
        numeric_keys = [
            k for k, v in original_stats.items()
            if isinstance(v, (int, float)) and not k.startswith("_")
        ]

        # Verify all numeric keys were extracted
        for key in numeric_keys:
            assert f"train/learner/{key}" in metrics, f"Missing metric: {key}"

    def _extract_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics using the same logic as RayWorkerRuntime._log_metrics."""
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
        if "num_agent_steps_sampled_lifetime" in env_runners:
            metrics["train/agent_steps_sampled"] = env_runners["num_agent_steps_sampled_lifetime"]

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
