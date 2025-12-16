"""Tests for the unified evaluation system."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestEvalRegistry:
    """Tests for eval_registry.py"""

    def test_eval_registry_contains_all_algorithms(self):
        """EVAL_REGISTRY should contain entries for all supported algorithms."""
        from cleanrl_worker.eval_registry import EVAL_REGISTRY

        expected_algos = [
            "ppo",
            "ppo_continuous_action",
            "ppo_atari",
            "ppo_atari_lstm",
            "ppo_atari_envpool",
            "ppo_procgen",
            "dqn",
            "dqn_atari",
            "c51",
            "c51_atari",
            "ddpg_continuous_action",
            "td3_continuous_action",
            "sac_continuous_action",
        ]

        for algo in expected_algos:
            assert algo in EVAL_REGISTRY, f"Algorithm '{algo}' not in EVAL_REGISTRY"

    def test_get_eval_entry_returns_unified_entry(self):
        """get_eval_entry should return _UnifiedEvalEntry for registered algos."""
        from cleanrl_worker.eval_registry import get_eval_entry, _UnifiedEvalEntry

        entry = get_eval_entry("ppo")
        assert entry is not None
        assert isinstance(entry, _UnifiedEvalEntry)
        assert entry.algo_name == "ppo"

    def test_get_eval_entry_returns_none_for_unknown(self):
        """get_eval_entry should return None for unknown algorithms."""
        from cleanrl_worker.eval_registry import get_eval_entry

        entry = get_eval_entry("unknown_algo_xyz")
        assert entry is None

    def test_unified_eval_entry_has_agent_path(self):
        """_UnifiedEvalEntry should have correct agent_path."""
        from cleanrl_worker.eval_registry import get_eval_entry

        # Test PPO
        ppo_entry = get_eval_entry("ppo")
        assert ppo_entry is not None
        assert "Agent" in ppo_entry.agent_path

        # Test DQN
        dqn_entry = get_eval_entry("dqn")
        assert dqn_entry is not None
        assert "QNetwork" in dqn_entry.agent_path

    def test_unified_eval_entry_has_make_env_path(self):
        """_UnifiedEvalEntry should have correct make_env_path."""
        from cleanrl_worker.eval_registry import get_eval_entry

        entry = get_eval_entry("ppo")
        assert entry is not None
        assert "make_env" in entry.make_env_path

    def test_unified_eval_entry_evaluate_is_callable(self):
        """_UnifiedEvalEntry.evaluate should be a callable."""
        from cleanrl_worker.eval_registry import get_eval_entry

        entry = get_eval_entry("ppo")
        assert entry is not None
        assert callable(entry.evaluate)


class TestUnifiedEvalRegistry:
    """Tests for unified_eval/registry.py"""

    def test_adapter_registry_has_all_algorithms(self):
        """ADAPTER_REGISTRY should contain adapters for all supported algorithms."""
        from cleanrl_worker.unified_eval.registry import ADAPTER_REGISTRY

        expected = [
            "ppo", "ppo_continuous_action", "ppo_atari",
            "dqn", "dqn_atari",
            "c51", "c51_atari",
            "ddpg_continuous_action",
            "td3_continuous_action",
            "sac_continuous_action",
        ]

        for algo in expected:
            assert algo in ADAPTER_REGISTRY, f"Algorithm '{algo}' not in ADAPTER_REGISTRY"

    def test_get_adapter_returns_correct_type(self):
        """get_adapter should return the correct adapter type."""
        from cleanrl_worker.unified_eval.registry import get_adapter
        from cleanrl_worker.unified_eval.adapters import (
            PPOSelector,
            DQNSelector,
            C51Selector,
            DDPGSelector,
            TD3Selector,
            SACSelector,
        )

        # Test each adapter type
        assert isinstance(get_adapter("ppo"), PPOSelector)
        assert isinstance(get_adapter("dqn"), DQNSelector)
        assert isinstance(get_adapter("c51"), C51Selector)
        assert isinstance(get_adapter("ddpg_continuous_action"), DDPGSelector)
        assert isinstance(get_adapter("td3_continuous_action"), TD3Selector)
        assert isinstance(get_adapter("sac_continuous_action"), SACSelector)

    def test_get_adapter_returns_none_for_unknown(self):
        """get_adapter should return None for unknown algorithms."""
        from cleanrl_worker.unified_eval.registry import get_adapter

        assert get_adapter("unknown_algo") is None

    def test_list_supported_algorithms(self):
        """list_supported_algorithms should return sorted list."""
        from cleanrl_worker.unified_eval.registry import list_supported_algorithms

        algos = list_supported_algorithms()
        assert isinstance(algos, list)
        assert len(algos) > 0
        assert algos == sorted(algos)  # Should be sorted


class TestAdapters:
    """Tests for unified_eval/adapters/"""

    def test_ppo_selector_accepts_model_cls(self):
        """PPOSelector.load should accept model_cls parameter."""
        from cleanrl_worker.unified_eval.adapters import PPOSelector

        selector = PPOSelector()
        # Check signature accepts model_cls
        import inspect
        sig = inspect.signature(selector.load)
        assert "model_cls" in sig.parameters

    def test_dqn_selector_accepts_model_cls(self):
        """DQNSelector.load should accept model_cls parameter."""
        from cleanrl_worker.unified_eval.adapters import DQNSelector

        selector = DQNSelector()
        import inspect
        sig = inspect.signature(selector.load)
        assert "model_cls" in sig.parameters

    def test_ddpg_selector_accepts_model_cls(self):
        """DDPGSelector.load should accept model_cls parameter."""
        from cleanrl_worker.unified_eval.adapters import DDPGSelector

        selector = DDPGSelector()
        import inspect
        sig = inspect.signature(selector.load)
        assert "model_cls" in sig.parameters

    def test_td3_selector_accepts_model_cls(self):
        """TD3Selector.load should accept model_cls parameter."""
        from cleanrl_worker.unified_eval.adapters import TD3Selector

        selector = TD3Selector()
        import inspect
        sig = inspect.signature(selector.load)
        assert "model_cls" in sig.parameters

    def test_sac_selector_accepts_model_cls(self):
        """SACSelector.load should accept model_cls parameter."""
        from cleanrl_worker.unified_eval.adapters import SACSelector

        selector = SACSelector()
        import inspect
        sig = inspect.signature(selector.load)
        assert "model_cls" in sig.parameters

    def test_c51_selector_accepts_model_cls(self):
        """C51Selector.load should accept model_cls parameter."""
        from cleanrl_worker.unified_eval.adapters import C51Selector

        selector = C51Selector()
        import inspect
        sig = inspect.signature(selector.load)
        assert "model_cls" in sig.parameters


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_eval_result_from_episodes_empty(self):
        """EvalResult.from_episodes should handle empty input."""
        from cleanrl_worker.unified_eval.evaluator import EvalResult

        result = EvalResult.from_episodes([], [])
        assert result.episodes == 0
        assert result.avg_return == 0.0
        assert result.returns == []

    def test_eval_result_from_episodes_single(self):
        """EvalResult.from_episodes should handle single episode."""
        from cleanrl_worker.unified_eval.evaluator import EvalResult

        result = EvalResult.from_episodes([100.0], [50])
        assert result.episodes == 1
        assert result.avg_return == 100.0
        assert result.min_return == 100.0
        assert result.max_return == 100.0
        assert result.std_return == 0.0

    def test_eval_result_from_episodes_multiple(self):
        """EvalResult.from_episodes should calculate correct statistics."""
        from cleanrl_worker.unified_eval.evaluator import EvalResult

        returns = [100.0, 200.0, 300.0]
        lengths = [10, 20, 30]
        result = EvalResult.from_episodes(returns, lengths)

        assert result.episodes == 3
        assert result.avg_return == 200.0
        assert result.min_return == 100.0
        assert result.max_return == 300.0
        assert result.avg_length == 20.0


class TestEvaluator:
    """Tests for the evaluate function."""

    def test_evaluate_handles_old_gymnasium_api(self):
        """evaluate should handle old Gymnasium API (final_info)."""
        from cleanrl_worker.unified_eval.evaluator import evaluate

        # Create mock selector
        mock_selector = MagicMock()
        mock_selector.select_action.return_value = np.array([0])

        # Create mock env with old Gymnasium API
        mock_envs = MagicMock()
        mock_envs.reset.return_value = (np.array([[0.0]]), {})

        # Simulate 2 episodes completing via old API
        call_count = [0]
        def step_side_effect(actions):
            call_count[0] += 1
            if call_count[0] <= 2:
                return (
                    np.array([[0.0]]),
                    np.array([0.0]),
                    np.array([False]),
                    np.array([False]),
                    {"final_info": [{"episode": {"r": 100.0 * call_count[0], "l": 10}}]}
                )
            return (
                np.array([[0.0]]),
                np.array([0.0]),
                np.array([False]),
                np.array([False]),
                {"final_info": [None]}
            )

        mock_envs.step.side_effect = step_side_effect

        result = evaluate(mock_selector, mock_envs, eval_episodes=2)

        assert result.episodes == 2
        assert len(result.returns) == 2

    def test_evaluate_handles_new_gymnasium_api(self):
        """evaluate should handle new Gymnasium 1.0+ API (_episode)."""
        from cleanrl_worker.unified_eval.evaluator import evaluate

        # Create mock selector
        mock_selector = MagicMock()
        mock_selector.select_action.return_value = np.array([0])

        # Create mock env with new Gymnasium API
        mock_envs = MagicMock()
        mock_envs.reset.return_value = (np.array([[0.0]]), {})

        # Simulate 2 episodes completing via new API
        call_count = [0]
        def step_side_effect(actions):
            call_count[0] += 1
            if call_count[0] <= 2:
                return (
                    np.array([[0.0]]),
                    np.array([0.0]),
                    np.array([False]),
                    np.array([False]),
                    {
                        "episode": {"r": np.array([100.0 * call_count[0]]), "l": np.array([10])},
                        "_episode": np.array([True])
                    }
                )
            return (
                np.array([[0.0]]),
                np.array([0.0]),
                np.array([False]),
                np.array([False]),
                {
                    "episode": {"r": np.array([0.0]), "l": np.array([0])},
                    "_episode": np.array([False])
                }
            )

        mock_envs.step.side_effect = step_side_effect

        result = evaluate(mock_selector, mock_envs, eval_episodes=2)

        assert result.episodes == 2
        assert len(result.returns) == 2


class TestIntegration:
    """Integration tests for the unified evaluation system."""

    def test_full_registry_to_adapter_flow(self):
        """Test the full flow from registry to adapter."""
        from cleanrl_worker.eval_registry import get_eval_entry
        from cleanrl_worker.unified_eval.registry import get_adapter

        # For each algorithm in the registry, verify we can get an adapter
        algos_to_test = ["ppo", "dqn", "c51", "ddpg_continuous_action", "td3_continuous_action", "sac_continuous_action"]

        for algo in algos_to_test:
            entry = get_eval_entry(algo)
            assert entry is not None, f"No entry for {algo}"

            adapter = get_adapter(algo)
            assert adapter is not None, f"No adapter for {algo}"

            # Verify evaluate is callable
            assert callable(entry.evaluate), f"evaluate not callable for {algo}"

    def test_no_removed_files_imported(self):
        """Verify removed files are not imported anywhere."""
        import importlib.util

        # These files were removed
        removed_modules = [
            "cleanrl_worker.eval.ppo",
            "cleanrl_worker.eval.ppo_eval_raw",
        ]

        for module_name in removed_modules:
            spec = importlib.util.find_spec(module_name)
            assert spec is None, f"Removed module {module_name} still exists"
