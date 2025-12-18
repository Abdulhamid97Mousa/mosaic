"""Tests for XuanCe algorithm registry."""

from __future__ import annotations

import pytest

from xuance_worker.algorithm_registry import (
    Backend,
    Paradigm,
    AlgorithmInfo,
    BACKEND_ALGORITHMS,
    ALGORITHM_INFO,
    get_algorithms,
    get_algorithms_for_backend,
    get_algorithms_for_paradigm,
    get_algorithm_info,
    get_algorithm_choices,
    get_algorithms_by_category,
    is_algorithm_available,
    get_backend_summary,
)


class TestBackendEnum:
    """Tests for Backend enum."""

    def test_backend_values(self) -> None:
        """Test backend enum values."""
        assert Backend.TORCH.value == "torch"
        assert Backend.TENSORFLOW.value == "tensorflow"
        assert Backend.MINDSPORE.value == "mindspore"

    def test_backend_from_string(self) -> None:
        """Test creating backend from string."""
        assert Backend("torch") == Backend.TORCH
        assert Backend("tensorflow") == Backend.TENSORFLOW
        assert Backend("mindspore") == Backend.MINDSPORE

    def test_backend_invalid_string(self) -> None:
        """Test invalid backend string raises ValueError."""
        with pytest.raises(ValueError):
            Backend("invalid")


class TestParadigmEnum:
    """Tests for Paradigm enum."""

    def test_paradigm_values(self) -> None:
        """Test paradigm enum values."""
        assert Paradigm.SINGLE_AGENT.value == "single_agent"
        assert Paradigm.MULTI_AGENT.value == "multi_agent"

    def test_paradigm_from_string(self) -> None:
        """Test creating paradigm from string."""
        assert Paradigm("single_agent") == Paradigm.SINGLE_AGENT
        assert Paradigm("multi_agent") == Paradigm.MULTI_AGENT


class TestAlgorithmInfo:
    """Tests for AlgorithmInfo dataclass."""

    def test_algorithm_info_creation(self) -> None:
        """Test creating AlgorithmInfo."""
        info = AlgorithmInfo(
            key="PPO_Clip",
            display_name="PPO (Clip)",
            category="Policy Optimization",
            paradigm=Paradigm.SINGLE_AGENT,
            description="Proximal Policy Optimization with clipping",
        )

        assert info.key == "PPO_Clip"
        assert info.display_name == "PPO (Clip)"
        assert info.category == "Policy Optimization"
        assert info.paradigm == Paradigm.SINGLE_AGENT
        assert info.description == "Proximal Policy Optimization with clipping"

    def test_algorithm_info_frozen(self) -> None:
        """Test AlgorithmInfo is immutable."""
        info = AlgorithmInfo(
            key="DQN",
            display_name="DQN",
            category="Value-based",
            paradigm=Paradigm.SINGLE_AGENT,
        )

        with pytest.raises(AttributeError):
            info.key = "DDQN"  # type: ignore[misc]


class TestBackendAlgorithms:
    """Tests for backend algorithm registry."""

    def test_torch_has_most_algorithms(self) -> None:
        """Test that PyTorch backend has the most algorithms."""
        torch_count = len(BACKEND_ALGORITHMS[Backend.TORCH])
        tf_count = len(BACKEND_ALGORITHMS[Backend.TENSORFLOW])
        ms_count = len(BACKEND_ALGORITHMS[Backend.MINDSPORE])

        assert torch_count >= tf_count
        assert torch_count >= ms_count

    def test_tensorflow_mindspore_equal(self) -> None:
        """Test TensorFlow and MindSpore have same algorithms."""
        tf_algos = BACKEND_ALGORITHMS[Backend.TENSORFLOW]
        ms_algos = BACKEND_ALGORITHMS[Backend.MINDSPORE]

        assert tf_algos == ms_algos

    def test_core_algorithms_in_all_backends(self) -> None:
        """Test core algorithms available in all backends."""
        core_algos = ["DQN", "PPO_Clip", "SAC", "MAPPO", "QMIX"]

        for algo in core_algos:
            for backend in Backend:
                assert algo in BACKEND_ALGORITHMS[backend], (
                    f"{algo} should be in {backend.value}"
                )

    def test_torch_only_algorithms(self) -> None:
        """Test PyTorch-only algorithms not in other backends."""
        torch_only = ["DreamerV2", "DreamerV3", "NPG", "CURL", "SPR", "DrQ"]

        torch_algos = BACKEND_ALGORITHMS[Backend.TORCH]
        tf_algos = BACKEND_ALGORITHMS[Backend.TENSORFLOW]

        for algo in torch_only:
            assert algo in torch_algos, f"{algo} should be in torch"
            assert algo not in tf_algos, f"{algo} should not be in tensorflow"


class TestGetAlgorithmsForBackend:
    """Tests for get_algorithms_for_backend function."""

    def test_get_torch_algorithms(self) -> None:
        """Test getting PyTorch algorithms."""
        algos = get_algorithms_for_backend("torch")

        assert "DQN" in algos
        assert "PPO_Clip" in algos
        assert "DreamerV3" in algos  # PyTorch-only

    def test_get_tensorflow_algorithms(self) -> None:
        """Test getting TensorFlow algorithms."""
        algos = get_algorithms_for_backend("tensorflow")

        assert "DQN" in algos
        assert "PPO_Clip" in algos
        assert "DreamerV3" not in algos  # Not in TensorFlow

    def test_get_algorithms_with_enum(self) -> None:
        """Test getting algorithms with Backend enum."""
        algos = get_algorithms_for_backend(Backend.TORCH)

        assert isinstance(algos, frozenset)
        assert len(algos) > 0


class TestGetAlgorithmsForParadigm:
    """Tests for get_algorithms_for_paradigm function."""

    def test_get_single_agent_algorithms(self) -> None:
        """Test getting single-agent algorithms."""
        algos = get_algorithms_for_paradigm(Paradigm.SINGLE_AGENT)

        assert "DQN" in algos
        assert "PPO_Clip" in algos
        assert "SAC" in algos
        assert "MAPPO" not in algos  # Multi-agent

    def test_get_multi_agent_algorithms(self) -> None:
        """Test getting multi-agent algorithms."""
        algos = get_algorithms_for_paradigm(Paradigm.MULTI_AGENT)

        assert "MAPPO" in algos
        assert "QMIX" in algos
        assert "VDN" in algos
        assert "DQN" not in algos  # Single-agent

    def test_get_algorithms_with_string(self) -> None:
        """Test getting algorithms with string paradigm."""
        algos = get_algorithms_for_paradigm("single_agent")

        assert isinstance(algos, frozenset)
        assert "PPO_Clip" in algos


class TestGetAlgorithms:
    """Tests for get_algorithms function."""

    def test_get_torch_single_agent(self) -> None:
        """Test filtering by torch + single-agent."""
        algos = get_algorithms("torch", "single_agent")

        assert "DQN" in algos
        assert "PPO_Clip" in algos
        assert "DreamerV3" in algos
        assert "MAPPO" not in algos  # Multi-agent

    def test_get_torch_multi_agent(self) -> None:
        """Test filtering by torch + multi-agent."""
        algos = get_algorithms("torch", "multi_agent")

        assert "MAPPO" in algos
        assert "QMIX" in algos
        assert "CommNet" in algos  # PyTorch-only multi-agent
        assert "DQN" not in algos  # Single-agent

    def test_get_tensorflow_single_agent(self) -> None:
        """Test filtering by tensorflow + single-agent."""
        algos = get_algorithms("tensorflow", "single_agent")

        assert "DQN" in algos
        assert "PPO_Clip" in algos
        assert "DreamerV3" not in algos  # Not in TensorFlow

    def test_get_tensorflow_multi_agent(self) -> None:
        """Test filtering by tensorflow + multi-agent."""
        algos = get_algorithms("tensorflow", "multi_agent")

        assert "MAPPO" in algos
        assert "QMIX" in algos
        assert "CommNet" not in algos  # PyTorch-only


class TestGetAlgorithmInfo:
    """Tests for get_algorithm_info function."""

    def test_get_existing_algorithm(self) -> None:
        """Test getting info for existing algorithm."""
        info = get_algorithm_info("PPO_Clip")

        assert info is not None
        assert info.key == "PPO_Clip"
        assert info.display_name == "PPO (Clip)"
        assert info.paradigm == Paradigm.SINGLE_AGENT

    def test_get_nonexistent_algorithm(self) -> None:
        """Test getting info for nonexistent algorithm."""
        info = get_algorithm_info("NonexistentAlgo")

        assert info is None

    def test_all_algorithms_have_info(self) -> None:
        """Test all registered algorithms have info."""
        for algo in BACKEND_ALGORITHMS[Backend.TORCH]:
            info = get_algorithm_info(algo)
            assert info is not None, f"Missing info for {algo}"


class TestGetAlgorithmChoices:
    """Tests for get_algorithm_choices function."""

    def test_choices_format(self) -> None:
        """Test choices are in correct format."""
        choices = get_algorithm_choices("torch", "single_agent")

        assert isinstance(choices, list)
        assert len(choices) > 0

        for key, display_name in choices:
            assert isinstance(key, str)
            assert isinstance(display_name, str)

    def test_choices_sorted(self) -> None:
        """Test choices are sorted by display name."""
        choices = get_algorithm_choices("torch", "single_agent")

        display_names = [name for _, name in choices]
        assert display_names == sorted(display_names)


class TestGetAlgorithmsByCategory:
    """Tests for get_algorithms_by_category function."""

    def test_categories_structure(self) -> None:
        """Test category grouping structure."""
        categories = get_algorithms_by_category("torch", "single_agent")

        assert isinstance(categories, dict)
        assert len(categories) > 0

        for cat_name, algos in categories.items():
            assert isinstance(cat_name, str)
            assert isinstance(algos, list)
            assert all(isinstance(a, AlgorithmInfo) for a in algos)

    def test_expected_categories(self) -> None:
        """Test expected categories exist."""
        categories = get_algorithms_by_category("torch", "single_agent")

        assert "Value-based" in categories
        assert "Policy Optimization" in categories

    def test_multi_agent_categories(self) -> None:
        """Test multi-agent categories."""
        categories = get_algorithms_by_category("torch", "multi_agent")

        assert "Value Decomposition" in categories
        assert "Centralized" in categories


class TestIsAlgorithmAvailable:
    """Tests for is_algorithm_available function."""

    def test_available_in_torch(self) -> None:
        """Test algorithm available in PyTorch."""
        assert is_algorithm_available("DreamerV3", "torch") is True
        assert is_algorithm_available("PPO_Clip", "torch") is True

    def test_not_available_in_tensorflow(self) -> None:
        """Test PyTorch-only algorithm not in TensorFlow."""
        assert is_algorithm_available("DreamerV3", "tensorflow") is False
        assert is_algorithm_available("NPG", "tensorflow") is False

    def test_core_available_everywhere(self) -> None:
        """Test core algorithms available in all backends."""
        for backend in ["torch", "tensorflow", "mindspore"]:
            assert is_algorithm_available("DQN", backend) is True
            assert is_algorithm_available("MAPPO", backend) is True


class TestGetBackendSummary:
    """Tests for get_backend_summary function."""

    def test_summary_structure(self) -> None:
        """Test summary has correct structure."""
        summary = get_backend_summary()

        assert "torch" in summary
        assert "tensorflow" in summary
        assert "mindspore" in summary

        for backend_name, counts in summary.items():
            assert "single_agent" in counts
            assert "multi_agent" in counts
            assert "total" in counts

    def test_counts_are_positive(self) -> None:
        """Test all counts are positive."""
        summary = get_backend_summary()

        for backend_name, counts in summary.items():
            assert counts["single_agent"] > 0
            assert counts["multi_agent"] > 0
            assert counts["total"] > 0

    def test_total_equals_sum(self) -> None:
        """Test total equals sum of single and multi-agent."""
        summary = get_backend_summary()

        for backend_name, counts in summary.items():
            assert counts["total"] == counts["single_agent"] + counts["multi_agent"]

    def test_torch_has_most(self) -> None:
        """Test PyTorch has the most algorithms."""
        summary = get_backend_summary()

        torch_total = summary["torch"]["total"]
        tf_total = summary["tensorflow"]["total"]
        ms_total = summary["mindspore"]["total"]

        assert torch_total >= tf_total
        assert torch_total >= ms_total
