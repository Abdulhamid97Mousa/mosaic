"""Tests for XuanCe algorithm registry.

These tests validate:
- Backend and paradigm enums
- Algorithm availability by backend
- Algorithm filtering functions
- Category grouping
- Summary statistics
"""

from __future__ import annotations

import pytest

from xuance_worker.algorithm_registry import (
    Backend,
    Paradigm,
    AlgorithmInfo,
    BACKEND_ALGORITHMS,
    ALGORITHM_INFO,
    get_algorithms_for_backend,
    get_algorithms_for_paradigm,
    get_algorithms,
    get_algorithm_info,
    get_algorithm_choices,
    get_algorithms_by_category,
    is_algorithm_available,
    get_backend_summary,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestBackendEnum:
    """Tests for Backend enum."""

    def test_backend_values(self) -> None:
        """Test Backend enum values."""
        assert Backend.TORCH.value == "torch"
        assert Backend.TENSORFLOW.value == "tensorflow"
        assert Backend.MINDSPORE.value == "mindspore"

    def test_backend_from_string(self) -> None:
        """Test creating Backend from string."""
        assert Backend("torch") == Backend.TORCH
        assert Backend("tensorflow") == Backend.TENSORFLOW
        assert Backend("mindspore") == Backend.MINDSPORE

    def test_backend_invalid_raises_error(self) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError):
            Backend("invalid_backend")


class TestParadigmEnum:
    """Tests for Paradigm enum."""

    def test_paradigm_values(self) -> None:
        """Test Paradigm enum values."""
        assert Paradigm.SINGLE_AGENT.value == "single_agent"
        assert Paradigm.MULTI_AGENT.value == "multi_agent"

    def test_paradigm_from_string(self) -> None:
        """Test creating Paradigm from string."""
        assert Paradigm("single_agent") == Paradigm.SINGLE_AGENT
        assert Paradigm("multi_agent") == Paradigm.MULTI_AGENT

    def test_paradigm_invalid_raises_error(self) -> None:
        """Test that invalid paradigm raises ValueError."""
        with pytest.raises(ValueError):
            Paradigm("invalid_paradigm")


# =============================================================================
# AlgorithmInfo Tests
# =============================================================================


class TestAlgorithmInfo:
    """Tests for AlgorithmInfo dataclass."""

    def test_algorithm_info_creation(self) -> None:
        """Test creating AlgorithmInfo."""
        info = AlgorithmInfo(
            key="PPO_Clip",
            display_name="PPO (Clip)",
            category="Policy Optimization",
            paradigm=Paradigm.SINGLE_AGENT,
            description="Proximal Policy Optimization",
        )

        assert info.key == "PPO_Clip"
        assert info.display_name == "PPO (Clip)"
        assert info.category == "Policy Optimization"
        assert info.paradigm == Paradigm.SINGLE_AGENT
        assert info.description == "Proximal Policy Optimization"

    def test_algorithm_info_immutability(self) -> None:
        """Test that AlgorithmInfo is frozen."""
        info = AlgorithmInfo(
            key="DQN",
            display_name="DQN",
            category="Value-based",
            paradigm=Paradigm.SINGLE_AGENT,
        )

        with pytest.raises(AttributeError):
            info.key = "modified"  # type: ignore

    def test_algorithm_info_default_description(self) -> None:
        """Test default empty description."""
        info = AlgorithmInfo(
            key="SAC",
            display_name="SAC",
            category="Continuous Control",
            paradigm=Paradigm.SINGLE_AGENT,
        )

        assert info.description == ""


# =============================================================================
# Backend Algorithms Tests
# =============================================================================


class TestBackendAlgorithms:
    """Tests for BACKEND_ALGORITHMS mapping."""

    def test_all_backends_have_algorithms(self) -> None:
        """Test that all backends have algorithm sets."""
        for backend in Backend:
            assert backend in BACKEND_ALGORITHMS
            assert len(BACKEND_ALGORITHMS[backend]) > 0

    def test_pytorch_has_most_algorithms(self) -> None:
        """Test that PyTorch has the most algorithms."""
        torch_count = len(BACKEND_ALGORITHMS[Backend.TORCH])
        tf_count = len(BACKEND_ALGORITHMS[Backend.TENSORFLOW])
        ms_count = len(BACKEND_ALGORITHMS[Backend.MINDSPORE])

        assert torch_count >= tf_count
        assert torch_count >= ms_count

    def test_tensorflow_and_mindspore_equal(self) -> None:
        """Test that TensorFlow and MindSpore have equal algorithm sets."""
        tf_algos = BACKEND_ALGORITHMS[Backend.TENSORFLOW]
        ms_algos = BACKEND_ALGORITHMS[Backend.MINDSPORE]

        assert tf_algos == ms_algos

    def test_core_algorithms_in_all_backends(self) -> None:
        """Test that core algorithms are available in all backends."""
        core_algorithms = ["DQN", "PPO_Clip", "SAC", "MAPPO", "QMIX"]

        for algo in core_algorithms:
            for backend in Backend:
                assert algo in BACKEND_ALGORITHMS[backend], (
                    f"{algo} not in {backend.value}"
                )


# =============================================================================
# get_algorithms_for_backend Tests
# =============================================================================


class TestGetAlgorithmsForBackend:
    """Tests for get_algorithms_for_backend function."""

    def test_torch_backend(self) -> None:
        """Test getting algorithms for PyTorch backend."""
        algos = get_algorithms_for_backend("torch")

        assert isinstance(algos, frozenset)
        assert "PPO_Clip" in algos
        assert "DQN" in algos
        assert "DreamerV3" in algos  # PyTorch-only

    def test_tensorflow_backend(self) -> None:
        """Test getting algorithms for TensorFlow backend."""
        algos = get_algorithms_for_backend("tensorflow")

        assert isinstance(algos, frozenset)
        assert "PPO_Clip" in algos
        assert "DQN" in algos
        # PyTorch-only algorithms should NOT be in TensorFlow
        assert "DreamerV3" not in algos

    def test_mindspore_backend(self) -> None:
        """Test getting algorithms for MindSpore backend."""
        algos = get_algorithms_for_backend("mindspore")

        assert isinstance(algos, frozenset)
        assert "PPO_Clip" in algos

    def test_backend_enum_input(self) -> None:
        """Test using Backend enum as input."""
        algos = get_algorithms_for_backend(Backend.TORCH)

        assert isinstance(algos, frozenset)
        assert len(algos) > 0


# =============================================================================
# get_algorithms_for_paradigm Tests
# =============================================================================


class TestGetAlgorithmsForParadigm:
    """Tests for get_algorithms_for_paradigm function."""

    def test_single_agent_paradigm(self) -> None:
        """Test getting single-agent algorithms."""
        algos = get_algorithms_for_paradigm("single_agent")

        assert isinstance(algos, frozenset)
        assert "DQN" in algos
        assert "PPO_Clip" in algos
        assert "SAC" in algos
        # Multi-agent algorithms should NOT be here
        assert "MAPPO" not in algos
        assert "QMIX" not in algos

    def test_multi_agent_paradigm(self) -> None:
        """Test getting multi-agent algorithms."""
        algos = get_algorithms_for_paradigm("multi_agent")

        assert isinstance(algos, frozenset)
        assert "MAPPO" in algos
        assert "QMIX" in algos
        assert "MADDPG" in algos
        # Single-agent algorithms should NOT be here
        assert "DQN" not in algos
        assert "PPO_Clip" not in algos

    def test_paradigm_enum_input(self) -> None:
        """Test using Paradigm enum as input."""
        algos = get_algorithms_for_paradigm(Paradigm.SINGLE_AGENT)

        assert isinstance(algos, frozenset)
        assert len(algos) > 0


# =============================================================================
# get_algorithms Tests
# =============================================================================


class TestGetAlgorithms:
    """Tests for get_algorithms function (combined filtering)."""

    def test_torch_single_agent(self) -> None:
        """Test PyTorch + single-agent filtering."""
        algos = get_algorithms("torch", "single_agent")

        assert "PPO_Clip" in algos
        assert "DQN" in algos
        assert "DreamerV3" in algos  # PyTorch-only
        # Multi-agent should NOT be here
        assert "MAPPO" not in algos

    def test_torch_multi_agent(self) -> None:
        """Test PyTorch + multi-agent filtering."""
        algos = get_algorithms("torch", "multi_agent")

        assert "MAPPO" in algos
        assert "QMIX" in algos
        assert "CommNet" in algos  # PyTorch-only
        # Single-agent should NOT be here
        assert "DQN" not in algos

    def test_tensorflow_single_agent(self) -> None:
        """Test TensorFlow + single-agent filtering."""
        algos = get_algorithms("tensorflow", "single_agent")

        assert "PPO_Clip" in algos
        assert "DQN" in algos
        # PyTorch-only should NOT be here
        assert "DreamerV3" not in algos

    def test_tensorflow_multi_agent(self) -> None:
        """Test TensorFlow + multi-agent filtering."""
        algos = get_algorithms("tensorflow", "multi_agent")

        assert "MAPPO" in algos
        assert "QMIX" in algos
        # PyTorch-only should NOT be here
        assert "CommNet" not in algos

    def test_enum_inputs(self) -> None:
        """Test using enum inputs."""
        algos = get_algorithms(Backend.TORCH, Paradigm.SINGLE_AGENT)

        assert isinstance(algos, frozenset)
        assert len(algos) > 0


# =============================================================================
# get_algorithm_info Tests
# =============================================================================


class TestGetAlgorithmInfo:
    """Tests for get_algorithm_info function."""

    def test_existing_algorithm(self) -> None:
        """Test getting info for existing algorithm."""
        info = get_algorithm_info("PPO_Clip")

        assert info is not None
        assert info.key == "PPO_Clip"
        assert info.display_name == "PPO (Clip)"
        assert info.paradigm == Paradigm.SINGLE_AGENT

    def test_dqn_algorithm(self) -> None:
        """Test getting info for DQN."""
        info = get_algorithm_info("DQN")

        assert info is not None
        assert info.key == "DQN"
        assert info.category == "Value-based"

    def test_mappo_algorithm(self) -> None:
        """Test getting info for MAPPO."""
        info = get_algorithm_info("MAPPO")

        assert info is not None
        assert info.paradigm == Paradigm.MULTI_AGENT
        assert info.category == "Centralized"

    def test_nonexistent_algorithm(self) -> None:
        """Test getting info for nonexistent algorithm."""
        info = get_algorithm_info("FAKE_ALGORITHM")

        assert info is None

    def test_all_algorithms_have_info(self) -> None:
        """Test that all registered algorithms have info."""
        for backend in Backend:
            for algo in BACKEND_ALGORITHMS[backend]:
                info = get_algorithm_info(algo)
                assert info is not None, f"No info for {algo}"


# =============================================================================
# get_algorithm_choices Tests
# =============================================================================


class TestGetAlgorithmChoices:
    """Tests for get_algorithm_choices function."""

    def test_returns_list_of_tuples(self) -> None:
        """Test that function returns list of tuples."""
        choices = get_algorithm_choices("torch", "single_agent")

        assert isinstance(choices, list)
        assert all(isinstance(c, tuple) for c in choices)
        assert all(len(c) == 2 for c in choices)

    def test_tuple_format(self) -> None:
        """Test that tuples are (key, display_name)."""
        choices = get_algorithm_choices("torch", "single_agent")

        # Find PPO
        ppo_choices = [c for c in choices if c[0] == "PPO_Clip"]
        assert len(ppo_choices) == 1
        assert ppo_choices[0] == ("PPO_Clip", "PPO (Clip)")

    def test_choices_are_sorted(self) -> None:
        """Test that choices are sorted by display name."""
        choices = get_algorithm_choices("torch", "single_agent")

        display_names = [c[1] for c in choices]
        assert display_names == sorted(display_names)

    def test_multi_agent_choices(self) -> None:
        """Test getting multi-agent choices."""
        choices = get_algorithm_choices("torch", "multi_agent")

        keys = [c[0] for c in choices]
        assert "MAPPO" in keys
        assert "QMIX" in keys
        # Single-agent should NOT be here
        assert "DQN" not in keys


# =============================================================================
# get_algorithms_by_category Tests
# =============================================================================


class TestGetAlgorithmsByCategory:
    """Tests for get_algorithms_by_category function."""

    def test_returns_dict_of_lists(self) -> None:
        """Test that function returns dict of lists."""
        categories = get_algorithms_by_category("torch", "single_agent")

        assert isinstance(categories, dict)
        assert all(isinstance(v, list) for v in categories.values())

    def test_single_agent_categories(self) -> None:
        """Test single-agent category structure."""
        categories = get_algorithms_by_category("torch", "single_agent")

        expected_categories = [
            "Policy Gradient",
            "Policy Optimization",
            "Value-based",
            "Continuous Control",
        ]

        for cat in expected_categories:
            assert cat in categories, f"Category {cat} not found"

    def test_multi_agent_categories(self) -> None:
        """Test multi-agent category structure."""
        categories = get_algorithms_by_category("torch", "multi_agent")

        expected_categories = [
            "Independent",
            "Value Decomposition",
            "Centralized",
        ]

        for cat in expected_categories:
            assert cat in categories, f"Category {cat} not found"

    def test_algorithms_sorted_within_category(self) -> None:
        """Test that algorithms are sorted within each category."""
        categories = get_algorithms_by_category("torch", "single_agent")

        for cat_algos in categories.values():
            display_names = [a.display_name for a in cat_algos]
            assert display_names == sorted(display_names)


# =============================================================================
# is_algorithm_available Tests
# =============================================================================


class TestIsAlgorithmAvailable:
    """Tests for is_algorithm_available function."""

    def test_ppo_available_all_backends(self) -> None:
        """Test that PPO is available in all backends."""
        for backend in Backend:
            assert is_algorithm_available("PPO_Clip", backend)

    def test_dreamer_only_torch(self) -> None:
        """Test that Dreamer is only available in PyTorch."""
        assert is_algorithm_available("DreamerV3", "torch")
        assert not is_algorithm_available("DreamerV3", "tensorflow")
        assert not is_algorithm_available("DreamerV3", "mindspore")

    def test_commnet_only_torch(self) -> None:
        """Test that CommNet is only available in PyTorch."""
        assert is_algorithm_available("CommNet", "torch")
        assert not is_algorithm_available("CommNet", "tensorflow")
        assert not is_algorithm_available("CommNet", "mindspore")

    def test_fake_algorithm_not_available(self) -> None:
        """Test that fake algorithms are not available."""
        assert not is_algorithm_available("FAKE_ALGO", "torch")
        assert not is_algorithm_available("FAKE_ALGO", "tensorflow")


# =============================================================================
# get_backend_summary Tests
# =============================================================================


class TestGetBackendSummary:
    """Tests for get_backend_summary function."""

    def test_summary_structure(self) -> None:
        """Test summary structure."""
        summary = get_backend_summary()

        assert isinstance(summary, dict)
        assert "torch" in summary
        assert "tensorflow" in summary
        assert "mindspore" in summary

    def test_summary_keys(self) -> None:
        """Test that each backend has expected keys."""
        summary = get_backend_summary()

        for backend_summary in summary.values():
            assert "single_agent" in backend_summary
            assert "multi_agent" in backend_summary
            assert "total" in backend_summary

    def test_summary_positive_counts(self) -> None:
        """Test that all counts are positive."""
        summary = get_backend_summary()

        for backend_summary in summary.values():
            assert backend_summary["single_agent"] > 0
            assert backend_summary["multi_agent"] > 0
            assert backend_summary["total"] > 0

    def test_total_equals_sum(self) -> None:
        """Test that total equals sum of single + multi agent."""
        summary = get_backend_summary()

        for backend_summary in summary.values():
            expected_total = (
                backend_summary["single_agent"] + backend_summary["multi_agent"]
            )
            assert backend_summary["total"] == expected_total

    def test_torch_has_most(self) -> None:
        """Test that PyTorch has the most algorithms."""
        summary = get_backend_summary()

        torch_total = summary["torch"]["total"]
        tf_total = summary["tensorflow"]["total"]
        ms_total = summary["mindspore"]["total"]

        assert torch_total >= tf_total
        assert torch_total >= ms_total


# =============================================================================
# Coverage Tests
# =============================================================================


class TestAlgorithmCoverage:
    """Tests to ensure expected algorithms are registered."""

    @pytest.mark.parametrize(
        "algorithm",
        [
            "DQN", "DDQN", "Duel_DQN", "NoisyDQN", "PerDQN",
            "C51DQN", "QRDQN", "DRQN",
            "PG", "A2C", "PPO_Clip", "PPO_KL", "PPG",
            "DDPG", "TD3", "SAC",
            "PDQN", "MPDQN", "SPDQN",
        ],
    )
    def test_core_single_agent_algorithms(self, algorithm: str) -> None:
        """Test that core single-agent algorithms are registered."""
        info = get_algorithm_info(algorithm)
        assert info is not None, f"{algorithm} not registered"
        assert info.paradigm == Paradigm.SINGLE_AGENT

    @pytest.mark.parametrize(
        "algorithm",
        [
            "IQL", "IAC", "IPPO", "IDDPG", "ISAC",
            "VDN", "QMIX", "CWQMIX", "OWQMIX",
            "QTRAN_base", "QTRAN_alt", "DCG", "DCG_S",
            "VDAC", "COMA", "MADDPG", "MAPPO", "MASAC", "MATD3",
            "MFQ", "MFAC",
        ],
    )
    def test_core_multi_agent_algorithms(self, algorithm: str) -> None:
        """Test that core multi-agent algorithms are registered."""
        info = get_algorithm_info(algorithm)
        assert info is not None, f"{algorithm} not registered"
        assert info.paradigm == Paradigm.MULTI_AGENT

    @pytest.mark.parametrize(
        "algorithm",
        ["NPG", "DreamerV2", "DreamerV3", "CURL", "SPR", "DrQ", "TD3BC"],
    )
    def test_torch_only_single_agent(self, algorithm: str) -> None:
        """Test that PyTorch-only single-agent algorithms are registered."""
        info = get_algorithm_info(algorithm)
        assert info is not None, f"{algorithm} not registered"
        assert is_algorithm_available(algorithm, "torch")
        assert not is_algorithm_available(algorithm, "tensorflow")

    @pytest.mark.parametrize(
        "algorithm",
        ["CommNet", "IC3Net", "TarMAC"],
    )
    def test_torch_only_multi_agent(self, algorithm: str) -> None:
        """Test that PyTorch-only multi-agent algorithms are registered."""
        info = get_algorithm_info(algorithm)
        assert info is not None, f"{algorithm} not registered"
        assert is_algorithm_available(algorithm, "torch")
        assert not is_algorithm_available(algorithm, "tensorflow")
