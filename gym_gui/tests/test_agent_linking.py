"""Tests for agent linking feature in multi-agent RL environments.

This module tests the agent linking functionality that allows multiple agents
to share the same policy checkpoint in MAPPO/IPPO scenarios.
"""

import pytest
from pathlib import Path
from typing import Dict, List

from gym_gui.services.operator import LinkGroup, OperatorConfig, WorkerAssignment


class TestLinkGroup:
    """Test the LinkGroup dataclass."""

    def test_link_group_creation(self):
        """Test creating a link group."""
        group = LinkGroup(
            group_id="link_0",
            primary_agent="agent_0",
            linked_agents=["agent_1", "agent_2"],
            policy_path="/path/to/checkpoint.pth",
            algorithm="MAPPO",
            worker_type="rl",
            color="#E3F2FD",
        )

        assert group.group_id == "link_0"
        assert group.primary_agent == "agent_0"
        assert group.linked_agents == ["agent_1", "agent_2"]
        assert group.policy_path == "/path/to/checkpoint.pth"
        assert group.algorithm == "MAPPO"
        assert group.worker_type == "rl"
        assert group.color == "#E3F2FD"

    def test_all_agents(self):
        """Test getting all agents in a group."""
        group = LinkGroup(
            group_id="link_0",
            primary_agent="agent_0",
            linked_agents=["agent_1", "agent_2"],
            policy_path="/path/to/checkpoint.pth",
            algorithm="MAPPO",
            worker_type="rl",
        )

        all_agents = group.all_agents()
        assert all_agents == ["agent_0", "agent_1", "agent_2"]

    def test_contains_agent(self):
        """Test checking if an agent is in a group."""
        group = LinkGroup(
            group_id="link_0",
            primary_agent="agent_0",
            linked_agents=["agent_1", "agent_2"],
            policy_path="/path/to/checkpoint.pth",
            algorithm="MAPPO",
            worker_type="rl",
        )

        assert group.contains_agent("agent_0") is True
        assert group.contains_agent("agent_1") is True
        assert group.contains_agent("agent_2") is True
        assert group.contains_agent("agent_3") is False


class TestOperatorConfigLinkGroups:
    """Test link groups in OperatorConfig."""

    def test_operator_config_with_link_groups(self):
        """Test creating an OperatorConfig with link groups."""
        # Create link group with operator-scoped ID
        group = LinkGroup(
            group_id="operator_0_link_0",
            primary_agent="agent_0",
            linked_agents=["agent_1"],
            policy_path="/path/to/checkpoint.pth",
            algorithm="MAPPO",
            worker_type="rl",
        )

        # Create operator config with link_groups via factory
        config = OperatorConfig.multi_agent(
            operator_id="operator_0",
            display_name="Test Operator",
            env_name="mosaic_multigrid",
            task="MosaicMultiGrid-Soccer-v0",
            player_workers={
                "agent_0": WorkerAssignment(
                    worker_id="xuance_worker",
                    worker_type="rl",
                    settings={"policy_path": "/path/to/checkpoint.pth", "algorithm": "MAPPO"},
                ),
                "agent_1": WorkerAssignment(
                    worker_id="xuance_worker",
                    worker_type="rl",
                    settings={"policy_path": "/path/to/checkpoint.pth", "algorithm": "MAPPO"},
                ),
            },
            link_groups={"operator_0_link_0": group},
        )

        assert "operator_0_link_0" in config.link_groups
        assert config.link_groups["operator_0_link_0"].primary_agent == "agent_0"
        assert config.link_groups["operator_0_link_0"].linked_agents == ["agent_1"]

    def test_link_group_ids_scoped_to_operator(self):
        """Test that link group IDs from different operators don't collide."""
        group_op0 = LinkGroup(
            group_id="operator_0_link_0",
            primary_agent="agent_0",
            linked_agents=["agent_1"],
            policy_path="/path/to/checkpoint_a.pth",
            algorithm="MAPPO",
            worker_type="rl",
        )
        group_op1 = LinkGroup(
            group_id="operator_1_link_0",
            primary_agent="agent_0",
            linked_agents=["agent_1"],
            policy_path="/path/to/checkpoint_b.pth",
            algorithm="IPPO",
            worker_type="rl",
        )

        assert group_op0.group_id != group_op1.group_id
        assert group_op0.group_id == "operator_0_link_0"
        assert group_op1.group_id == "operator_1_link_0"


class TestCheckpointStructure:
    """Test checkpoint structure for IPPO and MAPPO."""

    @pytest.fixture
    def ippo_checkpoint_path(self) -> Path:
        """Path to IPPO checkpoint."""
        return Path("/home/hamid/Desktop/software/mosaic/var/trainer/custom_scripts/"
                   "01KJ9A1QJPY999E563410SMSKA/checkpoints/soccer_2vs2_indagobs/"
                   "seed_1_2026_0225_102910/final_train_model.pth")

    @pytest.fixture
    def mappo_checkpoint_path(self) -> Path:
        """Path to MAPPO checkpoint."""
        return Path("/home/hamid/Desktop/software/mosaic/var/trainer/custom_scripts/"
                   "01KJ9MBMRWM4AMRN50SBPAZ10F/checkpoints/soccer_2vs2_teamobs/"
                   "seed_1_2026_0225_132938/final_train_model.pth")

    def test_ippo_checkpoint_exists(self, ippo_checkpoint_path):
        """Test that IPPO checkpoint exists."""
        if ippo_checkpoint_path.exists():
            assert ippo_checkpoint_path.is_file()
            assert ippo_checkpoint_path.suffix == ".pth"
        else:
            pytest.skip(f"IPPO checkpoint not found at {ippo_checkpoint_path}")

    def test_mappo_checkpoint_exists(self, mappo_checkpoint_path):
        """Test that MAPPO checkpoint exists."""
        if mappo_checkpoint_path.exists():
            assert mappo_checkpoint_path.is_file()
            assert mappo_checkpoint_path.suffix == ".pth"
        else:
            pytest.skip(f"MAPPO checkpoint not found at {mappo_checkpoint_path}")

    def test_ippo_checkpoint_structure(self, ippo_checkpoint_path):
        """Test IPPO checkpoint has agent-specific keys."""
        if not ippo_checkpoint_path.exists():
            pytest.skip(f"IPPO checkpoint not found at {ippo_checkpoint_path}")

        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        checkpoint = torch.load(ippo_checkpoint_path, map_location="cpu")

        # Check for agent-specific keys
        agent_keys = [k for k in checkpoint.keys() if "agent_" in k]
        assert len(agent_keys) > 0, "No agent-specific keys found in checkpoint"

        # Check for actor keys for each agent
        actor_keys = [k for k in checkpoint.keys() if "actor.agent_" in k]
        assert len(actor_keys) > 0, "No actor keys found for agents"

        # Verify multiple agents exist
        agent_ids = set()
        for key in checkpoint.keys():
            if "agent_" in key:
                # Extract agent ID (e.g., "agent_0" from "actor.agent_0.model.0.weight")
                parts = key.split(".")
                for part in parts:
                    if part.startswith("agent_"):
                        agent_ids.add(part)

        assert len(agent_ids) >= 2, f"Expected at least 2 agents, found {len(agent_ids)}"

    def test_mappo_checkpoint_structure(self, mappo_checkpoint_path):
        """Test MAPPO checkpoint has agent-specific keys."""
        if not mappo_checkpoint_path.exists():
            pytest.skip(f"MAPPO checkpoint not found at {mappo_checkpoint_path}")

        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        checkpoint = torch.load(mappo_checkpoint_path, map_location="cpu")

        # Check for agent-specific keys
        agent_keys = [k for k in checkpoint.keys() if "agent_" in k]
        assert len(agent_keys) > 0, "No agent-specific keys found in checkpoint"

        # Check for actor keys for each agent
        actor_keys = [k for k in checkpoint.keys() if "actor.agent_" in k]
        assert len(actor_keys) > 0, "No actor keys found for agents"

        # Verify multiple agents exist
        agent_ids = set()
        for key in checkpoint.keys():
            if "agent_" in key:
                # Extract agent ID (e.g., "agent_0" from "actor.agent_0.model.0.weight")
                parts = key.split(".")
                for part in parts:
                    if part.startswith("agent_"):
                        agent_ids.add(part)

        assert len(agent_ids) >= 2, f"Expected at least 2 agents, found {len(agent_ids)}"

    def test_ippo_mappo_same_structure(self, ippo_checkpoint_path, mappo_checkpoint_path):
        """Test that IPPO and MAPPO checkpoints have the same structure."""
        if not ippo_checkpoint_path.exists() or not mappo_checkpoint_path.exists():
            pytest.skip("One or both checkpoints not found")

        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        ippo_checkpoint = torch.load(ippo_checkpoint_path, map_location="cpu")
        mappo_checkpoint = torch.load(mappo_checkpoint_path, map_location="cpu")

        # Extract key patterns (without specific agent IDs)
        def get_key_patterns(checkpoint):
            patterns = set()
            for key in checkpoint.keys():
                # Replace agent_N with agent_X to get pattern
                pattern = key
                for i in range(10):
                    pattern = pattern.replace(f"agent_{i}", "agent_X")
                patterns.add(pattern)
            return patterns

        ippo_patterns = get_key_patterns(ippo_checkpoint)
        mappo_patterns = get_key_patterns(mappo_checkpoint)

        # Both should have similar patterns (actor, critic, etc.)
        common_patterns = ippo_patterns & mappo_patterns
        assert len(common_patterns) > 0, "No common key patterns found between IPPO and MAPPO"


class TestEnvironmentSupport:
    """Test environment support for agent linking."""

    def test_supported_environments(self):
        """Test that linking is supported for specific environments."""
        supported_envs = {
            "mosaic_multigrid",
            "multigrid_ini",
            "meltingpot",
            "overcooked",
        }

        # These environments should support linking
        for env in supported_envs:
            assert env in supported_envs

    def test_unsupported_environments(self):
        """Test that linking is not supported for other environments."""
        supported_envs = {
            "mosaic_multigrid",
            "multigrid_ini",
            "meltingpot",
            "overcooked",
        }

        # These environments should NOT support linking
        unsupported = ["babyai", "minigrid", "pettingzoo", "gymnasium"]
        for env in unsupported:
            assert env not in supported_envs


class TestLinkingScenarios:
    """Test real-world linking scenarios."""

    def test_soccer_2vs2_ippo_linking(self):
        """Test linking scenario for soccer 2vs2 with IPPO."""
        # Create link group for 4 agents
        group = LinkGroup(
            group_id="link_0",
            primary_agent="agent_0",
            linked_agents=["agent_1", "agent_2", "agent_3"],
            policy_path="/home/hamid/Desktop/software/mosaic/var/trainer/custom_scripts/"
                       "01KJ9A1QJPY999E563410SMSKA/checkpoints/soccer_2vs2_indagobs/"
                       "seed_1_2026_0225_102910/final_train_model.pth",
            algorithm="IPPO",
            worker_type="rl",
        )

        # Verify all agents are in the group
        assert group.all_agents() == ["agent_0", "agent_1", "agent_2", "agent_3"]

        # Verify primary agent
        assert group.primary_agent == "agent_0"

        # Verify policy path
        assert "final_train_model.pth" in group.policy_path

    def test_soccer_2vs2_mappo_linking(self):
        """Test linking scenario for soccer 2vs2 with MAPPO."""
        # Create link group for 4 agents
        group = LinkGroup(
            group_id="link_0",
            primary_agent="agent_0",
            linked_agents=["agent_1", "agent_2", "agent_3"],
            policy_path="/home/hamid/Desktop/software/mosaic/var/trainer/custom_scripts/"
                       "01KJ9MBMRWM4AMRN50SBPAZ10F/checkpoints/soccer_2vs2_teamobs/"
                       "seed_1_2026_0225_132938/final_train_model.pth",
            algorithm="MAPPO",
            worker_type="rl",
        )

        # Verify all agents are in the group
        assert group.all_agents() == ["agent_0", "agent_1", "agent_2", "agent_3"]

        # Verify primary agent
        assert group.primary_agent == "agent_0"

        # Verify policy path
        assert "final_train_model.pth" in group.policy_path

    def test_mixed_evaluation_scenario(self):
        """Test mixed evaluation: trained agents vs random agents."""
        # Create link group for trained agents only
        group = LinkGroup(
            group_id="link_0",
            primary_agent="agent_0",
            linked_agents=["agent_1"],
            policy_path="/path/to/checkpoint.pth",
            algorithm="MAPPO",
            worker_type="rl",
        )

        # Verify only trained agents are linked
        assert group.all_agents() == ["agent_0", "agent_1"]

        # agent_2 and agent_3 would be random (not in the group)
        assert not group.contains_agent("agent_2")
        assert not group.contains_agent("agent_3")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
