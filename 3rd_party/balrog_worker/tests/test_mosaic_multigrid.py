"""Tests for MOSAIC MultiGrid Extension.

Tests the novel MOSAIC contributions:
1. 3-level coordination prompts
2. 2-mode Theory of Mind observations
3. Integration with MultiGrid environments
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mosaic_extension.multigrid import prompts, observations


class TestMultiGridPrompts:
    """Test prompt generation for all coordination levels."""

    def test_level1_soccer_prompt(self):
        """Test Level 1 (Emergent) prompt for Soccer."""
        prompt = prompts.get_instruction_prompt_level1(
            agent_id=0, team=0, env_id="MultiGrid-Soccer-v0"
        )

        assert "Agent 0" in prompt
        assert "Red" in prompt  # Team 0 is Red
        assert "2v2 soccer" in prompt
        assert "pickup" in prompt  # Correct action name
        assert "left" in prompt  # Check action is mentioned
        # Should NOT have coordination tips (Level 1)
        assert "Spread out" not in prompt
        assert "FORWARD" not in prompt  # No role assignment
        print("✓ Level 1 Soccer prompt generated")

    def test_level2_soccer_prompt(self):
        """Test Level 2 (Basic Hints) prompt for Soccer."""
        prompt = prompts.get_instruction_prompt_level2(
            agent_id=1, team=0, env_id="MultiGrid-Soccer-v0"
        )

        assert "Agent 1" in prompt
        assert "Red" in prompt
        assert "Tips for Coordination" in prompt
        assert "Spread out" in prompt or "cluster" in prompt
        # Should NOT have role assignment (Level 2)
        assert "FORWARD" not in prompt
        assert "DEFENDER" not in prompt
        print("✓ Level 2 Soccer prompt generated")

    def test_level3_soccer_forward_prompt(self):
        """Test Level 3 (Role-Based) prompt for Soccer - Forward role."""
        prompt = prompts.get_instruction_prompt_level3(
            agent_id=0, team=0, role="forward", env_id="MultiGrid-Soccer-v0"
        )

        assert "Agent 0" in prompt
        assert "FORWARD" in prompt
        assert "Offensive" in prompt or "attacking" in prompt
        assert "Score goals" in prompt or "advance toward opponent goal" in prompt
        # Should mention teammate's role
        assert "DEFENDER" in prompt
        print("✓ Level 3 Soccer Forward prompt generated")

    def test_level3_soccer_defender_prompt(self):
        """Test Level 3 (Role-Based) prompt for Soccer - Defender role."""
        prompt = prompts.get_instruction_prompt_level3(
            agent_id=1, team=0, role="defender", env_id="MultiGrid-Soccer-v0"
        )

        assert "Agent 1" in prompt
        assert "DEFENDER" in prompt
        assert "Defensive" in prompt or "Protect goal" in prompt
        assert "FORWARD" in prompt  # Mentions teammate's role
        print("✓ Level 3 Soccer Defender prompt generated")

    def test_level1_collect_prompt(self):
        """Test Level 1 prompt for Collect environment."""
        prompt = prompts.get_instruction_prompt_level1(
            agent_id=0, team=0, env_id="MultiGrid-Collect-v0"
        )

        assert "Agent 0" in prompt
        assert "Red" in prompt  # Agent 0 is Red
        assert "ball collection" in prompt
        assert "Opponents" in prompt
        print("✓ Level 1 Collect prompt generated")

    def test_action_list_complete(self):
        """Test that all 8 MultiGrid actions are defined with correct names."""
        actions = prompts.MULTIGRID_ACTIONS
        action_space = prompts.MULTIGRID_ACTION_SPACE

        # Check dictionary
        assert len(actions) == 8
        assert "still" in actions
        assert "left" in actions
        assert "right" in actions
        assert "forward" in actions
        assert "pickup" in actions
        assert "drop" in actions
        assert "toggle" in actions
        assert "done" in actions
        print("✓ All 8 MultiGrid actions defined in MULTIGRID_ACTIONS")

        # Check action space list
        assert len(action_space) == 8
        assert action_space == ["still", "left", "right", "forward", "pickup", "drop", "toggle", "done"]
        print("✓ MULTIGRID_ACTION_SPACE matches MultiGrid action indices")

    def test_action_parser(self):
        """Test action parsing from LLM text output."""
        # Test exact matches
        assert prompts.parse_action("still") == 0
        assert prompts.parse_action("left") == 1
        assert prompts.parse_action("right") == 2
        assert prompts.parse_action("forward") == 3
        assert prompts.parse_action("pickup") == 4
        assert prompts.parse_action("drop") == 5
        assert prompts.parse_action("toggle") == 6
        assert prompts.parse_action("done") == 7

        # Test variations
        assert prompts.parse_action("turn left") == 1
        assert prompts.parse_action("turn right") == 2
        assert prompts.parse_action("go forward") == 3
        assert prompts.parse_action("pick up") == 4
        assert prompts.parse_action("wait") == 0

        # Test with full sentences
        assert prompts.parse_action("I will go forward to the ball") == 3
        assert prompts.parse_action("Let me pickup the ball") == 4
        assert prompts.parse_action("I should stay still") == 0

        # Test invalid action defaults to still
        assert prompts.parse_action("invalid action") == 0

        print("✓ Action parser works correctly")


class TestMultiGridObservations:
    """Test observation text conversion for both modes."""

    def create_mock_observation(self):
        """Create a mock MultiGrid observation array."""
        # 7x7 grid with 6 channels
        obs = np.zeros((7, 7, 6), dtype=np.uint8)

        # Place a red ball at (4, 2) - relative to agent at center (3, 3)
        obs[2, 4, 0] = 5  # Object type: ball
        obs[2, 4, 1] = 0  # Color: red

        # Place a green goal at (3, 1)
        obs[1, 3, 0] = 7  # Object type: goal
        obs[1, 3, 1] = 1  # Color: green

        # Place another agent at (5, 3) carrying a ball
        obs[3, 5, 0] = 9  # Object type: agent
        obs[3, 5, 1] = 2  # Color: blue
        obs[3, 5, 3] = 5  # Carrying: ball
        obs[3, 5, 4] = 0  # Carrying color: red
        obs[3, 5, 5] = 0  # Direction: EAST

        return obs

    def test_egocentric_observation(self):
        """Test egocentric observation mode."""
        obs = self.create_mock_observation()

        description = observations.describe_observation_egocentric(
            obs, agent_direction=0, carrying=None
        )

        assert "You see:" in description
        assert "red ball" in description
        assert "green goal" in description
        assert "agent" in description
        assert "You are facing: EAST" in description
        assert "You are carrying: nothing" in description
        # Should NOT have teammate section
        assert "Visible Teammates" not in description
        print("✓ Egocentric observation generated")
        print(f"Sample output:\n{description}\n")

    def test_egocentric_carrying(self):
        """Test egocentric observation when agent is carrying something."""
        obs = self.create_mock_observation()

        description = observations.describe_observation_egocentric(
            obs, agent_direction=2, carrying="red ball"
        )

        assert "You are facing: WEST" in description
        assert "You are carrying: red ball" in description
        print("✓ Egocentric observation with carrying generated")

    def test_visible_teammates_mode(self):
        """Test visible teammates observation mode."""
        obs = self.create_mock_observation()

        # Create mock teammate info
        visible_teammates = [
            {
                "id": 1,
                "position": (5, 4),
                "direction": 1,
                "carrying": "ball",
                "color": "red",
            }
        ]

        description = observations.describe_observation_with_teammates(
            obs,
            agent_id=0,
            visible_teammates=visible_teammates,
            agent_direction=0,
            carrying=None,
        )

        assert "You see:" in description
        assert "Visible Teammates:" in description
        assert "Teammate Agent 1" in description
        assert "red" in description.lower()
        assert "facing SOUTH" in description
        assert "carrying: ball" in description
        print("✓ Visible teammates observation generated")
        print(f"Sample output:\n{description}\n")

    def test_no_visible_teammates(self):
        """Test visible teammates mode when no teammates are visible."""
        obs = self.create_mock_observation()

        description = observations.describe_observation_with_teammates(
            obs,
            agent_id=0,
            visible_teammates=[],
            agent_direction=0,
            carrying=None,
        )

        assert "Visible Teammates: none in view" in description
        print("✓ No visible teammates case handled")


class TestMultiGridIntegration:
    """Test integration with actual MultiGrid environment."""

    @pytest.fixture
    def multigrid_env(self):
        """Create a MultiGrid Soccer environment."""
        try:
            import sys
            from pathlib import Path

            # Add mosaic_multigrid to path
            multigrid_path = Path(__file__).parent.parent.parent.parent / "mosaic_multigrid"
            if multigrid_path.exists():
                sys.path.insert(0, str(multigrid_path))

            from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2

            env = SoccerGame4HEnv10x15N2()
            obs = env.reset()
            return env, obs
        except ImportError as e:
            pytest.skip(f"MultiGrid not available: {e}")

    def test_environment_creation(self, multigrid_env):
        """Test that MultiGrid environment can be created."""
        env, obs = multigrid_env

        assert env is not None
        assert len(obs) == 4  # Soccer has 4 agents
        assert all(isinstance(o, np.ndarray) for o in obs)
        print("✓ MultiGrid Soccer environment created")

    def test_observation_conversion(self, multigrid_env):
        """Test converting actual MultiGrid observations to text."""
        env, obs_list = multigrid_env

        # Test first agent's observation
        agent_obs = obs_list[0]

        # Egocentric mode
        description = observations.describe_observation_egocentric(
            agent_obs, agent_direction=0, carrying=None
        )

        assert isinstance(description, str)
        assert len(description) > 50  # Should have substantial content
        assert "You see:" in description
        print("✓ Real MultiGrid observation converted to text")
        print(f"Real observation sample:\n{description}\n")

    def test_all_agents_observations(self, multigrid_env):
        """Test generating observations for all 4 agents."""
        env, obs_list = multigrid_env

        for agent_id, agent_obs in enumerate(obs_list):
            description = observations.describe_observation_egocentric(
                agent_obs, agent_direction=0, carrying=None
            )
            assert "You see:" in description
            print(f"✓ Agent {agent_id} observation generated")


class TestEnvironmentHelpers:
    """Test helper functions in environments.py."""

    def test_multigrid_description_generation(self):
        """Test generate_multigrid_description helper."""
        from balrog_worker.environments import generate_multigrid_description

        # Create mock observation
        obs = np.zeros((7, 7, 6), dtype=np.uint8)
        obs[2, 4, 0] = 5  # Ball
        obs[2, 4, 1] = 0  # Red

        # Mock environment
        class MockEnv:
            agents = []

        # Egocentric mode
        description = generate_multigrid_description(
            obs, agent_id=0, env=MockEnv(), observation_mode="egocentric"
        )

        assert isinstance(description, str)
        assert "You see:" in description
        assert "Visible Teammates" not in description
        print("✓ generate_multigrid_description works (egocentric)")

        # Visible teammates mode
        description = generate_multigrid_description(
            obs, agent_id=0, env=MockEnv(), observation_mode="visible_teammates"
        )

        assert "Visible Teammates" in description
        print("✓ generate_multigrid_description works (teammates)")

    def test_multigrid_instruction_prompt_helper(self):
        """Test get_multigrid_instruction_prompt helper."""
        from balrog_worker.environments import get_multigrid_instruction_prompt

        # Level 1
        prompt = get_multigrid_instruction_prompt(
            agent_id=0, env_id="MultiGrid-Soccer-v0", coordination_level=1
        )
        assert "Agent 0" in prompt
        assert "Tips for Coordination" not in prompt
        print("✓ get_multigrid_instruction_prompt works (Level 1)")

        # Level 2
        prompt = get_multigrid_instruction_prompt(
            agent_id=0, env_id="MultiGrid-Soccer-v0", coordination_level=2
        )
        assert "Tips for Coordination" in prompt
        print("✓ get_multigrid_instruction_prompt works (Level 2)")

        # Level 3
        prompt = get_multigrid_instruction_prompt(
            agent_id=0, env_id="MultiGrid-Soccer-v0", coordination_level=3, role="forward"
        )
        assert "FORWARD" in prompt
        print("✓ get_multigrid_instruction_prompt works (Level 3)")


def run_tests():
    """Run all tests and print summary."""
    print("\n" + "="*70)
    print("MOSAIC MultiGrid Extension Tests")
    print("="*70 + "\n")

    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print statements
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
