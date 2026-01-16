"""Tests for MiniGrid/BabyAI integration with MOSAIC LLM Worker.

Tests the description generation, prompt formatting, and action parsing
for single-agent MiniGrid environments.

Run with: pytest tests/test_minigrid_integration.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any


# =============================================================================
# Test BabyAI Description Generation
# =============================================================================

class TestBabyAIDescriptions:
    """Test the BabyAI observation description generator."""

    def test_generate_descriptions_with_goal(self):
        """Test that descriptions are generated for a goal in view."""
        from llm_worker.env_utils import generate_babyai_descriptions, BABYAI_OBJECT_NAMES

        # Create a mock observation with a goal visible
        # MiniGrid obs image is (7, 7, 3) - [obj_type, color, state]
        image = np.zeros((7, 7, 3), dtype=np.uint8)

        # Place a green goal (type=7, color=1) in front of agent
        # Agent is at (3, 6), so place goal at (3, 4) = 2 steps ahead
        image[3, 4, 0] = 7  # goal
        image[3, 4, 1] = 1  # green

        obs = {"image": image, "mission": "get to the green goal square"}

        descriptions = generate_babyai_descriptions(obs)

        assert len(descriptions) >= 1
        assert any("goal" in d.lower() for d in descriptions)

    def test_generate_descriptions_with_key(self):
        """Test that descriptions are generated for a key in view."""
        from llm_worker.env_utils import generate_babyai_descriptions

        image = np.zeros((7, 7, 3), dtype=np.uint8)

        # Place a red key (type=4, color=0) to the left of agent
        # Agent at (3, 6), so (2, 5) is 1 step left and 1 step ahead
        image[2, 5, 0] = 4  # key
        image[2, 5, 1] = 0  # red

        obs = {"image": image, "mission": "pick up the red key"}

        descriptions = generate_babyai_descriptions(obs)

        assert len(descriptions) >= 1
        assert any("key" in d.lower() for d in descriptions)

    def test_generate_descriptions_empty_room(self):
        """Test descriptions when nothing interesting is in view."""
        from llm_worker.env_utils import generate_babyai_descriptions

        # Empty room - only walls and floor
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        image[:, :, 0] = 2  # floor everywhere

        obs = {"image": image, "mission": "get to the green goal square"}

        descriptions = generate_babyai_descriptions(obs)

        assert descriptions == ["You see nothing special."]

    def test_generate_descriptions_no_image(self):
        """Test fallback when image is missing."""
        from llm_worker.env_utils import generate_babyai_descriptions

        obs = {"mission": "get to the green goal square"}

        descriptions = generate_babyai_descriptions(obs)

        assert descriptions == ["You see nothing special."]


# =============================================================================
# Test BabyAI Description Wrapper
# =============================================================================

class TestBabyAIDescriptionWrapper:
    """Test the BabyAI environment wrapper."""

    @pytest.fixture
    def minigrid_env(self):
        """Create a MiniGrid environment for testing."""
        import gymnasium as gym
        try:
            import minigrid
            minigrid.register_minigrid_envs()
        except ImportError:
            pytest.skip("MiniGrid not installed")

        env = gym.make("MiniGrid-Empty-8x8-v0", render_mode=None)
        yield env
        env.close()

    def test_wrapper_adds_descriptions_on_reset(self, minigrid_env):
        """Test that wrapper adds descriptions to info on reset."""
        from llm_worker.env_utils import BabyAIDescriptionWrapper

        wrapped_env = BabyAIDescriptionWrapper(minigrid_env)
        obs, info = wrapped_env.reset(seed=42)

        assert "descriptions" in info
        assert isinstance(info["descriptions"], list)
        assert len(info["descriptions"]) >= 1

    def test_wrapper_adds_descriptions_on_step(self, minigrid_env):
        """Test that wrapper adds descriptions to info on step."""
        from llm_worker.env_utils import BabyAIDescriptionWrapper

        wrapped_env = BabyAIDescriptionWrapper(minigrid_env)
        obs, info = wrapped_env.reset(seed=42)

        # Take a step (forward = action 2)
        obs, reward, terminated, truncated, info = wrapped_env.step(2)

        assert "descriptions" in info
        assert isinstance(info["descriptions"], list)

    def test_wrapper_preserves_observation(self, minigrid_env):
        """Test that wrapper doesn't modify the observation."""
        from llm_worker.env_utils import BabyAIDescriptionWrapper

        wrapped_env = BabyAIDescriptionWrapper(minigrid_env)
        obs, info = wrapped_env.reset(seed=42)

        assert "image" in obs
        assert "mission" in obs
        assert "direction" in obs


# =============================================================================
# Test make_env Factory
# =============================================================================

class TestMakeEnv:
    """Test the environment factory function."""

    def test_make_minigrid_env(self):
        """Test creating a MiniGrid environment."""
        from llm_worker.env_utils import make_env, BabyAIDescriptionWrapper

        try:
            env = make_env(
                env_name="minigrid",
                task="MiniGrid-Empty-8x8-v0",
                render_mode=None,
            )

            # Should be wrapped with BabyAIDescriptionWrapper
            assert isinstance(env, BabyAIDescriptionWrapper)

            obs, info = env.reset(seed=42)
            assert "descriptions" in info

            env.close()
        except ImportError:
            pytest.skip("MiniGrid not installed")

    def test_make_babyai_env(self):
        """Test creating a BabyAI environment."""
        from llm_worker.env_utils import make_env, BabyAIDescriptionWrapper

        try:
            env = make_env(
                env_name="babyai",
                task="MiniGrid-Empty-5x5-v0",
                render_mode=None,
            )

            assert isinstance(env, BabyAIDescriptionWrapper)
            env.close()
        except ImportError:
            pytest.skip("MiniGrid not installed")


# =============================================================================
# Test BabyAI Prompt Generator
# =============================================================================

class TestBabyAIPromptGenerator:
    """Test the BabyAI prompt generator."""

    def test_system_prompt_generation(self):
        """Test system prompt includes mission and actions."""
        from llm_worker.environments.babyai_text import BabyAIPromptGenerator

        generator = BabyAIPromptGenerator(task="MiniGrid-Empty-8x8-v0")
        generator._mission = "get to the green goal square"

        system_prompt = generator.get_system_prompt()

        assert "green goal" in system_prompt.lower()
        assert "forward" in system_prompt.lower()
        assert "turn left" in system_prompt.lower()

    def test_format_observation_with_descriptions(self):
        """Test observation formatting with descriptions."""
        from llm_worker.environments.babyai_text import BabyAIPromptGenerator

        generator = BabyAIPromptGenerator(task="MiniGrid-Empty-8x8-v0")

        obs = {"image": np.zeros((7, 7, 3)), "mission": "get to the green goal square"}
        info = {"descriptions": ["You see a green goal 2 steps ahead"]}

        formatted = generator.format_observation(obs, agent_id=0, info=info)

        assert "green goal" in formatted.lower()
        assert "2 steps ahead" in formatted.lower()

    def test_format_observation_without_descriptions(self):
        """Test observation formatting falls back gracefully."""
        from llm_worker.environments.babyai_text import BabyAIPromptGenerator

        generator = BabyAIPromptGenerator(task="MiniGrid-Empty-8x8-v0")

        obs = {"image": np.zeros((7, 7, 3)), "mission": "get to the green goal square"}
        info = {}  # No descriptions

        formatted = generator.format_observation(obs, agent_id=0, info=info)

        # Should at least include the mission
        assert "goal" in formatted.lower()


# =============================================================================
# Test Action Parsing
# =============================================================================

class TestActionParsing:
    """Test parsing LLM output to action indices."""

    def test_parse_exact_match(self):
        """Test parsing exact action names."""
        from llm_worker.environments.babyai_text import parse_babyai_action

        assert parse_babyai_action("forward") == 2
        assert parse_babyai_action("turn left") == 0
        assert parse_babyai_action("turn right") == 1
        assert parse_babyai_action("pickup") == 3
        assert parse_babyai_action("drop") == 4
        assert parse_babyai_action("toggle") == 5

    def test_parse_case_insensitive(self):
        """Test case-insensitive parsing."""
        from llm_worker.environments.babyai_text import parse_babyai_action

        assert parse_babyai_action("FORWARD") == 2
        assert parse_babyai_action("Turn Left") == 0
        assert parse_babyai_action("TURN RIGHT") == 1

    def test_parse_with_explanation(self):
        """Test parsing action embedded in explanation."""
        from llm_worker.environments.babyai_text import parse_babyai_action

        # LLM might output explanation with action
        assert parse_babyai_action("I should move forward to reach the goal") == 2
        assert parse_babyai_action("turn left to face the door") == 0

    def test_parse_go_forward_variant(self):
        """Test parsing 'go forward' variant."""
        from llm_worker.environments.babyai_text import parse_babyai_action

        assert parse_babyai_action("go forward") == 2
        assert parse_babyai_action("move forward") == 2

    def test_parse_just_direction(self):
        """Test parsing just 'left' or 'right' without 'turn'."""
        from llm_worker.environments.babyai_text import parse_babyai_action

        assert parse_babyai_action("left") == 0
        assert parse_babyai_action("right") == 1

    def test_parse_default_on_unknown(self):
        """Test that unknown input defaults to forward."""
        from llm_worker.environments.babyai_text import parse_babyai_action

        # Unknown actions should default to forward (2)
        assert parse_babyai_action("xyz123") == 2
        assert parse_babyai_action("") == 2


# =============================================================================
# Test Action Space
# =============================================================================

class TestActionSpace:
    """Test the action space constants."""

    def test_action_space_length(self):
        """Test that action space has 6 actions."""
        from llm_worker.environments.babyai_text import BABYAI_ACTION_SPACE

        assert len(BABYAI_ACTION_SPACE) == 6

    def test_action_descriptions_match(self):
        """Test that descriptions exist for all actions."""
        from llm_worker.environments.babyai_text import (
            BABYAI_ACTION_SPACE,
            BABYAI_ACTION_DESCRIPTIONS,
        )

        for action in BABYAI_ACTION_SPACE:
            assert action in BABYAI_ACTION_DESCRIPTIONS


# =============================================================================
# Integration Test: Full Flow
# =============================================================================

class TestFullIntegration:
    """Integration tests for the full observation -> action flow."""

    @pytest.fixture
    def wrapped_env(self):
        """Create a wrapped MiniGrid environment."""
        from llm_worker.env_utils import make_env

        try:
            env = make_env(
                env_name="minigrid",
                task="MiniGrid-Empty-8x8-v0",
                render_mode=None,
            )
            yield env
            env.close()
        except ImportError:
            pytest.skip("MiniGrid not installed")

    def test_reset_produces_descriptions(self, wrapped_env):
        """Test that reset produces usable descriptions."""
        from llm_worker.environments.babyai_text import BabyAIPromptGenerator

        obs, info = wrapped_env.reset(seed=42)

        # Descriptions should be in info
        assert "descriptions" in info

        # Prompt generator should format them
        generator = BabyAIPromptGenerator(task="MiniGrid-Empty-8x8-v0")
        formatted = generator.format_observation(obs, agent_id=0, info=info)

        # Should have some content
        assert len(formatted) > 10

    def test_step_produces_descriptions(self, wrapped_env):
        """Test that step produces usable descriptions."""
        from llm_worker.environments.babyai_text import BabyAIPromptGenerator

        obs, info = wrapped_env.reset(seed=42)

        # Take a step
        obs, reward, terminated, truncated, info = wrapped_env.step(2)  # forward

        assert "descriptions" in info

        generator = BabyAIPromptGenerator(task="MiniGrid-Empty-8x8-v0")
        formatted = generator.format_observation(obs, agent_id=0, info=info)

        assert len(formatted) > 10

    def test_episode_loop(self, wrapped_env):
        """Test running a simple episode loop."""
        from llm_worker.environments.babyai_text import BabyAIPromptGenerator

        generator = BabyAIPromptGenerator(task="MiniGrid-Empty-8x8-v0")

        obs, info = wrapped_env.reset(seed=42)

        for step in range(10):
            # Format observation
            formatted = generator.format_observation(obs, agent_id=0, info=info)
            assert isinstance(formatted, str)

            # Simulate LLM output
            llm_output = "forward"
            action = generator.parse_action(llm_output)
            assert 0 <= action < 6

            # Take step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            if terminated or truncated:
                break


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
