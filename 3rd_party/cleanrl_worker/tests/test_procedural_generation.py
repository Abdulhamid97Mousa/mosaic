"""
Test procedural generation wrapper functionality.

This test verifies that:
1. Procedural mode creates different levels each episode
2. Fixed mode creates the same level each episode
3. The toggle works correctly
"""

import pytest
import gymnasium as gym
import minigrid
from cleanrl_worker.wrappers.procedural_generation import ProceduralGenerationWrapper

# Register MiniGrid environments
minigrid.register_minigrid_envs()


class TestProceduralGenerationWrapper:
    """Test suite for ProceduralGenerationWrapper."""

    def test_procedural_mode_creates_different_levels(self):
        """Test that procedural mode generates different layouts each episode."""
        env = gym.make("BabyAI-GoToLocal-v0")
        env = ProceduralGenerationWrapper(env, procedural=True, fixed_seed=42)

        layouts = []
        for _ in range(5):
            obs, info = env.reset()
            assert info['procedural_generation'] is True
            assert 'episode_seed' in info
            layout = {
                'agent_pos': env.unwrapped.agent_pos,
                'mission': env.unwrapped.mission,
                'seed': info['episode_seed']
            }
            layouts.append(layout)

        # Check that we got different layouts
        agent_positions = [layout['agent_pos'] for layout in layouts]
        missions = [layout['mission'] for layout in layouts]
        seeds = [layout['seed'] for layout in layouts]

        # All seeds should be different
        assert len(set(seeds)) == 5, "All episode seeds should be unique"

        # Most layouts should be different (allow for small chance of collision)
        unique_positions = len(set(agent_positions))
        assert unique_positions >= 3, f"Expected at least 3 unique positions, got {unique_positions}"

        env.close()

    def test_fixed_mode_creates_same_level(self):
        """Test that fixed mode generates identical layouts each episode."""
        env = gym.make("BabyAI-GoToLocal-v0")
        env = ProceduralGenerationWrapper(env, procedural=False, fixed_seed=42)

        layouts = []
        for _ in range(5):
            obs, info = env.reset()
            assert info['procedural_generation'] is False
            assert info['fixed_seed'] == 42
            layout = {
                'agent_pos': env.unwrapped.agent_pos,
                'agent_dir': env.unwrapped.agent_dir,
                'mission': env.unwrapped.mission,
            }
            layouts.append(layout)

        # Check that all layouts are identical
        first_layout = layouts[0]
        for layout in layouts[1:]:
            assert layout['agent_pos'] == first_layout['agent_pos'], "Agent positions should be identical"
            assert layout['agent_dir'] == first_layout['agent_dir'], "Agent directions should be identical"
            assert layout['mission'] == first_layout['mission'], "Missions should be identical"

        env.close()

    def test_procedural_vs_fixed_difference(self):
        """Test that procedural and fixed modes behave differently."""
        # Procedural mode
        env_proc = gym.make("BabyAI-GoToLocal-v0")
        env_proc = ProceduralGenerationWrapper(env_proc, procedural=True, fixed_seed=42)

        proc_layouts = []
        for _ in range(3):
            obs, info = env_proc.reset()
            proc_layouts.append((env_proc.unwrapped.agent_pos, env_proc.unwrapped.mission))

        env_proc.close()

        # Fixed mode
        env_fixed = gym.make("BabyAI-GoToLocal-v0")
        env_fixed = ProceduralGenerationWrapper(env_fixed, procedural=False, fixed_seed=42)

        fixed_layouts = []
        for _ in range(3):
            obs, info = env_fixed.reset()
            fixed_layouts.append((env_fixed.unwrapped.agent_pos, env_fixed.unwrapped.mission))

        env_fixed.close()

        # Procedural should have variety
        unique_proc = len(set(proc_layouts))
        assert unique_proc > 1, f"Procedural mode should create variety, got {unique_proc} unique layouts"

        # Fixed should be identical
        unique_fixed = len(set(fixed_layouts))
        assert unique_fixed == 1, f"Fixed mode should be identical, got {unique_fixed} unique layouts"

    def test_episode_count_increments(self):
        """Test that episode count increments correctly."""
        env = gym.make("BabyAI-GoToLocal-v0")
        env = ProceduralGenerationWrapper(env, procedural=True, fixed_seed=42)

        for expected_episode in range(5):
            obs, info = env.reset()
            assert info['episode_number'] == expected_episode, \
                f"Expected episode {expected_episode}, got {info['episode_number']}"

        env.close()

    def test_different_fixed_seeds_create_different_levels(self):
        """Test that different fixed seeds create different fixed levels."""
        layouts_seed_1 = []
        env1 = gym.make("BabyAI-GoToLocal-v0")
        env1 = ProceduralGenerationWrapper(env1, procedural=False, fixed_seed=1)
        for _ in range(2):
            obs, info = env1.reset()
            layouts_seed_1.append(env1.unwrapped.agent_pos)
        env1.close()

        layouts_seed_2 = []
        env2 = gym.make("BabyAI-GoToLocal-v0")
        env2 = ProceduralGenerationWrapper(env2, procedural=False, fixed_seed=2)
        for _ in range(2):
            obs, info = env2.reset()
            layouts_seed_2.append(env2.unwrapped.agent_pos)
        env2.close()

        # Within same seed, should be identical
        assert layouts_seed_1[0] == layouts_seed_1[1], "Same seed should produce identical layouts"
        assert layouts_seed_2[0] == layouts_seed_2[1], "Same seed should produce identical layouts"

        # Different seeds should (likely) produce different layouts
        # Note: There's a small chance they could be the same, but very unlikely
        assert layouts_seed_1[0] != layouts_seed_2[0], "Different seeds should produce different layouts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
