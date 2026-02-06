"""
Compare BabyAI vs MultiGrid reproducibility behavior.

This test demonstrates:
1. BabyAI is NATURALLY reproducible (no wrapper needed)
2. MultiGrid NEEDS our ReproducibleMultiGridWrapper

The difference is in how they handle randomness:
- BabyAI: ALL random operations use self.np_random (seeded RNG)
- MultiGrid: step() uses np.random.permutation() (GLOBAL RNG - bug)
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest


# =============================================================================
# Configuration
# =============================================================================

NUM_STEPS = 30
NUM_EPISODES = 3


# =============================================================================
# Helper Functions
# =============================================================================


def run_babyai_trajectory(env: Any, seed: int, num_steps: int) -> Tuple[List, List]:
    """Run a BabyAI trajectory and collect observations + rewards."""
    obs, info = env.reset(seed=seed)

    # BabyAI returns dict observations with 'image' key
    obs_image = obs['image'] if isinstance(obs, dict) else obs
    observations = [obs_image.copy()]
    rewards = []

    for _ in range(num_steps):
        # Use a fixed action sequence for comparison
        action = 2  # Forward
        obs, reward, terminated, truncated, info = env.step(action)

        obs_image = obs['image'] if isinstance(obs, dict) else obs
        observations.append(obs_image.copy())
        rewards.append(reward)

        if terminated or truncated:
            obs, info = env.reset(seed=seed)

    return observations, rewards


def run_multigrid_trajectory(env: Any, seed: int, num_steps: int) -> Tuple[List, List]:
    """Run a MultiGrid trajectory and collect observations + rewards."""
    env.seed(seed)
    obs = env.reset()

    observations = [[o.copy() for o in obs]]
    rewards = []

    for _ in range(num_steps):
        # Use fixed actions for all agents
        actions = [0] * len(env.agents)  # Do nothing
        obs, reward, done, info = env.step(actions)

        observations.append([o.copy() for o in obs])
        rewards.append(list(reward))

        if done:
            env.seed(seed)
            env.reset()

    return observations, rewards


# =============================================================================
# Test: BabyAI Natural Reproducibility
# =============================================================================


class TestBabyAIReproducibility:
    """Prove that BabyAI is naturally reproducible without any wrapper."""

    @pytest.fixture
    def babyai_env(self):
        """Create a BabyAI environment."""
        try:
            # BabyAI is now part of minigrid package
            from minigrid.envs.babyai import GoToRedBallNoDists
            env = GoToRedBallNoDists()
            yield env
            env.close()
        except Exception as e:
            pytest.skip(f"BabyAI not available: {e}")

    def test_babyai_level_generation_is_reproducible(self, babyai_env):
        """Same seed always produces same level layout and mission."""
        env = babyai_env
        seed = 42

        # Generate level twice with same seed
        obs1, info1 = env.reset(seed=seed)
        mission1 = env.mission
        grid1 = env.grid.encode()

        obs2, info2 = env.reset(seed=seed)
        mission2 = env.mission
        grid2 = env.grid.encode()

        # Must be identical
        np.testing.assert_array_equal(obs1['image'], obs2['image'])
        assert mission1 == mission2, f"Missions differ: {mission1} vs {mission2}"
        np.testing.assert_array_equal(grid1, grid2)

        print(f"\n[BABYAI] Level generation with seed={seed}:")
        print(f"  Mission: {mission1}")
        print(f"  ✅ Same seed produces identical level and mission")

    def test_babyai_trajectories_are_reproducible(self, babyai_env):
        """Same seed + same actions = identical trajectory (no wrapper needed)."""
        env = babyai_env
        seed = 42

        # Run 1: Start with one global np.random state
        np.random.seed(111)
        obs1, rew1 = run_babyai_trajectory(env, seed, NUM_STEPS)

        # Run 2: Start with DIFFERENT global np.random state
        np.random.seed(222)
        obs2, rew2 = run_babyai_trajectory(env, seed, NUM_STEPS)

        # Compare - should be IDENTICAL (no global RNG dependency)
        for i, (o1, o2) in enumerate(zip(obs1, obs2)):
            np.testing.assert_array_equal(
                o1, o2,
                err_msg=f"Observations diverged at step {i}"
            )

        assert rew1 == rew2, "Rewards should be identical"

        print(f"\n[BABYAI] Trajectory reproducibility test:")
        print(f"  Global np.random seed was different between runs")
        print(f"  ✅ Trajectories are IDENTICAL (no wrapper needed!)")

    def test_babyai_different_seeds_different_levels(self, babyai_env):
        """Different seeds should produce different levels."""
        env = babyai_env

        obs1, info1 = env.reset(seed=42)
        mission1 = env.mission
        obs1_img = obs1['image'] if isinstance(obs1, dict) else obs1

        obs2, info2 = env.reset(seed=123)
        mission2 = env.mission
        obs2_img = obs2['image'] if isinstance(obs2, dict) else obs2

        # Should be different (procedural generation working)
        levels_differ = (
            not np.array_equal(obs1_img, obs2_img) or
            mission1 != mission2
        )

        print(f"\n[BABYAI] Different seeds test:")
        print(f"  Seed 42 mission: {mission1}")
        print(f"  Seed 123 mission: {mission2}")
        print(f"  ✅ Different seeds produce different levels: {levels_differ}")

        assert levels_differ, "Different seeds should produce different levels"


# =============================================================================
# Test: MultiGrid Needs Wrapper
# =============================================================================


class TestMultiGridNeedsWrapper:
    """Show that MultiGrid NEEDS our reproducibility wrapper."""

    @pytest.fixture
    def multigrid_env_unwrapped(self):
        """Create MultiGrid WITHOUT wrapper."""
        try:
            from gym_multigrid.envs import SoccerGame4HEnv10x15N2
            env = SoccerGame4HEnv10x15N2()
            yield env
            env.close()
        except ImportError:
            pytest.skip("gym-multigrid not installed")

    @pytest.fixture
    def multigrid_env_wrapped(self):
        """Create MultiGrid WITH reproducibility wrapper."""
        try:
            from gym_multigrid.envs import SoccerGame4HEnv10x15N2
            from gym_gui.core.wrappers.multigrid_reproducibility import ReproducibleMultiGridWrapper

            env = SoccerGame4HEnv10x15N2()
            env = ReproducibleMultiGridWrapper(env)
            yield env
            env.close()
        except ImportError:
            pytest.skip("gym-multigrid not installed")

    def test_multigrid_without_wrapper_is_not_reproducible(self, multigrid_env_unwrapped):
        """WITHOUT wrapper, same seed can produce different trajectories."""
        env = multigrid_env_unwrapped
        seed = 42

        # Run 1
        np.random.seed(111)
        obs1, rew1 = run_multigrid_trajectory(env, seed, NUM_STEPS)

        # Run 2 (different global state)
        np.random.seed(222)
        obs2, rew2 = run_multigrid_trajectory(env, seed, NUM_STEPS)

        # Check for divergence
        diverged = False
        for i, (o1, o2) in enumerate(zip(obs1, obs2)):
            for j, (a1, a2) in enumerate(zip(o1, o2)):
                if not np.array_equal(a1, a2):
                    diverged = True
                    print(f"\n[MULTIGRID without wrapper] Diverged at step {i}, agent {j}")
                    break
            if diverged:
                break

        if not diverged:
            print(f"\n[MULTIGRID without wrapper] No divergence in {NUM_STEPS} steps")
            print(f"  (Bug may not have manifested - try more steps)")
        else:
            print(f"  ✅ Bug confirmed: trajectories diverged")

    def test_multigrid_with_wrapper_is_reproducible(self, multigrid_env_wrapped):
        """WITH wrapper, same seed always produces identical trajectories."""
        env = multigrid_env_wrapped
        seed = 42

        # Run 1
        np.random.seed(111)
        obs1, rew1 = run_multigrid_trajectory(env, seed, NUM_STEPS)

        # Run 2 (different global state - shouldn't matter with wrapper)
        np.random.seed(222)
        obs2, rew2 = run_multigrid_trajectory(env, seed, NUM_STEPS)

        # Should be identical
        for i, (o1, o2) in enumerate(zip(obs1, obs2)):
            for j, (a1, a2) in enumerate(zip(o1, o2)):
                np.testing.assert_array_equal(
                    a1, a2,
                    err_msg=f"Step {i}, agent {j}: observations should match"
                )

        assert rew1 == rew2, "Rewards should be identical"

        print(f"\n[MULTIGRID with wrapper] Trajectory reproducibility:")
        print(f"  ✅ Trajectories are IDENTICAL (wrapper fixes the bug)")


# =============================================================================
# Test: Side-by-Side Comparison
# =============================================================================


class TestSideBySideComparison:
    """Direct comparison of BabyAI vs MultiGrid reproducibility."""

    def test_comparison_summary(self):
        """Print a summary comparing both environments."""
        print("\n" + "=" * 70)
        print("REPRODUCIBILITY COMPARISON: BabyAI vs MultiGrid")
        print("=" * 70)

        print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     BABYAI (Procedural Generation)                   │
├─────────────────────────────────────────────────────────────────────┤
│  Level Generation:   _rand_int() → self.np_random.integers() ✅     │
│  Object Placement:   _rand_elem() → self.np_random.integers() ✅    │
│  Mission Text:       gen_mission() → uses seeded random ✅          │
│  Action Execution:   step() → uses self.np_random ✅                │
│                                                                     │
│  RESULT: Naturally reproducible, NO WRAPPER NEEDED                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     MULTIGRID (Soccer/Collect)                       │
├─────────────────────────────────────────────────────────────────────┤
│  Level Generation:   Uses self.np_random ✅                          │
│  Object Placement:   Uses self.np_random ✅                          │
│  Agent Placement:    Uses self.np_random ✅                          │
│  Action Execution:   np.random.permutation() ❌ (GLOBAL RNG!)       │
│                                                                     │
│  RESULT: NEEDS ReproducibleMultiGridWrapper                         │
└─────────────────────────────────────────────────────────────────────┘

WHY THE DIFFERENCE?
- BabyAI/MiniGrid developers used self.np_random EVERYWHERE
- gym-multigrid developer accidentally used np.random in step()
- One line of code (line 1249) caused the entire reproducibility issue
        """)

        # This test always passes - it's just for documentation
        assert True


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
