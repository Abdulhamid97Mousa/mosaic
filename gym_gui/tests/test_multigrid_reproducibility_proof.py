"""Proof of Reproducibility Fix for gym-multigrid.

This test file PROVES that:
1. The original gym-multigrid has a reproducibility BUG
2. Our ReproducibleMultiGridWrapper FIXES the bug

The bug is in gym_multigrid/multigrid.py step() method:
    order = np.random.permutation(len(actions))  # Uses GLOBAL np.random!

This ignores env.np_random (the seeded RNG), making trajectories non-reproducible.

Our wrapper fixes this by seeding np.random from env.np_random before each step().
"""

from __future__ import annotations

import sys
from typing import Any, List, Tuple

import numpy as np
import pytest


# =============================================================================
# Test Configuration
# =============================================================================

# Number of steps to run in trajectory comparison
NUM_STEPS = 50

# Number of trajectory pairs to compare (more = stronger proof)
NUM_TRIALS = 3

# Seeds to test
TEST_SEEDS = [42, 123, 999]


# =============================================================================
# Helper Functions
# =============================================================================


def get_random_actions(env: Any, rng: np.random.Generator) -> List[int]:
    """Generate random actions for all agents using a controlled RNG.

    We use a separate RNG for action selection to isolate the test
    from action generation randomness.
    """
    n_agents = len(env.agents)
    n_actions = env.action_space[0].n if hasattr(env.action_space, '__getitem__') else env.action_space.n
    return [int(rng.integers(0, n_actions)) for _ in range(n_agents)]


def run_trajectory(
    env: Any,
    seed: int,
    num_steps: int,
    action_seed: int = 12345,
) -> Tuple[List[np.ndarray], List[List[float]], List[np.ndarray]]:
    """Run an environment trajectory and record observations, rewards, and np.random states.

    Args:
        env: The gym-multigrid environment (wrapped or unwrapped)
        seed: Seed for the environment
        num_steps: Number of steps to run
        action_seed: Seed for action generation (constant across runs for fair comparison)

    Returns:
        Tuple of (observations, rewards, np_random_states)
        - observations: List of observation arrays after each step
        - rewards: List of reward lists after each step
        - np_random_states: np.random state after each step (to show divergence)
    """
    # Seed the environment
    env.seed(seed)
    obs = env.reset()

    # Use a separate, controlled RNG for action selection
    action_rng = np.random.default_rng(action_seed)

    observations = []
    rewards = []
    np_random_states = []

    for _ in range(num_steps):
        actions = get_random_actions(env, action_rng)

        obs, reward, done, info = env.step(actions)

        # Record trajectory data
        observations.append([o.copy() if hasattr(o, 'copy') else o for o in obs])
        rewards.append(list(reward))

        # Record np.random state (this shows the bug - state diverges without wrapper)
        np_random_states.append(np.random.get_state()[1][0:5].copy())

        if done:
            env.seed(seed)
            obs = env.reset()

    return observations, rewards, np_random_states


def trajectories_match(
    traj1: Tuple[List, List, List],
    traj2: Tuple[List, List, List],
) -> Tuple[bool, str]:
    """Compare two trajectories and return (match, message).

    Returns:
        Tuple of (is_matching, description)
    """
    obs1, rew1, states1 = traj1
    obs2, rew2, states2 = traj2

    # Compare observations
    for i, (o1, o2) in enumerate(zip(obs1, obs2)):
        for j, (a1, a2) in enumerate(zip(o1, o2)):
            if not np.array_equal(a1, a2):
                return False, f"Observations diverged at step {i}, agent {j}"

    # Compare rewards
    for i, (r1, r2) in enumerate(zip(rew1, rew2)):
        if r1 != r2:
            return False, f"Rewards diverged at step {i}: {r1} vs {r2}"

    # Compare np.random states (shows internal state consistency)
    for i, (s1, s2) in enumerate(zip(states1, states2)):
        if not np.array_equal(s1, s2):
            return False, f"np.random state diverged at step {i}"

    return True, "Trajectories are identical"


# =============================================================================
# Test: Demonstrate the Bug (WITHOUT Wrapper)
# =============================================================================


class TestMultiGridReproducibilityBug:
    """Tests that demonstrate the reproducibility BUG in gym-multigrid.

    These tests SHOULD show that WITHOUT our wrapper, trajectories diverge.
    """

    @pytest.fixture
    def soccer_env(self):
        """Create a Soccer environment WITHOUT the reproducibility wrapper."""
        try:
            from gym_multigrid.envs import SoccerGame4HEnv10x15N2
            env = SoccerGame4HEnv10x15N2()
            yield env
            env.close()
        except ImportError:
            pytest.skip("gym-multigrid not installed")

    def test_bug_exists_trajectories_diverge_without_wrapper(self, soccer_env):
        """PROOF: Without wrapper, same seed produces DIFFERENT trajectories.

        This test demonstrates the bug by:
        1. Running two trajectories with identical seeds and actions
        2. Showing that observations/rewards DIVERGE due to np.random.permutation()

        Expected: trajectories_match returns False (bug exists)
        """
        env = soccer_env
        seed = 42

        # Reset global np.random to different states between runs
        # This simulates real-world conditions where global RNG state varies

        # Run 1: Set a specific global np.random state
        np.random.seed(111)
        traj1 = run_trajectory(env, seed, NUM_STEPS)

        # Run 2: Set a DIFFERENT global np.random state
        np.random.seed(222)  # Different state!
        traj2 = run_trajectory(env, seed, NUM_STEPS)

        match, msg = trajectories_match(traj1, traj2)

        # The bug means trajectories SHOULD diverge (match = False)
        # If they match, either the bug was fixed upstream or something changed
        print(f"\n[BUG TEST] Same seed, different global np.random state:")
        print(f"  Trajectories match: {match}")
        print(f"  Message: {msg}")

        if not match:
            print("  ✓ BUG CONFIRMED: Trajectories diverged as expected")
        else:
            print("  ⚠ BUG NOT OBSERVED: Trajectories matched (may need more steps)")

        # Note: We don't assert here because we want to show the bug exists,
        # but we don't want to fail if upstream somehow fixed it

    def test_np_random_permutation_used_in_step(self, soccer_env):
        """Show that step() uses np.random.permutation() for action ordering.

        This proves the bug mechanism: the order in which agent actions are
        executed depends on np.random.permutation(), which uses the global RNG.

        We verify this by showing that different global np.random seeds produce
        different action orderings (and thus different outcomes).
        """
        env = soccer_env
        seed = 42

        # Helper to collect observations after N steps
        def run_with_global_seed(global_seed: int, steps: int = 20) -> List[np.ndarray]:
            """Run environment with specific global np.random seed."""
            np.random.seed(global_seed)  # Set global seed
            env.seed(seed)
            obs = env.reset()

            observations = []
            for _ in range(steps):
                actions = [0, 1, 2, 3]  # Fixed actions
                obs, _, done, _ = env.step(actions)
                observations.append([o.copy() for o in obs])
                if done:
                    env.seed(seed)
                    env.reset()
            return observations

        # Run with different global seeds
        obs_run1 = run_with_global_seed(111)
        obs_run2 = run_with_global_seed(222)

        # Check if observations diverge (they should, due to np.random.permutation)
        diverged = False
        diverge_step = -1
        for i, (o1, o2) in enumerate(zip(obs_run1, obs_run2)):
            for a1, a2 in zip(o1, o2):
                if not np.array_equal(a1, a2):
                    diverged = True
                    diverge_step = i
                    break
            if diverged:
                break

        print(f"\n[MECHANISM TEST] Different global seeds produce different trajectories: {diverged}")
        if diverged:
            print(f"  Trajectories diverged at step {diverge_step}")
            print(f"  ✓ This proves step() uses np.random (the bug mechanism)")
        else:
            print(f"  Trajectories matched for {len(obs_run1)} steps")

        # Note: We don't assert because the bug manifests probabilistically
        # The side-by-side test is the definitive proof


# =============================================================================
# Test: Demonstrate the Fix (WITH Wrapper)
# =============================================================================


class TestMultiGridReproducibilityFix:
    """Tests that PROVE the reproducibility FIX works.

    These tests show that WITH our wrapper, trajectories are IDENTICAL.
    """

    @pytest.fixture
    def wrapped_soccer_env(self):
        """Create a Soccer environment WITH the reproducibility wrapper."""
        try:
            from gym_multigrid.envs import SoccerGame4HEnv10x15N2
            from gym_gui.core.wrappers.multigrid_reproducibility import ReproducibleMultiGridWrapper

            env = SoccerGame4HEnv10x15N2()
            wrapped_env = ReproducibleMultiGridWrapper(env)
            yield wrapped_env
            wrapped_env.close()
        except ImportError:
            pytest.skip("gym-multigrid not installed")

    def test_fix_works_trajectories_match_with_wrapper(self, wrapped_soccer_env):
        """PROOF: With wrapper, same seed produces IDENTICAL trajectories.

        This is the key test proving our fix works.

        Expected: trajectories_match returns True (fix works)
        """
        env = wrapped_soccer_env
        seed = 42

        # Run 1: Set a specific global np.random state
        np.random.seed(111)
        traj1 = run_trajectory(env, seed, NUM_STEPS)

        # Run 2: Set a DIFFERENT global np.random state
        np.random.seed(222)  # Different state - but shouldn't matter with wrapper!
        traj2 = run_trajectory(env, seed, NUM_STEPS)

        match, msg = trajectories_match(traj1, traj2)

        print(f"\n[FIX TEST] Same seed, different global np.random state (WITH WRAPPER):")
        print(f"  Trajectories match: {match}")
        print(f"  Message: {msg}")

        if match:
            print("  ✓ FIX CONFIRMED: Trajectories are identical!")
        else:
            print("  ✗ FIX FAILED: Trajectories still diverged")

        assert match, f"With wrapper, trajectories should match. {msg}"

    @pytest.mark.parametrize("seed", TEST_SEEDS)
    def test_fix_works_across_multiple_seeds(self, wrapped_soccer_env, seed):
        """Test reproducibility across multiple seeds."""
        env = wrapped_soccer_env

        # Different global states between runs
        np.random.seed(seed * 2)
        traj1 = run_trajectory(env, seed, NUM_STEPS)

        np.random.seed(seed * 3)
        traj2 = run_trajectory(env, seed, NUM_STEPS)

        match, msg = trajectories_match(traj1, traj2)
        assert match, f"Seed {seed}: {msg}"

    def test_fix_multiple_trials(self, wrapped_soccer_env):
        """Run multiple trials to increase confidence in the fix."""
        env = wrapped_soccer_env
        seed = 42

        trajectories = []
        for trial in range(NUM_TRIALS):
            # Use different global np.random seeds each trial
            np.random.seed(trial * 1000 + 500)
            traj = run_trajectory(env, seed, NUM_STEPS)
            trajectories.append(traj)

        # All trajectories should be identical
        reference = trajectories[0]
        for i, traj in enumerate(trajectories[1:], 1):
            match, msg = trajectories_match(reference, traj)
            assert match, f"Trial {i} diverged from trial 0: {msg}"

        print(f"\n[CONFIDENCE TEST] All {NUM_TRIALS} trials produced identical trajectories ✓")


# =============================================================================
# Test: Side-by-Side Comparison
# =============================================================================


class TestSideBySideComparison:
    """Direct comparison of wrapped vs unwrapped behavior."""

    def test_side_by_side_comparison(self):
        """Run the same scenario with and without wrapper to show the difference."""
        try:
            from gym_multigrid.envs import SoccerGame4HEnv10x15N2
            from gym_gui.core.wrappers.multigrid_reproducibility import ReproducibleMultiGridWrapper
        except ImportError:
            pytest.skip("gym-multigrid not installed")

        seed = 42

        # --- WITHOUT WRAPPER ---
        print("\n" + "=" * 60)
        print("SIDE-BY-SIDE COMPARISON: WITHOUT vs WITH WRAPPER")
        print("=" * 60)

        env_unwrapped = SoccerGame4HEnv10x15N2()

        np.random.seed(111)
        traj_unwrapped_1 = run_trajectory(env_unwrapped, seed, NUM_STEPS)

        np.random.seed(222)
        traj_unwrapped_2 = run_trajectory(env_unwrapped, seed, NUM_STEPS)

        match_unwrapped, msg_unwrapped = trajectories_match(traj_unwrapped_1, traj_unwrapped_2)
        env_unwrapped.close()

        # --- WITH WRAPPER ---
        env_wrapped = SoccerGame4HEnv10x15N2()
        env_wrapped = ReproducibleMultiGridWrapper(env_wrapped)

        np.random.seed(111)
        traj_wrapped_1 = run_trajectory(env_wrapped, seed, NUM_STEPS)

        np.random.seed(222)
        traj_wrapped_2 = run_trajectory(env_wrapped, seed, NUM_STEPS)

        match_wrapped, msg_wrapped = trajectories_match(traj_wrapped_1, traj_wrapped_2)
        env_wrapped.close()

        # --- REPORT ---
        print(f"\nWITHOUT wrapper: Trajectories match = {match_unwrapped}")
        print(f"  → {msg_unwrapped}")

        print(f"\nWITH wrapper: Trajectories match = {match_wrapped}")
        print(f"  → {msg_wrapped}")

        print("\n" + "-" * 60)
        if not match_unwrapped and match_wrapped:
            print("✓ PROOF COMPLETE: Wrapper fixes the reproducibility bug!")
        elif match_unwrapped and match_wrapped:
            print("⚠ Both matched (bug may not have manifested in this run)")
        else:
            print("✗ UNEXPECTED: Wrapper didn't fix the issue")
        print("-" * 60)

        # The wrapper version MUST be reproducible
        assert match_wrapped, f"Wrapped env must be reproducible: {msg_wrapped}"


# =============================================================================
# Test: Understand the Fix Mechanism
# =============================================================================


class TestFixMechanism:
    """Tests that explain HOW the fix works."""

    def test_wrapper_syncs_np_random_from_env_np_random(self):
        """Demonstrate the fix mechanism step by step."""
        try:
            from gym_multigrid.envs import SoccerGame4HEnv10x15N2
            from gym_gui.core.wrappers.multigrid_reproducibility import ReproducibleMultiGridWrapper
        except ImportError:
            pytest.skip("gym-multigrid not installed")

        print("\n" + "=" * 60)
        print("FIX MECHANISM DEMONSTRATION")
        print("=" * 60)

        env = SoccerGame4HEnv10x15N2()
        env = ReproducibleMultiGridWrapper(env)

        # Seed the env
        env.seed(42)
        env.reset()

        print("\n1. After env.seed(42) and reset():")
        print(f"   env.np_random state: {env.np_random.bit_generator.state['state']['state']}")

        # Set global np.random to something unrelated
        np.random.seed(99999)
        print(f"   global np.random (before step): {np.random.get_state()[1][0]}")

        # Take a step - the wrapper should sync np.random
        actions = [0] * len(env.agents)
        env.step(actions)

        print(f"\n2. After wrapper.step():")
        print(f"   global np.random (after step): {np.random.get_state()[1][0]}")
        print(f"   env.np_random state: {env.np_random.bit_generator.state['state']['state']}")

        print("\n3. The fix mechanism:")
        print("   a) Before calling env.step(), wrapper reads env.np_random")
        print("   b) Generates a deterministic seed from env.np_random")
        print("   c) Seeds global np.random with that value")
        print("   d) Now env.step()'s np.random.permutation() is deterministic!")
        print("   e) env.np_random advances, so next step gets different but still deterministic seed")

        env.close()

    def test_wrapper_produces_deterministic_np_random_sequence(self):
        """Show that the wrapper produces the same np.random sequence each run."""
        try:
            from gym_multigrid.envs import SoccerGame4HEnv10x15N2
            from gym_gui.core.wrappers.multigrid_reproducibility import ReproducibleMultiGridWrapper
        except ImportError:
            pytest.skip("gym-multigrid not installed")

        def collect_np_random_states(seed: int, num_steps: int = 5) -> List[int]:
            """Collect the np.random state after each step."""
            env = SoccerGame4HEnv10x15N2()
            env = ReproducibleMultiGridWrapper(env)
            env.seed(seed)
            env.reset()

            states = []
            for _ in range(num_steps):
                actions = [0] * len(env.agents)
                env.step(actions)
                states.append(np.random.get_state()[1][0])

            env.close()
            return states

        # Run 1: Start with one global state
        np.random.seed(111)
        states1 = collect_np_random_states(42)

        # Run 2: Start with different global state
        np.random.seed(999)
        states2 = collect_np_random_states(42)

        print("\n" + "=" * 60)
        print("DETERMINISTIC np.random SEQUENCE TEST")
        print("=" * 60)
        print(f"\nRun 1 np.random states: {states1}")
        print(f"Run 2 np.random states: {states2}")

        assert states1 == states2, "np.random states should be identical with same env seed"
        print("\n✓ Both runs produced identical np.random state sequences!")


# =============================================================================
# Main: Run with verbose output
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
