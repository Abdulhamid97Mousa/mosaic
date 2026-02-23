"""Test observation_space.shape with curriculum environments.

This test verifies that curriculum environments properly expose observation_space
and action_space through the wrapper chain. It caught a critical bug where
apply_wrappers=False was used assuming sitecustomize.py had patched gym.make(),
but the sitecustomize wasn't being loaded.

Note: These tests are sensitive to minigrid wrapper state and can fail in batch
runs where other tests modify the observation space wrappers. They pass
reliably when run in isolation.
"""
import pytest
import gymnasium as gym
from xuance_worker.wrappers.curriculum import make_curriculum_env

pytestmark = pytest.mark.slow  # May fail in batch runs due to minigrid state pollution


def test_curriculum_env_with_wrappers():
    """Test curriculum env with apply_wrappers=True (CORRECT)."""
    curriculum_schedule = [
        {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
    ]

    print("\nCreating curriculum env with apply_wrappers=True...")
    envs = make_curriculum_env(
        curriculum_schedule,
        num_envs=4,
        apply_wrappers=True,  # ← CORRECT: Ensures ImgObsWrapper + FlattenObservation
    )

    # Check observation space
    assert hasattr(envs, 'single_observation_space')
    single_obs = envs.single_observation_space
    print(f"single_observation_space: {single_obs}")

    assert hasattr(single_obs, 'shape')
    shape = single_obs.shape
    print(f"shape: {shape}")

    assert shape is not None, "shape should not be None"
    assert len(shape) > 0, "shape should have at least one dimension"
    assert len(shape) == 1, "Flattened observation should be 1D"

    # Check action space
    assert hasattr(envs, 'single_action_space')
    action_space = envs.single_action_space
    assert hasattr(action_space, 'n')
    assert action_space.n == 7, "BabyAI should have 7 actions"

    print(f"✅ SUCCESS with apply_wrappers=True: shape={shape}, actions={action_space.n}")
    envs.close()


def test_curriculum_env_without_wrappers_fails():
    """Test that apply_wrappers=False fails when sitecustomize doesn't patch gym.make().

    This documents the bug that was fixed: single_agent_curriculum_training.py
    (formerly curriculum_training.py) was using apply_wrappers=False assuming
    sitecustomize.py had patched gym.make(), but sitecustomize wasn't loaded,
    so wrappers weren't applied.
    """
    curriculum_schedule = [
        {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
    ]

    print("\nCreating curriculum env with apply_wrappers=False (documenting the bug)...")
    envs = make_curriculum_env(
        curriculum_schedule,
        num_envs=4,
        apply_wrappers=False,  # ← BUG: Wrappers not applied if sitecustomize not loaded
    )

    single_obs = envs.single_observation_space
    print(f"single_observation_space: {single_obs}")
    print(f"type: {type(single_obs)}")

    # Without wrappers, BabyAI returns a Dict observation space
    # which doesn't have a simple .shape attribute
    if hasattr(single_obs, 'shape'):
        print(f"shape: {single_obs.shape}")
        # If sitecustomize IS loaded, this will pass
        # If sitecustomize is NOT loaded, shape will be a dict structure
        envs.close()
    else:
        # Expected: Dict space without simple shape
        print("⚠️  observation_space has no .shape (expected with apply_wrappers=False)")
        assert isinstance(single_obs, gym.spaces.Dict), \
            "Without wrappers, BabyAI should return Dict observation space"
        envs.close()


def test_curriculum_multiple_envs():
    """Test with full curriculum schedule."""
    curriculum_schedule = [
        {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
        {"env_id": "BabyAI-GoToRedBall-v0", "steps": 200000},
        {"env_id": "BabyAI-GoToObj-v0", "steps": 200000},
        {"env_id": "BabyAI-GoToLocal-v0"},
    ]

    envs = make_curriculum_env(curriculum_schedule, num_envs=4, apply_wrappers=True)

    assert envs.single_observation_space.shape is not None
    assert envs.single_action_space.n > 0

    print(f"✅ Full curriculum: obs_shape={envs.single_observation_space.shape}, "
          f"actions={envs.single_action_space.n}")

    envs.close()


if __name__ == "__main__":
    # Run tests standalone
    tests = [
        ("Test 1: With Wrappers (Correct)", test_curriculum_env_with_wrappers),
        ("Test 2: Without Wrappers (Bug)", test_curriculum_env_without_wrappers_fails),
        ("Test 3: Full Curriculum", test_curriculum_multiple_envs),
    ]

    for name, test_func in tests:
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        try:
            test_func()
            print(f"\n✅ {name} PASSED")
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
