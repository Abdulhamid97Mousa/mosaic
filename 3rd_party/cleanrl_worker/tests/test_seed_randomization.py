"""
Test to determine what happens when we change seeds in BabyAI/MiniGrid environments.

This test will answer:
1. Does changing the seed change the level layout?
2. What exactly changes (agent position, object positions, grid size)?
3. Are episodes truly different or identical?
"""

import gymnasium as gym
import minigrid
import numpy as np

# Register MiniGrid environments
minigrid.register_minigrid_envs()


def test_seed_changes_layout():
    """Test if different seeds produce different level layouts."""
    print("=" * 80)
    print("TEST 1: Do different seeds create different level layouts?")
    print("=" * 80)

    # Test multiple environments
    test_envs = [
        "BabyAI-GoToLocal-v0",
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-DoorKey-8x8-v0",
    ]

    for env_name in test_envs:
        print(f"\n{'─' * 80}")
        print(f"Environment: {env_name}")
        print('─' * 80)

        try:
            env = gym.make(env_name)

            # Reset with 3 different seeds
            seeds = [0, 1, 42]
            layouts = []

            for seed in seeds:
                obs, info = env.reset(seed=seed)

                # Extract layout information
                layout_info = {
                    'seed': seed,
                    'agent_pos': env.unwrapped.agent_pos,
                    'agent_dir': env.unwrapped.agent_dir,
                    'grid_size': (env.unwrapped.width, env.unwrapped.height),
                }

                # Get all objects in the grid
                objects = []
                for i in range(env.unwrapped.width):
                    for j in range(env.unwrapped.height):
                        cell = env.unwrapped.grid.get(i, j)
                        if cell is not None and cell.type != 'wall':
                            objects.append({
                                'type': cell.type,
                                'color': cell.color,
                                'pos': (i, j)
                            })
                layout_info['objects'] = objects

                # Get mission if it exists (BabyAI)
                if hasattr(env.unwrapped, 'mission'):
                    layout_info['mission'] = env.unwrapped.mission

                layouts.append(layout_info)

                print(f"\nSeed {seed}:")
                print(f"  Agent position: {layout_info['agent_pos']}")
                print(f"  Agent direction: {layout_info['agent_dir']}")
                print(f"  Grid size: {layout_info['grid_size']}")
                if 'mission' in layout_info:
                    print(f"  Mission: {layout_info['mission']}")
                print(f"  Objects: {len(objects)} found")
                for obj in objects[:5]:  # Show first 5 objects
                    print(f"    - {obj['color']} {obj['type']} at {obj['pos']}")
                if len(objects) > 5:
                    print(f"    ... and {len(objects) - 5} more")

            # Compare layouts
            print(f"\n{'─' * 40}")
            print("COMPARISON:")
            print('─' * 40)

            all_same = True

            # Compare agent positions
            agent_positions = [layout['agent_pos'] for layout in layouts]
            if len(set(agent_positions)) > 1:
                print(f"[DIFFER] Agent positions: {agent_positions}")
                all_same = False
            else:
                print(f"[SAME] Agent positions: {agent_positions[0]}")

            # Compare agent directions
            agent_dirs = [layout['agent_dir'] for layout in layouts]
            if len(set(agent_dirs)) > 1:
                print(f"[DIFFER] Agent directions: {agent_dirs}")
                all_same = False
            else:
                print(f"[SAME] Agent directions: {agent_dirs[0]}")

            # Compare object positions
            for i in range(len(layouts)):
                for j in range(i + 1, len(layouts)):
                    obj_str_i = str(sorted([(o['type'], o['pos']) for o in layouts[i]['objects']]))
                    obj_str_j = str(sorted([(o['type'], o['pos']) for o in layouts[j]['objects']]))
                    if obj_str_i != obj_str_j:
                        print(f"[DIFFER] Object layouts between seed {seeds[i]} and {seeds[j]}")
                        all_same = False
                        break
                if not all_same:
                    break

            if all_same:
                print("[SAME] All object layouts are IDENTICAL")

            # Compare missions (BabyAI)
            if 'mission' in layouts[0]:
                missions = [layout['mission'] for layout in layouts]
                if len(set(missions)) > 1:
                    print(f"[DIFFER] Missions")
                    for i, mission in enumerate(missions):
                        print(f"    Seed {seeds[i]}: {mission}")
                    all_same = False
                else:
                    print(f"[SAME] Missions: {missions[0]}")

            print("\n" + "=" * 40)
            if all_same:
                print(f"[FAIL] CONCLUSION: Changing seed does NOT change level layout!")
            else:
                print(f"[PASS] CONCLUSION: Changing seed DOES change level layout!")
            print("=" * 40)

            env.close()

        except Exception as e:
            print(f"[ERROR] Error testing {env_name}: {e}")
            import traceback
            traceback.print_exc()


def test_babyai_mixed_train_local():
    """Specific test for BabyAI-MixedTrainLocal-v0 (the one used in BALROG)."""
    print("\n\n" + "=" * 80)
    print("TEST 2: BabyAI-MixedTrainLocal-v0 (BALROG Environment)")
    print("=" * 80)

    try:
        # This is the base environment - it randomly selects action kinds
        env = gym.make("BabyAI-MixedTrainLocal-v0")

        print("\nResetting environment 5 times with different seeds...")
        print("This will show us if the 'action_kinds' (tasks) vary\n")

        for seed in [0, 1, 2, 3, 4]:
            obs, info = env.reset(seed=seed)
            action_kind = env.unwrapped.action_kinds[0] if hasattr(env.unwrapped, 'action_kinds') else 'unknown'
            mission = env.unwrapped.mission if hasattr(env.unwrapped, 'mission') else 'no mission'

            print(f"Seed {seed}:")
            print(f"  Action kind: {action_kind}")
            print(f"  Mission: {mission}")
            print(f"  Agent pos: {env.unwrapped.agent_pos}")
            print(f"  Grid size: ({env.unwrapped.width}, {env.unwrapped.height})")
            print()

        env.close()

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


def test_same_seed_multiple_resets():
    """Test if resetting with the SAME seed produces identical layouts."""
    print("\n\n" + "=" * 80)
    print("TEST 3: Same seed = same layout? (Reproducibility test)")
    print("=" * 80)

    env_name = "BabyAI-GoToLocal-v0"
    env = gym.make(env_name)

    print(f"\nResetting {env_name} with seed=42 THREE times...")
    print("If layouts are identical, seed behavior is deterministic.\n")

    layouts = []
    for i in range(3):
        obs, info = env.reset(seed=42)
        layout = {
            'agent_pos': env.unwrapped.agent_pos,
            'agent_dir': env.unwrapped.agent_dir,
            'mission': getattr(env.unwrapped, 'mission', 'no mission'),
        }
        layouts.append(layout)
        print(f"Reset {i+1}: agent_pos={layout['agent_pos']}, agent_dir={layout['agent_dir']}")
        print(f"         mission={layout['mission']}")

    # Check if all identical
    all_identical = all(
        layouts[i] == layouts[0] for i in range(len(layouts))
    )

    print("\n" + "=" * 40)
    if all_identical:
        print("[PASS] SAME seed produces IDENTICAL layouts (deterministic)")
    else:
        print("[FAIL] SAME seed produces DIFFERENT layouts (non-deterministic)")
    print("=" * 40)

    env.close()


def test_episode_rollout_comparison():
    """Test full episode rollouts with different seeds to see if policy would need observations."""
    print("\n\n" + "=" * 80)
    print("TEST 4: Would a policy need to use observations?")
    print("=" * 80)

    env_name = "MiniGrid-Empty-5x5-v0"
    env = gym.make(env_name)

    print(f"\nEnvironment: {env_name}")
    print("Running 3 episodes with different seeds using FIXED action sequence...")
    print("If all episodes succeed, observations are NOT needed (just memorization).")
    print("If episodes fail, observations ARE needed.\n")

    # Fixed action sequence (just go forward)
    fixed_actions = [2] * 10  # action 2 = go forward

    for seed in [0, 1, 2]:
        obs, info = env.reset(seed=seed)
        initial_pos = env.unwrapped.agent_pos

        print(f"Seed {seed}:")
        print(f"  Initial agent position: {initial_pos}")
        print(f"  Goal position: {env.unwrapped.goal_pos if hasattr(env.unwrapped, 'goal_pos') else 'unknown'}")

        # Execute fixed action sequence
        for step, action in enumerate(fixed_actions):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"  [SUCCESS] Reached goal at step {step+1}!")
                break
            if truncated:
                print(f"  [TRUNCATED] Episode truncated at step {step+1}")
                break
        else:
            print(f"  [FAILED] Failed to reach goal after {len(fixed_actions)} steps")

        print(f"  Final agent position: {env.unwrapped.agent_pos}")
        print()

    print("=" * 40)
    print("INTERPRETATION:")
    print("If the SAME action sequence works for ALL seeds,")
    print("then the policy can IGNORE observations (just memorize).")
    print("\nIf the action sequence ONLY works for SOME seeds,")
    print("then the policy MUST use observations (spatial reasoning).")
    print("=" * 40)

    env.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EMPIRICAL TEST: Seed Randomization in BabyAI/MiniGrid")
    print("=" * 80)
    print("\nThis test will determine:")
    print("1. Whether changing seeds actually changes level layouts")
    print("2. What specific elements change (positions, missions, etc.)")
    print("3. Whether RL agents need to use observations or can just memorize")
    print("\n")

    test_seed_changes_layout()
    test_babyai_mixed_train_local()
    test_same_seed_multiple_resets()
    test_episode_rollout_comparison()

    print("\n\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
    print("\nThese results will tell us:")
    print("- Whether BALROG's seed randomization actually creates different levels")
    print("- Whether RL training needs environment randomization")
    print("- How to properly configure training for fair comparison")
    print("\n")
