"""
Test to understand how seeds work during RL training based on BabyAI paper methodology.

Based on research:
1. BabyAI paper used PPO and A2C for RL training
2. They recommend "varying --seed" across different training runs
3. Modern Gymnasium allows two approaches:
   - Set seed once, then let internal random state evolve
   - Set different seed per episode

This test will determine:
1. What happens when we set seed ONCE then reset multiple times?
2. Do we get different level layouts (good for generalization)?
3. Or same layout (bad - leads to memorization)?
"""

import gymnasium as gym
import minigrid

# Register MiniGrid environments
minigrid.register_minigrid_envs()


def test_single_seed_multiple_resets():
    """
    Test what happens in TYPICAL RL training:
    - Set seed once at initialization
    - Reset multiple times without specifying seed
    - Check if layouts differ (generalization) or stay same (memorization)
    """
    print("=" * 80)
    print("TEST: RL Training Behavior - Single Seed, Multiple Resets")
    print("=" * 80)
    print("\nThis simulates TYPICAL RL training where:")
    print("1. Environment is created with initial seed")
    print("2. Training loop resets environment many times")
    print("3. Each reset() is called WITHOUT specifying a seed\n")

    test_envs = [
        "BabyAI-GoToLocal-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-Empty-8x8-v0",
    ]

    for env_name in test_envs:
        print(f"\n{'─' * 80}")
        print(f"Environment: {env_name}")
        print('─' * 80)

        env = gym.make(env_name)

        # TYPICAL RL TRAINING PATTERN:
        # Set seed ONCE at start
        print("\nStep 1: Set seed=42 at initialization")
        obs, info = env.reset(seed=42)

        initial_agent_pos = env.unwrapped.agent_pos
        initial_mission = getattr(env.unwrapped, 'mission', 'N/A')
        print(f"  Initial: agent_pos={initial_agent_pos}, mission={initial_mission}")

        # Now reset multiple times WITHOUT specifying seed (like in training loop)
        print("\nStep 2: Reset 5 times WITHOUT specifying seed (training simulation)")

        layouts = []
        for episode in range(5):
            obs, info = env.reset()  # NO SEED PARAMETER

            layout = {
                'episode': episode,
                'agent_pos': env.unwrapped.agent_pos,
                'agent_dir': env.unwrapped.agent_dir,
                'mission': getattr(env.unwrapped, 'mission', 'N/A'),
            }
            layouts.append(layout)

            print(f"  Episode {episode}: agent_pos={layout['agent_pos']}, "
                  f"agent_dir={layout['agent_dir']}, mission={layout['mission']}")

        # Analysis
        print(f"\n{'─' * 40}")
        print("ANALYSIS:")
        print('─' * 40)

        # Check if all layouts are identical
        agent_positions = [layout['agent_pos'] for layout in layouts]
        agent_dirs = [layout['agent_dir'] for layout in layouts]
        missions = [layout['mission'] for layout in layouts]

        all_positions_same = len(set(agent_positions)) == 1
        all_dirs_same = len(set(agent_dirs)) == 1
        all_missions_same = len(set(missions)) == 1

        if all_positions_same and all_dirs_same:
            print("[MEMORIZATION] All episodes have IDENTICAL layouts!")
            print("  Agent always starts at:", agent_positions[0])
            print("  Agent always faces:", agent_dirs[0])
            if all_missions_same:
                print("  Mission always:", missions[0])
            print("\n  IMPLICATION: RL agent can MEMORIZE a fixed action sequence")
            print("  No need to use partial observations!")
        else:
            print("[GENERALIZATION] Episodes have DIFFERENT layouts!")
            print(f"  Agent positions vary: {set(agent_positions)}")
            print(f"  Agent directions vary: {set(agent_dirs)}")
            if not all_missions_same:
                print(f"  Missions vary: {set(missions)}")
            print("\n  IMPLICATION: RL agent MUST use partial observations")
            print("  Cannot memorize - must learn spatial reasoning!")

        env.close()

    print("\n" + "=" * 80)


def test_explicit_different_seeds_per_episode():
    """
    Test what happens if we EXPLICITLY set different seeds per episode.
    This is what might be needed for fair comparison to LLM evaluation.
    """
    print("\n\n" + "=" * 80)
    print("TEST: Explicit Different Seeds Per Episode")
    print("=" * 80)
    print("\nThis tests if we MANUALLY set different seeds each reset:")
    print("Like: env.reset(seed=0), env.reset(seed=1), env.reset(seed=2), ...\n")

    env_name = "BabyAI-GoToLocal-v0"
    env = gym.make(env_name)

    print(f"Environment: {env_name}\n")

    layouts = []
    for episode in range(5):
        obs, info = env.reset(seed=episode)  # DIFFERENT SEED EACH TIME

        layout = {
            'episode': episode,
            'seed': episode,
            'agent_pos': env.unwrapped.agent_pos,
            'mission': getattr(env.unwrapped, 'mission', 'N/A'),
        }
        layouts.append(layout)

        print(f"Episode {episode} (seed={episode}): agent_pos={layout['agent_pos']}, "
              f"mission={layout['mission']}")

    # Analysis
    print(f"\n{'─' * 40}")
    print("ANALYSIS:")
    print('─' * 40)

    agent_positions = [layout['agent_pos'] for layout in layouts]
    missions = [layout['mission'] for layout in layouts]

    all_same = len(set(agent_positions)) == 1 and len(set(missions)) == 1

    if all_same:
        print("[SAME] All episodes are identical despite different seeds!")
    else:
        print("[DIFFERENT] Each episode has unique layout!")
        print(f"  Unique positions: {len(set(agent_positions))}")
        print(f"  Unique missions: {len(set(missions))}")
        print("\n  This approach ensures maximum diversity for training")

    env.close()


def test_no_seed_ever():
    """
    Test what happens if we NEVER set a seed (truly random each time).
    """
    print("\n\n" + "=" * 80)
    print("TEST: No Seed Specification (Random Training)")
    print("=" * 80)
    print("\nThis tests if we NEVER set any seed (purely random):\n")

    env_name = "BabyAI-GoToLocal-v0"
    env = gym.make(env_name)

    print(f"Environment: {env_name}\n")

    layouts = []
    for episode in range(5):
        obs, info = env.reset()  # NO SEED AT ALL

        layout = {
            'episode': episode,
            'agent_pos': env.unwrapped.agent_pos,
            'mission': getattr(env.unwrapped, 'mission', 'N/A'),
        }
        layouts.append(layout)

        print(f"Episode {episode}: agent_pos={layout['agent_pos']}, "
              f"mission={layout['mission']}")

    # Analysis
    print(f"\n{'─' * 40}")
    print("ANALYSIS:")
    print('─' * 40)

    agent_positions = [layout['agent_pos'] for layout in layouts]
    missions = [layout['mission'] for layout in layouts]

    unique_positions = len(set(agent_positions))
    unique_missions = len(set(missions))

    print(f"Unique positions: {unique_positions}/5")
    print(f"Unique missions: {unique_missions}/5")

    if unique_positions > 1 or unique_missions > 1:
        print("\n[RANDOM] Episodes are randomized without seed!")
        print("  Training will have environment diversity")
    else:
        print("\n[FIXED] Episodes are identical even without seed!")

    env.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EMPIRICAL TEST: How Do Seeds Work in RL Training?")
    print("=" * 80)
    print("\nBased on BabyAI paper and Gymnasium documentation:")
    print("- BabyAI paper used PPO/A2C for RL training")
    print("- Recommended 'varying --seed' across training runs")
    print("- Question: What happens WITHIN a single training run?\n")
    print("Sources:")
    print("- https://github.com/mila-iqia/babyai")
    print("- https://gymnasium.farama.org/introduction/basic_usage/")
    print("\n")

    test_single_seed_multiple_resets()
    test_explicit_different_seeds_per_episode()
    test_no_seed_ever()

    print("\n\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
    print("\nConclusions will show:")
    print("1. Whether current RL training has environment diversity")
    print("2. Whether agents can memorize or must use observations")
    print("3. What changes (if any) are needed for proper training")
    print("\n")
