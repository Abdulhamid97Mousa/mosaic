"""End-to-end test with actual LLM for MOSAIC MultiGrid Extension.

Tests the complete pipeline:
1. Generate instruction prompt
2. Generate observation description
3. Send to LLM via OpenRouter
4. Parse LLM response to extract action
5. Execute action in MultiGrid environment
"""

import os
import sys
from pathlib import Path

import requests

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gym-multigrid"))

from mosaic_extension.multigrid import (
    get_instruction_prompt_level1,
    get_instruction_prompt_level2,
    get_instruction_prompt_level3,
    describe_observation_egocentric,
    describe_observation_with_teammates,
    extract_visible_teammates,
    parse_action,
    MULTIGRID_ACTION_SPACE,
)


def call_llm(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Call OpenRouter API with prompts.

    Args:
        system_prompt: Instruction prompt for the agent
        user_prompt: Current observation
        api_key: OpenRouter API key

    Returns:
        LLM response text
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "anthropic/claude-3.5-haiku",  # Fast and cheap for testing
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 100,  # Short response expected
        "temperature": 0.7,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def test_llm_level1_soccer(api_key: str):
    """Test Level 1 (Emergent) prompt with real LLM."""
    print("\n" + "="*70)
    print("Testing Level 1 (Emergent) Coordination - Soccer")
    print("="*70)

    from gym_multigrid.envs import SoccerGame4HEnv10x15N2

    # Create environment
    env = SoccerGame4HEnv10x15N2()
    obs_list = env.reset()

    # Get agent 0's observation
    agent_id = 0
    team = 0
    obs = obs_list[agent_id]

    # Generate instruction prompt
    instruction_prompt = get_instruction_prompt_level1(
        agent_id=agent_id,
        team=team,
        env_id="MultiGrid-Soccer-v0"
    )

    # Generate observation description
    observation_text = describe_observation_egocentric(
        obs, agent_direction=0, carrying=None
    )

    # Combine into user prompt
    user_prompt = f"{observation_text}\n\nWhat action do you take? Respond with just the action name."

    print("\n--- Instruction Prompt ---")
    print(instruction_prompt[:300] + "...")

    print("\n--- Observation ---")
    print(observation_text)

    print("\n--- Sending to LLM ---")
    llm_response = call_llm(instruction_prompt, user_prompt, api_key)

    print("\n--- LLM Response ---")
    print(llm_response)

    # Parse action
    action_idx = parse_action(llm_response)
    action_name = MULTIGRID_ACTION_SPACE[action_idx]

    print(f"\n--- Parsed Action ---")
    print(f"Action: {action_idx} ({action_name})")

    # Execute in environment
    print("\n--- Executing Action ---")
    obs_list, rewards, done, info = env.step([action_idx, 0, 0, 0])
    print(f"Rewards: {rewards}")
    print(f"Done: {done}")

    print("\n✅ Level 1 LLM test passed!")
    env.close()
    return True


def test_llm_level2_soccer(api_key: str):
    """Test Level 2 (Basic Hints) prompt with real LLM."""
    print("\n" + "="*70)
    print("Testing Level 2 (Basic Hints) Coordination - Soccer")
    print("="*70)

    from gym_multigrid.envs import SoccerGame4HEnv10x15N2

    # Create environment
    env = SoccerGame4HEnv10x15N2()
    obs_list = env.reset()

    # Get agent 0's observation
    agent_id = 0
    team = 0
    obs = obs_list[agent_id]

    # Generate instruction prompt
    instruction_prompt = get_instruction_prompt_level2(
        agent_id=agent_id,
        team=team,
        env_id="MultiGrid-Soccer-v0"
    )

    # Generate observation description
    observation_text = describe_observation_egocentric(
        obs, agent_direction=0, carrying=None
    )

    # Combine into user prompt
    user_prompt = f"{observation_text}\n\nWhat action do you take? Respond with just the action name."

    print("\n--- Instruction Prompt (with coordination tips) ---")
    print(instruction_prompt[:400] + "...")

    print("\n--- Observation ---")
    print(observation_text)

    print("\n--- Sending to LLM ---")
    llm_response = call_llm(instruction_prompt, user_prompt, api_key)

    print("\n--- LLM Response ---")
    print(llm_response)

    # Parse action
    action_idx = parse_action(llm_response)
    action_name = MULTIGRID_ACTION_SPACE[action_idx]

    print(f"\n--- Parsed Action ---")
    print(f"Action: {action_idx} ({action_name})")

    # Execute in environment
    print("\n--- Executing Action ---")
    obs_list, rewards, done, info = env.step([action_idx, 0, 0, 0])
    print(f"Rewards: {rewards}")
    print(f"Done: {done}")

    print("\n✅ Level 2 LLM test passed!")
    env.close()
    return True


def test_llm_level3_soccer(api_key: str):
    """Test Level 3 (Role-Based) prompt with real LLM."""
    print("\n" + "="*70)
    print("Testing Level 3 (Role-Based) Coordination - Soccer Forward")
    print("="*70)

    from gym_multigrid.envs import SoccerGame4HEnv10x15N2

    # Create environment
    env = SoccerGame4HEnv10x15N2()
    obs_list = env.reset()

    # Get agent 0's observation
    agent_id = 0
    team = 0
    role = "forward"
    obs = obs_list[agent_id]

    # Generate instruction prompt
    instruction_prompt = get_instruction_prompt_level3(
        agent_id=agent_id,
        team=team,
        role=role,
        env_id="MultiGrid-Soccer-v0"
    )

    # Generate observation description
    observation_text = describe_observation_egocentric(
        obs, agent_direction=0, carrying=None
    )

    # Combine into user prompt
    user_prompt = f"{observation_text}\n\nWhat action do you take? Respond with just the action name."

    print("\n--- Instruction Prompt (with role: FORWARD) ---")
    print(instruction_prompt[:400] + "...")

    print("\n--- Observation ---")
    print(observation_text)

    print("\n--- Sending to LLM ---")
    llm_response = call_llm(instruction_prompt, user_prompt, api_key)

    print("\n--- LLM Response ---")
    print(llm_response)

    # Parse action
    action_idx = parse_action(llm_response)
    action_name = MULTIGRID_ACTION_SPACE[action_idx]

    print(f"\n--- Parsed Action ---")
    print(f"Action: {action_idx} ({action_name})")

    # Execute in environment
    print("\n--- Executing Action ---")
    obs_list, rewards, done, info = env.step([action_idx, 0, 0, 0])
    print(f"Rewards: {rewards}")
    print(f"Done: {done}")

    print("\n✅ Level 3 LLM test passed!")
    env.close()
    return True


def test_llm_observation_mode_egocentric(api_key: str):
    """Test Observation Mode 1: Egocentric (Own view only)."""
    print("\n" + "="*70)
    print("Testing Observation Mode: EGOCENTRIC (Own view only)")
    print("="*70)

    from gym_multigrid.envs import SoccerGame4HEnv10x15N2

    # Create environment
    env = SoccerGame4HEnv10x15N2()
    obs_list = env.reset()

    agent_id = 0
    team = 0
    obs = obs_list[agent_id]

    # Generate instruction prompt
    instruction_prompt = get_instruction_prompt_level2(
        agent_id=agent_id,
        team=team,
        env_id="MultiGrid-Soccer-v0"
    )

    # EGOCENTRIC observation (no teammate info)
    observation_text = describe_observation_egocentric(
        obs, agent_direction=0, carrying=None
    )

    user_prompt = f"{observation_text}\n\nWhat action do you take? Respond with just the action name."

    print("\n--- Observation Mode: Egocentric ---")
    print(observation_text)
    print("\nNote: No teammate information included (decentralized)")

    print("\n--- Sending to LLM ---")
    llm_response = call_llm(instruction_prompt, user_prompt, api_key)

    print("\n--- LLM Response ---")
    print(llm_response)

    # Parse and execute
    action_idx = parse_action(llm_response)
    action_name = MULTIGRID_ACTION_SPACE[action_idx]
    print(f"\n--- Parsed Action: {action_idx} ({action_name}) ---")

    obs_list, rewards, done, info = env.step([action_idx, 0, 0, 0])
    print(f"Executed successfully! Reward: {rewards[agent_id]}")

    print("\n✅ Egocentric observation mode test passed!")
    env.close()
    return True


def test_llm_observation_mode_teammates(api_key: str):
    """Test Observation Mode 2: Visible Teammates (Theory of Mind)."""
    print("\n" + "="*70)
    print("Testing Observation Mode: VISIBLE TEAMMATES (Theory of Mind)")
    print("="*70)

    from gym_multigrid.envs import SoccerGame4HEnv10x15N2

    # Create environment
    env = SoccerGame4HEnv10x15N2()
    obs_list = env.reset()

    agent_id = 0
    team = 0
    obs = obs_list[agent_id]

    # Generate instruction prompt
    instruction_prompt = get_instruction_prompt_level2(
        agent_id=agent_id,
        team=team,
        env_id="MultiGrid-Soccer-v0"
    )

    # VISIBLE TEAMMATES observation (includes teammate info)
    visible_teammates = extract_visible_teammates(env, agent_id, team)

    observation_text = describe_observation_with_teammates(
        obs,
        agent_id=agent_id,
        visible_teammates=visible_teammates,
        agent_direction=0,
        carrying=None
    )

    user_prompt = f"{observation_text}\n\nWhat action do you take? Respond with just the action name."

    print("\n--- Observation Mode: Visible Teammates ---")
    print(observation_text)
    print(f"\nNote: Includes {len(visible_teammates)} visible teammate(s) (Theory of Mind)")

    print("\n--- Sending to LLM ---")
    llm_response = call_llm(instruction_prompt, user_prompt, api_key)

    print("\n--- LLM Response ---")
    print(llm_response)

    # Parse and execute
    action_idx = parse_action(llm_response)
    action_name = MULTIGRID_ACTION_SPACE[action_idx]
    print(f"\n--- Parsed Action: {action_idx} ({action_name}) ---")

    obs_list, rewards, done, info = env.step([action_idx, 0, 0, 0])
    print(f"Executed successfully! Reward: {rewards[agent_id]}")

    print("\n✅ Visible Teammates observation mode test passed!")
    env.close()
    return True


def test_llm_compare_observation_modes(api_key: str):
    """Compare LLM responses with both observation modes."""
    print("\n" + "="*70)
    print("Comparing Observation Modes: Egocentric vs Visible Teammates")
    print("="*70)

    from gym_multigrid.envs import SoccerGame4HEnv10x15N2

    # Create environment
    env = SoccerGame4HEnv10x15N2()
    obs_list = env.reset()

    agent_id = 0
    team = 0
    obs = obs_list[agent_id]

    # Instruction prompt
    instruction_prompt = get_instruction_prompt_level2(
        agent_id=agent_id,
        team=team,
        env_id="MultiGrid-Soccer-v0"
    )

    # Test 1: Egocentric
    print("\n--- MODE 1: Egocentric Only ---")
    obs_egocentric = describe_observation_egocentric(
        obs, agent_direction=0, carrying=None
    )
    print(obs_egocentric)

    user_prompt_1 = f"{obs_egocentric}\n\nWhat action do you take? Respond with just the action name."
    llm_response_1 = call_llm(instruction_prompt, user_prompt_1, api_key)
    action_1 = parse_action(llm_response_1)
    print(f"\nLLM Decision: {llm_response_1.strip()} → {MULTIGRID_ACTION_SPACE[action_1]}")

    # Test 2: Visible Teammates
    print("\n--- MODE 2: Visible Teammates ---")
    visible_teammates = extract_visible_teammates(env, agent_id, team)
    obs_teammates = describe_observation_with_teammates(
        obs,
        agent_id=agent_id,
        visible_teammates=visible_teammates,
        agent_direction=0,
        carrying=None
    )
    print(obs_teammates)

    user_prompt_2 = f"{obs_teammates}\n\nWhat action do you take? Respond with just the action name."
    llm_response_2 = call_llm(instruction_prompt, user_prompt_2, api_key)
    action_2 = parse_action(llm_response_2)
    print(f"\nLLM Decision: {llm_response_2.strip()} → {MULTIGRID_ACTION_SPACE[action_2]}")

    # Compare
    print("\n--- Comparison ---")
    print(f"Egocentric mode: {MULTIGRID_ACTION_SPACE[action_1]}")
    print(f"Teammates mode:  {MULTIGRID_ACTION_SPACE[action_2]}")
    if action_1 == action_2:
        print("Both modes produced same action")
    else:
        print("Modes produced different actions (Theory of Mind may affect decision)")

    print("\n✅ Observation mode comparison test passed!")
    env.close()
    return True


def test_llm_multi_turn(api_key: str):
    """Test multi-turn interaction with LLM."""
    print("\n" + "="*70)
    print("Testing Multi-Turn LLM Interaction (5 steps)")
    print("="*70)

    from gym_multigrid.envs import SoccerGame4HEnv10x15N2

    # Create environment
    env = SoccerGame4HEnv10x15N2()
    obs_list = env.reset()

    agent_id = 0
    team = 0

    # Generate instruction prompt once
    instruction_prompt = get_instruction_prompt_level2(
        agent_id=agent_id,
        team=team,
        env_id="MultiGrid-Soccer-v0"
    )

    print(f"\n--- Playing 5 steps with Agent {agent_id} ---\n")

    for step in range(5):
        obs = obs_list[agent_id]

        # Generate observation
        observation_text = describe_observation_egocentric(
            obs, agent_direction=0, carrying=None
        )

        # Create prompt
        user_prompt = f"{observation_text}\n\nWhat action do you take? Respond with just the action name."

        print(f"Step {step + 1}:")
        print(f"  Observation: {observation_text[:80]}...")

        # Get LLM action
        llm_response = call_llm(instruction_prompt, user_prompt, api_key)
        print(f"  LLM says: {llm_response.strip()}")

        # Parse and execute
        action_idx = parse_action(llm_response)
        action_name = MULTIGRID_ACTION_SPACE[action_idx]
        print(f"  Action: {action_idx} ({action_name})")

        obs_list, rewards, done, info = env.step([action_idx, 0, 0, 0])
        print(f"  Reward: {rewards[agent_id]}")

        if done:
            print("  Episode done!")
            break
        print()

    print("\n✅ Multi-turn LLM test passed!")
    env.close()
    return True


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Usage: OPENROUTER_API_KEY='your-key' python test_llm_integration.py")
        sys.exit(1)

    print("\n" + "="*70)
    print("MOSAIC MultiGrid Extension - LLM Integration Tests")
    print("="*70)
    print(f"Using OpenRouter API")
    print(f"Model: anthropic/claude-3.5-haiku")

    try:
        # Test all coordination levels
        test_llm_level1_soccer(api_key)
        test_llm_level2_soccer(api_key)
        test_llm_level3_soccer(api_key)

        # Test both observation modes
        test_llm_observation_mode_egocentric(api_key)
        test_llm_observation_mode_teammates(api_key)
        test_llm_compare_observation_modes(api_key)

        # Test multi-turn
        test_llm_multi_turn(api_key)

        print("\n" + "="*70)
        print("✅ ALL LLM INTEGRATION TESTS PASSED!")
        print("  - 3 Coordination Levels ✓")
        print("  - 2 Observation Modes ✓")
        print("  - Multi-turn interaction ✓")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
