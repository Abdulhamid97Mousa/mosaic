"""Integration tests for MiniGrid with real vLLM server.

These tests require a running vLLM server at http://127.0.0.1:8000
with the Qwen/Qwen2.5-1.5B-Instruct model loaded.

Run with:
    cd /home/hamid/Desktop/Projects/GUI_BDI_RL
    source .venv/bin/activate
    pytest 3rd_party/mosaic/llm_worker/llm_worker/tests/test_minigrid_vllm.py -v -s
"""

import os
import warnings

# CRITICAL: Clear ALL proxy environment variables BEFORE importing OpenAI/httpx
# httpx (used by OpenAI client) doesn't support SOCKS proxies and will fail
# These must be cleared before any imports that might trigger httpx initialization
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_var, None)

# Set NO_PROXY for localhost
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import pytest
import requests

# Suppress gym deprecation warnings
warnings.filterwarnings("ignore", message=".*Gym.*")

# Check if vLLM server is available
VLLM_BASE_URL = "http://127.0.0.1:8000"
VLLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def vllm_available():
    """Check if vLLM server is running."""
    try:
        response = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


# Skip all tests if vLLM is not available
pytestmark = pytest.mark.skipif(
    not vllm_available(),
    reason="vLLM server not available at http://127.0.0.1:8000"
)


class TestMiniGridEnvironment:
    """Test MiniGrid environment creation and wrappers."""

    def test_create_minigrid_env(self):
        """Test that MiniGrid environment can be created with wrappers."""
        from llm_worker.environments.minigrid import make_minigrid_env

        class MockConfig:
            pass

        config = MockConfig()
        env = make_minigrid_env("minigrid", "MiniGrid-Empty-5x5-v0", config, render_mode="rgb_array")

        assert env is not None
        obs, info = env.reset(seed=42)

        # Check observation structure
        assert "text" in obs
        assert "long_term_context" in obs["text"]
        assert "mission" in obs

        # Check info has descriptions
        assert "descriptions" in info

        env.close()

    def test_minigrid_action_space(self):
        """Test that action space is correctly defined."""
        from llm_worker.environments.minigrid import make_minigrid_env
        from llm_worker.environments.minigrid.clean_lang_wrapper import MINIGRID_ACTION_SPACE

        expected_actions = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]
        assert MINIGRID_ACTION_SPACE == expected_actions

    def test_minigrid_object_names(self):
        """Test that object names are correctly mapped."""
        from llm_worker.environments.minigrid import MINIGRID_OBJECT_NAMES

        # Goal should be index 8, not 7 (which is box)
        assert MINIGRID_OBJECT_NAMES[8] == "goal"
        assert MINIGRID_OBJECT_NAMES[9] == "lava"
        assert MINIGRID_OBJECT_NAMES[7] == "box"


class TestCaseInsensitiveActions:
    """Test case-insensitive action matching."""

    def test_env_wrapper_case_insensitive(self):
        """Test that env_wrapper normalizes actions to lowercase."""
        from llm_worker.environments import make_env

        class MockConfig:
            pass

        config = MockConfig()
        env = make_env("minigrid", "MiniGrid-Empty-5x5-v0", config, render_mode="rgb_array")

        # Test various case variations - all should normalize to lowercase
        test_cases = [
            ("turn left", "turn left"),
            ("Turn Left", "turn left"),
            ("TURN LEFT", "turn left"),
            ("  turn left  ", "turn left"),
            ("Go Forward", "go forward"),
            ("GO FORWARD", "go forward"),
        ]

        for input_action, expected in test_cases:
            result = env.check_action_validity(input_action)
            assert result == expected, f"Input '{input_action}' should normalize to '{expected}', got '{result}'"

        # Test that invalid actions raise ValueError
        with pytest.raises(ValueError, match="Invalid action"):
            env.check_action_validity("invalid action")

        env.close()

    def test_minigrid_wrapper_step_case_insensitive(self):
        """Test that MiniGrid wrapper step handles case variations."""
        from llm_worker.environments.minigrid import make_minigrid_env

        class MockConfig:
            pass

        config = MockConfig()
        env = make_minigrid_env("minigrid", "MiniGrid-Empty-5x5-v0", config, render_mode="rgb_array")
        obs, info = env.reset(seed=42)

        # Test that capitalized actions work
        test_actions = ["Go forward", "TURN LEFT", "turn right", "GO FORWARD"]

        for action in test_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            if terminated or truncated:
                obs, info = env.reset(seed=42)

        # Test that invalid actions raise ValueError
        with pytest.raises(ValueError, match="Invalid action"):
            env.step("invalid action")

        env.close()


class TestVLLMIntegration:
    """Test integration with real vLLM server."""

    def test_vllm_chat_completion(self):
        """Test basic chat completion with vLLM."""
        from openai import OpenAI

        client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="not-needed")

        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "user", "content": "Say 'hello' and nothing else."}
            ],
            max_tokens=10,
            temperature=0.0
        )

        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_vllm_minigrid_action(self):
        """Test that vLLM can generate valid MiniGrid actions."""
        from openai import OpenAI
        from llm_worker.environments.minigrid import get_instruction_prompt

        client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="not-needed")

        # Get the system prompt
        system_prompt = get_instruction_prompt(mission="get to the green goal square")

        # Create a simple observation
        observation = """Mission: get to the green goal square

Environment:
a green goal 3 steps ahead"""

        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "user", "content": f"{system_prompt}\n\nObservation:\n{observation}\n\nWhat action do you take? Reply with only the action."}
            ],
            max_tokens=10,
            temperature=0.0
        )

        action = response.choices[0].message.content.strip().lower()
        valid_actions = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]

        # The action should be one of the valid actions (case-insensitive)
        assert action in valid_actions, f"LLM returned invalid action: '{action}'"


class TestFullIntegration:
    """Full end-to-end integration tests."""

    def test_minigrid_single_episode(self):
        """Run a single episode of MiniGrid with LLM agent."""
        from openai import OpenAI
        from llm_worker.environments import make_env
        from llm_worker.environments.minigrid import get_instruction_prompt

        client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="not-needed")

        class MockConfig:
            pass

        config = MockConfig()
        env = make_env("minigrid", "MiniGrid-Empty-5x5-v0", config, render_mode="rgb_array")

        obs, info = env.reset(seed=42)
        system_prompt = get_instruction_prompt(mission=obs.get("mission", "navigate"))

        max_steps = 20
        total_reward = 0.0
        actions_taken = []

        invalid_action_count = 0

        for step in range(max_steps):
            # Get observation text
            obs_text = obs.get("text", {}).get("long_term_context", "nothing special")
            mission = obs.get("mission", "get to the green goal square")

            full_obs = f"Mission: {mission}\n\nEnvironment:\n{obs_text}"

            # Get LLM action
            response = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {"role": "user", "content": f"{system_prompt}\n\nCurrent Observation:\n{full_obs}\n\nOutput only the action:"}
                ],
                max_tokens=10,
                temperature=0.7
            )

            raw_action = response.choices[0].message.content.strip()

            # Validate action (case-insensitive) - now raises ValueError for invalid actions
            try:
                action = env.check_action_validity(raw_action)
            except ValueError:
                invalid_action_count += 1
                print(f"Step {step}: INVALID action '{raw_action}' - skipping")
                continue

            actions_taken.append(action)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"Step {step}: action='{action}' (raw='{raw_action}'), reward={reward}")

            if terminated or truncated:
                print(f"Episode finished at step {step} with total reward {total_reward}")
                break

        print(f"Invalid actions encountered: {invalid_action_count}")

        env.close()

        # Check that we took various actions (not just "go forward")
        unique_actions = set(actions_taken)
        print(f"Unique actions taken: {unique_actions}")

        # The LLM should use at least 2 different action types
        assert len(unique_actions) >= 2, f"LLM only used actions: {unique_actions}"

    def test_description_shows_goal_not_lava(self):
        """Test that goal is described as 'goal' not 'lava'."""
        from llm_worker.environments.minigrid import make_minigrid_env, generate_minigrid_descriptions
        import gymnasium as gym
        import minigrid

        minigrid.register_minigrid_envs()

        class MockConfig:
            pass

        config = MockConfig()

        # Use a small grid where goal is likely visible
        env = make_minigrid_env("minigrid", "MiniGrid-Empty-5x5-v0", config, render_mode="rgb_array")

        # Try different seeds to find one where goal is visible
        goal_found = False
        for seed in range(50):
            obs, info = env.reset(seed=seed)
            descriptions = info.get("descriptions", [])
            desc_text = " ".join(descriptions).lower()

            if "goal" in desc_text:
                print(f"Seed {seed}: Found goal in description: {descriptions}")
                assert "lava" not in desc_text, f"Goal incorrectly labeled as lava: {descriptions}"
                goal_found = True
                break

        if not goal_found:
            # Walk around to find the goal
            for _ in range(30):
                obs, reward, term, trunc, info = env.step("go forward")
                descriptions = info.get("descriptions", [])
                desc_text = " ".join(descriptions).lower()

                if "goal" in desc_text:
                    print(f"Found goal after walking: {descriptions}")
                    assert "lava" not in desc_text, f"Goal incorrectly labeled as lava"
                    goal_found = True
                    break

                if term or trunc:
                    break

        env.close()
        assert goal_found, "Could not find goal in any observation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
