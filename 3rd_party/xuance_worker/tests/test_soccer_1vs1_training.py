"""Tests for 1vs1 Soccer MAPPO training flow.

Validates the full integration chain from MultiGrid_Env through XuanCe:
1. Info dict compatibility with OnPolicyMARLAgents.store_experience()
2. DummyVecMultiAgentEnv compatibility with our MultiGrid_Env
3. RunnerMARL selection for soccer_1vs1 (not RunnerCompetition)
4. YAML config correctness
5. End-to-end env step loop

These tests use the REAL mosaic_multigrid environment (not mocked) to
validate actual observation shapes, reward values, and info dict structure
against what XuanCe's on-policy agents expect.

Root cause of previous failure:
    RunnerCompetition.train() calls agent.store_experience() with the
    OFF-POLICY signature: (obs, avail, actions, next_obs, next_avail, rew, term, info).
    But MAPPO (on-policy) expects: (obs, avail, actions, log_pi, rew, values, term, info).
    When use_actions_mask=False, next_avail_actions=None flows into the
    on-policy rewards_dict parameter, causing TypeError: NoneType not iterable.
    Fix: Use RunnerMARL which delegates to the agent's own train() loop.
"""

from __future__ import annotations

import importlib
from operator import itemgetter
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml


# =============================================================================
# Skip if mosaic_multigrid is not installed
# =============================================================================

try:
    from mosaic_multigrid.envs import SoccerGame2HIndAgObsEnv16x11N2
    _HAS_MOSAIC = True
except ImportError:
    _HAS_MOSAIC = False

requires_mosaic = pytest.mark.skipif(
    not _HAS_MOSAIC,
    reason="mosaic_multigrid not installed"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def soccer_1vs1_config():
    """Create config for 1vs1 Soccer environment."""
    return SimpleNamespace(
        env_name="multigrid",
        env_id="soccer_1vs1",
        env_seed=42,
        render_mode=None,
        training_mode="competitive",
    )


@pytest.fixture
def multigrid_env(soccer_1vs1_config):
    """Create a real MultiGrid_Env for 1vs1 soccer."""
    if not _HAS_MOSAIC:
        pytest.skip("mosaic_multigrid not installed")
    from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env
    env = MultiGrid_Env(soccer_1vs1_config)
    yield env
    env.close()


@pytest.fixture
def env_after_reset(multigrid_env):
    """MultiGrid_Env after calling reset()."""
    obs, info = multigrid_env.reset()
    return multigrid_env, obs, info


# =============================================================================
# Test: Info Dict Compatibility with OnPolicyMARLAgents
# =============================================================================


class TestInfoDictCompatibility:
    """Validate that step() info dict has all fields required by MAPPO.

    OnPolicyMARLAgents.store_experience() reads from info:
      - info[e]['agent_mask'][k]  -- every step (line 86)
      - info[e]['episode_step']   -- episode-end logging (line 342)
      - info[e]['episode_score']  -- episode-end logging (line 349)

    DummyVecMultiAgentEnv.step_wait() adds on episode end:
      - info[e]['reset_obs']           -- auto-reset observations
      - info[e]['reset_avail_actions'] -- auto-reset action masks
      - info[e]['reset_state']         -- auto-reset global state
    """

    @requires_mosaic
    def test_step_info_has_agent_mask(self, env_after_reset):
        """agent_mask must be present in every step's info dict."""
        env, obs, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        _, _, _, _, info = env.step(actions)

        assert "agent_mask" in info, (
            "info dict missing 'agent_mask' -- required by "
            "OnPolicyMARLAgents.store_experience() line 86"
        )
        mask = info["agent_mask"]
        assert isinstance(mask, dict)
        for agent in env.agents:
            assert agent in mask, f"agent_mask missing key '{agent}'"
            assert mask[agent] is True, "All soccer agents are always alive"

    @requires_mosaic
    def test_step_info_has_episode_step(self, env_after_reset):
        """episode_step must be present for logging on episode end."""
        env, obs, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        _, _, _, _, info = env.step(actions)

        assert "episode_step" in info, (
            "info dict missing 'episode_step' -- required by "
            "MARL runner logging (line 342)"
        )
        assert isinstance(info["episode_step"], int)
        assert info["episode_step"] == 1  # First step after reset

    @requires_mosaic
    def test_step_info_has_episode_score(self, env_after_reset):
        """episode_score must be dict keyed by agent names."""
        env, obs, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        _, _, _, _, info = env.step(actions)

        assert "episode_score" in info, (
            "info dict missing 'episode_score' -- required by "
            "MARL runner logging (line 349): "
            "np.mean(itemgetter(*agent_keys)(info[i]['episode_score']))"
        )
        score = info["episode_score"]
        assert isinstance(score, dict)
        for agent in env.agents:
            assert agent in score, f"episode_score missing key '{agent}'"
            assert isinstance(score[agent], float)

    @requires_mosaic
    def test_episode_score_works_with_itemgetter(self, env_after_reset):
        """XuanCe uses itemgetter(*agent_keys) on episode_score."""
        env, obs, _ = env_after_reset
        agent_keys = env.agents
        actions = {agent: 0 for agent in agent_keys}
        _, _, _, _, info = env.step(actions)

        # Replicate XuanCe's exact access pattern (on_policy_marl.py line 349)
        episode_score = info["episode_score"]
        mean_score = np.mean(itemgetter(*agent_keys)(episode_score))
        assert isinstance(mean_score, (float, np.floating))

    @requires_mosaic
    def test_agent_mask_works_with_store_experience_pattern(self, env_after_reset):
        """Replicate the exact access pattern from store_experience line 86."""
        env, obs, _ = env_after_reset
        agent_keys = env.agents

        # Run 3 steps to build a list of info dicts (simulates vectorized env)
        info_list = []
        for _ in range(3):
            actions = {agent: np.random.randint(7) for agent in agent_keys}
            _, _, _, _, info = env.step(actions)
            info_list.append(info)

        # Replicate XuanCe's exact access pattern (on_policy_marl.py line 86):
        # {k: np.array([data['agent_mask'][k] for data in info]) for k in agent_keys}
        agent_mask_arrays = {
            k: np.array([data['agent_mask'][k] for data in info_list])
            for k in agent_keys
        }

        for k in agent_keys:
            assert agent_mask_arrays[k].shape == (3,)
            assert np.all(agent_mask_arrays[k])  # All alive


# =============================================================================
# Test: DummyVecMultiAgentEnv Compatibility
# =============================================================================


class TestVecEnvCompatibility:
    """Validate that MultiGrid_Env works with DummyVecMultiAgentEnv."""

    @requires_mosaic
    def test_reset_info_has_state(self, multigrid_env):
        """DummyVecMultiAgentEnv.reset() reads info['state']."""
        _, info = multigrid_env.reset()
        assert "state" in info
        assert isinstance(info["state"], np.ndarray)

    @requires_mosaic
    def test_reset_info_has_avail_actions(self, multigrid_env):
        """DummyVecMultiAgentEnv.reset() reads info['avail_actions']."""
        _, info = multigrid_env.reset()
        assert "avail_actions" in info
        assert isinstance(info["avail_actions"], dict)

    @requires_mosaic
    def test_step_info_has_state(self, env_after_reset):
        """DummyVecMultiAgentEnv.step_wait() reads info['state']."""
        env, _, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        _, _, _, _, info = env.step(actions)
        assert "state" in info
        assert isinstance(info["state"], np.ndarray)

    @requires_mosaic
    def test_step_info_has_avail_actions(self, env_after_reset):
        """DummyVecMultiAgentEnv.step_wait() reads info['avail_actions']."""
        env, _, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        _, _, _, _, info = env.step(actions)
        assert "avail_actions" in info


# =============================================================================
# Test: Observation and Action Space Shapes
# =============================================================================


class TestObsActionShapes:
    """Validate that observation and action spaces are MLP-compatible."""

    @requires_mosaic
    def test_observation_space_is_flat(self, multigrid_env):
        """Observation space must be flat (27,) not (3,3,3) for Basic_MLP."""
        for agent in multigrid_env.agents:
            obs_space = multigrid_env.observation_space[agent]
            assert len(obs_space.shape) == 1, (
                f"Observation space must be 1D for Basic_MLP, "
                f"got shape {obs_space.shape}"
            )
            assert obs_space.shape[0] == 27, (
                f"Expected flattened (3*3*3=27), got {obs_space.shape[0]}"
            )

    @requires_mosaic
    def test_observations_match_space(self, env_after_reset):
        """Actual observations must match the declared observation space."""
        env, obs, _ = env_after_reset
        for agent in env.agents:
            assert obs[agent].shape == env.observation_space[agent].shape, (
                f"Obs shape {obs[agent].shape} != "
                f"space shape {env.observation_space[agent].shape}"
            )
            assert obs[agent].dtype == np.float32

    @requires_mosaic
    def test_step_observations_match_space(self, env_after_reset):
        """Step observations must also match the declared space."""
        env, _, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
        for agent in env.agents:
            assert obs[agent].shape == env.observation_space[agent].shape
            assert obs[agent].dtype == np.float32

    @requires_mosaic
    def test_action_space_discrete_7(self, multigrid_env):
        """1vs1 soccer has 7 discrete actions per agent."""
        for agent in multigrid_env.agents:
            assert multigrid_env.action_space[agent].n == 7

    @requires_mosaic
    def test_num_agents_is_2(self, multigrid_env):
        """1vs1 soccer has exactly 2 agents."""
        assert multigrid_env.num_agents == 2
        assert len(multigrid_env.agents) == 2
        assert multigrid_env.agents == ["agent_0", "agent_1"]


# =============================================================================
# Test: Step Return Format
# =============================================================================


class TestStepReturnFormat:
    """Validate step() returns the 5-tuple format XuanCe expects."""

    @requires_mosaic
    def test_step_returns_5_tuple(self, env_after_reset):
        """step() must return (obs, rew, term, trunc, info)."""
        env, _, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        result = env.step(actions)
        assert len(result) == 5, f"step() must return 5-tuple, got {len(result)}"

    @requires_mosaic
    def test_step_obs_is_dict(self, env_after_reset):
        """Observations must be dict keyed by agent names."""
        env, _, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
        assert isinstance(obs, dict)
        for agent in env.agents:
            assert agent in obs

    @requires_mosaic
    def test_step_rewards_is_dict(self, env_after_reset):
        """Rewards must be dict keyed by agent names."""
        env, _, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        _, rewards, _, _, _ = env.step(actions)
        assert isinstance(rewards, dict)
        for agent in env.agents:
            assert agent in rewards
            assert isinstance(rewards[agent], float)

    @requires_mosaic
    def test_step_terminated_is_dict(self, env_after_reset):
        """Terminated must be dict keyed by agent names."""
        env, _, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        _, _, terminated, _, _ = env.step(actions)
        assert isinstance(terminated, dict)
        for agent in env.agents:
            assert agent in terminated
            assert isinstance(terminated[agent], bool)

    @requires_mosaic
    def test_step_truncated_is_bool(self, env_after_reset):
        """Truncated must be a single bool (not per-agent)."""
        env, _, _ = env_after_reset
        actions = {agent: 0 for agent in env.agents}
        _, _, _, truncated, _ = env.step(actions)
        assert isinstance(truncated, bool)


# =============================================================================
# Test: RunnerMARL Selection (not RunnerCompetition)
# =============================================================================


class TestRunnerSelection:
    """Validate that soccer_1vs1 uses RunnerMARL, not RunnerCompetition."""

    def test_soccer_1vs1_not_in_competition_groups(self):
        """soccer_1vs1 must NOT be in _COMPETITION_ENV_GROUPS."""
        from xuance_worker.runtime import _get_competition_num_groups

        result = _get_competition_num_groups("multigrid", "soccer_1vs1")
        assert result is None, (
            "soccer_1vs1 must NOT be in _COMPETITION_ENV_GROUPS. "
            "RunnerCompetition uses the off-policy store_experience signature "
            "which is incompatible with on-policy MAPPO."
        )

    def test_soccer_2vs2_still_in_competition_groups(self):
        """soccer (2vs2) should still use RunnerCompetition."""
        from xuance_worker.runtime import _get_competition_num_groups

        result = _get_competition_num_groups("multigrid", "soccer")
        assert result == 2, (
            "soccer (2vs2) should still be in _COMPETITION_ENV_GROUPS "
            "with 2 groups"
        )

    def test_method_is_string_not_list_for_soccer_1vs1(self):
        """When num_groups is None, method must be string (not list)."""
        from xuance_worker.runtime import _get_competition_num_groups

        num_groups = _get_competition_num_groups("multigrid", "soccer_1vs1")
        normalized_method = "mappo"

        if num_groups is not None:
            method_for_runner = [normalized_method] * num_groups
        else:
            method_for_runner = normalized_method

        assert isinstance(method_for_runner, str), (
            "For soccer_1vs1, method must be a string (triggers RunnerMARL), "
            f"but got {type(method_for_runner).__name__}"
        )

    def test_config_path_is_string_not_list_for_soccer_1vs1(self):
        """When num_groups is None, config_path must be str (not list)."""
        from xuance_worker.runtime import _resolve_custom_config_path

        config_path = _resolve_custom_config_path(
            method="mappo",
            env="multigrid",
            env_id="soccer_1vs1",
            num_groups=None,
            config_path=None,
        )

        # Should be a string path (not a list)
        if config_path is not None:
            assert isinstance(config_path, str), (
                f"config_path should be str for RunnerMARL, got {type(config_path)}"
            )


# =============================================================================
# Test: YAML Config Correctness
# =============================================================================


class TestYamlConfig:
    """Validate soccer_1vs1.yaml has correct settings."""

    @pytest.fixture
    def yaml_config(self):
        """Load the soccer_1vs1 YAML config."""
        import pathlib
        config_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "xuance_worker" / "configs" / "mappo" / "multigrid" / "soccer_1vs1.yaml"
        )
        assert config_path.exists(), f"Config not found: {config_path}"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_runner_is_marl(self, yaml_config):
        """Runner must be 'MARL' (registry key for RunnerMARL), not RunnerCompetition."""
        assert yaml_config["runner"] == "MARL", (
            f"Expected runner='MARL', got '{yaml_config['runner']}'. "
            "RunnerCompetition uses off-policy store_experience which is "
            "incompatible with on-policy MAPPO. "
            "Note: XuanCe REGISTRY_Runner uses 'MARL' as the key, not 'RunnerMARL'."
        )

    def test_agent_is_mappo(self, yaml_config):
        """Agent must be MAPPO."""
        assert yaml_config["agent"] == "MAPPO"

    def test_parameter_sharing_enabled(self, yaml_config):
        """use_parameter_sharing must be True for symmetric game (shared policy)."""
        assert yaml_config["use_parameter_sharing"] is True

    def test_representation_is_mlp(self, yaml_config):
        """Basic_MLP requires flat observation space."""
        assert yaml_config["representation"] == "Basic_MLP"

    def test_vectorize_is_multi_agent(self, yaml_config):
        """Must use DummyVecMultiAgentEnv for multi-agent environments."""
        assert yaml_config["vectorize"] == "DummyVecMultiAgentEnv"

    def test_use_gae(self, yaml_config):
        """MAPPO requires GAE for advantage estimation."""
        assert yaml_config.get("use_gae", False) is True

    def test_buffer_size_matches_episode_length(self, yaml_config):
        """Buffer size should match max_steps (200) for on-policy training."""
        assert yaml_config["buffer_size"] == 200


# =============================================================================
# Test: Full Reset-Step Loop
# =============================================================================


class TestResetStepLoop:
    """Validate a full reset -> step -> check cycle."""

    @requires_mosaic
    def test_multi_step_loop(self, multigrid_env):
        """Run 10 steps and validate info dict every step."""
        obs, info = multigrid_env.reset()

        for step_num in range(1, 11):
            # Random actions
            actions = {
                agent: np.random.randint(7)
                for agent in multigrid_env.agents
            }
            obs, rewards, terminated, truncated, info = multigrid_env.step(actions)

            # Validate info has all required fields
            assert "agent_mask" in info, f"Missing agent_mask at step {step_num}"
            assert "episode_step" in info, f"Missing episode_step at step {step_num}"
            assert "episode_score" in info, f"Missing episode_score at step {step_num}"
            assert "state" in info, f"Missing state at step {step_num}"
            assert "avail_actions" in info, f"Missing avail_actions at step {step_num}"

            # Validate episode_step increments
            assert info["episode_step"] == step_num

            # Validate observation shapes
            for agent in multigrid_env.agents:
                assert obs[agent].shape == (27,)
                assert obs[agent].dtype == np.float32

    @requires_mosaic
    def test_episode_score_accumulates(self, multigrid_env):
        """Episode score should accumulate rewards over steps."""
        obs, info = multigrid_env.reset()

        total_rewards = {agent: 0.0 for agent in multigrid_env.agents}

        for _ in range(5):
            actions = {
                agent: np.random.randint(7)
                for agent in multigrid_env.agents
            }
            obs, rewards, terminated, truncated, info = multigrid_env.step(actions)

            for agent in multigrid_env.agents:
                total_rewards[agent] += rewards[agent]

        # episode_score should match accumulated rewards
        for agent in multigrid_env.agents:
            assert abs(info["episode_score"][agent] - total_rewards[agent]) < 1e-6, (
                f"episode_score[{agent}]={info['episode_score'][agent]} "
                f"!= accumulated {total_rewards[agent]}"
            )


# =============================================================================
# Test: GymToGymnasiumWrapper Reset
# =============================================================================


class TestGymToGymnasiumWrapper:
    """Validate GymToGymnasiumWrapper passes through Gymnasium API."""

    @requires_mosaic
    def test_wrapper_passthrough_gymnasium_reset(self):
        """Wrapper should NOT double-wrap Gymnasium API reset."""
        from xuance_worker.environments.mosaic_multigrid import GymToGymnasiumWrapper

        # Create a mock env that returns Gymnasium-style reset
        mock_obs = {"agent_0": np.zeros(27), "agent_1": np.zeros(27)}
        mock_info = {"state": np.zeros(54)}

        mock_env = MagicMock()
        mock_env.observation_space = {0: MagicMock()}
        mock_env.action_space = {0: MagicMock()}
        mock_env.max_steps = 200
        mock_env.reset.return_value = (mock_obs, mock_info)

        wrapper = GymToGymnasiumWrapper(mock_env)
        obs, info = wrapper.reset()

        # Should be the original dicts, not wrapped again
        assert obs is mock_obs, "Wrapper double-wrapped Gymnasium reset!"
        assert info is mock_info
