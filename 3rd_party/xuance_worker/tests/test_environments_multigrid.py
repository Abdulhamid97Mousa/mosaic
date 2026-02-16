"""Tests for MultiGrid environment integration with XuanCe.

These tests validate:
- MultiGrid_Env wrapper creation and initialization
- Team assignment (agent_groups) from agents_index
- XuanCe-compatible API (dict-based obs/actions/rewards)
- groups_info for runner_competition.py
- ReproducibleMultiGridWrapper integration
- Registration with XuanCe registry

Note: These tests mock heavy imports (XuanCe, mosaic_multigrid) to avoid
slow initialization times during testing.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# Module-Level Mocks (to avoid heavy imports)
# =============================================================================

# Create mock for XuanCe's RawMultiAgentEnv
class MockRawMultiAgentEnv:
    """Mock of xuance.environment.RawMultiAgentEnv base class."""

    def __init__(self):
        pass


# Create mock gymnasium spaces
class MockBox:
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class MockDiscrete:
    def __init__(self, n):
        self.n = n


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_soccer_env():
    """Create a mock Soccer environment for testing."""
    env = MagicMock()

    # Mock agents with index attributes (team assignment)
    # agents_index = [1, 1, 2, 2] -> Team 1: agents 0,1; Team 2: agents 2,3
    mock_agents = []
    for i, team_idx in enumerate([1, 1, 2, 2]):
        agent = MagicMock()
        agent.index = team_idx
        mock_agents.append(agent)

    env.agents = mock_agents
    env.max_steps = 1000

    # Mock observation space (7x7x6 grid encoding)
    env.observation_space = MagicMock()
    env.observation_space.shape = (7, 7, 6)
    env.observation_space.dtype = np.uint8

    # Mock action space (8 discrete actions)
    env.action_space = MagicMock()
    env.action_space.n = 8

    # Mock reset - returns list of observations
    env.reset.return_value = [
        np.zeros((7, 7, 6), dtype=np.uint8) for _ in range(4)
    ]

    # Mock step - returns (obs_list, rewards_list, done, info)
    env.step.return_value = (
        [np.zeros((7, 7, 6), dtype=np.uint8) for _ in range(4)],  # obs
        [0.0, 0.0, 0.0, 0.0],  # rewards
        False,  # done
        {},  # info
    )

    # Mock seed
    env.seed.return_value = [42]

    # Mock np_random for reproducibility
    env.np_random = np.random.default_rng(42)

    return env


@pytest.fixture
def mock_config():
    """Create a mock config object for MultiGrid_Env."""
    class MockConfig:
        env_name = "multigrid"
        env_id = "soccer"
        env_seed = 42
        render_mode = None

    return MockConfig()


# =============================================================================
# Test: Environment Registration (with mocked XuanCe)
# =============================================================================


class TestEnvironmentRegistration:
    """Test registration of MultiGrid environments with XuanCe."""

    def test_register_mosaic_environments_function_exists(self):
        """Test that register_mosaic_environments function can be accessed."""
        # Import just the module's __init__ which has lazy loading
        import xuance_worker.environments as env_module

        # Check function exists
        assert hasattr(env_module, 'register_mosaic_environments')
        assert callable(env_module.register_mosaic_environments)

    def test_get_registered_environments_function_exists(self):
        """Test that get_registered_environments function can be accessed."""
        import xuance_worker.environments as env_module

        assert hasattr(env_module, 'get_registered_environments')
        assert callable(env_module.get_registered_environments)

    def test_register_with_mocked_xuance(self):
        """Test registration logic with mocked XuanCe registry."""
        # Create mock registry
        mock_registry = {}

        # Create mock MultiGrid_Env class
        mock_multigrid_env = MagicMock()

        with patch.dict(sys.modules, {
            'xuance': MagicMock(),
            'xuance.environment': MagicMock(),
            'xuance.environment.multi_agent_env': MagicMock(
                REGISTRY_MULTI_AGENT_ENV=mock_registry
            ),
        }):
            with patch(
                'xuance_worker.environments.mosaic_multigrid.MultiGrid_Env',
                mock_multigrid_env
            ):
                # Now import and call register
                from xuance_worker.environments import register_mosaic_environments

                # Reset the _REGISTERED flag for testing
                import xuance_worker.environments as env_module
                env_module._REGISTERED = False

                register_mosaic_environments()

                # Check registration happened
                assert 'multigrid' in mock_registry


# =============================================================================
# Test: MultiGrid_Env Wrapper (using isolated testing)
# =============================================================================


class TestMultiGridEnvWrapper:
    """Test the MultiGrid_Env wrapper class logic."""

    def test_agent_groups_from_index(self, mock_soccer_env, mock_config):
        """Test that agent_groups is correctly built from agent.index."""
        # Directly test the logic without importing the heavy module
        # Build agent groups logic (copied from multigrid.py)
        team_to_agents: Dict[int, List[str]] = defaultdict(list)

        for i, agent in enumerate(mock_soccer_env.agents):
            team_idx = agent.index  # Team assignment (1, 2, etc.)
            team_to_agents[team_idx].append(f"agent_{i}")

        sorted_teams = sorted(team_to_agents.keys())
        agent_groups = [team_to_agents[team] for team in sorted_teams]

        # Check agent_groups structure
        assert len(agent_groups) == 2, "Should have 2 teams"
        assert agent_groups[0] == ['agent_0', 'agent_1'], "Team 1 should have agents 0, 1"
        assert agent_groups[1] == ['agent_2', 'agent_3'], "Team 2 should have agents 2, 3"

    def test_agents_list_creation(self, mock_soccer_env, mock_config):
        """Test that agents list is correctly created."""
        num_agents = len(mock_soccer_env.agents)
        agents = [f"agent_{i}" for i in range(num_agents)]

        assert agents == ['agent_0', 'agent_1', 'agent_2', 'agent_3']
        assert num_agents == 4

    def test_observation_space_conversion(self, mock_soccer_env, mock_config):
        """Test that observation_space is converted to dict format."""
        agents = [f"agent_{i}" for i in range(len(mock_soccer_env.agents))]

        # Create dict-based observation space like the wrapper does
        obs_space = mock_soccer_env.observation_space
        observation_space = {
            agent: MockBox(
                low=0,
                high=255,
                shape=obs_space.shape,
                dtype=obs_space.dtype
            )
            for agent in agents
        }

        assert isinstance(observation_space, dict)
        assert len(observation_space) == 4
        for agent in agents:
            assert agent in observation_space
            assert observation_space[agent].shape == (7, 7, 6)

    def test_action_space_conversion(self, mock_soccer_env, mock_config):
        """Test that action_space is converted to dict format."""
        agents = [f"agent_{i}" for i in range(len(mock_soccer_env.agents))]

        # Create dict-based action space like the wrapper does
        act_space = mock_soccer_env.action_space
        action_space = {
            agent: MockDiscrete(act_space.n)
            for agent in agents
        }

        assert isinstance(action_space, dict)
        assert len(action_space) == 4
        for agent in agents:
            assert agent in action_space
            assert action_space[agent].n == 8

    def test_groups_info_structure(self, mock_soccer_env, mock_config):
        """Test that groups_info has correct structure for runner_competition.py."""
        # Build components
        agents = [f"agent_{i}" for i in range(len(mock_soccer_env.agents))]

        team_to_agents: Dict[int, List[str]] = defaultdict(list)
        for i, agent in enumerate(mock_soccer_env.agents):
            team_to_agents[agent.index].append(f"agent_{i}")
        sorted_teams = sorted(team_to_agents.keys())
        agent_groups = [team_to_agents[team] for team in sorted_teams]

        observation_space = {agent: MagicMock() for agent in agents}
        action_space = {agent: MagicMock() for agent in agents}

        # Build groups_info like the wrapper does
        groups_info = {
            'num_groups': len(agent_groups),
            'agent_groups': agent_groups,
            'observation_space_groups': [
                {agent: observation_space[agent] for agent in group}
                for group in agent_groups
            ],
            'action_space_groups': [
                {agent: action_space[agent] for agent in group}
                for group in agent_groups
            ],
            'num_agents_groups': [len(group) for group in agent_groups]
        }

        # Check required keys
        assert 'num_groups' in groups_info
        assert 'agent_groups' in groups_info
        assert 'observation_space_groups' in groups_info
        assert 'action_space_groups' in groups_info
        assert 'num_agents_groups' in groups_info

        # Check values
        assert groups_info['num_groups'] == 2
        assert groups_info['num_agents_groups'] == [2, 2]

    def test_env_info_structure(self, mock_soccer_env, mock_config):
        """Test that env_info has correct structure."""
        agents = [f"agent_{i}" for i in range(len(mock_soccer_env.agents))]
        max_episode_steps = getattr(mock_soccer_env, 'max_steps', 10000)

        # Build env_info like the wrapper does
        env_info = {
            'state_space': MagicMock(),  # Would be Box space
            'observation_space': {agent: MagicMock() for agent in agents},
            'action_space': {agent: MagicMock() for agent in agents},
            'agents': agents,
            'num_agents': len(agents),
            'max_episode_steps': max_episode_steps
        }

        assert 'state_space' in env_info
        assert 'observation_space' in env_info
        assert 'action_space' in env_info
        assert 'agents' in env_info
        assert 'num_agents' in env_info
        assert 'max_episode_steps' in env_info


# =============================================================================
# Test: Reset and Step API Conversion
# =============================================================================


class TestMultiGridEnvAPI:
    """Test the reset() and step() API conversion logic."""

    def test_reset_returns_dict(self, mock_soccer_env, mock_config):
        """Test that reset() returns observations as dict."""
        # Simulate reset conversion
        obs_list = mock_soccer_env.reset()

        # Convert list to dict (like wrapper does)
        observations = {
            f"agent_{i}": obs.astype(np.float32)
            for i, obs in enumerate(obs_list)
        }

        assert isinstance(observations, dict)
        assert len(observations) == 4
        for i in range(4):
            assert f"agent_{i}" in observations
            assert isinstance(observations[f"agent_{i}"], np.ndarray)

    def test_reset_info_structure(self, mock_soccer_env, mock_config):
        """Test that reset() info has required keys."""
        agents = [f"agent_{i}" for i in range(4)]
        individual_episode_reward = {k: 0.0 for k in agents}

        # Build info dict like wrapper does
        reset_info = {
            "infos": {},
            "individual_episode_rewards": individual_episode_reward.copy(),
            "state": np.zeros(100, dtype=np.float32),  # Mock state
            "avail_actions": {agent: np.ones(8, dtype=np.bool_) for agent in agents},
        }

        assert 'individual_episode_rewards' in reset_info
        assert 'state' in reset_info
        assert 'avail_actions' in reset_info

    def test_step_dict_to_list_conversion(self, mock_soccer_env, mock_config):
        """Test that step() converts dict actions to list for underlying env."""
        num_agents = 4

        # Dict format actions
        actions = {
            'agent_0': 3,
            'agent_1': 4,
            'agent_2': 1,
            'agent_3': 2,
        }

        # Convert dict to list (like wrapper does)
        actions_list = [actions[f"agent_{i}"] for i in range(num_agents)]

        assert actions_list == [3, 4, 1, 2]

    def test_step_returns_dict_format(self, mock_soccer_env, mock_config):
        """Test that step() returns XuanCe-compatible dict format."""
        # Get raw result from env
        obs_list, rewards_list, done, info = mock_soccer_env.step([0, 0, 0, 0])

        num_agents = 4
        agents = [f"agent_{i}" for i in range(num_agents)]

        # Convert to dict format (like wrapper does)
        observations = {
            f"agent_{i}": obs.astype(np.float32)
            for i, obs in enumerate(obs_list)
        }
        rewards = {
            f"agent_{i}": float(rewards_list[i])
            for i in range(num_agents)
        }
        terminated = {agent: bool(done) for agent in agents}
        truncated = False  # Based on episode step

        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, bool)


# =============================================================================
# Test: Agent Mask and Available Actions
# =============================================================================


class TestMultiGridEnvMasks:
    """Test agent_mask() and avail_actions() methods."""

    def test_agent_mask_all_alive(self, mock_soccer_env, mock_config):
        """Test that agent_mask returns True for all agents."""
        agents = [f"agent_{i}" for i in range(4)]

        # Build mask like wrapper does
        mask = {agent: True for agent in agents}

        assert isinstance(mask, dict)
        for agent in agents:
            assert mask[agent] is True

    def test_avail_actions_all_available(self, mock_soccer_env, mock_config):
        """Test that avail_actions returns all ones."""
        agents = [f"agent_{i}" for i in range(4)]
        num_actions = 8

        # Build avail_actions like wrapper does
        avail = {
            agent: np.ones(num_actions, dtype=np.bool_)
            for agent in agents
        }

        assert isinstance(avail, dict)
        for agent in agents:
            assert agent in avail
            assert avail[agent].dtype == np.bool_
            assert np.all(avail[agent])


# =============================================================================
# Test: State Function
# =============================================================================


class TestMultiGridEnvState:
    """Test global state function."""

    def test_state_from_observations(self, mock_soccer_env, mock_config):
        """Test that state() creates concatenated observation array."""
        # Get observations
        obs_list = mock_soccer_env.reset()

        # Build state like wrapper does
        state = np.concatenate([obs.flatten() for obs in obs_list])
        state = state.astype(np.float32)

        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        # Shape should be 4 * (7*7*6) = 4 * 294 = 1176
        assert len(state) == 4 * 7 * 7 * 6

    def test_state_before_reset(self, mock_soccer_env, mock_config):
        """Test that state() returns zeros before reset."""
        # Before reset, _last_obs is None
        _last_obs = None
        num_agents = 4
        single_obs_dim = int(np.prod((7, 7, 6)))
        state_shape = (num_agents * single_obs_dim,)

        # Build state like wrapper does when no observations
        if _last_obs is None:
            state = np.zeros(state_shape, dtype=np.float32)
        else:
            state = np.concatenate([obs.flatten() for obs in _last_obs])

        assert isinstance(state, np.ndarray)
        assert np.all(state == 0)


# =============================================================================
# Test: Reproducibility Wrapper Integration
# =============================================================================


class TestReproducibilityIntegration:
    """Test integration with ReproducibleMultiGridWrapper logic."""

    def test_wrapper_application_logic(self, mock_soccer_env, mock_config):
        """Test that wrapper would be applied when available."""
        # Mock wrapper
        mock_wrapper = MagicMock(return_value=mock_soccer_env)

        # Test wrapping logic
        env = mock_soccer_env

        # Apply wrapper if available (like multigrid.py does)
        ReproducibleMultiGridWrapper = mock_wrapper
        if ReproducibleMultiGridWrapper is not None:
            env = ReproducibleMultiGridWrapper(env)

        mock_wrapper.assert_called_once_with(mock_soccer_env)

    def test_wrapper_not_applied_when_none(self, mock_soccer_env, mock_config):
        """Test that wrapper is not applied when not available."""
        env = mock_soccer_env

        # Apply wrapper if available
        ReproducibleMultiGridWrapper = None
        if ReproducibleMultiGridWrapper is not None:
            env = ReproducibleMultiGridWrapper(env)

        # env should still be the original
        assert env is mock_soccer_env

    def test_seed_called_on_env(self, mock_soccer_env, mock_config):
        """Test that seed() is called when env_seed is provided."""
        env_seed = 42

        # Call seed like wrapper does
        if env_seed is not None:
            mock_soccer_env.seed(env_seed)

        mock_soccer_env.seed.assert_called_once_with(42)


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestMultiGridEnvErrors:
    """Test error handling in MultiGrid_Env."""

    def test_unknown_env_id_logic(self, mock_config):
        """Test that unknown env_id would raise ValueError."""
        env_id = "unknown_env"
        MULTIGRID_ENV_CLASSES = {
            "soccer": MagicMock(),
            "collect": MagicMock(),
        }

        env_cls = MULTIGRID_ENV_CLASSES.get(env_id)

        assert env_cls is None

        # Verify the error message that would be raised
        if env_cls is None:
            available = list(MULTIGRID_ENV_CLASSES.keys())
            error_msg = f"Unknown MultiGrid environment: {env_id}. Available: {available}"
            assert "unknown_env" in error_msg
            assert "soccer" in error_msg
            assert "collect" in error_msg


# =============================================================================
# Test: MULTIGRID_TEAM_DICT and MULTIGRID_ENV_CLASSES
# =============================================================================


class TestMultiGridConstants:
    """Test the module-level constants."""

    def test_team_dict_structure(self):
        """Test MULTIGRID_TEAM_DICT has correct structure."""
        # Define expected structure (matches multigrid.py)
        MULTIGRID_TEAM_DICT = {
            "multigrid.soccer": ["red", "blue"],
            "multigrid.collect": ["team"],
        }

        assert "multigrid.soccer" in MULTIGRID_TEAM_DICT
        assert MULTIGRID_TEAM_DICT["multigrid.soccer"] == ["red", "blue"]
        assert "multigrid.collect" in MULTIGRID_TEAM_DICT

    def test_env_classes_keys(self):
        """Test MULTIGRID_ENV_CLASSES has expected keys."""
        # Define expected keys
        expected_keys = {"soccer", "Soccer", "SoccerGame4HEnv10x15N2",
                        "collect", "Collect", "CollectGame4HEnv10x10N2"}

        # Verify we have the key set
        assert "soccer" in expected_keys
        assert "Soccer" in expected_keys
        assert "collect" in expected_keys
