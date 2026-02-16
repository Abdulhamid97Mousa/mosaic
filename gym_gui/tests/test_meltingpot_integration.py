"""Test Melting Pot integration with Shimmy PettingZoo wrapper.

Tests that Melting Pot environments:
1. Can be created and loaded via Shimmy
2. Provide valid action spaces for all agents
3. Support parallel stepping (simultaneous multi-agent)
4. Handle observation and reward structures correctly
5. Work with substrates documented in MOSAIC
"""

from __future__ import annotations

import pytest
import numpy as np
from typing import Any, Dict


class TestMeltingPotEnvironmentLoads:
    """Test that Melting Pot environments can be created via Shimmy."""

    @pytest.fixture
    def collaborative_cooking_env(self):
        """Create a Collaborative Cooking environment."""
        from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0

        env = MeltingPotCompatibilityV0(substrate_name='collaborative_cooking__circuit')
        yield env
        env.close()

    def test_env_loads_successfully(self, collaborative_cooking_env) -> None:
        """Melting Pot environment loads successfully."""
        assert collaborative_cooking_env is not None
        assert hasattr(collaborative_cooking_env, 'agents')
        assert hasattr(collaborative_cooking_env, 'action_space')
        assert hasattr(collaborative_cooking_env, 'observation_space')

    def test_env_has_agents(self, collaborative_cooking_env) -> None:
        """Environment has agent list."""
        agents = collaborative_cooking_env.agents
        assert isinstance(agents, list)
        assert len(agents) > 0
        # Melting Pot agents are named player_0, player_1, etc.
        assert all(agent.startswith('player_') for agent in agents)

    def test_env_reset(self, collaborative_cooking_env) -> None:
        """Environment reset returns valid observations."""
        obs, info = collaborative_cooking_env.reset()

        assert isinstance(obs, dict)
        assert len(obs) == len(collaborative_cooking_env.agents)

        # Each agent should have an observation dict with RGB and COLLECTIVE_REWARD
        for agent in collaborative_cooking_env.agents:
            assert agent in obs
            assert isinstance(obs[agent], dict)
            assert 'RGB' in obs[agent]
            assert 'COLLECTIVE_REWARD' in obs[agent]
            assert isinstance(obs[agent]['RGB'], np.ndarray)


class TestMeltingPotActionSpaces:
    """Test action space structures for Melting Pot environments."""

    @pytest.fixture
    def env(self):
        """Create a Melting Pot environment for testing."""
        from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0

        env = MeltingPotCompatibilityV0(substrate_name='collaborative_cooking__circuit')
        env.reset()
        yield env
        env.close()

    def test_action_space_type(self, env) -> None:
        """Action spaces are Discrete for all agents."""
        from gymnasium.spaces import Discrete

        for agent in env.agents:
            action_space = env.action_space(agent)
            assert isinstance(action_space, Discrete)
            # Melting Pot typically has 8 actions (move + turn + interact)
            assert action_space.n > 0

    def test_action_space_consistent_across_agents(self, env) -> None:
        """All agents have the same action space structure."""
        action_spaces = [env.action_space(agent) for agent in env.agents]

        # All action spaces should have the same number of actions
        action_counts = [space.n for space in action_spaces]
        assert len(set(action_counts)) == 1, "All agents should have same action space size"

    def test_valid_action_sampling(self, env) -> None:
        """Can sample valid actions for all agents."""
        for agent in env.agents:
            action = env.action_space(agent).sample()
            assert isinstance(action, (int, np.integer))
            assert 0 <= action < env.action_space(agent).n


class TestMeltingPotObservationSpaces:
    """Test observation space structures for Melting Pot environments."""

    @pytest.fixture
    def env(self):
        """Create a Melting Pot environment for testing."""
        from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0

        env = MeltingPotCompatibilityV0(substrate_name='collaborative_cooking__circuit')
        env.reset()
        yield env
        env.close()

    def test_observation_space_type(self, env) -> None:
        """Observation spaces are Dict with RGB and COLLECTIVE_REWARD for all agents."""
        from gymnasium.spaces import Box, Dict

        for agent in env.agents:
            obs_space = env.observation_space(agent)
            assert isinstance(obs_space, Dict)
            # Should have RGB and COLLECTIVE_REWARD keys
            assert 'RGB' in obs_space.spaces
            assert 'COLLECTIVE_REWARD' in obs_space.spaces
            # RGB should be a Box space
            assert isinstance(obs_space.spaces['RGB'], Box)
            # RGB image should have 3 channels
            assert len(obs_space.spaces['RGB'].shape) == 3
            assert obs_space.spaces['RGB'].shape[2] == 3  # RGB

    def test_observation_dimensions(self, env) -> None:
        """Observations have valid RGB dimensions."""
        for agent in env.agents:
            obs_space = env.observation_space(agent)
            rgb_space = obs_space.spaces['RGB']
            height, width, channels = rgb_space.shape

            assert height > 0
            assert width > 0
            assert channels == 3  # RGB


class TestMeltingPotParallelStepping:
    """Test parallel (simultaneous) stepping paradigm."""

    @pytest.fixture
    def env(self):
        """Create a Melting Pot environment for testing."""
        from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0

        env = MeltingPotCompatibilityV0(substrate_name='collaborative_cooking__circuit')
        env.reset()
        yield env
        env.close()

    def test_parallel_step(self, env) -> None:
        """Step with actions for all agents simultaneously."""
        # Sample actions for all agents
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Check return values
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)
        assert isinstance(info, dict)

        # All agents should have entries
        for agent in env.agents:
            assert agent in obs
            assert agent in rewards
            assert agent in terminated
            assert agent in truncated

    def test_rewards_are_numeric(self, env) -> None:
        """Rewards are numeric values (possibly 0-d arrays)."""
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)

        for agent in env.agents:
            # Rewards may be 0-d numpy arrays or scalars
            reward = rewards[agent]
            if isinstance(reward, np.ndarray):
                # 0-d array
                assert reward.ndim == 0
                reward_val = float(reward)
            else:
                reward_val = reward
            assert isinstance(reward_val, (int, float, np.number))

    def test_terminated_truncated_are_bool(self, env) -> None:
        """Terminated and truncated flags are booleans."""
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)

        for agent in env.agents:
            assert isinstance(terminated[agent], (bool, np.bool_))
            assert isinstance(truncated[agent], (bool, np.bool_))


class TestMeltingPotSubstrates:
    """Test multiple Melting Pot substrates documented in MOSAIC."""

    @pytest.mark.parametrize("substrate_name", [
        "collaborative_cooking__circuit",
        pytest.param("clean_up__repeated", marks=pytest.mark.skip(reason="Substrate config issue")),
        "commons_harvest__open",
        "territory__rooms",
        pytest.param("king_of_the_hill__repeated", marks=pytest.mark.skip(reason="Substrate config issue")),
        "prisoners_dilemma_in_the_matrix__repeated",
        "stag_hunt_in_the_matrix__repeated",
        "allelopathic_harvest__open",
    ])
    def test_substrate_loads(self, substrate_name: str) -> None:
        """Test that documented substrates load successfully.

        Note: Some substrates may have configuration issues in the Melting Pot
        distribution and are skipped until resolved upstream.
        """
        from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0

        env = MeltingPotCompatibilityV0(substrate_name=substrate_name)
        assert env is not None

        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert len(obs) > 0

        env.close()


class TestMeltingPotGameIds:
    """Test that Melting Pot GameIds are registered in MOSAIC enums."""

    def test_meltingpot_game_ids_exist(self) -> None:
        """Melting Pot GameIds are defined in enums."""
        from gym_gui.core.enums import GameId

        expected_ids = [
            "MELTINGPOT_COLLABORATIVE_COOKING",
            "MELTINGPOT_CLEAN_UP",
            "MELTINGPOT_COMMONS_HARVEST",
            "MELTINGPOT_TERRITORY",
            "MELTINGPOT_KING_OF_THE_HILL",
            "MELTINGPOT_PRISONERS_DILEMMA",
            "MELTINGPOT_STAG_HUNT",
            "MELTINGPOT_ALLELOPATHIC_HARVEST",
        ]

        for game_id_name in expected_ids:
            assert hasattr(GameId, game_id_name), f"GameId.{game_id_name} not found"

    def test_meltingpot_family_exists(self) -> None:
        """MELTINGPOT environment family is defined."""
        from gym_gui.core.enums import EnvironmentFamily

        assert hasattr(EnvironmentFamily, 'MELTINGPOT')
        assert EnvironmentFamily.MELTINGPOT.value == 'meltingpot'

    def test_meltingpot_in_family_mapping(self) -> None:
        """Melting Pot games are mapped to MELTINGPOT family."""
        from gym_gui.core.enums import (
            EnvironmentFamily,
            GameId,
            ENVIRONMENT_FAMILY_BY_GAME,
        )

        meltingpot_game_ids = [
            GameId.MELTINGPOT_COLLABORATIVE_COOKING,
            GameId.MELTINGPOT_CLEAN_UP,
            GameId.MELTINGPOT_COMMONS_HARVEST,
            GameId.MELTINGPOT_TERRITORY,
            GameId.MELTINGPOT_KING_OF_THE_HILL,
            GameId.MELTINGPOT_PRISONERS_DILEMMA,
            GameId.MELTINGPOT_STAG_HUNT,
            GameId.MELTINGPOT_ALLELOPATHIC_HARVEST,
        ]

        for game_id in meltingpot_game_ids:
            assert game_id in ENVIRONMENT_FAMILY_BY_GAME
            assert ENVIRONMENT_FAMILY_BY_GAME[game_id] == EnvironmentFamily.MELTINGPOT

    def test_meltingpot_stepping_paradigm(self) -> None:
        """Melting Pot uses SIMULTANEOUS stepping paradigm."""
        from gym_gui.core.enums import (
            EnvironmentFamily,
            SteppingParadigm,
            DEFAULT_PARADIGM_BY_FAMILY,
        )

        assert EnvironmentFamily.MELTINGPOT in DEFAULT_PARADIGM_BY_FAMILY
        assert DEFAULT_PARADIGM_BY_FAMILY[EnvironmentFamily.MELTINGPOT] == SteppingParadigm.SIMULTANEOUS


class TestMeltingPotDocumentation:
    """Test that Melting Pot documentation is registered."""

    def test_meltingpot_game_info_registered(self) -> None:
        """Melting Pot substrates return documentation via get_game_info()."""
        from gym_gui.core.enums import GameId
        from gym_gui.game_docs import get_game_info

        # One representative variant per documented base substrate
        meltingpot_game_ids = [
            GameId.MELTINGPOT_COLLABORATIVE_COOKING__RING,
            GameId.MELTINGPOT_CLEAN_UP,
            GameId.MELTINGPOT_COMMONS_HARVEST__OPEN,
            GameId.MELTINGPOT_TERRITORY__ROOMS,
            GameId.MELTINGPOT_PAINTBALL__KING_OF_THE_HILL,
            GameId.MELTINGPOT_PRISONERS_DILEMMA_IN_THE_MATRIX__ARENA,
            GameId.MELTINGPOT_STAG_HUNT_IN_THE_MATRIX__ARENA,
            GameId.MELTINGPOT_ALLELOPATHIC_HARVEST__OPEN,
        ]

        for game_id in meltingpot_game_ids:
            doc = get_game_info(game_id)
            assert isinstance(doc, str), f"{game_id} doc is not str"
            assert len(doc) > 0, f"{game_id} doc is empty"
            assert "Documentation unavailable" not in doc, (
                f"{game_id} has no specific doc"
            )

    def test_documentation_has_expected_content(self) -> None:
        """Documentation contains expected information."""
        from gym_gui.core.enums import GameId
        from gym_gui.game_docs import get_game_info

        doc = get_game_info(GameId.MELTINGPOT_COLLABORATIVE_COOKING__RING)

        assert 'Melting Pot' in doc or 'melting' in doc.lower()
        assert 'agent' in doc.lower() or 'Agent' in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
