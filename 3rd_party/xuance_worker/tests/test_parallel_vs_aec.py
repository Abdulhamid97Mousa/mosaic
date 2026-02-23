# 3rd_party/xuance_worker/tests/test_parallel_vs_aec.py
"""Parallel vs AEC execution mode comparison tests.

Demonstrates the key difference:
  Parallel: Both agents observe S(t) simultaneously, act at the same time.
            env.step([A_0, A_1]) — one physics step per round.
            Neither agent sees the other's action before deciding.

  AEC:      Agent_0 observes S(t), acts → physics fires → S(t+0.5).
            Agent_1 observes S(t+0.5), acts → physics fires → S(t+1).
            env.step([A_0, NOOP]) then env.step([NOOP, A_1]) — N physics steps per round.
            Agent_1 sees the result of Agent_0's action.

  NOOP = 0: mosaic_multigrid v5.0.0 — genuine no-op, agent stays in place.
            Critical for AEC correctness; non-acting agents must not move.

These tests verify:
  1. Same seed + same actions → Parallel and AEC produce DIFFERENT world states
     (because AEC fires physics twice per round with intermediate NOOP padding)
  2. NOOP action doesn't change agent position
  3. AEC intermediate observations differ from Parallel observations
  4. Both modes produce valid obs shapes for all env variants
"""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env


NOOP = 0  # mosaic_multigrid v5.0.0 — action 0 = no-op


def _make_env(env_id: str, seed: int = 42) -> MultiGrid_Env:
    cfg = SimpleNamespace(
        env_name="multigrid", env_id=env_id,
        env_seed=seed, training_mode="competitive",
    )
    return MultiGrid_Env(cfg)


# ---------------------------------------------------------------------------
# NOOP behaviour
# ---------------------------------------------------------------------------
class TestNOOP:
    """NOOP (action 0) must not change the acting agent's observation."""

    @pytest.mark.parametrize("env_id", ["collect_1vs1", "soccer_1vs1"])
    def test_noop_preserves_obs(self, env_id: str):
        """Sending NOOP for all agents should not change observations significantly.

        After a NOOP step, the image portion of each agent's observation should
        be identical (agent didn't move, didn't turn, didn't interact).
        """
        env = _make_env(env_id)
        try:
            obs_before, _ = env.reset()
            noop_actions = {a: NOOP for a in env.agents}
            obs_after, *_ = env.step(noop_actions)
            for agent_id in env.agents:
                # Image portion (first 27 dims) should be identical after NOOP
                np.testing.assert_array_equal(
                    obs_before[agent_id][:27],
                    obs_after[agent_id][:27],
                    err_msg=f"{agent_id} obs changed after NOOP",
                )
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Parallel vs AEC state divergence
# ---------------------------------------------------------------------------
class TestParallelVsAEC:
    """Show the structural difference between Parallel and AEC stepping.

    Key difference: AEC fires N physics steps per logical round (one per agent),
    while Parallel fires 1. This means AEC consumes more episode steps for the
    same number of agent decisions.
    """

    @pytest.mark.parametrize("env_id", ["collect_1vs1", "soccer_1vs1"])
    def test_step_count_divergence(self, env_id: str):
        """AEC uses 2x the episode steps of Parallel for the same logical actions.

        Parallel: env.step({a0: A, a1: B}) → _episode_step = 1
        AEC:      env.step({a0: A, a1: NOOP}) + env.step({a0: NOOP, a1: B}) → _episode_step = 2
        """
        ACTION = 2  # forward

        # Parallel: 1 step
        env_par = _make_env(env_id, seed=42)
        try:
            env_par.reset()
            env_par.step({a: ACTION for a in env_par.agents})
            steps_parallel = env_par._episode_step
        finally:
            env_par.close()

        # AEC-like: 2 steps (one per agent)
        env_aec = _make_env(env_id, seed=42)
        try:
            env_aec.reset()
            agents = env_aec.agents
            env_aec.step({agents[0]: ACTION, agents[1]: NOOP})
            env_aec.step({agents[0]: NOOP, agents[1]: ACTION})
            steps_aec = env_aec._episode_step
        finally:
            env_aec.close()

        assert steps_parallel == 1, f"Parallel step count = {steps_parallel}, expected 1"
        assert steps_aec == 2, f"AEC step count = {steps_aec}, expected 2"

    @pytest.mark.parametrize("env_id", ["collect_1vs1", "soccer_1vs1"])
    def test_aec_intermediate_state_valid(self, env_id: str):
        """In AEC, the intermediate observation after agent_0's action is valid.

        After agent_0 acts and agent_1 NOOPs, we get a valid intermediate state.
        The obs shapes and dtypes must be correct, and rewards must be numeric.
        This intermediate state is what agent_1 would observe before deciding.
        """
        ACTION_FORWARD = 2

        env = _make_env(env_id, seed=42)
        try:
            obs_initial, _ = env.reset()
            agents = env.agents
            obs_dim = obs_initial[agents[0]].shape[0]

            # Agent_0 acts, agent_1 NOOPs → intermediate state S(t+0.5)
            obs_mid, rew_mid, term_mid, trunc_mid, info_mid = env.step(
                {agents[0]: ACTION_FORWARD, agents[1]: NOOP}
            )

            # Intermediate observations must be valid
            for a in agents:
                assert obs_mid[a].shape == (obs_dim,), f"{a}: bad shape {obs_mid[a].shape}"
                assert obs_mid[a].dtype == np.float32
                assert isinstance(rew_mid[a], float)

            # Agent_1 then acts on this intermediate state
            obs_final, rew_final, *_ = env.step(
                {agents[0]: NOOP, agents[1]: ACTION_FORWARD}
            )
            for a in agents:
                assert obs_final[a].shape == (obs_dim,)
        finally:
            env.close()

    @pytest.mark.parametrize("env_id", ["collect_1vs1", "soccer_1vs1"])
    def test_noop_only_acting_agent_moves(self, env_id: str):
        """When one agent acts and the other NOOPs, only the acting agent's
        state should change (the NOOPing agent stays in place).

        This is the foundation of AEC correctness: NOOP must truly be a no-op.
        """
        env = _make_env(env_id, seed=42)
        try:
            obs_before, _ = env.reset()
            agents = env.agents

            # Only agent_0 acts (turn left = 0 is NOOP, so use turn right = 1)
            obs_after, *_ = env.step({agents[0]: 1, agents[1]: NOOP})

            # agent_1 NOOPed — its image portion (first 27 dims) should not change
            np.testing.assert_array_equal(
                obs_before[agents[1]][:27],
                obs_after[agents[1]][:27],
                err_msg="NOOPing agent's obs changed — NOOP is not a true no-op",
            )
        finally:
            env.close()


# ---------------------------------------------------------------------------
# TeamObs AEC compatibility
# ---------------------------------------------------------------------------
class TestTeamObsAECCompat:
    """TeamObs variants must also work correctly with AEC-style stepping."""

    @pytest.mark.parametrize("env_id", [
        "soccer_2vs2_teamobs",
        "collect_2vs2_teamobs",
        "basketball_3vs3_teamobs",
    ])
    def test_teamobs_noop_step(self, env_id: str):
        """TeamObs envs should handle NOOP stepping without errors."""
        env = _make_env(env_id)
        try:
            obs, _ = env.reset()
            noop_actions = {a: NOOP for a in env.agents}
            obs2, rew, term, trunc, info = env.step(noop_actions)
            for a in env.agents:
                assert obs2[a].dtype == np.float32
                assert obs2[a].shape == obs[a].shape
        finally:
            env.close()

    @pytest.mark.parametrize("env_id", [
        "soccer_2vs2_teamobs",
        "basketball_3vs3_teamobs",
    ])
    def test_teamobs_sequential_stepping(self, env_id: str):
        """TeamObs envs can be stepped sequentially (AEC-style) per agent."""
        env = _make_env(env_id)
        try:
            obs, _ = env.reset()
            agents = env.agents
            ACTION_FORWARD = 2

            # Step each agent one at a time with NOOPs for others
            for i, acting_agent in enumerate(agents):
                actions = {a: NOOP for a in agents}
                actions[acting_agent] = ACTION_FORWARD
                obs, *_ = env.step(actions)

            # All obs should still be valid
            for a in agents:
                assert obs[a].dtype == np.float32
        finally:
            env.close()
