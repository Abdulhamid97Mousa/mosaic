# gym_gui/services/aec_wrapper.py

"""AEC per-agent physics wrapper for mosaic_multigrid and MeltingPot environments.

Two execution modes
-------------------
MOSAIC supports exactly two execution modes:

  Parallel (simultaneous):
      Both agents observe S(t) and act simultaneously.
      env.step([A_0, A_1]) fires ONCE — one physics update per round.
      Agent_1 does NOT see Agent_0's result when deciding.

  AEC (sequential physics):
      Agent_0 acts first; physics fires immediately.
      Agent_1 observes the intermediate state S(t+0.5) before deciding.
      env.step([A_0, NOOP]) → S(t+0.5) → env.step([NOOP, A_1]) → S(t+1)
      N env.step() calls per round (one per agent).

This module implements AEC.

Why NOOP=0 is required
-----------------------
Non-acting agents must receive a placeholder action.  mosaic_multigrid v5.0.0
introduced action 0 = NOOP (a genuine no-op: the agent stays in place, does
not turn, does not interact with objects).  Earlier versions had action 0 =
LEFT, so non-acting agents would silently turn left — corrupting episodes.
NOOP=0 makes AEC correct and safe.

Academic context
----------------
Terry et al. (2021), "PettingZoo: Gym for Multi-Agent Reinforcement Learning",
NeurIPS 2021 (arXiv:2009.14471) formally define the AEC model: after each
individual agent action the environment updates, and the next agent observes
the post-action state.  This wrapper realises that definition on top of a
standard Gymnasium Parallel env.

Usage
-----
    import gymnasium
    import mosaic_multigrid.envs  # registers environments

    raw_env = gymnasium.make("MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0")
    aec_env = GymnasiumMultiAgentAECWrapper(raw_env)

    obs, info = aec_env.reset(seed=42)
    while not aec_env.done:
        agent = aec_env.agent_selection       # "agent_0", then "agent_1", …
        observation = aec_env.observe(agent)
        action = my_policy(observation)
        aec_env.step(action)                  # physics fires NOW for this agent only
        # The next agent will observe S(t+0.5) — not the stale S(t)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

NOOP: int = 0
"""mosaic_multigrid v5.0.0 — action 0 = noop.
Passed to all non-acting agents so the physics step advances only the
currently acting agent's action.
"""


class GymnasiumMultiAgentAECWrapper:
    """AEC per-agent physics wrapper for mosaic_multigrid environments.

    Each call to ``step(action)`` triggers exactly one ``env.step()`` on the
    underlying gymnasium env, passing NOOP for all agents except the currently
    acting agent.  The next agent therefore observes the intermediate post-action
    world state S(t+0.5) produced by its predecessor's action.

    Interface
    ---------
    agent_selection : str
        The currently active agent ID ("agent_0", "agent_1", …).
    agents : list[str]
        Active agents.  Cleared to [] when the episode ends.
    done : bool
        True if any agent terminated or truncated.
    observations : dict[str, obs]
        Latest observation for every agent (updated after each step).
    rewards : dict[str, float]
        Rewards accumulated during the current round (reset each new round).
    terminations : dict[str, bool]
        Termination flags from the most recent env.step() call.
    truncations : dict[str, bool]
        Truncation flags from the most recent env.step() call.
    infos : dict[str, dict]
        Info dicts from the most recent env.step() call.

    Methods
    -------
    reset(seed, options) → (obs_dict, info_dict)
    observe(agent) → obs
    step(action) → None

    Args
    ----
    gymnasium_env :
        A mosaic_multigrid env returned by ``gymnasium.make()``.
        Must expose ``action_space`` and ``observation_space`` as
        ``gymnasium.spaces.Dict`` with integer keys 0, 1, …
    """

    def __init__(self, gymnasium_env: Any) -> None:
        self._env = gymnasium_env

        # Determine agent count from the Dict action space (keys: 0, 1, ...)
        raw_spaces = self._env.action_space.spaces  # OrderedDict {0: Discrete(8), …}
        self._int_keys: List[int] = sorted(raw_spaces.keys())
        self._n_agents: int = len(self._int_keys)

        # String agent IDs (matches PettingZoo / MOSAIC convention)
        self.possible_agents: List[str] = [f"agent_{k}" for k in self._int_keys]
        self.agents: List[str] = []

        # Per-agent spaces (indexed by string ID)
        self._obs_spaces: Dict[str, Any] = {
            f"agent_{k}": self._env.observation_space.spaces[k]
            for k in self._int_keys
        }
        self._act_spaces: Dict[str, Any] = {
            f"agent_{k}": self._env.action_space.spaces[k]
            for k in self._int_keys
        }

        # Episode state
        self._agent_idx: int = 0        # index into possible_agents (cycling)
        self._done: bool = True

        self.observations: Dict[str, Any] = {}
        self.rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agent_selection(self) -> str:
        """The currently active agent ID."""
        return self.possible_agents[self._agent_idx]

    @property
    def done(self) -> bool:
        """True if the episode has ended (any agent terminated or truncated)."""
        return self._done

    def observation_space(self, agent: str) -> Any:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> Any:
        return self._act_spaces[agent]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and return initial observations and infos."""
        obs_int, info_int = self._env.reset(seed=seed, options=options)

        self.agents = list(self.possible_agents)
        self._agent_idx = 0
        self._done = False

        self.observations = self._to_str_keys(obs_int)
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = (
            self._to_str_keys(info_int)
            if isinstance(info_int, dict)
            else {a: {} for a in self.possible_agents}
        )

        return dict(self.observations), dict(self.infos)

    def observe(self, agent: str) -> Any:
        """Return the current observation for the given agent.

        In AEC this reflects the world state *after* all preceding agents
        in the current round have acted — the genuine intermediate state
        S(t+0.5), not the stale S(t).
        """
        return self.observations.get(agent)

    def step(self, action: Any) -> None:
        """Apply ``action`` for the current agent; physics advances immediately.

        All non-acting agents receive NOOP (action=0).  This produces a
        genuine intermediate world state S(t+0.5) that the next agent will
        observe via ``observe()``.

        After all agents have acted once (end of round), accumulated rewards
        are reset and the cycle restarts from agent_0.

        Args
        ----
        action :
            An integer action valid for ``self.agent_selection``.

        Raises
        ------
        RuntimeError
            If called before ``reset()``.
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() before calling step()."
            )

        acting_idx = self._agent_idx

        # Build action list: NOOP for all agents except the currently acting one.
        # NOOP=0 (mosaic_multigrid v5.0.0) — the agent stays in place and does nothing.
        action_list = [NOOP] * self._n_agents
        action_list[acting_idx] = action

        # Apply to physics immediately — this is the key AEC behaviour.
        # The next agent's observe() will return the updated intermediate state.
        obs_int, rew_int, term_int, trunc_int, info_int = self._env.step(action_list)

        # All agents now see the updated (intermediate) world state
        self.observations = self._to_str_keys(obs_int)

        # Accumulate rewards for this round; store latest flags and infos
        for i, k in enumerate(self._int_keys):
            agent_id = f"agent_{k}"
            raw_rew = rew_int.get(k, 0.0) if isinstance(rew_int, dict) else rew_int[i]
            self.rewards[agent_id] = self.rewards.get(agent_id, 0.0) + float(raw_rew)
            raw_term = (
                term_int.get(k, False) if isinstance(term_int, dict) else term_int[i]
            )
            raw_trunc = (
                trunc_int.get(k, False) if isinstance(trunc_int, dict) else trunc_int[i]
            )
            self.terminations[agent_id] = bool(raw_term)
            self.truncations[agent_id] = bool(raw_trunc)

        self.infos = (
            self._to_str_keys(info_int)
            if isinstance(info_int, dict)
            else {f"agent_{k}": {} for k in self._int_keys}
        )

        # Check episode termination after this physics step
        if any(self.terminations.values()) or any(self.truncations.values()):
            self._done = True
            self.agents = []
            return

        # Advance to next agent (cyclic)
        next_idx = (acting_idx + 1) % self._n_agents
        self._agent_idx = next_idx

        # Reset per-round reward accumulator when a new round begins
        if next_idx == 0:
            self.rewards = {a: 0.0 for a in self.possible_agents}

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_str_keys(self, int_dict: Any) -> Dict[str, Any]:
        """Convert {0: v, 1: v, …} → {"agent_0": v, "agent_1": v, …}."""
        if isinstance(int_dict, dict):
            return {f"agent_{k}": v for k, v in int_dict.items()}
        # Fallback: treat as a sequence indexed by position
        return {f"agent_{i}": v for i, v in enumerate(int_dict)}
