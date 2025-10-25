"""Custom BDI actions for SPADE-BDI + Q-Learning integration.

This module defines all 15 custom actions that bridge AgentSpeak (ASL) plans
to Python RL/environment capabilities. Actions are registered with the
GLOBAL_ACTIONS registry before ASL parsing to avoid "no such action" warnings.

Actions:
  - .reset_environment/0 - Reset environment to initial state
  - .set_goal/2 - Change goal and reset Q-table
  - .check_cached_policy/3 - Check if cached policy exists
  - .get_state_from_pos/3 - Convert (x,y) to state
  - .get_best_action/2 - Get best action from Q-table
  - .set_epsilon/1 - Set exploration rate
  - .execute_action/1 - Execute action and update Q-table
  - .exec_cached_seq/1 - Execute cached action sequence
  - .rl_propose_seq/1 - Propose sequence from Q-table
  - .cache_policy/5 - Cache successful policy
  - .clear_episode_flags/0 - Clear episode outcome flags
  - .remove_cached_policy/0 - Remove cached policy for current goal
  - .clear_policy_store/0 - Clear all cached policies
  - .save_policies/0 - Save policies to disk
  - .load_policies/0 - Load policies from disk
"""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any, Dict, List, Optional

import agentspeak

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_POLICY_EVENT,
    LOG_WORKER_POLICY_WARNING,
    LOG_WORKER_POLICY_ERROR,
)

_LOGGER = logging.getLogger(__name__)
_log = partial(log_constant, _LOGGER)

# Global action registry - will be populated by register_actions()
GLOBAL_ACTIONS = agentspeak.Actions()


def _extract_value(term: Any) -> Any:
    """Extract Python value from AgentSpeak term."""
    if isinstance(term, agentspeak.Literal):
        if term.args:
            return _extract_value(term.args[0])
        return term.functor
    elif isinstance(term, (int, float, str, bool)):
        return term
    elif isinstance(term, list):
        return [_extract_value(t) for t in term]
    else:
        return str(term)


# ============================================================================
# CUSTOM BDI ACTIONS (15 total)
# ============================================================================

@GLOBAL_ACTIONS.add(".reset_environment", 0)
def _act_reset_environment(agent, term, intention):
    """Reset environment to initial state."""
    try:
        if hasattr(agent, "adapter"):
            agent.current_state, _ = agent.adapter.reset(seed=None)
            agent.episode_steps = 0
            agent.episode_count += 1
            x, y = agent.adapter.state_to_pos(agent.current_state)
            _log(LOG_WORKER_POLICY_EVENT, message=f"[RESET] Episode {agent.episode_count}: Environment reset to ({x},{y})")
    finally:
        yield


@GLOBAL_ACTIONS.add(".set_goal", 2)
def _act_set_goal(agent, term, intention):
    """Change goal and reset Q-table."""
    try:
        gx = int(_extract_value(term.args[0]))
        gy = int(_extract_value(term.args[1]))
        # Update goal in adapter (if supported)
        if hasattr(agent.adapter, "set_goal"):
            agent.adapter.set_goal(gx, gy)
        # Reset Q-table
        if hasattr(agent, "rl_agent") and hasattr(agent.rl_agent, "q_table"):
            agent.rl_agent.q_table[:] = 0.0
            agent.rl_agent.epsilon = 0.1
        _log(LOG_WORKER_POLICY_EVENT, message=f"[GOAL] Set goal to ({gx},{gy}) and reset Q-table")
    finally:
        yield


@GLOBAL_ACTIONS.add(".check_cached_policy", 3)
def _act_check_cached_policy(agent, term, intention):
    """Check if cached policy exists for goal."""
    try:
        gx = int(_extract_value(term.args[0]))
        gy = int(_extract_value(term.args[1]))
        min_conf = float(_extract_value(term.args[2]))
        key = f"goal_{gx}_{gy}"
        
        if hasattr(agent, "cached_policies") and key in agent.cached_policies:
            policy = agent.cached_policies[key]
            conf = float(policy.get("confidence", 0.0))
            if conf >= min_conf:
                seq = policy.get("sequence", [])
                seq_str = "[" + ",".join(seq) + "]"
                lit = agentspeak.Literal("has_policy", [
                    agentspeak.Literal(key),
                    agentspeak.Literal(seq_str)
                ])
                agent.add_belief(lit, intention.scope)
                _log(LOG_WORKER_POLICY_EVENT, message=f"[OK] Found cached policy: {key} (conf={conf:.3f})")
            else:
                _log(LOG_WORKER_POLICY_EVENT, message=f"[SKIP] Cached policy {key} below min conf {min_conf}")
        else:
            _log(LOG_WORKER_POLICY_EVENT, message=f"[NO_CACHE] No cached policy found for {key}")
    finally:
        yield


@GLOBAL_ACTIONS.add(".get_state_from_pos", 3)
def _act_get_state_from_pos(agent, term, intention):
    """Convert (x,y) position to state."""
    try:
        x = int(_extract_value(term.args[0]))
        y = int(_extract_value(term.args[1]))
        if hasattr(agent.adapter, "_ncol"):
            state = y * agent.adapter._ncol + x
        else:
            state = y * 8 + x  # Default 8x8
        agent.set_result(int(state), intention)
    finally:
        yield


@GLOBAL_ACTIONS.add(".get_best_action", 2)
def _act_get_best_action(agent, term, intention):
    """Get best action from Q-table."""
    try:
        state = int(_extract_value(term.args[0]))
        if hasattr(agent, "runtime") and hasattr(agent.runtime, "get_action"):
            a_idx = agent.runtime.get_action(state, training=True)
        else:
            import numpy as np
            a_idx = int(np.argmax(agent.rl_agent.q_table[state]))
        action_names = ["left", "down", "right", "up"]
        name = action_names[a_idx % 4]
        agent.set_result(agentspeak.Literal(name), intention)
    finally:
        yield


@GLOBAL_ACTIONS.add(".set_epsilon", 1)
def _act_set_epsilon(agent, term, intention):
    """Set exploration rate."""
    try:
        eps = float(_extract_value(term.args[0]))
        if hasattr(agent, "rl_agent"):
            agent.rl_agent.epsilon = eps
        _log(LOG_WORKER_POLICY_EVENT, message=f"[EPS] epsilon <- {eps:.3f}")
    finally:
        yield


@GLOBAL_ACTIONS.add(".execute_action", 1)
def _act_execute_action(agent, term, intention):
    """Execute action and update Q-table."""
    try:
        action_name = _extract_value(term.args[0])
        a_map = {"left": 0, "down": 1, "right": 2, "up": 3}
        a = a_map.get(action_name, 0)
        s = agent.current_state
        ns, r, done, truncated, info = agent.adapter.step(a)
        
        # Update Q-table
        if hasattr(agent, "runtime"):
            agent.runtime.update_q_online(s, a, float(r), ns, done)
        
        agent.current_state = ns
        agent.episode_steps += 1
        x, y = agent.adapter.state_to_pos(ns)
        
        if done:
            if float(r) > 0:
                agent.add_belief(agentspeak.Literal("goal_reached", []), intention.scope)
                _log(LOG_WORKER_POLICY_EVENT, message=f"[ONLINE] Successfully reached goal at ({x},{y}) in {agent.episode_steps} steps")
            else:
                agent.add_belief(agentspeak.Literal("fell_in_hole", []), intention.scope)
                _log(LOG_WORKER_POLICY_EVENT, message=f"[ONLINE] Fell in hole at ({x},{y}) after {agent.episode_steps} steps")
    finally:
        yield


@GLOBAL_ACTIONS.add(".exec_cached_seq", 1)
def _act_exec_cached_seq(agent, term, intention):
    """Execute cached action sequence."""
    try:
        raw = _extract_value(term.args[0])
        seq = [] if raw == "[]" else [s.strip() for s in raw.strip("[]").split(",") if s.strip()]
        agent.rl_agent.epsilon = 0.0  # deterministic
        a_map = {"left": 0, "down": 1, "right": 2, "up": 3}
        
        for name in seq:
            if name not in a_map:
                _log(LOG_WORKER_POLICY_ERROR, message=f"Invalid action '{name}' in cached sequence")
                agent.add_belief(agentspeak.Literal("cached_policy_failed", []), intention.scope)
                return
            
            a = a_map[name]
            s = agent.current_state
            ns, r, done, truncated, info = agent.adapter.step(a)
            agent.runtime.update_q_online(s, a, float(r), ns, done)
            agent.current_state = ns
            agent.episode_steps += 1
            
            if done:
                if float(r) > 0:
                    agent.add_belief(agentspeak.Literal("goal_reached", []), intention.scope)
                    _log(LOG_WORKER_POLICY_EVENT, message="[CACHED] Successfully reached goal with cached policy")
                else:
                    agent.add_belief(agentspeak.Literal("fell_in_hole", []), intention.scope)
                    agent.add_belief(agentspeak.Literal("cached_policy_failed", []), intention.scope)
                    _log(LOG_WORKER_POLICY_WARNING, message="[CACHED] Cached policy failed - fell in hole")
                break
    finally:
        yield


@GLOBAL_ACTIONS.add(".rl_propose_seq", 1)
def _act_rl_propose_seq(agent, term, intention):
    """Propose sequence from Q-table."""
    try:
        s0 = int(_extract_value(term.args[0]))
        if hasattr(agent.runtime, "propose_sequence"):
            seq, conf = agent.runtime.propose_sequence(s0, max_len=30)
        else:
            seq, conf = [], 0.0
        seq_str = "[" + ",".join(seq) + "]"
        agent.set_result(agentspeak.Literal(seq_str), intention)
        agent.add_belief(agentspeak.Literal("seq_confidence", [agentspeak.Literal(str(round(conf, 3)))]), intention.scope)
        agent.add_belief(agentspeak.Literal("proposed_seq", [agentspeak.Literal(seq_str)]), intention.scope)
    finally:
        yield


@GLOBAL_ACTIONS.add(".cache_policy", 5)
def _act_cache_policy(agent, term, intention):
    """Cache successful policy."""
    try:
        gx = int(_extract_value(term.args[0]))
        gy = int(_extract_value(term.args[1]))
        seq_str = _extract_value(term.args[2])
        conf = float(_extract_value(term.args[3]))
        min_conf = float(_extract_value(term.args[4]))
        key = f"goal_{gx}_{gy}"
        
        if conf >= min_conf:
            seq = [] if seq_str == "[]" else [s.strip() for s in seq_str.strip("[]").split(",") if s.strip()]
            if not hasattr(agent, "cached_policies"):
                agent.cached_policies = {}
            agent.cached_policies[key] = {"sequence": seq, "confidence": conf, "episode": agent.episode_count}
            seq_b = "[" + ",".join(seq) + "]"
            lit = agentspeak.Literal("has_policy", [agentspeak.Literal(key), agentspeak.Literal(seq_b)])
            agent.add_belief(lit, intention.scope)
            _log(LOG_WORKER_POLICY_EVENT, message=f"[CACHE] {key} (conf={conf:.3f}): {seq}")
    finally:
        yield


@GLOBAL_ACTIONS.add(".clear_episode_flags", 0)
def _act_clear_episode_flags(agent, term, intention):
    """Clear episode outcome flags."""
    try:
        for flag in ("goal_reached", "fell_in_hole", "episode_timeout", "cached_policy_failed"):
            for b in list(agent.beliefs):
                if b.functor == flag:
                    agent.remove_belief(b, intention)
    finally:
        yield


@GLOBAL_ACTIONS.add(".remove_cached_policy", 0)
def _act_remove_cached_policy(agent, term, intention):
    """Remove cached policy for current goal."""
    try:
        gx, gy = agent.adapter.goal_pos()
        key = f"goal_{gx}_{gy}"
        if hasattr(agent, "cached_policies") and key in agent.cached_policies:
            del agent.cached_policies[key]
            _log(LOG_WORKER_POLICY_EVENT, message=f"[REMOVE] Removed cached policy for {key}")
    finally:
        yield


@GLOBAL_ACTIONS.add(".clear_policy_store", 0)
def _act_clear_policy_store(agent, term, intention):
    """Clear all cached policies."""
    try:
        if hasattr(agent, "cached_policies"):
            agent.cached_policies.clear()
        _log(LOG_WORKER_POLICY_EVENT, message="[CLEAR] Cleared all cached policies")
    finally:
        yield


@GLOBAL_ACTIONS.add(".save_policies", 0)
def _act_save_policies(agent, term, intention):
    """Save policies to disk."""
    try:
        if hasattr(agent, "cached_policies") and hasattr(agent, "policy_store"):
            agent.policy_store.save(agent.cached_policies)
            _log(LOG_WORKER_POLICY_EVENT, message=f"[SAVE] Saved {len(agent.cached_policies)} policies")
    finally:
        yield


@GLOBAL_ACTIONS.add(".load_policies", 0)
def _act_load_policies(agent, term, intention):
    """Load policies from disk."""
    try:
        if hasattr(agent, "policy_store"):
            agent.cached_policies = agent.policy_store.load()
            _log(LOG_WORKER_POLICY_EVENT, message=f"[LOAD] Loaded {len(agent.cached_policies)} policies")
    finally:
        yield


def register_actions() -> agentspeak.Actions:
    """Return the global actions registry for BDI agent initialization."""
    return GLOBAL_ACTIONS


__all__ = ["GLOBAL_ACTIONS", "register_actions"]
