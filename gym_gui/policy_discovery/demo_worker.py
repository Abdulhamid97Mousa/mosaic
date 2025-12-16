#!/usr/bin/env python3
"""
Demo worker that emits JSONL telemetry to stdout.

This worker simulates a simple training run by generating random
episodes with steps, emitting JSONL events that the telemetry proxy
can consume and forward to the daemon via gRPC.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import Any, Dict


def emit_jsonl(event: Dict[str, Any]) -> None:
    """Emit a single JSONL event to stdout."""
    json.dump(event, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_demo(
    run_id: str,
    agent_id: str,
    num_episodes: int = 3,
    steps_per_episode: int = 15,
    step_delay: float = 0.03,
) -> None:
    """
    Run a simple demo training loop that emits telemetry.
    
    Args:
        run_id: Run identifier for correlation
        agent_id: Agent identifier for correlation
        num_episodes: Number of episodes to run
        steps_per_episode: Number of steps per episode
        step_delay: Delay between steps in seconds
    """
    # Write startup message to stderr (won't be parsed as JSONL)
    sys.stderr.write(f"[demo_worker] Starting run_id={run_id}, agent_id={agent_id}\n")
    sys.stderr.flush()

    for episode_idx in range(num_episodes):
        total_reward = 0.0
        
        for step_idx in range(steps_per_episode):
            # Simulate observations and actions
            observation = {
                "position": [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)],
                "velocity": [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)],
            }
            action = random.randint(0, 3)
            reward = random.choice([0.0, 0.0, 0.0, 1.0])  # Sparse rewards
            total_reward += reward
            
            terminated = (step_idx == steps_per_episode - 1)
            
            # Emit step event as JSONL
            step_event = {
                "type": "step",
                "run_id": run_id,
                "agent_id": agent_id,
                "episode_index": episode_idx,
                "step_index": step_idx,
                "action_json": json.dumps(action),
                "observation_json": json.dumps(observation),
                "reward": reward,
                "terminated": terminated,
                "truncated": False,
                "policy_label": "demo_policy",
                "backend": "demo",
                "ts_unix_ns": int(time.time_ns()),
            }
            emit_jsonl(step_event)
            
            # Simulate step delay
            time.sleep(step_delay)
        
        # Emit episode summary event with rich metadata
        episode_event = {
            "type": "episode",
            "run_id": run_id,
            "agent_id": agent_id,
            "episode_index": episode_idx,
            "total_reward": total_reward,
            "steps": steps_per_episode,
            "terminated": True,
            "truncated": False,
            "metadata_json": json.dumps({
                "note": "demo episode",
                "avg_reward": total_reward / steps_per_episode,
                "seed": 42 + episode_idx,  # For GUI display
                "episode_index": episode_idx,  # For GUI sorting
                "game_id": "DemoEnv-v0",  # For GUI display
                "control_mode": "agent",  # Track control mode: human/agent/multi_agent/hybrid
                "run_id": run_id,  # Include for reference
                "policy_label": "demo_policy",
                "backend": "demo_worker",
            }),
            "ts_unix_ns": int(time.time_ns()),
        }
        emit_jsonl(episode_event)
        
        sys.stderr.write(
            f"[demo_worker] Episode {episode_idx} complete: "
            f"reward={total_reward:.2f}, steps={steps_per_episode}\n"
        )
        sys.stderr.flush()
    
    sys.stderr.write(f"[demo_worker] All {num_episodes} episodes complete\n")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for demo worker."""
    parser = argparse.ArgumentParser(description="Demo worker for telemetry testing")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--agent-id", default="agent_1", help="Agent identifier")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=15, help="Steps per episode")
    parser.add_argument("--delay", type=float, default=0.03, help="Delay between steps")
    
    args = parser.parse_args(argv)
    
    try:
        run_demo(
            run_id=args.run_id,
            agent_id=args.agent_id,
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            step_delay=args.delay,
        )
        return 0
    except KeyboardInterrupt:
        sys.stderr.write("\n[demo_worker] Interrupted\n")
        return 130
    except Exception as exc:
        sys.stderr.write(f"[demo_worker] Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
