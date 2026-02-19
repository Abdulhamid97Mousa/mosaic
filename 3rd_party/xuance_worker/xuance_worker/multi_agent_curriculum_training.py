"""
Multi-Agent Curriculum Training for XuanCe.

Single-process curriculum training for multi-agent environments (e.g. multigrid).
Uses environment swapping between phases -- the runner, policy weights, optimizer
state, and LR schedule persist across phase transitions in memory.

Unlike the single-agent curriculum (which uses Syllabus-RL and a custom PPO loop),
this module uses XuanCe's native RunnerMARL and agents.train() with environment
hot-swapping between phases.

Why not Syllabus here?
    Syllabus wraps gym.Env (single-agent API). XuanCe MARL uses RawMultiAgentEnv
    (dict-based API) with DummyVecMultiAgentEnv. The GymnasiumSyncWrapper can't
    parse multi-agent reward dicts to track curriculum progress.

Usage from CLI:
    python -m xuance_worker.cli --config curriculum_config.json

Where curriculum_config.json has:
    {
        "method": "mappo",
        "env": "multigrid",
        "env_id": "collect_1vs1",
        "running_steps": 2000000,
        "extras": {
            "curriculum_schedule": [
                {"env_id": "collect_1vs1", "steps": 1000000},
                {"env_id": "soccer_1vs1", "steps": 1000000}
            ],
            "training_mode": "competitive"
        }
    }
"""

from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .config import XuanCeWorkerConfig

# Disable tqdm progress bars BEFORE XuanCe imports tqdm.
# XuanCe's on_policy_marl.py uses `for _ in tqdm(range(n_steps))` which writes
# to stderr on every iteration. When stderr is a pipe (e.g. telemetry proxy),
# the 1MB pipe buffer fills up and blocks the training process (pipe deadlock).
# This MUST be set before any `from xuance import ...` call.
os.environ.setdefault("TQDM_DISABLE", "1")

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiAgentCurriculumConfig:
    """Configuration for multi-agent curriculum training."""

    # Base training config (used for runner creation)
    worker_config: XuanCeWorkerConfig

    # Curriculum phases: [{"env_id": "collect_1vs1", "steps": 1000000}, ...]
    curriculum_schedule: List[Dict[str, Any]]

    @classmethod
    def from_worker_config(cls, config: XuanCeWorkerConfig) -> "MultiAgentCurriculumConfig":
        extras = config.extras or {}
        schedule = extras.get("curriculum_schedule")
        if not schedule:
            raise ValueError(
                "curriculum_schedule is required in extras for multi-agent "
                "curriculum training. Example: [{'env_id': 'collect_1vs1', "
                "'steps': 1000000}, {'env_id': 'soccer_1vs1', 'steps': 1000000}]"
            )

        # Validate schedule entries
        for i, phase in enumerate(schedule):
            if "env_id" not in phase:
                raise ValueError(f"Phase {i} missing 'env_id' in curriculum_schedule")
            if "steps" not in phase:
                raise ValueError(f"Phase {i} missing 'steps' in curriculum_schedule")

        return cls(worker_config=config, curriculum_schedule=schedule)

    @property
    def total_steps(self) -> int:
        return sum(p["steps"] for p in self.curriculum_schedule)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_multi_agent_curriculum_config(config: XuanCeWorkerConfig) -> bool:
    """Check if config specifies multi-agent curriculum training.

    Multi-agent curriculum requires BOTH:
    - curriculum_schedule in extras
    - A multi-agent environment family (e.g. multigrid)
    """
    extras = config.extras or {}
    has_schedule = bool(extras.get("curriculum_schedule"))
    is_marl = config.env in ("multigrid",)
    return has_schedule and is_marl


# ---------------------------------------------------------------------------
# Runner creation (reuses runtime.py infrastructure)
# ---------------------------------------------------------------------------

def _build_parser_args(config: XuanCeWorkerConfig) -> SimpleNamespace:
    """Build XuanCe parser_args from worker config."""
    args = SimpleNamespace()
    args.dl_toolbox = config.dl_toolbox
    args.device = config.device
    args.parallels = config.parallels
    args.running_steps = config.running_steps
    args.env_id = config.env_id

    if config.seed is not None:
        args.seed = config.seed
        args.env_seed = config.seed

    # Apply extras (training_mode, num_envs, tensorboard_dir, etc.)
    for key, value in config.extras.items():
        if key != "curriculum_schedule":  # Don't pass schedule to XuanCe
            setattr(args, key, value)

    return args


def _create_runner(config: XuanCeWorkerConfig, env_id: str):
    """Create a XuanCe RunnerMARL for the given env_id.

    Reuses the same config resolution logic as runtime.py:
    normalize method -> resolve custom YAML config -> get_runner().
    """
    from xuance import get_runner
    from .runtime import (
        _normalize_method_name,
        _resolve_custom_config_path,
        _get_competition_num_groups,
    )

    normalized_method = _normalize_method_name(config.method)

    # Build parser_args with the phase's env_id
    parser_args = _build_parser_args(config)
    parser_args.env_id = env_id

    # Set output directories from extras
    extras = config.extras or {}
    if extras.get("tensorboard_dir"):
        parser_args.log_dir = extras["tensorboard_dir"]
    if extras.get("checkpoint_dir"):
        parser_args.model_dir = extras["checkpoint_dir"]

    # Resolve config path (finds configs/mappo/multigrid/collect_1vs1.yaml etc.)
    num_groups = _get_competition_num_groups(config.env, env_id)
    resolved_config_path = _resolve_custom_config_path(
        method=normalized_method,
        env=config.env,
        env_id=env_id,
        num_groups=num_groups,
        config_path=config.config_path,
    )

    if num_groups is not None:
        method_for_runner = [normalized_method] * num_groups
    else:
        method_for_runner = normalized_method

    LOGGER.info(
        "Creating runner: method=%s env=%s env_id=%s config=%s",
        method_for_runner, config.env, env_id, resolved_config_path,
    )

    runner = get_runner(
        algo=method_for_runner,
        env=config.env,
        env_id=env_id,
        config_path=resolved_config_path,
        parser_args=parser_args,
    )

    LOGGER.info("Created runner: %s", type(runner).__name__)
    return runner


def _create_envs(runner_config, env_id: str):
    """Create new vectorized multi-agent envs for a different env_id.

    Uses the runner's merged config (which has all XuanCe internals like
    vectorize, env_seed, etc.) with just env_id swapped.
    """
    from xuance.environment import make_envs

    phase_config = copy.deepcopy(runner_config)
    phase_config.env_id = env_id
    return make_envs(phase_config)


from ._patches import apply_xuance_patches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_multi_agent_curriculum_training(
    config: MultiAgentCurriculumConfig,
) -> Dict[str, Any]:
    """Run multi-agent curriculum training in a single process.

    Creates one RunnerMARL for the first phase, then swaps environments
    between phases. Policy weights, optimizer state, LR schedule, and step
    counter persist across all phase transitions.

    Args:
        config: Multi-agent curriculum configuration.

    Returns:
        Dictionary with training results.
    """
    schedule = config.curriculum_schedule
    worker_config = config.worker_config
    start_time = time.time()

    LOGGER.info("=" * 60)
    LOGGER.info("  Multi-Agent Curriculum Training")
    LOGGER.info("=" * 60)
    LOGGER.info("Phases: %d", len(schedule))
    for i, phase in enumerate(schedule):
        LOGGER.info("  Phase %d: %s (%d steps)", i + 1, phase["env_id"], phase["steps"])
    LOGGER.info("Total steps: %d", config.total_steps)
    LOGGER.info("=" * 60)

    # Reset FastLane slot counter before creating any environments
    try:
        from .fastlane import reset_slot_counter
        reset_slot_counter()
        LOGGER.info("FastLane slot counter reset")
    except ImportError:
        pass

    # Apply all XuanCe v1.4.0 shim-layer patches before creating runners
    apply_xuance_patches()

    # Create runner for the first phase
    first_env_id = schedule[0]["env_id"]
    runner = _create_runner(worker_config, first_env_id)

    phase_results = []

    for phase_idx, phase in enumerate(schedule):
        env_id = phase["env_id"]
        phase_steps = phase["steps"]
        phase_start = time.time()

        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("  Phase %d/%d: %s (%d steps)",
                     phase_idx + 1, len(schedule), env_id, phase_steps)
        LOGGER.info("=" * 60)

        if phase_idx > 0:
            # -- Environment swap --
            LOGGER.info("Swapping environments: -> %s", env_id)

            # Close old envs FIRST (releases their FastLane writers)
            old_envs = runner.envs
            old_envs.close()

            # Unlink old FastLane shared memory buffer so the new envs
            # (which may have different frame dimensions) create a fresh one.
            # Reset slot counter so new envs get slots 0..N-1.
            try:
                from .fastlane import cleanup_fastlane_buffer, reset_slot_counter
                cleanup_fastlane_buffer()
                reset_slot_counter()
            except ImportError:
                pass

            new_envs = _create_envs(runner.config, env_id)
            new_envs.reset()  # Populate buf_obs for agents.train()

            # CRITICAL: XuanCe's training loop uses agent.train_envs (not
            # agent.envs).  Setting the wrong attribute silently keeps
            # training on the OLD environment -- the swap looks successful
            # in logs but the agent never sees the new env.
            runner.envs = new_envs
            runner.agent.train_envs = new_envs

            # Sanity check: verify the agent will actually use the new envs
            assert runner.agent.train_envs is new_envs, (
                "Environment swap failed: runner.agent.train_envs does not "
                "point to the new environments.  The training loop reads "
                "from train_envs, so swapping any other attribute is a no-op."
            )

            LOGGER.info(
                "Environment swap complete: %s (step counter at %d)",
                env_id, runner.agent.current_step,
            )

        # Train this phase
        n_train_steps = phase_steps // runner.n_envs
        LOGGER.info(
            "Training %d iterations (%d steps / %d envs)",
            n_train_steps, phase_steps, runner.n_envs,
        )

        runner.agent.train(n_train_steps)

        # Save phase checkpoint
        model_name = f"phase{phase_idx + 1}_model.pth"
        runner.agent.save_model(model_name)

        phase_elapsed = time.time() - phase_start
        LOGGER.info(
            "Phase %d complete: %s | %d steps in %.1fs",
            phase_idx + 1, env_id, phase_steps, phase_elapsed,
        )

        phase_results.append({
            "phase": phase_idx + 1,
            "env_id": env_id,
            "steps": phase_steps,
            "elapsed_seconds": phase_elapsed,
        })

    # Final save
    runner.agent.save_model("final_train_model.pth")
    runner.agent.finish()
    runner.envs.close()

    total_elapsed = time.time() - start_time

    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("  Curriculum Training Complete!")
    LOGGER.info("=" * 60)
    LOGGER.info("Total steps: %d", config.total_steps)
    LOGGER.info("Total time: %.1fs", total_elapsed)
    for pr in phase_results:
        LOGGER.info(
            "  Phase %d (%s): %d steps, %.1fs",
            pr["phase"], pr["env_id"], pr["steps"], pr["elapsed_seconds"],
        )
    LOGGER.info("=" * 60)

    return {
        "total_steps": config.total_steps,
        "total_elapsed_seconds": total_elapsed,
        "phases": phase_results,
    }
