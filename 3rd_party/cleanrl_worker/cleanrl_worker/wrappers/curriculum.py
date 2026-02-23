"""
Curriculum Learning Wrappers using Syllabus-RL.

This module provides algorithm-agnostic curriculum wrappers that work with
any CleanRL algorithm (PPO, DQN, SAC, etc.) by operating at the environment level.

The key insight is that curriculum learning should happen at the ENVIRONMENT level,
not the ALGORITHM level. This way, any algorithm can benefit from curriculum learning
without any modifications to the training code.

Usage:
    from cleanrl_worker.wrappers.curriculum import make_curriculum_env

    # Define curriculum schedule
    curriculum_schedule = [
        {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
        {"env_id": "BabyAI-GoToRedBall-v0", "steps": 200000},
        {"env_id": "BabyAI-GoToObj-v0", "steps": 200000},
        {"env_id": "BabyAI-GoToLocal-v0", "steps": 200000},
    ]

    # Create curriculum-enabled environment
    envs = make_curriculum_env(curriculum_schedule, num_envs=4)

    # Use with ANY algorithm - no changes needed to training code!
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
from .minigrid import is_minigrid_env

# Register BabyAI/MiniGrid environments before gym.make()
try:
    import minigrid  # noqa: F401 - registers MiniGrid envs
except ImportError:
    pass

try:
    import babyai  # noqa: F401 - registers BabyAI envs
except ImportError:
    pass

from syllabus.core import ReinitTaskWrapper
from syllabus.core import GymnasiumSyncWrapper, make_multiprocessing_curriculum
from syllabus.curricula import SequentialCurriculum, Constant
from syllabus.task_space import DiscreteTaskSpace

_LOGGER = logging.getLogger(__name__)


class BabyAITaskWrapper(ReinitTaskWrapper):
    """
    Task wrapper for BabyAI environments that switches between different env_ids.

    This wrapper allows seamless switching between BabyAI environments during training,
    enabling curriculum learning where the agent progresses through increasingly
    difficult environments.

    Example tasks:
        0: BabyAI-GoToRedBallNoDists-v0  (easiest)
        1: BabyAI-GoToRedBall-v0
        2: BabyAI-GoToObj-v0
        3: BabyAI-GoToLocal-v0           (hardest)
    """

    def __init__(
        self,
        env: gym.Env,
        env_ids: List[str],
        make_env_fn: Optional[Callable[[str, int], gym.Env]] = None,
        apply_wrappers: bool = True,
    ):
        """
        Initialize the BabyAI task wrapper.

        Args:
            env: Initial environment instance
            env_ids: List of environment IDs for the curriculum
            make_env_fn: Optional custom function to create environments.
                         Signature: (env_id: str, idx: int) -> gym.Env
            apply_wrappers: Whether to apply standard MiniGrid wrappers
        """
        self.env_ids = env_ids
        self.apply_wrappers = apply_wrappers
        self._make_env_fn = make_env_fn or self._default_make_env

        # Create task space - tasks are the env_ids themselves (strings)
        task_space = DiscreteTaskSpace(len(env_ids), env_ids)

        # Create env factory function for ReinitTaskWrapper
        # NOTE: Syllabus passes the TASK (env_id string), not the index
        def env_fn(task: Any) -> gym.Env:
            # Handle both integer index and string env_id
            if isinstance(task, int):
                env_id = env_ids[task]
            else:
                # Task is the env_id string directly
                env_id = task
            # Skip FastLane wrapping for replacement envs - the original
            # envs already have FastLane wrappers with proper slot assignments.
            # Without this, replacement envs get slots >= grid_limit and
            # won't contribute frames to GRID mode.
            os.environ["FASTLANE_SKIP_WRAP"] = "1"
            try:
                return self._make_env_fn(env_id, 0)
            finally:
                os.environ.pop("FASTLANE_SKIP_WRAP", None)

        super().__init__(env, env_fn, task_space)

        # Start with the first task
        self.task = 0

        _LOGGER.info(
            "BabyAITaskWrapper initialized with %d environments: %s",
            len(env_ids),
            env_ids,
        )

    def _default_make_env(self, env_id: str, _idx: int) -> gym.Env:
        """Default environment creation with standard MiniGrid wrappers."""
        env = gym.make(env_id)

        if self.apply_wrappers:
            _is_mg = is_minigrid_env(env_id)

            # Apply standard MiniGrid observation wrappers
            if _is_mg:
                try:
                    from minigrid.wrappers import ImgObsWrapper
                    env = ImgObsWrapper(env)
                except ImportError:
                    pass
                # NOTE: Do NOT flatten — MinigridAgent (CNN) expects raw (7,7,3) images
            else:
                env = gym.wrappers.FlattenObservation(env)

        return env

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None, new_task: Optional[Any] = None) -> Tuple[Any, Dict]:
        """
        Reset the environment, optionally switching to a new task.

        Args:
            seed: Random seed for the environment
            options: Additional options for reset
            new_task: Task to switch to (can be int index or string env_id)

        Returns:
            Tuple of (observation, info)
        """
        if new_task is not None:
            self.change_task(new_task)
            # Get env_id for logging (task could be int or string)
            env_id = new_task if isinstance(new_task, str) else self.env_ids[new_task]
            _LOGGER.debug(
                "Switched to task %s: %s",
                new_task,
                env_id,
            )

        return self.env.reset(seed=seed, options=options)

    @property
    def current_env_id(self) -> str:
        """Return the current environment ID."""
        if self.task is None:
            return self.env_ids[0]
        # Task can be int index or string env_id
        if isinstance(self.task, str):
            return self.task
        return self.env_ids[self.task]


def make_babyai_curriculum(
    curriculum_schedule: List[Dict[str, Any]],
    task_space: Optional[DiscreteTaskSpace] = None,
) -> SequentialCurriculum:
    """
    Create a SequentialCurriculum for BabyAI environments.

    Args:
        curriculum_schedule: List of dicts with 'env_id' and stopping condition.
            Each dict should have:
            - 'env_id': The BabyAI environment ID
            - 'steps': Number of steps before advancing (optional)
            - 'episodes': Number of episodes before advancing (optional)
            - 'episode_return': Average return threshold (optional)

    Returns:
        SequentialCurriculum instance

    Example:
        curriculum_schedule = [
            {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
            {"env_id": "BabyAI-GoToRedBall-v0", "steps": 200000},
            {"env_id": "BabyAI-GoToObj-v0", "steps": 200000},
            {"env_id": "BabyAI-GoToLocal-v0"},  # Final stage, no stopping condition
        ]
    """
    env_ids = [stage["env_id"] for stage in curriculum_schedule]

    if task_space is None:
        task_space = DiscreteTaskSpace(len(env_ids), env_ids)

    # Build curriculum list (one Constant curriculum per stage)
    curriculum_list = []
    for i, stage in enumerate(curriculum_schedule):
        # Each stage uses the corresponding task index
        stage_task_space = DiscreteTaskSpace(1, [i])
        curriculum_list.append(Constant(i, stage_task_space))

    # Build stopping conditions
    stopping_conditions = []
    for stage in curriculum_schedule[:-1]:  # All but the last stage
        conditions = []

        if "steps" in stage:
            conditions.append(f"steps>={stage['steps']}")
        if "episodes" in stage:
            conditions.append(f"episodes>={stage['episodes']}")
        if "episode_return" in stage:
            conditions.append(f"episode_return>={stage['episode_return']}")

        # Default to steps if no condition specified
        if not conditions:
            conditions.append("steps>=100000")

        # Combine conditions with OR (any condition triggers advancement)
        stopping_conditions.append("|".join(conditions))

    curriculum = SequentialCurriculum(
        curriculum_list=curriculum_list,
        stopping_conditions=stopping_conditions,
        task_space=task_space,
    )

    _LOGGER.info(
        "Created BabyAI curriculum with %d stages: %s",
        len(env_ids),
        env_ids,
    )

    return curriculum


def make_curriculum_env(
    curriculum_schedule: List[Dict[str, Any]],
    num_envs: int = 4,
    seed: Optional[int] = None,  # Reserved for future seeding support
    capture_video: bool = False,
    run_name: str = "curriculum_run",
    apply_wrappers: bool = True,
    max_episode_steps: int = 256,
) -> gym.vector.VectorEnv:
    """
    Create a vectorized environment with curriculum learning support.

    This is the main entry point for curriculum learning. It creates a
    VectorEnv that automatically switches between environments based on
    the curriculum schedule.

    Args:
        curriculum_schedule: List of dicts defining the curriculum stages.
            Each dict should have:
            - 'env_id': The environment ID
            - 'steps': Number of steps before advancing (optional)
            - 'episodes': Number of episodes before advancing (optional)
            - 'episode_return': Average return threshold (optional)
        num_envs: Number of parallel environments
        seed: Random seed for reproducibility
        capture_video: Whether to capture video (only for env 0)
        run_name: Name for video recording folder
        apply_wrappers: Whether to apply standard observation wrappers

    Returns:
        VectorEnv with curriculum learning support

    Example:
        curriculum = [
            {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
            {"env_id": "BabyAI-GoToRedBall-v0", "steps": 200000},
            {"env_id": "BabyAI-GoToObj-v0", "steps": 200000},
            {"env_id": "BabyAI-GoToLocal-v0"},
        ]

        envs = make_curriculum_env(curriculum, num_envs=4)

        # Now use with ANY CleanRL algorithm!
        # The curriculum switching happens automatically during training.
    """
    env_ids = [stage["env_id"] for stage in curriculum_schedule]
    task_space = DiscreteTaskSpace(len(env_ids), env_ids)

    # Create the base curriculum
    base_curriculum = make_babyai_curriculum(curriculum_schedule, task_space)

    # Wrap with multiprocessing sync wrapper (required for GymnasiumSyncWrapper)
    # This works for both single-process and multi-process training
    mp_curriculum = make_multiprocessing_curriculum(base_curriculum, start=True)

    # Detect whether the curriculum targets MiniGrid/BabyAI environments
    _is_mg = is_minigrid_env(curriculum_schedule[0]["env_id"])

    def make_env_fn(env_id: str, idx: int) -> gym.Env:
        """Create a single environment with wrappers."""
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        if apply_wrappers:
            if _is_mg:
                # MiniGrid/BabyAI: ImgObsWrapper converts Dict→image Box.
                # Do NOT flatten — MinigridAgent (CNN) expects raw (7,7,3).
                try:
                    from minigrid.wrappers import ImgObsWrapper
                    env = ImgObsWrapper(env)
                except ImportError:
                    pass
            else:
                env = gym.wrappers.FlattenObservation(env)

        # TimeLimit BEFORE RecordEpisodeStatistics so truncation
        # triggers the "episode" info dict correctly.
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    def make_curriculum_env_fn(idx: int) -> gym.Env:
        """Create a curriculum-wrapped environment with Syllabus sync wrapper."""
        # IMPORTANT: Disable FastLane monkey-patching for INNER envs.
        # ReinitTaskWrapper.change_task() replaces self.env entirely, which
        # would disconnect any FastLane wrapper applied inside the task wrapper.
        # Instead, we apply FastLane at the OUTERMOST level after all Syllabus wrappers.
        os.environ["FASTLANE_SKIP_WRAP"] = "1"
        try:
            # Start with the first environment (no FastLane wrapper inside)
            initial_env = make_env_fn(env_ids[0], idx)
        finally:
            os.environ.pop("FASTLANE_SKIP_WRAP", None)

        # Wrap with task wrapper - this handles environment switching
        task_env = BabyAITaskWrapper(
            initial_env,
            env_ids,
            make_env_fn=lambda env_id, _i: make_env_fn(env_id, idx),
            apply_wrappers=False,  # Already applied above
        )

        # Wrap with Syllabus sync wrapper for curriculum communication
        # mp_curriculum.components contains the MultiProcessingComponents
        sync_env = GymnasiumSyncWrapper(
            task_env,
            task_space,
            components=mp_curriculum.components,
        )

        # Apply FastLane wrapper at the OUTERMOST level.
        # This ensures step() calls always flow through FastLane regardless
        # of internal task switching by Syllabus/ReinitTaskWrapper.
        from cleanrl_worker.fastlane import maybe_wrap_env
        outer_env = maybe_wrap_env(sync_env)

        return outer_env

    # Reset FastLane slot counter before creating envs to ensure proper slot assignments
    from cleanrl_worker.fastlane import reset_slot_counter
    reset_slot_counter()

    # Create vectorized environment (same for both modes now)
    envs = gym.vector.SyncVectorEnv([lambda i=i: make_curriculum_env_fn(i) for i in range(num_envs)])

    _LOGGER.info(
        "Created curriculum VectorEnv with %d parallel environments",
        num_envs,
    )

    return envs


# Convenience presets for common BabyAI curricula
BABYAI_GOTO_CURRICULUM = [
    {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
    {"env_id": "BabyAI-GoToRedBall-v0", "steps": 200000},
    {"env_id": "BabyAI-GoToObj-v0", "steps": 200000},
    {"env_id": "BabyAI-GoToLocal-v0"},
]

BABYAI_DOORKEY_CURRICULUM = [
    {"env_id": "MiniGrid-DoorKey-5x5-v0", "steps": 100000},
    {"env_id": "MiniGrid-DoorKey-6x6-v0", "steps": 150000},
    {"env_id": "MiniGrid-DoorKey-8x8-v0", "steps": 200000},
    {"env_id": "MiniGrid-DoorKey-16x16-v0"},
]


__all__ = [
    "BabyAITaskWrapper",
    "make_babyai_curriculum",
    "make_curriculum_env",
    "BABYAI_GOTO_CURRICULUM",
    "BABYAI_DOORKEY_CURRICULUM",
]
