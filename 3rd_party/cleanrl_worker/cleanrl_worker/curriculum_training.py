"""
Curriculum Training Module for CleanRL.

This module provides a unified training loop that uses Syllabus-RL for
curriculum learning. Unlike the bash-based phase scripts, this runs everything
in a SINGLE PROCESS, preserving neural network weights across environment
transitions.

The key insight: curriculum learning should happen at the ENVIRONMENT level,
not by spawning separate training processes per phase.

Usage from CLI:
    python -m cleanrl_worker.cli --config curriculum_config.json

Where curriculum_config.json has:
    {
        "run_id": "curriculum-test",
        "algo": "ppo",
        "env_id": "BabyAI-GoToRedBallNoDists-v0",  # Starting env
        "total_timesteps": 800000,
        "extras": {
            "curriculum_schedule": [
                {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
                {"env_id": "BabyAI-GoToRedBall-v0", "steps": 200000},
                {"env_id": "BabyAI-GoToObj-v0", "steps": 200000},
                {"env_id": "BabyAI-GoToLocal-v0"}
            ]
        }
    }
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

# CRITICAL: Import sitecustomize EARLY to patch gym.make() with FastLane wrapper
# This must happen before any gym.make() calls to enable FastLane telemetry
from . import sitecustomize  # noqa: F401 - patches gym.make for FastLane

# Register BabyAI/MiniGrid environments before gym.make()
try:
    import minigrid  # noqa: F401 - registers MiniGrid envs
except ImportError:
    pass

try:
    import babyai  # noqa: F401 - registers BabyAI envs
except ImportError:
    pass
import torch.nn as nn
import torch.optim as optim

from .agents import MinigridAgent, MLPAgent
from .config import CleanRLWorkerConfig
from .wrappers.curriculum import make_curriculum_env
from .wrappers.minigrid import is_minigrid_env

_LOGGER = logging.getLogger(__name__)


@dataclass
class CurriculumTrainingConfig:
    """Configuration for curriculum training."""

    run_id: str
    curriculum_schedule: List[Dict[str, Any]]
    total_timesteps: int
    num_envs: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_steps: int = 128  # Steps per rollout
    max_episode_steps: int = 256  # Maximum steps per episode
    anneal_lr: bool = True
    seed: Optional[int] = None
    capture_video: bool = False
    cuda: bool = True
    tensorboard_dir: Optional[str] = None
    checkpoint_freq: int = 50000  # Save checkpoint every N steps
    checkpoint_dir: Optional[str] = None
    # FastLane streaming configuration
    fastlane_only: bool = True  # Enable FastLane by default for curriculum
    fastlane_slot: int = 0
    fastlane_video_mode: str = "grid"
    fastlane_grid_limit: Optional[int] = None  # Defaults to num_envs if None

    @classmethod
    def from_worker_config(cls, config: CleanRLWorkerConfig) -> "CurriculumTrainingConfig":
        """Create CurriculumTrainingConfig from CleanRLWorkerConfig."""
        extras = config.extras or {}
        algo_params = extras.get("algo_params", {})

        curriculum_schedule = extras.get("curriculum_schedule")
        if not curriculum_schedule:
            raise ValueError(
                "curriculum_schedule is required in extras for curriculum training. "
                "Example: {'curriculum_schedule': [{'env_id': 'BabyAI-GoToRedBallNoDists-v0', 'steps': 200000}, ...]}"
            )

        num_envs = algo_params.get("num_envs", extras.get("num_envs", 4))
        return cls(
            run_id=config.run_id,
            curriculum_schedule=curriculum_schedule,
            total_timesteps=config.total_timesteps,
            num_envs=num_envs,
            learning_rate=algo_params.get("learning_rate", 2.5e-4),
            gamma=algo_params.get("gamma", 0.99),
            gae_lambda=algo_params.get("gae_lambda", 0.95),
            num_minibatches=algo_params.get("num_minibatches", 4),
            update_epochs=algo_params.get("update_epochs", 4),
            clip_coef=algo_params.get("clip_coef", 0.2),
            ent_coef=algo_params.get("ent_coef", 0.01),
            vf_coef=algo_params.get("vf_coef", 0.5),
            max_grad_norm=algo_params.get("max_grad_norm", 0.5),
            num_steps=algo_params.get("num_steps", 128),
            max_episode_steps=algo_params.get("max_episode_steps", extras.get("max_episode_steps", 256)),
            anneal_lr=algo_params.get("anneal_lr", True),
            seed=config.seed,
            capture_video=extras.get("capture_video", False),
            cuda=extras.get("cuda", True),
            tensorboard_dir=extras.get("tensorboard_dir"),
            checkpoint_freq=extras.get("checkpoint_freq", 50000),
            checkpoint_dir=extras.get("checkpoint_dir"),
            # FastLane settings - enabled by default for curriculum training
            fastlane_only=extras.get("fastlane_only", True),
            fastlane_slot=extras.get("fastlane_slot", 0),
            fastlane_video_mode=extras.get("fastlane_video_mode", "grid"),
            fastlane_grid_limit=extras.get("fastlane_grid_limit"),  # None = use num_envs
        )


def run_curriculum_training(config: CurriculumTrainingConfig) -> Dict[str, Any]:
    """
    Run curriculum training with automatic environment transitions.

    This function implements PPO training with Syllabus-RL curriculum learning.
    The key difference from standard training: environments switch automatically
    based on the curriculum schedule, while the agent's weights are preserved.

    Args:
        config: Curriculum training configuration

    Returns:
        Dictionary with training results and final checkpoint path

    Note:
        FastLane environment variables (GYM_GUI_FASTLANE_*, CLEANRL_RUN_ID, etc.)
        must be set by the calling script (e.g., bash custom_script) BEFORE
        launching this training. The script is the source of truth for FastLane
        configuration to ensure run_id matches what the GUI expects.
    """
    # Log FastLane config from environment (set by parent script)
    _LOGGER.info(
        "FastLane from environment: enabled=%s, mode=%s, run_id=%s",
        os.environ.get("GYM_GUI_FASTLANE_ONLY", "not set"),
        os.environ.get("GYM_GUI_FASTLANE_VIDEO_MODE", "not set"),
        os.environ.get("CLEANRL_RUN_ID", "not set"),
    )

    # Reset FastLane slot counter BEFORE creating environments.
    # This is critical for curriculum learning where Syllabus may recreate envs
    # during task transitions. Without resetting, replacement envs get slots
    # >= grid_limit and won't contribute frames to GRID mode rendering.
    from .fastlane import reset_slot_counter
    reset_slot_counter()
    _LOGGER.info("FastLane slot counter reset for curriculum training")

    run_name = f"{config.run_id}_curriculum_{int(time.time())}"

    # Set up device
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
    _LOGGER.info(f"Using device: {device}")

    # Set random seeds
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    # Create curriculum-enabled vectorized environment
    _LOGGER.info("Creating curriculum environment...")
    _LOGGER.info(f"Schedule: {[s['env_id'] for s in config.curriculum_schedule]}")

    # NOTE: apply_wrappers=True so BabyAI/MiniGrid environments get
    # ImgObsWrapper (Dict obs → image Box).  FlattenObservation is only
    # applied for non-MiniGrid envs — MiniGrid uses a CNN that needs
    # raw (7,7,3) images.
    envs = make_curriculum_env(
        config.curriculum_schedule,
        num_envs=config.num_envs,
        seed=config.seed,
        capture_video=config.capture_video,
        run_name=run_name,
        apply_wrappers=True,
        max_episode_steps=config.max_episode_steps,
    )

    # Select agent architecture based on env type — must mirror
    # algorithms/ppo.py so checkpoints are compatible with the
    # interactive runtime.
    starting_env_id = config.curriculum_schedule[0]["env_id"]
    if is_minigrid_env(starting_env_id):
        _LOGGER.info("Using MinigridAgent (CNN) for MiniGrid curriculum")
        agent = MinigridAgent(envs).to(device)
    else:
        _LOGGER.info("Using MLPAgent (MLP) for curriculum")
        agent = MLPAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Set up TensorBoard logging if configured
    writer = None
    if config.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_path = Path(config.tensorboard_dir)
            tb_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(tb_path))
            _LOGGER.info(f"TensorBoard logging to: {tb_path}")
        except ImportError:
            _LOGGER.warning("tensorboard not installed, skipping logging")

    # Set up checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else Path(f"checkpoints/{run_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Storage for rollouts
    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = config.total_timesteps // batch_size

    obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.num_steps, config.num_envs)).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)

    # Training loop
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_envs).to(device)

    episode_rewards = []
    episode_lengths = []

    _LOGGER.info(f"Starting curriculum training for {config.total_timesteps} timesteps...")
    _LOGGER.info(f"Batch size: {batch_size}, Updates: {num_updates}")

    for update in range(1, num_updates + 1):
        # Annealing learning rate
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Execute action
            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)

            # Track episode statistics
            # Gymnasium SyncVectorEnv uses "episode" key with "_episode" boolean mask
            # (the old "final_info" pattern is from pre-Gymnasium versions)
            if "episode" in infos:
                episode_mask = infos.get("_episode", np.ones(config.num_envs, dtype=bool))
                for env_idx in range(config.num_envs):
                    if episode_mask[env_idx]:
                        ep_return = float(infos["episode"]["r"][env_idx])
                        ep_length = int(infos["episode"]["l"][env_idx])
                        episode_rewards.append(ep_return)
                        episode_lengths.append(ep_length)

                        if writer:
                            writer.add_scalar("charts/episodic_return", ep_return, global_step)
                            writer.add_scalar("charts/episodic_length", ep_length, global_step)

        # Compute advantages using GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batches
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy and value networks
        b_inds = np.arange(batch_size)
        clipfracs = []
        for _ in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

        # Logging
        if update % 10 == 0:
            sps = int(global_step / (time.time() - start_time))
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            _LOGGER.info(
                f"Update {update}/{num_updates} | "
                f"Steps: {global_step}/{config.total_timesteps} | "
                f"SPS: {sps} | "
                f"Avg Reward (100 ep): {avg_reward:.2f}"
            )

            if writer:
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        # Checkpointing
        if global_step % config.checkpoint_freq < batch_size:
            checkpoint_path = checkpoint_dir / f"checkpoint_{global_step}.pt"
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "config": config.__dict__,
            }, checkpoint_path)
            _LOGGER.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "config": config.__dict__,
    }, final_path)
    _LOGGER.info(f"Training complete! Final model: {final_path}")

    envs.close()
    if writer:
        writer.close()

    return {
        "final_checkpoint": str(final_path),
        "total_steps": global_step,
        "total_episodes": len(episode_rewards),
        "final_avg_reward": np.mean(episode_rewards[-100:]) if episode_rewards else 0,
        "training_time_seconds": time.time() - start_time,
    }


def is_curriculum_config(config: CleanRLWorkerConfig) -> bool:
    """Check if config specifies curriculum training."""
    extras = config.extras or {}
    return "curriculum_schedule" in extras and extras["curriculum_schedule"]
