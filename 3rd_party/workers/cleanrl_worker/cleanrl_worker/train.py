"""
Unified RL Training Entry Point with Paper Baseline Presets.

This module provides a single entry point for launching RL training with
pre-configured hyperparameters from published papers (BabyAI ICLR 2019, etc.).

This replaces the need for algorithm-specific files like "ppo_with_save.py",
making the infrastructure algorithm-agnostic.

Usage:
    # Train with BabyAI paper hyperparameters
    python -m cleanrl_worker.train --preset babyai-paper \\
        --env-id BabyAI-GoToRedBallNoDists-v0 --total-timesteps 1000000

    # Train with default hyperparameters
    python -m cleanrl_worker.train --env-id CartPole-v1 --total-timesteps 100000

    # Override specific hyperparameters
    python -m cleanrl_worker.train --preset babyai-paper \\
        --env-id BabyAI-GoToLocal-v0 --learning-rate 0.0002

    # List available presets
    python -m cleanrl_worker.train --list-presets
"""

import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import tyro

logger = logging.getLogger(__name__)


# =============================================================================
# Paper Hyperparameter Presets
# =============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "babyai-paper": {
        # BabyAI ICLR 2019 Paper - Table 4 hyperparameters
        # "BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning"
        # Chevalier-Boisvert et al., ICLR 2019
        "description": "BabyAI ICLR 2019 paper PPO (gae_lambda=0.99, reward_scale=20)",
        "algo": "ppo",
        "num_envs": 64,           # paper: 64 parallel processes
        "num_steps": 40,          # paper: frames_per_proc = 40
        "learning_rate": 0.0001,  # paper: 1e-4
        "gamma": 0.99,            # paper: discount = 0.99
        "gae_lambda": 0.99,       # paper: 0.99 (KEY: higher than typical 0.95!)
        "update_epochs": 4,       # paper: 4 PPO epochs
        "clip_coef": 0.2,         # paper: 0.2
        "ent_coef": 0.01,         # paper: entropy_coef = 0.01
        "vf_coef": 0.5,           # paper: value_loss_coef = 0.5
        "max_grad_norm": 0.5,     # paper: 0.5
        "reward_scale": 20.0,     # paper: reward_scale = 20
        "max_episode_steps": 256, # reasonable episode length
        "anneal_lr": True,
        "norm_adv": True,
        "clip_vloss": True,
    },
    "babyai-fast": {
        # Faster training variant (fewer envs for single machine)
        "description": "BabyAI paper hyperparameters adapted for single machine (16 envs)",
        "algo": "ppo",
        "num_envs": 16,           # reduced for single machine
        "num_steps": 80,          # increased to compensate
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "gae_lambda": 0.99,
        "update_epochs": 4,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "reward_scale": 20.0,
        "max_episode_steps": 256,
        "anneal_lr": True,
        "norm_adv": True,
        "clip_vloss": True,
    },
    "cleanrl-default": {
        # CleanRL default PPO hyperparameters
        "description": "CleanRL default PPO hyperparameters",
        "algo": "ppo",
        "num_envs": 4,
        "num_steps": 128,
        "learning_rate": 0.00025,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "update_epochs": 4,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "anneal_lr": True,
        "norm_adv": True,
        "clip_vloss": True,
    },
}

# BabyAI Paper Episode Requirements (from Table 2)
# Episodes needed to reach 99% success rate with RL
BABYAI_EPISODE_REQUIREMENTS = {
    "BabyAI-GoToRedBallGrey-v0": (15_900, 17_400),
    "BabyAI-GoToRedBall-v0": (261_100, 333_600),
    "BabyAI-GoToRedBallNoDists-v0": (15_900, 17_400),  # Similar to GoToRedBallGrey
    "BabyAI-GoToLocal-v0": (903_000, 1_114_000),
    "BabyAI-PickupLoc-v0": (1_447_000, 1_643_000),
    "BabyAI-PutNextLocal-v0": (2_186_000, 2_727_000),
    "BabyAI-GoTo-v0": (816_000, 1_964_000),
}


# =============================================================================
# Algorithm Registry
# =============================================================================

ALGO_REGISTRY = {
    "ppo": "cleanrl_worker.algorithms.ppo",
}


@dataclass
class TrainArgs:
    """Training arguments with preset support."""

    # Preset selection
    preset: Optional[str] = None
    """Use a preset configuration (babyai-paper, babyai-fast, cleanrl-default)"""

    list_presets: bool = False
    """List available presets and exit"""

    # Algorithm
    algo: str = "ppo"
    """RL algorithm to use"""

    # Environment
    env_id: str = "CartPole-v1"
    """Environment ID"""

    total_timesteps: int = 500000
    """Total training timesteps"""

    seed: int = 1
    """Random seed"""

    # Hyperparameters (override preset values)
    num_envs: Optional[int] = None
    num_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    gamma: Optional[float] = None
    gae_lambda: Optional[float] = None
    update_epochs: Optional[int] = None
    clip_coef: Optional[float] = None
    ent_coef: Optional[float] = None
    vf_coef: Optional[float] = None
    max_grad_norm: Optional[float] = None
    reward_scale: Optional[float] = None
    max_episode_steps: Optional[int] = None
    anneal_lr: Optional[bool] = None
    norm_adv: Optional[bool] = None
    clip_vloss: Optional[bool] = None

    # Training options
    save_model: bool = False
    capture_video: bool = False
    track: bool = False

    # Procedural generation
    procedural_generation: bool = True
    """Generate new random layouts each episode (standard RL training)"""


def list_presets_and_exit():
    """Print available presets and exit."""
    print("\n" + "=" * 70)
    print("Available Presets")
    print("=" * 70)
    for name, config in PRESETS.items():
        print(f"\n  --preset {name}")
        print(f"    {config['description']}")
        print(f"    Key hyperparameters:")
        for key in ["num_envs", "num_steps", "learning_rate", "gae_lambda", "reward_scale"]:
            if key in config:
                print(f"      {key}: {config[key]}")
    print("\n" + "=" * 70)
    print("BabyAI Paper Baseline Requirements (episodes to 99% success)")
    print("=" * 70)
    for env_id, (low, high) in BABYAI_EPISODE_REQUIREMENTS.items():
        print(f"  {env_id}: {low:,} - {high:,} episodes")
    print()
    sys.exit(0)


def build_command(args: TrainArgs) -> List[str]:
    """Build the training command with preset and override logic."""
    algo = args.algo
    if args.preset and "algo" in PRESETS.get(args.preset, {}):
        algo = PRESETS[args.preset]["algo"]

    if algo not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algo}. Available: {list(ALGO_REGISTRY.keys())}")

    module = ALGO_REGISTRY[algo]
    cmd = [sys.executable, "-m", module]

    # Get preset defaults
    preset_config = {}
    if args.preset:
        if args.preset not in PRESETS:
            raise ValueError(f"Unknown preset: {args.preset}. Use --list-presets to see options.")
        preset_config = PRESETS[args.preset].copy()
        preset_config.pop("description", None)
        preset_config.pop("algo", None)
        logger.info(f"Using preset: {args.preset}")

    # Required args
    cmd.extend(["--env-id", args.env_id])
    cmd.extend(["--total-timesteps", str(args.total_timesteps)])
    cmd.extend(["--seed", str(args.seed)])

    # Merge preset with explicit overrides
    hyperparams = [
        "num_envs", "num_steps", "learning_rate", "gamma", "gae_lambda",
        "update_epochs", "clip_coef", "ent_coef", "vf_coef", "max_grad_norm",
        "reward_scale", "max_episode_steps",
    ]

    for param in hyperparams:
        explicit = getattr(args, param)
        if explicit is not None:
            value = explicit
        elif param in preset_config:
            value = preset_config[param]
        else:
            continue
        cmd.extend([f"--{param.replace('_', '-')}", str(value)])

    # Boolean flags
    bool_params = ["anneal_lr", "norm_adv", "clip_vloss"]
    for param in bool_params:
        explicit = getattr(args, param)
        if explicit is not None:
            value = explicit
        elif param in preset_config:
            value = preset_config[param]
        else:
            continue
        if value:
            cmd.append(f"--{param.replace('_', '-')}")
        else:
            cmd.append(f"--no-{param.replace('_', '-')}")

    if args.save_model:
        cmd.append("--save-model")
    if args.capture_video:
        cmd.append("--capture-video")
    if args.track:
        cmd.append("--track")

    if not args.procedural_generation:
        cmd.append("--no-procedural-generation")

    return cmd


def estimate_training_time(args: TrainArgs):
    """Estimate training requirements for BabyAI environments."""
    if "BabyAI" in args.env_id and args.env_id in BABYAI_EPISODE_REQUIREMENTS:
        low, high = BABYAI_EPISODE_REQUIREMENTS[args.env_id]
        avg_episodes = (low + high) // 2
        steps_per_episode = 64  # rough average for successful episodes
        estimated_timesteps = avg_episodes * steps_per_episode

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"BabyAI Paper Baseline: {args.env_id}")
        logger.info(f"Episodes to 99%: {low:,} - {high:,}")
        logger.info(f"Estimated timesteps needed: ~{estimated_timesteps:,}")
        logger.info(f"Your training: {args.total_timesteps:,} timesteps")
        if args.total_timesteps < estimated_timesteps:
            logger.warning(f"⚠️  May need more timesteps to reach 99% success")
        logger.info("=" * 60)
        logger.info("")


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = tyro.cli(TrainArgs)

    if args.list_presets:
        list_presets_and_exit()

    # Show estimate for BabyAI
    estimate_training_time(args)

    try:
        cmd = build_command(args)
        logger.info(f"Command: {' '.join(cmd[:5])}...")
        logger.info("")

        result = subprocess.run(cmd, check=True)
        return result.returncode

    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with return code {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.exception(f"Launcher failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
