#!/bin/bash
# babyai_paper_baseline.sh - Reproduce BabyAI ICLR 2019 Paper Results
#
# @description: BabyAI paper baseline PPO (ICLR 2019 hyperparameters)
# @env_family: babyai
# @phases: 1
# @total_timesteps: 1000000
#
# This script reproduces the PPO training setup from:
# "BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning"
# Chevalier-Boisvert et al., ICLR 2019
#
# Original paper hyperparameters:
#   - Algorithm: PPO
#   - Parallel processes: 64
#   - Frames per process: 40
#   - Batch size: 1280 (64 * 40 / 2)
#   - Learning rate: 1e-4
#   - Discount (gamma): 0.99
#   - GAE lambda: 0.99 (higher than typical 0.95)
#   - PPO epochs: 4
#   - Entropy coefficient: 0.01
#   - Clip coefficient: 0.2
#   - Reward scale: 20.0
#
# Note: We adapt some parameters for single-machine training while
# preserving the key hyperparameters that affect learning dynamics.

set -e  # Exit on error

# ============================================================================
# Configuration from MOSAIC
# ============================================================================

CONFIG_FILE="${MOSAIC_CONFIG_FILE:?MOSAIC_CONFIG_FILE not set}"
RUN_ID="${MOSAIC_RUN_ID:?MOSAIC_RUN_ID not set}"
SCRIPTS_DIR="${MOSAIC_CUSTOM_SCRIPTS_DIR:?MOSAIC_CUSTOM_SCRIPTS_DIR not set}"

# ============================================================================
# Export FastLane environment variables for child Python processes
# ============================================================================
export GYM_GUI_FASTLANE_ONLY="${GYM_GUI_FASTLANE_ONLY:-1}"
export GYM_GUI_FASTLANE_SLOT="${GYM_GUI_FASTLANE_SLOT:-0}"
export GYM_GUI_FASTLANE_VIDEO_MODE="${GYM_GUI_FASTLANE_VIDEO_MODE:-grid}"
export GYM_GUI_FASTLANE_GRID_LIMIT="${GYM_GUI_FASTLANE_GRID_LIMIT:-8}"
export CLEANRL_RUN_ID="${CLEANRL_RUN_ID:-$RUN_ID}"
export CLEANRL_AGENT_ID="${CLEANRL_AGENT_ID:-cleanrl-agent}"
export TRACK_TENSORBOARD="${TRACK_TENSORBOARD:-1}"

# ============================================================================
# BabyAI Paper Hyperparameters (ICLR 2019)
# ============================================================================

# Environment (can override via env var before running script)
ENV_ID="${BABYAI_ENV_ID:-BabyAI-GoToRedBallNoDists-v0}"

# Training scale (adapted for single machine)
# Original paper: 64 processes, we use fewer but compensate with more steps
NUM_ENVS="${BABYAI_NUM_ENVS:-16}"
NUM_STEPS="${BABYAI_NUM_STEPS:-80}"  # frames_per_proc * 2 to match batch dynamics

# Total timesteps (paper ran for millions of frames)
TOTAL_TIMESTEPS="${BABYAI_TOTAL_TIMESTEPS:-1000000}"

# Learning parameters from paper
LEARNING_RATE="0.0001"      # Paper: 1e-4
GAMMA="0.99"                # Paper: 0.99
GAE_LAMBDA="0.99"           # Paper: 0.99 (key difference from typical 0.95!)
UPDATE_EPOCHS="4"           # Paper: 4 PPO epochs
CLIP_COEF="0.2"             # Paper: 0.2
ENT_COEF="0.01"             # Paper: 0.01
VF_COEF="0.5"               # Paper: 0.5
MAX_GRAD_NORM="0.5"         # Paper: 0.5

# MiniGrid-specific
MAX_EPISODE_STEPS="256"     # Reasonable episode length
REWARD_SCALE="20.0"         # Paper: 20.0 reward scaling

# Compute batch sizes
# Paper: batch_size = 1280 = 64 procs * 40 frames / 2
# We use: batch_size = NUM_ENVS * NUM_STEPS
NUM_MINIBATCHES="4"

# Artifact directory -- uses var/trainer/runs/{run_id} when launched from GUI
RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"

echo "=============================================="
echo "BabyAI Paper Baseline (ICLR 2019)"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Environment: $ENV_ID"
echo "Base Config: $CONFIG_FILE"
echo "FastLane: enabled=$GYM_GUI_FASTLANE_ONLY mode=$GYM_GUI_FASTLANE_VIDEO_MODE"
echo "Output Dir: $RUN_DIR"
echo ""
echo "Paper hyperparameters:"
echo "  num_envs: $NUM_ENVS (paper: 64)"
echo "  num_steps: $NUM_STEPS (paper: 40)"
echo "  learning_rate: $LEARNING_RATE"
echo "  gamma: $GAMMA"
echo "  gae_lambda: $GAE_LAMBDA (key: higher than typical 0.95)"
echo "  ppo_epochs: $UPDATE_EPOCHS"
echo "  reward_scale: $REWARD_SCALE"
echo "  total_timesteps: $TOTAL_TIMESTEPS"
echo ""

# ============================================================================
# Build training config
# ============================================================================

mkdir -p "$RUN_DIR/config"
TRAINING_CONFIG="$RUN_DIR/config/training_config.json"

jq --argjson steps "$TOTAL_TIMESTEPS" \
   --argjson num_envs "$NUM_ENVS" \
   --argjson num_steps "$NUM_STEPS" \
   --argjson num_minibatches "$NUM_MINIBATCHES" \
   --argjson lr "$LEARNING_RATE" \
   --argjson gamma "$GAMMA" \
   --argjson gae_lambda "$GAE_LAMBDA" \
   --argjson update_epochs "$UPDATE_EPOCHS" \
   --argjson clip_coef "$CLIP_COEF" \
   --argjson ent_coef "$ENT_COEF" \
   --argjson vf_coef "$VF_COEF" \
   --argjson max_grad_norm "$MAX_GRAD_NORM" \
   --argjson max_episode_steps "$MAX_EPISODE_STEPS" \
   --argjson reward_scale "$REWARD_SCALE" \
   --arg env_id "$ENV_ID" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" '
    .algo = "ppo" |
    .env_id = $env_id |
    .total_timesteps = $steps |
    .extras.num_envs = $num_envs |
    .extras.num_steps = $num_steps |
    .extras.num_minibatches = $num_minibatches |
    .extras.learning_rate = $lr |
    .extras.gamma = $gamma |
    .extras.gae_lambda = $gae_lambda |
    .extras.update_epochs = $update_epochs |
    .extras.clip_coef = $clip_coef |
    .extras.ent_coef = $ent_coef |
    .extras.vf_coef = $vf_coef |
    .extras.max_grad_norm = $max_grad_norm |
    .extras.max_episode_steps = $max_episode_steps |
    .extras.reward_scale = $reward_scale |
    .extras.anneal_lr = true |
    .extras.norm_adv = true |
    .extras.clip_vloss = true |
    .extras.tensorboard_dir = $tensorboard_dir |
    .extras.checkpoint_dir = $checkpoint_dir |
    .extras.save_model = true |
    .extras.algo_params.num_envs = $num_envs
' "$CONFIG_FILE" > "$TRAINING_CONFIG"

echo "Training config: $TRAINING_CONFIG"
echo ""

# ============================================================================
# Run PPO training
# ============================================================================

echo "Starting BabyAI Paper Baseline training..."
echo "(Using PPO with ICLR 2019 hyperparameters)"
echo ""

python -m cleanrl_worker.cli --config "$TRAINING_CONFIG"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "BabyAI Paper Baseline Training Complete!"
echo "=============================================="
echo "Environment: $ENV_ID"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Checkpoints: $RUN_DIR/checkpoints/"
echo "TensorBoard: $RUN_DIR/tensorboard/"
echo ""
echo "Key paper hyperparameters used:"
echo "  - Learning rate: $LEARNING_RATE (paper: 1e-4)"
echo "  - GAE lambda: $GAE_LAMBDA (paper: 0.99)"
echo "  - Reward scale: $REWARD_SCALE (paper: 20.0)"
echo ""
