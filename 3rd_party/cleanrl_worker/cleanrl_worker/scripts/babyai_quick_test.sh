#!/bin/bash
# babyai_quick_test.sh - Quick test of BabyAI training (5 minutes)
#
# @description: Quick 5-minute test to verify BabyAI training works
# @env_family: babyai
# @phases: 1
# @total_timesteps: 50000
#
# This script runs a quick training session to verify everything works.
# Good for testing before running the full baseline reproduction.

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
# Quick Test Configuration
# ============================================================================

ENV_ID="${BABYAI_ENV_ID:-BabyAI-GoToRedBallNoDists-v0}"
TOTAL_TIMESTEPS="${BABYAI_TIMESTEPS:-50000}"
NUM_ENVS="${CLEANRL_NUM_ENVS:-8}"

# Paper hyperparameters (for quick validation)
LEARNING_RATE="0.0001"
GAE_LAMBDA="0.99"
REWARD_SCALE="20.0"

# Create run-specific directory
RUN_DIR="$SCRIPTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"

echo "=============================================="
echo "BabyAI Quick Test (~5 minutes)"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Environment: $ENV_ID"
echo "Timesteps: $TOTAL_TIMESTEPS"
echo "FastLane: enabled=$GYM_GUI_FASTLANE_ONLY mode=$GYM_GUI_FASTLANE_VIDEO_MODE"
echo ""
echo "This is a quick test to verify training works."
echo "For full baseline reproduction, use babyai_paper_baseline.sh"
echo ""

# ============================================================================
# Build training config
# ============================================================================

TRAINING_CONFIG="$RUN_DIR/quick_test_config.json"

jq --argjson steps "$TOTAL_TIMESTEPS" \
   --argjson num_envs "$NUM_ENVS" \
   --argjson lr "$LEARNING_RATE" \
   --argjson gae_lambda "$GAE_LAMBDA" \
   --argjson reward_scale "$REWARD_SCALE" \
   --arg env_id "$ENV_ID" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" '
    .algo = "ppo" |
    .env_id = $env_id |
    .total_timesteps = $steps |
    .extras.num_envs = $num_envs |
    .extras.learning_rate = $lr |
    .extras.gae_lambda = $gae_lambda |
    .extras.reward_scale = $reward_scale |
    .extras.anneal_lr = true |
    .extras.norm_adv = true |
    .extras.tensorboard_dir = $tensorboard_dir |
    .extras.checkpoint_dir = $checkpoint_dir |
    .extras.algo_params.num_envs = $num_envs
' "$CONFIG_FILE" > "$TRAINING_CONFIG"

echo "Training config: $TRAINING_CONFIG"
echo ""

# ============================================================================
# Run Quick Test
# ============================================================================

echo "Starting quick test..."
echo ""

python -m cleanrl_worker.cli --config "$TRAINING_CONFIG"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "Quick Test Complete!"
echo "=============================================="
echo ""
echo "If you see episodic_return > 0, training is working correctly."
echo "For 99% success rate, run the full baseline with ~1M timesteps:"
echo "  ./babyai_paper_baseline.sh"
echo ""
