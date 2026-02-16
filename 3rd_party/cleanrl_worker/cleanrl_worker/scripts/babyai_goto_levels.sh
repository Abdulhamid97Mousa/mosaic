#!/bin/bash
# babyai_goto_levels.sh - Train on BabyAI GoTo Level Progression
#
# @description: Progressive training on BabyAI GoTo levels (easy to hard)
# @env_family: babyai
# @phases: 3
# @total_timesteps: 5000000
#
# This script trains on progressively harder GoTo levels from the BabyAI paper.
# Each level requires more episodes to reach 99% success rate:
#
#   Level                  Episodes to 99%    Difficulty
#   GoToRedBallNoDists     ~16k               Easy (single room, no distractors)
#   GoToRedBall            ~300k              Medium (obstacles)
#   GoToLocal              ~1M                Hard (larger maze)
#
# Note: This runs levels SEQUENTIALLY (not curriculum). For curriculum with
# weight preservation, use curriculum_babyai_goto.sh

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
export CLEANRL_NUM_ENVS="${CLEANRL_NUM_ENVS:-16}"
export TRACK_TENSORBOARD="${TRACK_TENSORBOARD:-1}"

# ============================================================================
# Level Configuration
# ============================================================================

# Which level(s) to run (1=easy, 2=medium, 3=hard, 0=all)
LEVEL="${BABYAI_LEVEL:-0}"

# BabyAI Paper hyperparameters
LEARNING_RATE="0.0001"
GAMMA="0.99"
GAE_LAMBDA="0.99"
REWARD_SCALE="20.0"
NUM_ENVS="${CLEANRL_NUM_ENVS:-16}"
NUM_STEPS="80"

# Level 1: Easy - GoToRedBallNoDists
LEVEL1_ENV="BabyAI-GoToRedBallNoDists-v0"
LEVEL1_TIMESTEPS="${BABYAI_LEVEL1_TIMESTEPS:-1000000}"

# Level 2: Medium - GoToRedBall
LEVEL2_ENV="BabyAI-GoToRedBall-v0"
LEVEL2_TIMESTEPS="${BABYAI_LEVEL2_TIMESTEPS:-2000000}"

# Level 3: Hard - GoToLocal
LEVEL3_ENV="BabyAI-GoToLocal-v0"
LEVEL3_TIMESTEPS="${BABYAI_LEVEL3_TIMESTEPS:-5000000}"

# Artifact directory -- uses var/trainer/runs/{run_id} when launched from GUI
RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"

echo "=============================================="
echo "BabyAI GoTo Level Progression"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Base Config: $CONFIG_FILE"
echo "FastLane: enabled=$GYM_GUI_FASTLANE_ONLY mode=$GYM_GUI_FASTLANE_VIDEO_MODE"
echo "Output Dir: $RUN_DIR"
echo ""
echo "Levels:"
echo "  1. $LEVEL1_ENV ($LEVEL1_TIMESTEPS steps)"
echo "  2. $LEVEL2_ENV ($LEVEL2_TIMESTEPS steps)"
echo "  3. $LEVEL3_ENV ($LEVEL3_TIMESTEPS steps)"
echo ""

# ============================================================================
# Training Function
# ============================================================================

train_level() {
    local level_num=$1
    local env_id=$2
    local timesteps=$3

    echo ""
    echo "=============================================="
    echo "Level $level_num: $env_id"
    echo "=============================================="
    echo "Timesteps: $timesteps"
    echo ""

    mkdir -p "$RUN_DIR/config"
    LEVEL_CONFIG="$RUN_DIR/config/level${level_num}_config.json"

    jq --argjson steps "$timesteps" \
       --argjson num_envs "$NUM_ENVS" \
       --argjson num_steps "$NUM_STEPS" \
       --argjson lr "$LEARNING_RATE" \
       --argjson gamma "$GAMMA" \
       --argjson gae_lambda "$GAE_LAMBDA" \
       --argjson reward_scale "$REWARD_SCALE" \
       --arg env_id "$env_id" \
       --arg checkpoint_dir "$RUN_DIR/checkpoints/level$level_num" \
       --arg tensorboard_dir "$RUN_DIR/tensorboard/level$level_num" '
        .algo = "ppo" |
        .env_id = $env_id |
        .total_timesteps = $steps |
        .extras.num_envs = $num_envs |
        .extras.num_steps = $num_steps |
        .extras.learning_rate = $lr |
        .extras.gamma = $gamma |
        .extras.gae_lambda = $gae_lambda |
        .extras.reward_scale = $reward_scale |
        .extras.anneal_lr = true |
        .extras.norm_adv = true |
        .extras.clip_vloss = true |
        .extras.tensorboard_dir = $tensorboard_dir |
        .extras.checkpoint_dir = $checkpoint_dir |
        .extras.save_model = true |
        .extras.algo_params.num_envs = $num_envs
    ' "$CONFIG_FILE" > "$LEVEL_CONFIG"

    mkdir -p "$RUN_DIR/checkpoints/level$level_num"
    mkdir -p "$RUN_DIR/tensorboard/level$level_num"

    python -m cleanrl_worker.cli --config "$LEVEL_CONFIG"

    echo ""
    echo "Level $level_num complete!"
}

# ============================================================================
# Run Selected Levels
# ============================================================================

if [ "$LEVEL" = "0" ] || [ "$LEVEL" = "1" ]; then
    train_level 1 "$LEVEL1_ENV" "$LEVEL1_TIMESTEPS"
fi

if [ "$LEVEL" = "0" ] || [ "$LEVEL" = "2" ]; then
    train_level 2 "$LEVEL2_ENV" "$LEVEL2_TIMESTEPS"
fi

if [ "$LEVEL" = "0" ] || [ "$LEVEL" = "3" ]; then
    train_level 3 "$LEVEL3_ENV" "$LEVEL3_TIMESTEPS"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "BabyAI GoTo Level Progression Complete!"
echo "=============================================="
echo "Checkpoints: $RUN_DIR/checkpoints/"
echo "TensorBoard: $RUN_DIR/tensorboard/"
echo ""
echo "Paper reference (ICLR 2019):"
echo "  Chevalier-Boisvert et al."
echo "  'BabyAI: A Platform to Study the Sample Efficiency"
echo "   of Grounded Language Learning'"
echo ""
