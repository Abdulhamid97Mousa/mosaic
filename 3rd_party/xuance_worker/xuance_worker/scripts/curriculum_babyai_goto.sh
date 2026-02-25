#!/bin/bash
# curriculum_babyai_goto.sh - BabyAI GoTo Curriculum Learning for XuanCe
#
# @description: BabyAI GoTo progression (simple → complex)
# @env_family: babyai
# @phases: 4
# @total_timesteps: 800000
#
# This script trains through BabyAI GoTo environments with increasing
# complexity - from simple object navigation to multi-room navigation.
#
# IMPORTANT: Uses Syllabus-RL for SINGLE-PROCESS curriculum training.
# The agent's weights are preserved across environment transitions!
#
# Curriculum stages:
#   1. GoToRedBallNoDists (200K) - Single object, no distractors
#   2. GoToRedBall (200K) - Single object with distractors
#   3. GoToObj (200K) - Any object type
#   4. GoToLocal (200K) - Local navigation with obstacles

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
export GYM_GUI_FASTLANE_GRID_LIMIT="${GYM_GUI_FASTLANE_GRID_LIMIT:-4}"
export XUANCE_RUN_ID="${XUANCE_RUN_ID:-$RUN_ID}"
export XUANCE_AGENT_ID="${XUANCE_AGENT_ID:-xuance-agent}"
export XUANCE_NUM_ENVS="${XUANCE_NUM_ENVS:-4}"
export XUANCE_SEED="${XUANCE_SEED:-}"
export TRACK_TENSORBOARD="${TRACK_TENSORBOARD:-1}"

# Artifact directory -- uses var/trainer/runs/{run_id} when launched from GUI
RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/logs"
mkdir -p "$RUN_DIR/models/ppo"

# Total timesteps across all curriculum stages
TOTAL_TIMESTEPS=800000

echo "=============================================="
echo "BabyAI GoTo Curriculum Learning (XuanCe)"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Base Config: $CONFIG_FILE"
echo "FastLane: enabled=$GYM_GUI_FASTLANE_ONLY mode=$GYM_GUI_FASTLANE_VIDEO_MODE"
echo "Output Dir: $RUN_DIR"
echo ""
echo "Curriculum stages (single process, weights preserved):"
echo "  1. BabyAI-GoToRedBallNoDists-v0 (200K steps)"
echo "  2. BabyAI-GoToRedBall-v0 (200K steps)"
echo "  3. BabyAI-GoToObj-v0 (200K steps)"
echo "  4. BabyAI-GoToLocal-v0 (remaining steps)"
echo ""

# ============================================================================
# Build curriculum config with schedule
# ============================================================================

mkdir -p "$RUN_DIR/config"
CURRICULUM_CONFIG="$RUN_DIR/config/curriculum_config.json"

# Create curriculum schedule and inject into config
# The CLI will detect curriculum_schedule and use Syllabus-RL training
# NOTE: Use absolute paths so checkpoints/tensorboard go to the run-specific directory
jq --argjson steps "$TOTAL_TIMESTEPS" \
   --argjson num_envs "${XUANCE_NUM_ENVS:-4}" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" \
   --arg log_dir "$RUN_DIR/logs" \
   --arg model_dir "$RUN_DIR/models/ppo" '
    .method = "ppo" |
    .env = "minigrid" |
    .env_id = "BabyAI-GoToRedBallNoDists-v0" |
    .log_dir = $log_dir |
    .model_dir = $model_dir |
    .running_steps = $steps |
    .parallels = $num_envs |
    .extras.algo_params.num_envs = $num_envs |
    .extras.num_envs = $num_envs |
    .extras.tensorboard_dir = $tensorboard_dir |
    .extras.checkpoint_dir = $checkpoint_dir |
    .extras.curriculum_schedule = [
        {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
        {"env_id": "BabyAI-GoToRedBall-v0", "steps": 200000},
        {"env_id": "BabyAI-GoToObj-v0", "steps": 200000},
        {"env_id": "BabyAI-GoToLocal-v0"}
    ]
' "$CONFIG_FILE" > "$CURRICULUM_CONFIG"

echo "Curriculum config: $CURRICULUM_CONFIG"
echo ""

# ============================================================================
# Run single-process curriculum training
# ============================================================================

echo "Starting curriculum training..."
echo "(Agent weights preserved across all environment transitions)"
echo ""

python -m xuance_worker.cli --config "$CURRICULUM_CONFIG"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "BabyAI Curriculum Training Complete!"
echo "=============================================="
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Curriculum stages: 4"
echo "Checkpoints: $RUN_DIR/checkpoints/"
echo ""
