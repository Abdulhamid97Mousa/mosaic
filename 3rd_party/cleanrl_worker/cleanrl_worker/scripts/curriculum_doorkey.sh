#!/bin/bash
# curriculum_doorkey.sh - DoorKey Curriculum Learning
#
# @description: DoorKey curriculum (5x5 → 8x8 → 16x16)
# @env_family: minigrid
# @phases: 3
# @total_timesteps: 600000
#
# This script trains a PPO agent through progressively harder DoorKey
# environments, a classic curriculum learning approach for MiniGrid.
#
# IMPORTANT: Uses Syllabus-RL for SINGLE-PROCESS curriculum training.
# The agent's weights are preserved across environment transitions!
#
# Curriculum stages:
#   1. MiniGrid-DoorKey-5x5-v0 (200K steps) - Learn basic mechanics
#   2. MiniGrid-DoorKey-8x8-v0 (200K steps) - Generalize to larger space
#   3. MiniGrid-DoorKey-16x16-v0 (remaining) - Master complex navigation

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
export CLEANRL_RUN_ID="${CLEANRL_RUN_ID:-$RUN_ID}"
export CLEANRL_AGENT_ID="${CLEANRL_AGENT_ID:-cleanrl-agent}"
export CLEANRL_NUM_ENVS="${CLEANRL_NUM_ENVS:-4}"
export CLEANRL_SEED="${CLEANRL_SEED:-}"
export TRACK_TENSORBOARD="${TRACK_TENSORBOARD:-1}"

# Create run-specific directory
RUN_DIR="$SCRIPTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/checkpoints"

# Total timesteps across all curriculum stages
TOTAL_TIMESTEPS=600000

echo "=============================================="
echo "DoorKey Curriculum Learning (Syllabus-RL)"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Base Config: $CONFIG_FILE"
echo "FastLane: enabled=$GYM_GUI_FASTLANE_ONLY mode=$GYM_GUI_FASTLANE_VIDEO_MODE"
echo "Output Dir: $RUN_DIR"
echo ""
echo "Curriculum stages (single process, weights preserved):"
echo "  1. MiniGrid-DoorKey-5x5-v0 (200K steps)"
echo "  2. MiniGrid-DoorKey-8x8-v0 (200K steps)"
echo "  3. MiniGrid-DoorKey-16x16-v0 (remaining steps)"
echo ""

# ============================================================================
# Build curriculum config with schedule
# ============================================================================

CURRICULUM_CONFIG="$RUN_DIR/curriculum_config.json"

# Create curriculum schedule and inject into config
# The CLI will detect curriculum_schedule and use Syllabus-RL training
# NOTE: Use absolute paths so checkpoints/tensorboard go to the run-specific directory
jq --argjson steps "$TOTAL_TIMESTEPS" \
   --argjson num_envs "${CLEANRL_NUM_ENVS:-4}" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" '
    .algo = "ppo" |
    .env_id = "MiniGrid-DoorKey-5x5-v0" |
    .total_timesteps = $steps |
    .extras.algo_params.num_envs = $num_envs |
    .extras.num_envs = $num_envs |
    .extras.tensorboard_dir = $tensorboard_dir |
    .extras.checkpoint_dir = $checkpoint_dir |
    .extras.curriculum_schedule = [
        {"env_id": "MiniGrid-DoorKey-5x5-v0", "steps": 200000},
        {"env_id": "MiniGrid-DoorKey-8x8-v0", "steps": 200000},
        {"env_id": "MiniGrid-DoorKey-16x16-v0"}
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

python -m cleanrl_worker.cli --config "$CURRICULUM_CONFIG"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "DoorKey Curriculum Training Complete!"
echo "=============================================="
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Curriculum stages: 3"
echo "Checkpoints: $RUN_DIR/checkpoints/"
echo ""
