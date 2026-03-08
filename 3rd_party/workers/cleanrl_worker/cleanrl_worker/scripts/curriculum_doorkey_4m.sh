#!/bin/bash
# curriculum_doorkey_4m.sh - DoorKey 4-Phase Curriculum Learning (4M steps)
#
# @description: DoorKey full curriculum (5x5 → 6x6 → 8x8 → 16x16), 1M steps each
# @env_family: minigrid
# @phases: 4
# @total_timesteps: 4000000
#
# Progressive curriculum through ALL four DoorKey grid sizes.
# Each phase trains for 1 million timesteps with a 100-step episode cap.
#
# IMPORTANT: Uses Syllabus-RL for SINGLE-PROCESS curriculum training.
# The agent's weights are preserved across environment transitions!
#
# Curriculum stages:
#   1. MiniGrid-DoorKey-5x5-v0   (1M steps) - Learn key/door mechanics
#   2. MiniGrid-DoorKey-6x6-v0   (1M steps) - Slightly larger grid
#   3. MiniGrid-DoorKey-8x8-v0   (1M steps) - Generalize navigation
#   4. MiniGrid-DoorKey-16x16-v0 (1M steps) - Master complex navigation

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

# Artifact directory -- uses var/trainer/runs/{run_id} when launched from GUI
RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"
mkdir -p "$RUN_DIR/checkpoints"

# Curriculum parameters
TOTAL_TIMESTEPS=4000000
STEPS_PER_PHASE=1000000
MAX_EPISODE_STEPS=100

echo "=============================================="
echo "DoorKey 4-Phase Curriculum Learning (4M steps)"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Base Config: $CONFIG_FILE"
echo "FastLane: enabled=$GYM_GUI_FASTLANE_ONLY mode=$GYM_GUI_FASTLANE_VIDEO_MODE"
echo "Output Dir: $RUN_DIR"
echo "Max episode steps: $MAX_EPISODE_STEPS"
echo ""
echo "Curriculum stages (single process, weights preserved):"
echo "  1. MiniGrid-DoorKey-5x5-v0   (1M steps) - Learn key/door mechanics"
echo "  2. MiniGrid-DoorKey-6x6-v0   (1M steps) - Slightly larger grid"
echo "  3. MiniGrid-DoorKey-8x8-v0   (1M steps) - Generalize navigation"
echo "  4. MiniGrid-DoorKey-16x16-v0 (1M steps) - Master complex navigation"
echo ""

# ============================================================================
# Build curriculum config with schedule
# ============================================================================

CURRICULUM_CONFIG="$RUN_DIR/config/curriculum_config.json"
mkdir -p "$RUN_DIR/config"

# Create curriculum schedule and inject into config
# The CLI will detect curriculum_schedule and use Syllabus-RL training
# NOTE: Use absolute paths so checkpoints/tensorboard go to the run-specific directory
jq --argjson steps "$TOTAL_TIMESTEPS" \
   --argjson num_envs "${CLEANRL_NUM_ENVS:-4}" \
   --argjson max_ep_steps "$MAX_EPISODE_STEPS" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" '
    .algo = "ppo" |
    .env_id = "MiniGrid-DoorKey-5x5-v0" |
    .total_timesteps = $steps |
    .extras.algo_params.num_envs = $num_envs |
    .extras.algo_params.max_episode_steps = $max_ep_steps |
    .extras.num_envs = $num_envs |
    .extras.max_episode_steps = $max_ep_steps |
    .extras.tensorboard_dir = $tensorboard_dir |
    .extras.checkpoint_dir = $checkpoint_dir |
    .extras.curriculum_schedule = [
        {"env_id": "MiniGrid-DoorKey-5x5-v0",   "steps": 1000000},
        {"env_id": "MiniGrid-DoorKey-6x6-v0",   "steps": 1000000},
        {"env_id": "MiniGrid-DoorKey-8x8-v0",   "steps": 1000000},
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
echo "DoorKey 4-Phase Curriculum Training Complete!"
echo "=============================================="
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Steps per phase: $STEPS_PER_PHASE"
echo "Max episode steps: $MAX_EPISODE_STEPS"
echo "Curriculum stages: 4"
echo "Checkpoints: $RUN_DIR/checkpoints/"
echo ""
