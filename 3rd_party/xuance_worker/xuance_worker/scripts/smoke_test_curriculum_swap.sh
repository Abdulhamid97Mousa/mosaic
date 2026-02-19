#!/bin/bash
# smoke_test_curriculum_swap.sh
#
# @description: Smoke test: MAPPO curriculum collect_1vs1 (50k) -> soccer_1vs1 (50k)
# @env_family: multigrid
# @environments: collect_1vs1, soccer_1vs1
# @method: MAPPO
# @phases: 2
# @total_timesteps: 100000
#
# Quick verification that multi-agent curriculum environment swap works.
# Uses small step counts (50k per phase) so it completes in ~3 minutes.
#
# Verifies:
#   - Phase 1 trains on collect_1vs1
#   - Environment swap to soccer_1vs1 succeeds
#   - Phase 2 trains on soccer_1vs1
#   - Checkpoints saved for both phases
#
# Usage:
#   # Launch via GUI (XuanCe Script Form) or manually:
#   export MOSAIC_CONFIG_FILE="/path/to/config.json"
#   export MOSAIC_RUN_ID="smoke_test_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash smoke_test_curriculum_swap.sh

set -e  # Exit on error

# ============================================================================
# Configuration from MOSAIC
# ============================================================================

CONFIG_FILE="${MOSAIC_CONFIG_FILE:?MOSAIC_CONFIG_FILE not set}"
RUN_ID="${MOSAIC_RUN_ID:?MOSAIC_RUN_ID not set}"
SCRIPTS_DIR="${MOSAIC_CUSTOM_SCRIPTS_DIR:?MOSAIC_CUSTOM_SCRIPTS_DIR not set}"

# ============================================================================
# Training Parameters -- small steps for smoke testing
# ============================================================================

PHASE1_STEPS="${PHASE1_STEPS:-50000}"
PHASE2_STEPS="${PHASE2_STEPS:-50000}"
NUM_ENVS="${XUANCE_NUM_ENVS:-4}"
SEED="${XUANCE_SEED:-}"
TRAINING_MODE="competitive"

# ============================================================================
# Export FastLane environment variables for child Python processes
# ============================================================================
export GYM_GUI_FASTLANE_ONLY="${GYM_GUI_FASTLANE_ONLY:-1}"
export GYM_GUI_FASTLANE_SLOT="${GYM_GUI_FASTLANE_SLOT:-0}"
export GYM_GUI_FASTLANE_VIDEO_MODE="${GYM_GUI_FASTLANE_VIDEO_MODE:-grid}"
export GYM_GUI_FASTLANE_GRID_LIMIT="${GYM_GUI_FASTLANE_GRID_LIMIT:-4}"
export XUANCE_RUN_ID="${XUANCE_RUN_ID:-$RUN_ID}"
export XUANCE_AGENT_ID="${XUANCE_AGENT_ID:-xuance-agent}"
export XUANCE_NUM_ENVS="$NUM_ENVS"
export XUANCE_PARALLELS="$NUM_ENVS"
export XUANCE_SEED="$SEED"
export TRACK_TENSORBOARD="${TRACK_TENSORBOARD:-1}"

# ============================================================================
# Directory structure (single unified output)
# ============================================================================

RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"

mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"
mkdir -p "$RUN_DIR/config"

# ============================================================================
# Display configuration
# ============================================================================

echo "============================================================"
echo "  Smoke Test: MAPPO Curriculum Swap"
echo "  collect_1vs1 (50k) -> soccer_1vs1 (50k)"
echo "============================================================"
echo ""
echo "Run Configuration:"
echo "  Run ID:           $RUN_ID"
echo "  Base Config:      $CONFIG_FILE"
echo "  Output Directory: $RUN_DIR"
echo ""
echo "Phase 1 - Ball Collection (collect_1vs1):"
echo "  Environment:      CollectGame2HIndAgObsEnv10x10N2"
echo "  Grid:             10x10, 3 balls, zero_sum=True"
echo "  Steps:            $PHASE1_STEPS"
echo ""
echo "Phase 2 - Soccer Transfer (soccer_1vs1):"
echo "  Environment:      SoccerGame2HIndAgObsEnv16x11N2"
echo "  Grid:             16x11, first-to-2 goals"
echo "  Steps:            $PHASE2_STEPS"
echo "  Transfer:         In-memory (optimizer + LR preserved)"
echo ""
echo "Shared Architecture:"
echo "  Agents:           2 (1v1 competitive, parameter sharing)"
echo "  Observations:     IndAgObs (3,3,3) = 27 flat"
echo "  Actions:          7 discrete"
echo "  Parallel Envs:    $NUM_ENVS"
echo ""
echo "============================================================"

# ============================================================================
# Build unified curriculum config with curriculum_schedule
# ============================================================================

CURRICULUM_CONFIG="$RUN_DIR/config/curriculum_config.json"

jq --argjson phase1_steps "$PHASE1_STEPS" \
   --argjson phase2_steps "$PHASE2_STEPS" \
   --argjson num_envs "$NUM_ENVS" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" \
   --arg training_mode "$TRAINING_MODE" \
   --arg seed "${SEED:-null}" '
    .method = "mappo" |
    .env = "multigrid" |
    .env_id = "collect_1vs1" |
    .running_steps = ($phase1_steps + $phase2_steps) |
    .parallels = $num_envs |
    .extras.training_mode = $training_mode |
    .extras.algo_params.num_envs = $num_envs |
    .extras.num_envs = $num_envs |
    .extras.tensorboard_dir = $tensorboard_dir |
    .extras.checkpoint_dir = $checkpoint_dir |
    .extras.curriculum_schedule = [
        {"env_id": "collect_1vs1", "steps": $phase1_steps},
        {"env_id": "soccer_1vs1", "steps": $phase2_steps}
    ] |
    if $seed != "null" and $seed != "" then
        .extras.seed = ($seed | tonumber)
    else
        .
    end
' "$CONFIG_FILE" > "$CURRICULUM_CONFIG"

echo ""
echo "Curriculum config: $CURRICULUM_CONFIG"
echo "Starting single-process curriculum training..."
echo ""

# ============================================================================
# Single-process training (environment swap happens in Python)
# ============================================================================

export MOSAIC_RUN_DIR="$RUN_DIR"
python -m xuance_worker.cli --config "$CURRICULUM_CONFIG"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo "  Smoke Test Complete!"
echo "============================================================"
echo ""
echo "Phase 1 (collect_1vs1): $PHASE1_STEPS steps"
echo "Phase 2 (soccer_1vs1): $PHASE2_STEPS steps"
echo "Total:                  $((PHASE1_STEPS + PHASE2_STEPS)) steps"
echo ""
echo "Checkpoints: $RUN_DIR/checkpoints/"
echo "TensorBoard: $RUN_DIR/tensorboard/"
echo ""
echo "View training:"
echo "  tensorboard --logdir $RUN_DIR/tensorboard"
echo ""
echo "============================================================"
