#!/bin/bash
# mappo_curriculum_collect_soccer_1v1_indagobs.sh
#
# @description: MAPPO curriculum: collect_1vs1 (1M) -> soccer_1vs1 (4M)
# @env_family: multigrid
# @environments: collect_1vs1, soccer_1vs1
# @method: MAPPO
# @phases: 2
# @total_timesteps: 5000000
#
# Single-Process 2-Phase Curriculum Transfer Learning:
#   Phase 1: Train MAPPO on collect_1vs1 (ball collection) for 1M steps
#            - Simpler task: collect 3 balls in 10x10 grid (+1 per ball)
#            - Agents learn: movement, spatial awareness, object interaction
#            - Natural termination when all balls collected (denser reward)
#   Phase 2: Swap environment to soccer_1vs1 and continue training for 1M steps
#            - Harder task: score goals in 16x11 FIFA-ratio grid
#            - Policy weights, optimizer state, LR schedule ALL preserved
#            - Agents only need to learn: carry to goal zone + drop
#
# Both environments share identical obs/action spaces:
#   - 2 agents (1v1), IndAgObs (3,3,3)=27 obs, 7 discrete actions
#   - Same network architecture: [64,64] MLP
#   - Environment swap preserves everything in memory (no checkpoint transfer)
#
# Architecture note:
#   Uses multi_agent_curriculum_training.py which creates ONE RunnerMARL,
#   trains Phase 1, then hot-swaps environments for Phase 2. Unlike the
#   2-process approach, this preserves Adam optimizer momentum, LR schedule
#   position, and produces a single continuous TensorBoard timeline.
#
# Usage:
#   # Launch via GUI (XuanCe Script Form) or manually:
#   export MOSAIC_CONFIG_FILE="/path/to/config.json"
#   export MOSAIC_RUN_ID="curriculum_collect_soccer_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash mappo_curriculum_collect_soccer_1v1_indagobs.sh

set -e  # Exit on error

# ============================================================================
# Configuration from MOSAIC
# ============================================================================

CONFIG_FILE="${MOSAIC_CONFIG_FILE:?MOSAIC_CONFIG_FILE not set}"
RUN_ID="${MOSAIC_RUN_ID:?MOSAIC_RUN_ID not set}"
SCRIPTS_DIR="${MOSAIC_CUSTOM_SCRIPTS_DIR:?MOSAIC_CUSTOM_SCRIPTS_DIR not set}"

# ============================================================================
# Training Parameters (configurable via environment variables)
# ============================================================================

PHASE1_STEPS="${PHASE1_STEPS:-1000000}"
PHASE2_STEPS="${PHASE2_STEPS:-4000000}"
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
echo "  MAPPO Curriculum: collect_1vs1 -> soccer_1vs1"
echo "  (Single Process - Zero Interruption)"
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
echo "  Policy:           1 shared policy (symmetric game, no collapse)"
echo "  Observations:     IndAgObs (3,3,3) = 27 flat"
echo "  Actions:          7 discrete"
echo "  Network:          MLP [64,64]"
echo "  Parallel Envs:    $NUM_ENVS"
echo "  Seed:             ${SEED:-random}"
echo ""
echo "FastLane:"
echo "  Enabled:          $GYM_GUI_FASTLANE_ONLY"
echo "  Video Mode:       $GYM_GUI_FASTLANE_VIDEO_MODE"
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
echo "  Curriculum Training Complete!"
echo "============================================================"
echo ""
echo "Phase 1 (collect_1vs1 - ball collection): $PHASE1_STEPS steps"
echo "Phase 2 (soccer_1vs1 - goal scoring):     $PHASE2_STEPS steps"
echo "Total:                                     $((PHASE1_STEPS + PHASE2_STEPS)) steps"
echo ""
echo "Checkpoints: $RUN_DIR/checkpoints/"
echo "TensorBoard: $RUN_DIR/tensorboard/"
echo ""
echo "View training:"
echo "  tensorboard --logdir $RUN_DIR/tensorboard"
echo ""
echo "============================================================"
