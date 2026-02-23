#!/bin/bash
# ippo_curriculum_collect_soccer_1v1_indagobs.sh
#
# @description: IPPO curriculum: collect_1vs1 (1M) -> soccer_1vs1 (4M)
# @env_family: multigrid
# @environments: collect_1vs1, soccer_1vs1
# @method: IPPO
# @phases: 2
# @total_timesteps: 5000000
#
# Single-Process 2-Phase Curriculum Transfer Learning (IPPO):
#   Phase 1: Train IPPO on collect_1vs1 (ball collection) for 1M steps
#            - Simpler task: collect 3 balls in 10x10 grid (+1 per ball)
#            - Dense rewards teach: movement, spatial awareness, ball pickup
#            - Both agents train independent networks from the start
#
#   Phase 2: Swap environment to soccer_1vs1 and continue for 4M steps
#            - Harder task: carry ball to correct goal zone, first to 2 goals
#            - Both independent policy networks (pi_agent_0, pi_agent_1) survive
#              the environment swap intact -- weights, optimizer, LR schedule
#            - Agents only need to learn: goal direction + drop at goal zone
#
# Why IPPO (not MAPPO) for Paper 2:
#   IPPO trains two SEPARATE networks with no one-hot agent index:
#     pi_agent_0: input_dim = 27 (obs only, no team encoding)
#     pi_agent_1: input_dim = 27 (obs only, no team encoding)
#   This means each policy is a standalone 27-dim model.
#   Deploying pi_agent_0 in MOSAIC with agent_0: just load checkpoint, pass obs -> action.
#   No n_agents dependency. No one-hot reconstruction needed.
#
# IMPORTANT -- Policy is Team-Dependent (read before deploying):
#   pi_agent_0 was trained as Green (specific goal direction).
#   pi_agent_1 was trained as Blue (opposite goal direction).
#   Do NOT swap them. Deploy pi_agent_0 as agent_0, pi_agent_1 as agent_1.
#   See: docs/Development_Progress/1.0_DAY_67/TASK_1/Training_MAPPO_IPPO_Analysis.md
#
# Shared architecture (MUST be identical across both phases for hot-swap):
#   - 2 agents (1v1), IndAgObs (3,3,3) = 27 obs, 7 discrete actions
#   - Network: MLP [64,64] (representation + actor + critic)
#   - use_parameter_sharing: False
#
# Usage:
#   export MOSAIC_CONFIG_FILE="/path/to/config.json"
#   export MOSAIC_RUN_ID="ippo_curriculum_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash ippo_curriculum_collect_soccer_1v1_indagobs.sh

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

PHASE1_STEPS="${PHASE1_STEPS:-1000000}"   # collect_1vs1: dense reward, teaches pickup
PHASE2_STEPS="${PHASE2_STEPS:-4000000}"   # soccer_1vs1:  sparse reward, teaches scoring
NUM_ENVS="${XUANCE_NUM_ENVS:-4}"
SEED="${XUANCE_SEED:-}"
TRAINING_MODE="competitive"               # 1v1 is always competitive (2 policies)

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
# Directory structure
# ============================================================================

RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"

mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"
mkdir -p "$RUN_DIR/config"

# ============================================================================
# Display configuration
# ============================================================================

echo "============================================================"
echo "  IPPO Curriculum: collect_1vs1 -> soccer_1vs1"
echo "  (Single Process - Hot-Swap, No Interruption)"
echo "============================================================"
echo ""
echo "Run Configuration:"
echo "  Run ID:           $RUN_ID"
echo "  Base Config:      $CONFIG_FILE"
echo "  Output Directory: $RUN_DIR"
echo ""
echo "Phase 1 - Ball Collection (collect_1vs1):"
echo "  Environment:      MosaicMultiGrid-Collect-1vs1-IndAgObs-v0"
echo "  Grid:             10x10, 3 balls"
echo "  Reward:           +1 per ball (dense)"
echo "  Steps:            $PHASE1_STEPS"
echo "  Goal:             Learn movement + ball pickup"
echo ""
echo "Phase 2 - Soccer Transfer (soccer_1vs1):"
echo "  Environment:      MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0"
echo "  Grid:             16x11, first-to-2 goals"
echo "  Reward:           +1 per goal (sparse)"
echo "  Steps:            $PHASE2_STEPS"
echo "  Transfer:         In-memory hot-swap (optimizer + LR preserved)"
echo "  Goal:             Learn goal direction + scoring"
echo ""
echo "Algorithm: IPPO (Independent PPO)"
echo "  Policy Structure:"
echo "    pi_agent_0 (Green): 27-dim input, no one-hot, standalone"
echo "    pi_agent_1 (Blue):  27-dim input, no one-hot, standalone"
echo "  use_parameter_sharing: False"
echo "  Both networks survive the Phase 1 -> Phase 2 hot-swap."
echo ""
echo "Shared Architecture (MUST match across phases):"
echo "  Observations:     IndAgObs (3,3,3) = 27 flat"
echo "  Actions:          7 discrete"
echo "  Network:          MLP [64,64]"
echo "  Parallel Envs:    $NUM_ENVS"
echo "  Seed:             ${SEED:-random}"
echo ""
echo "DEPLOYMENT NOTE:"
echo "  pi_agent_0 knows the Green team goal direction."
echo "  pi_agent_1 knows the Blue team goal direction."
echo "  Always deploy pi_agent_0 as agent_0, pi_agent_1 as agent_1."
echo "  Policies are NOT interchangeable between teams."
echo ""
echo "FastLane:"
echo "  Enabled:          $GYM_GUI_FASTLANE_ONLY"
echo "  Video Mode:       $GYM_GUI_FASTLANE_VIDEO_MODE"
echo ""
echo "============================================================"

# ============================================================================
# Build IPPO curriculum config
# ============================================================================

CURRICULUM_CONFIG="$RUN_DIR/config/ippo_curriculum_config.json"

jq --argjson phase1_steps "$PHASE1_STEPS" \
   --argjson phase2_steps "$PHASE2_STEPS" \
   --argjson num_envs "$NUM_ENVS" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" \
   --arg training_mode "$TRAINING_MODE" \
   --arg seed "${SEED:-null}" '
    .method = "ippo" |
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
        {"env_id": "soccer_1vs1",  "steps": $phase2_steps}
    ] |
    if $seed != "null" and $seed != "" then
        .extras.seed = ($seed | tonumber)
    else
        .
    end
' "$CONFIG_FILE" > "$CURRICULUM_CONFIG"

echo ""
echo "Curriculum config: $CURRICULUM_CONFIG"
echo "Starting single-process IPPO curriculum training..."
echo ""

# ============================================================================
# Single-process training (environment hot-swap happens in Python)
# ============================================================================

export MOSAIC_RUN_DIR="$RUN_DIR"
python -m xuance_worker.cli --config "$CURRICULUM_CONFIG"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo "  IPPO Curriculum Training Complete!"
echo "============================================================"
echo ""
echo "Phase 1 (collect_1vs1 - ball pickup):    $PHASE1_STEPS steps"
echo "Phase 2 (soccer_1vs1  - goal scoring):   $PHASE2_STEPS steps"
echo "Total:                                    $((PHASE1_STEPS + PHASE2_STEPS)) steps"
echo ""
echo "Checkpoints: $RUN_DIR/checkpoints/"
echo "  final_train_model.pth  <- contains pi_agent_0 AND pi_agent_1"
echo "  phase1_model.pth       <- collect-only policy (for comparison)"
echo "  phase2_model.pth       <- alias for final (soccer policy)"
echo ""
echo "TensorBoard: tensorboard --logdir $RUN_DIR/tensorboard"
echo ""
echo "Deploying in MOSAIC:"
echo "  agent_0 (Green): load final_train_model.pth, player_id=agent_0"
echo "  agent_1 (Blue):  load final_train_model.pth, player_id=agent_1"
echo "  For Paper 2 hybrid: deploy pi_agent_0 as agent_0 alongside LLM"
echo ""
echo "============================================================"
