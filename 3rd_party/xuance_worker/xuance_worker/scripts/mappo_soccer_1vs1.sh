#!/bin/bash
# mappo_soccer_1vs1.sh - MAPPO Training for 1vs1 Soccer (IndAgObs)
#
# @description: MAPPO 1vs1 Soccer - separate Green/Blue policies
# @env_family: multigrid
# @environments: soccer_1vs1
# @method: MAPPO
# @total_timesteps: 2000000
#
# This script trains two separate policies using XuanCe's MAPPO algorithm
# on MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0 (SoccerGame2HIndAgObsEnv16x11N2).
#
# Environment Details:
#   - Grid: 16x11 (FIFA aspect ratio, 14x9 playable)
#   - Agents: 2 (agent_0 = Green team, agent_1 = Blue team)
#   - Observations: IndAgObs (3,3,3) per agent
#   - Actions: 7 discrete (left, right, forward, pickup, drop, toggle, done)
#   - Win condition: First to 2 goals
#   - Max steps: 200
#   - Rewards: Positive-only shared team rewards (zero_sum=False)
#   - No teleport pass (1v1 = no teammates)
#
# Training Mode:
#   - Always competitive: 2 policies (pi_green vs pi_blue)
#   - Policy 0 (Green): controls agent_0
#   - Policy 1 (Blue):  controls agent_1
#   - Both policies update via self-play
#
# Usage:
#   # Launch via GUI (XuanCe Script Form) or manually:
#   export MOSAIC_CONFIG_FILE="/path/to/config.json"
#   export MOSAIC_RUN_ID="soccer_1vs1_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash mappo_soccer_1vs1.sh

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

TOTAL_TIMESTEPS="${MAPPO_TOTAL_TIMESTEPS:-2000000}"
NUM_ENVS="${XUANCE_NUM_ENVS:-4}"
SEED="${XUANCE_SEED:-}"

# 1v1 is always competitive (2 policies, one per team)
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
export XUANCE_PARALLELS="$NUM_ENVS"  # Required by FastLane grid mode
export XUANCE_SEED="$SEED"
export TRACK_TENSORBOARD="${TRACK_TENSORBOARD:-1}"

# ============================================================================
# Create run-specific directory structure
# ============================================================================

# Artifact directory -- uses var/trainer/runs/{run_id} when launched from GUI
RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"

# ============================================================================
# Display training configuration
# ============================================================================

echo "============================================================"
echo "  MAPPO Training - 1vs1 Soccer (IndAgObs)"
echo "============================================================"
echo ""
echo "Run Configuration:"
echo "  Run ID:           $RUN_ID"
echo "  Base Config:      $CONFIG_FILE"
echo "  Output Directory: $RUN_DIR"
echo ""
echo "Environment:"
echo "  Family:           multigrid"
echo "  Environment:      MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0"
echo "  Class:            SoccerGame2HIndAgObsEnv16x11N2"
echo "  Grid Size:        16x11 (14x9 playable)"
echo "  Agents:           2"
echo "  Observations:     IndAgObs (3,3,3)"
echo "  Win Condition:    First to 2 goals"
echo "  Max Steps:        200"
echo "  Rewards:          Positive-only (zero_sum=False)"
echo ""
echo "Training Mode:"
echo "  Mode:             competitive (parameter sharing)"
echo "  Num Policies:     1 (shared by both agents)"
echo ""
echo "  Policy Structure:"
echo "    Shared Policy:    agent_0 + agent_1 (symmetric game)"
echo "    Benefit:          No competitive collapse possible"
echo ""
echo "Training Parameters:"
echo "  Algorithm:        MAPPO (Multi-Agent PPO)"
echo "  Total Timesteps:  $TOTAL_TIMESTEPS"
echo "  Parallel Envs:    $NUM_ENVS"
echo "  Seed:             ${SEED:-random}"
echo ""
echo "FastLane:"
echo "  Enabled:          $GYM_GUI_FASTLANE_ONLY"
echo "  Video Mode:       $GYM_GUI_FASTLANE_VIDEO_MODE"
echo ""
echo "============================================================"

# ============================================================================
# Build MAPPO configuration for 1vs1 Soccer
# ============================================================================

mkdir -p "$RUN_DIR/config"
MAPPO_CONFIG="$RUN_DIR/config/mappo_soccer_1vs1_config.json"

echo "Creating MAPPO configuration..."

# Build config using jq
jq --argjson steps "$TOTAL_TIMESTEPS" \
   --argjson num_envs "$NUM_ENVS" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" \
   --arg training_mode "$TRAINING_MODE" \
   --arg seed "${SEED:-null}" '
    # Algorithm selection
    .method = "mappo" |

    # Environment configuration
    .env = "multigrid" |
    .env_id = "soccer_1vs1" |

    # Training mode (always competitive for 1v1)
    .extras.training_mode = $training_mode |

    # Training steps
    .running_steps = $steps |

    # Parallel environments
    .parallels = $num_envs |

    # Extra configuration
    .extras.algo_params.num_envs = $num_envs |
    .extras.num_envs = $num_envs |
    .extras.tensorboard_dir = $tensorboard_dir |
    .extras.checkpoint_dir = $checkpoint_dir |

    # Seed (if provided)
    if $seed != "null" and $seed != "" then
        .extras.seed = ($seed | tonumber)
    else
        .
    end
' "$CONFIG_FILE" > "$MAPPO_CONFIG"

echo "Configuration saved to: $MAPPO_CONFIG"
echo ""

# ============================================================================
# Run MAPPO training
# ============================================================================

echo "Starting MAPPO training..."
echo ""
echo "Training Mode: competitive with parameter sharing (1v1)"
echo "  - Green + Blue share ONE policy (symmetric game)"
echo "  - No competitive collapse — both sides train the same network"
echo "  - Win rate should stay ~50% (identical policies)"
echo ""
echo "Monitor progress:"
echo "  - TensorBoard: tensorboard --logdir $RUN_DIR/tensorboard"
echo "  - Checkpoints: $RUN_DIR/checkpoints/"
echo ""
echo "------------------------------------------------------------"

python -m xuance_worker.cli --config "$MAPPO_CONFIG"

# ============================================================================
# Training complete - Summary
# ============================================================================

echo ""
echo "============================================================"
echo "  MAPPO 1vs1 Soccer Training Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Environment:     MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0"
echo "  Training Mode:   competitive (parameter sharing)"
echo "  Num Policies:    1 (shared by both agents)"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Checkpoints:     $RUN_DIR/checkpoints/"
echo "  TensorBoard:     $RUN_DIR/tensorboard/"
echo "  Config:          $MAPPO_CONFIG"
echo ""
echo "Policy Mapping:"
echo "  Shared Policy: agent_0 + agent_1 -> pi_shared"
echo "  (Symmetric game: one policy plays both sides)"
echo ""
echo "Next Steps:"
echo "  1. Review training curves in TensorBoard"
echo "  2. Load shared checkpoint for 2v2 evaluation"
echo "  3. Deploy: agents 0,1,2 use pi_shared; agent 3 = LLM/Random"
echo ""
echo "============================================================"
