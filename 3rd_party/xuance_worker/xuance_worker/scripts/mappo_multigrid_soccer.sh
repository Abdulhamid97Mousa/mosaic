#!/bin/bash
# mappo_multigrid_soccer.sh - MAPPO Training for MultiGrid Soccer
#
# @description: Train MAPPO on gym-multigrid SoccerGame with configurable training modes
# @env_family: multigrid
# @method: MAPPO
# @total_timesteps: 1000000
#
# This script trains multi-agent policies using XuanCe's MAPPO algorithm
# on the gym-multigrid SoccerGame4HEnv10x15N2 environment.
#
# Environment Details:
#   - Grid: 15x10
#   - Agents: 4 (indices [1,1,2,2])
#   - Default Teams: Red (agents 0,1) vs Blue (agents 2,3)
#   - Zero-sum: Yes
#
# Training Modes (set via TRAINING_MODE environment variable):
#   - competitive: Per-team policies (2 policies: Red, Blue) [DEFAULT]
#   - cooperative: Shared policy (1 policy for all 4 agents)
#   - independent: Per-agent policies (4 separate policies)
#
# Usage:
#   # Competitive training (default - 2 teams, 2 policies)
#   export MOSAIC_CONFIG_FILE="/path/to/config.json"
#   export MOSAIC_RUN_ID="soccer_competitive_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash mappo_multigrid_soccer.sh
#
#   # Cooperative training (shared policy)
#   export TRAINING_MODE="cooperative"
#   bash mappo_multigrid_soccer.sh
#
#   # Independent training (per-agent policies)
#   export TRAINING_MODE="independent"
#   bash mappo_multigrid_soccer.sh

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

TOTAL_TIMESTEPS="${MAPPO_TOTAL_TIMESTEPS:-1000000}"
NUM_ENVS="${XUANCE_NUM_ENVS:-4}"
SEED="${XUANCE_SEED:-}"

# Training mode: competitive (default), cooperative, or independent
TRAINING_MODE="${TRAINING_MODE:-competitive}"

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
export XUANCE_SEED="$SEED"
export TRACK_TENSORBOARD="${TRACK_TENSORBOARD:-1}"

# ============================================================================
# Create run-specific directory structure
# ============================================================================

RUN_DIR="$SCRIPTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"

# ============================================================================
# Validate training mode
# ============================================================================

case "$TRAINING_MODE" in
    competitive)
        MODE_DESC="Per-team policies (2 policies: Red team, Blue team)"
        NUM_POLICIES=2
        ;;
    cooperative)
        MODE_DESC="Shared policy (1 policy for all 4 agents)"
        NUM_POLICIES=1
        ;;
    independent)
        MODE_DESC="Per-agent policies (4 separate policies)"
        NUM_POLICIES=4
        ;;
    *)
        echo "ERROR: Unknown TRAINING_MODE '$TRAINING_MODE'"
        echo "Valid options: competitive, cooperative, independent"
        exit 1
        ;;
esac

# ============================================================================
# Display training configuration
# ============================================================================

echo "============================================================"
echo "  MAPPO Training - MultiGrid Soccer"
echo "============================================================"
echo ""
echo "Run Configuration:"
echo "  Run ID:           $RUN_ID"
echo "  Base Config:      $CONFIG_FILE"
echo "  Output Directory: $RUN_DIR"
echo ""
echo "Environment:"
echo "  Family:           multigrid"
echo "  Environment:      SoccerGame4HEnv10x15N2"
echo "  Grid Size:        15x10"
echo "  Agents:           4"
echo ""
echo "Training Mode:"
echo "  Mode:             $TRAINING_MODE"
echo "  Description:      $MODE_DESC"
echo "  Num Policies:     $NUM_POLICIES"
echo ""
if [ "$TRAINING_MODE" = "competitive" ]; then
echo "  Policy Structure:"
echo "    Policy 0 (Red):  agent_0, agent_1"
echo "    Policy 1 (Blue): agent_2, agent_3"
echo ""
fi
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
# Build MAPPO configuration for MultiGrid Soccer
# ============================================================================

MAPPO_CONFIG="$RUN_DIR/mappo_soccer_config.json"

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
    .env_id = "soccer" |

    # Training mode (competitive, cooperative, independent)
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
echo "Training Mode: $TRAINING_MODE"
case "$TRAINING_MODE" in
    competitive)
        echo "  - Team Red (Policy 0) vs Team Blue (Policy 1)"
        echo "  - Both teams update their policies via self-play"
        echo "  - Win rate should oscillate around 50% in balanced training"
        ;;
    cooperative)
        echo "  - All 4 agents share a single policy"
        echo "  - Parameter sharing enables faster learning"
        echo "  - Best for cooperative objectives"
        ;;
    independent)
        echo "  - Each agent learns independently"
        echo "  - 4 separate policies trained simultaneously"
        echo "  - Allows heterogeneous agent behaviors"
        ;;
esac
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
echo "  MAPPO Soccer Training Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Training Mode:   $TRAINING_MODE"
echo "  Num Policies:    $NUM_POLICIES"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Checkpoints:     $RUN_DIR/checkpoints/"
echo "  TensorBoard:     $RUN_DIR/tensorboard/"
echo "  Config:          $MAPPO_CONFIG"
echo ""
echo "Next Steps:"
echo "  1. Review training curves in TensorBoard"
echo "  2. Load checkpoint for evaluation"
if [ "$TRAINING_MODE" = "competitive" ]; then
echo "  3. For LLM vs RL: Load Blue team policy, control Red with LLM"
fi
echo ""
echo "See: docs/Development_Progress/1.0_DAY_63/TASK_1/MAPPO_MULTIGRID_TRAINING_GUIDE.md"
echo ""
echo "============================================================"
