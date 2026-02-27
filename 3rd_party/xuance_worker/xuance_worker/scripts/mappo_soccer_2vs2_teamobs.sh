#!/bin/bash
# mappo_soccer_2vs2_teamobs.sh - MAPPO Training for 2vs2 Soccer (TeamObs, view_size=7)
#
# @description: MAPPO 2vs2 Soccer - four separate policies (competitive)
# @env_family: multigrid
# @environments: soccer_2vs2_teamobs
# @method: MAPPO
# @total_timesteps: 5000000
#
# This script trains four separate policies using XuanCe's MAPPO algorithm
# on MosaicMultiGrid-Soccer-2vs2-TeamObs-v0 (SoccerTeamObsEnv).
#
# Environment Details:
#   - Grid: 16x11 (FIFA aspect ratio, 14x9 playable)
#   - Agents: 4 (agent_0, agent_1 = Green team; agent_2, agent_3 = Blue team)
#   - Observations: TeamObs (7,7,3) image + (4,) teammate = 151 per agent
#   - Actions: 7 discrete (left, right, forward, pickup, drop, toggle, done)
#   - Win condition: First to 2 goals
#   - Max steps: 200
#   - Rewards: Positive-only shared team rewards (zero_sum=False)
#   - Teleport pass enabled (2v2 = teammates available)
#   - view_size=7: Agents see a 7x7 grid (vs default 3x3)
#
# Training Mode:
#   - Competitive: 4 separate policies (one per agent)
#   - Policy 0 (Green): controls agent_0
#   - Policy 1 (Green): controls agent_1
#   - Policy 2 (Blue):  controls agent_2
#   - Policy 3 (Blue):  controls agent_3
#   - Green team vs Blue team via self-play
#
# Usage:
#   # Launch via GUI (XuanCe Script Form) or manually:
#   export MOSAIC_CONFIG_FILE="/path/to/config.json"
#   export MOSAIC_RUN_ID="soccer_2vs2_teamobs_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash mappo_soccer_2vs2_teamobs.sh

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

TOTAL_TIMESTEPS="${MAPPO_TOTAL_TIMESTEPS:-5000000}"
NUM_ENVS="${XUANCE_NUM_ENVS:-8}"
SEED="${XUANCE_SEED:-}"

# 2v2 competitive: 4 separate policies (one per agent)
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

# Critical: 7x7 view size for TeamObs environment
export MOSAIC_VIEW_SIZE=7

# ============================================================================
# Create run-specific directory structure
# ============================================================================

# Artifact directory -- uses var/trainer/runs/{run_id} when launched from GUI
RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"
mkdir -p "$RUN_DIR/logs"
mkdir -p "$RUN_DIR/models/mappo"

# ============================================================================
# Display training configuration
# ============================================================================

echo "============================================================"
echo "  MAPPO Training - 2vs2 Soccer (TeamObs, view_size=7)"
echo "============================================================"
echo ""
echo "Run Configuration:"
echo "  Run ID:           $RUN_ID"
echo "  Base Config:      $CONFIG_FILE"
echo "  Output Directory: $RUN_DIR"
echo ""
echo "Environment:"
echo "  Family:           multigrid"
echo "  Environment:      MosaicMultiGrid-Soccer-2vs2-TeamObs-v0"
echo "  Class:            SoccerTeamObsEnv"
echo "  Grid Size:        16x11 (14x9 playable)"
echo "  Agents:           4 (agent_0,1 = Green; agent_2,3 = Blue)"
echo "  Observations:     TeamObs (7,7,3) image + (4,) teammate = 151"
echo "  View Size:        7x7 (MOSAIC_VIEW_SIZE=7)"
echo "  Win Condition:    First to 2 goals"
echo "  Max Steps:        200"
echo "  Rewards:          Positive-only (zero_sum=False)"
echo ""
echo "Training Mode:"
echo "  Mode:             competitive (4 separate policies, one per agent)"
echo "  Num Policies:     4"
echo ""
echo "  Policy Structure:"
echo "    Policy 0 (Green): agent_0"
echo "    Policy 1 (Green): agent_1"
echo "    Policy 2 (Blue):  agent_2"
echo "    Policy 3 (Blue):  agent_3"
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
# Build MAPPO configuration for 2vs2 Soccer (TeamObs)
# ============================================================================

mkdir -p "$RUN_DIR/config"
MAPPO_CONFIG="$RUN_DIR/config/mappo_soccer_2vs2_teamobs_config.json"

echo "Creating MAPPO configuration..."

# Build config using jq
jq --argjson steps "$TOTAL_TIMESTEPS" \
   --argjson num_envs "$NUM_ENVS" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" \
   --arg log_dir "$RUN_DIR/logs" \
   --arg model_dir "$RUN_DIR/models/mappo" \
   --arg training_mode "$TRAINING_MODE" \
   --arg seed "${SEED:-null}" '
    # Algorithm selection
    .method = "mappo" |

    # Environment configuration
    .env = "multigrid" |
    .env_id = "soccer_2vs2_teamobs" |

    # Training mode (competitive: 4 separate policies)
    .extras.training_mode = $training_mode |

    # Training steps
    .running_steps = $steps |

    # Parallel environments
    .parallels = $num_envs |

    # Output directories
    .log_dir = $log_dir |
    .model_dir = $model_dir |

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
echo "Training Mode: competitive (2vs2 — 4 separate policies)"
echo "  - Green team: agent_0 (policy_0), agent_1 (policy_1)"
echo "  - Blue team:  agent_2 (policy_2), agent_3 (policy_3)"
echo "  - TeamObs: each agent sees 7x7 grid + teammate direction vector"
echo "  - Agents can coordinate via shared team rewards + teammate obs"
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
echo "  MAPPO 2vs2 Soccer (TeamObs) Training Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Environment:     MosaicMultiGrid-Soccer-2vs2-TeamObs-v0"
echo "  Training Mode:   competitive (4 separate policies)"
echo "  Num Policies:    4"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Checkpoints:     $RUN_DIR/checkpoints/"
echo "  TensorBoard:     $RUN_DIR/tensorboard/"
echo "  Config:          $MAPPO_CONFIG"
echo ""
echo "Policy Mapping:"
echo "  Policy 0 (Green): agent_0 -> pi_green_0"
echo "  Policy 1 (Green): agent_1 -> pi_green_1"
echo "  Policy 2 (Blue):  agent_2 -> pi_blue_0"
echo "  Policy 3 (Blue):  agent_3 -> pi_blue_1"
echo ""
echo "Next Steps:"
echo "  1. Review training curves in TensorBoard"
echo "  2. Load trained checkpoints for 2v2 evaluation"
echo "  3. Deploy with MAPPOActor for RL+LLM heterogeneous evaluation"
echo ""
echo "============================================================"
