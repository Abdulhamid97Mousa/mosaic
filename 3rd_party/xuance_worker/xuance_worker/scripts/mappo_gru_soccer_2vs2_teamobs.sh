#!/bin/bash
# mappo_gru_soccer_2vs2_teamobs.sh - MAPPO+GRU Training for 2vs2 Soccer (TeamObs)
#
# @description: MAPPO+GRU 2vs2 Soccer - competitive 4-agent training with TeamObs
# @env_family: multigrid
# @environments: soccer_2vs2_teamobs
# @method: MAPPO
# @total_timesteps: 5000000
#
# This script trains four separate policies using XuanCe's MAPPO algorithm
# with GRU recurrence on MosaicMultiGrid-Soccer-2vs2-TeamObs-v0 (SoccerTeamObsEnv).
#
# References:
#   - Yu et al. (2022) "The Surprising Effectiveness of PPO in Cooperative
#     Multi-Agent Games" NeurIPS
#   - Phan et al. (2023) "Attention-Based Recurrence for Multi-Agent RL under
#     Stochastic Partial Observability" ICML
#
# GRU provides memory for navigation under partial observability.
# Each agent maintains its own hidden state across timesteps, allowing the
# policy to reason about previously seen grid cells and teammate positions.
#
# Environment Details:
#   - Grid: 16x11 (FIFA aspect ratio, 14x9 playable)
#   - Agents: 4 (agent_0, agent_1 = Green team; agent_2, agent_3 = Blue team)
#   - Observations: TeamObs (7,7,3) image + (4,) teammate = 147 + 4 = 151
#   - View Size: 7x7 (set via MOSAIC_VIEW_SIZE=7)
#   - Actions: 7 discrete (left, right, forward, pickup, drop, toggle, done)
#   - Win condition: First to 2 goals
#   - Max steps: 200
#   - Rewards: Positive-only shared team rewards (zero_sum=False)
#   - TeamObs: agents can see their teammates' positions explicitly
#
# Training Mode:
#   - Competitive: 4 separate policies with per-agent hidden states
#   - Policy 0 (Green): controls agent_0
#   - Policy 1 (Green): controls agent_1
#   - Policy 2 (Blue):  controls agent_2
#   - Policy 3 (Blue):  controls agent_3
#   - All policies update via self-play
#
# No curriculum learning is used.
#
# Usage:
#   # Launch via GUI (XuanCe Script Form) or manually:
#   export MOSAIC_CONFIG_FILE="/path/to/config.json"
#   export MOSAIC_RUN_ID="soccer_2vs2_teamobs_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash mappo_gru_soccer_2vs2_teamobs.sh

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

# 2v2 competitive: 4 policies (one per agent)
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

# Critical: enable 7x7 view for TeamObs
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
echo "  MAPPO+GRU Training - 2vs2 Soccer (TeamObs, view_size=7)"
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
echo "  Agents:           4 (agent_0, agent_1 = Green; agent_2, agent_3 = Blue)"
echo "  Observations:     TeamObs (7,7,3) image + (4,) teammate = 151"
echo "  View Size:        7x7 (MOSAIC_VIEW_SIZE=$MOSAIC_VIEW_SIZE)"
echo "  Win Condition:    First to 2 goals"
echo "  Max Steps:        200"
echo "  Rewards:          Positive-only (zero_sum=False)"
echo ""
echo "Training Mode:"
echo "  Mode:             competitive (4 separate policies)"
echo "  Num Policies:     4 (one per agent)"
echo ""
echo "  Policy Structure:"
echo "    Policy 0 (Green): agent_0"
echo "    Policy 1 (Green): agent_1"
echo "    Policy 2 (Blue):  agent_2"
echo "    Policy 3 (Blue):  agent_3"
echo ""
echo "Training Parameters:"
echo "  Algorithm:        MAPPO+GRU (Multi-Agent PPO with GRU recurrence)"
echo "  Total Timesteps:  $TOTAL_TIMESTEPS"
echo "  Parallel Envs:    $NUM_ENVS"
echo "  Seed:             ${SEED:-random}"
echo ""
echo "Recurrence:"
echo "  RNN Type:         GRU"
echo "  Note:             GRU provides memory for navigation under"
echo "                    partial observability"
echo ""
echo "FastLane:"
echo "  Enabled:          $GYM_GUI_FASTLANE_ONLY"
echo "  Video Mode:       $GYM_GUI_FASTLANE_VIDEO_MODE"
echo ""
echo "============================================================"

# ============================================================================
# Build MAPPO+GRU configuration for 2vs2 Soccer (TeamObs)
# ============================================================================

mkdir -p "$RUN_DIR/config"
MAPPO_CONFIG="$RUN_DIR/config/mappo_gru_soccer_2vs2_teamobs_config.json"

echo "Creating MAPPO+GRU configuration..."

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

    # RNN configuration (GRU) -- override base YAML Basic_MLP
    .extras.representation = "Basic_RNN" |
    .extras.use_rnn = true |
    .extras.rnn = "GRU" |
    .extras.fc_hidden_sizes = [128, 128] |
    .extras.recurrent_hidden_size = 64 |
    .extras.N_recurrent_layers = 1 |
    .extras.actor_hidden_size = [] |
    .extras.critic_hidden_size = [] |
    .extras.n_epochs = 10 |
    .extras.n_minibatch = 1 |
    .extras.ent_coef = 0.01 |

    # Training mode (competitive for 2v2)
    .extras.training_mode = $training_mode |

    # Output directories
    .log_dir = $log_dir |
    .model_dir = $model_dir |

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
# Run MAPPO+GRU training
# ============================================================================

echo "Starting MAPPO+GRU training..."
echo ""
echo "Training Mode: competitive (2v2, 4 separate policies)"
echo "  - Green team: agent_0 + agent_1 (policies 0, 1)"
echo "  - Blue team:  agent_2 + agent_3 (policies 2, 3)"
echo "  - GRU hidden states maintained per agent across timesteps"
echo "  - TeamObs: agents see teammate positions explicitly"
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
echo "  MAPPO+GRU 2vs2 Soccer (TeamObs) Training Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Environment:     MosaicMultiGrid-Soccer-2vs2-TeamObs-v0"
echo "  Training Mode:   competitive (4 separate policies)"
echo "  Num Policies:    4 (one per agent, per-agent GRU hidden states)"
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
echo "  2. Deploy with MAPPOGRUActor for RL+LLM heterogeneous evaluation"
echo "  3. Evaluate team coordination via TeamObs advantage"
echo ""
echo "============================================================"
