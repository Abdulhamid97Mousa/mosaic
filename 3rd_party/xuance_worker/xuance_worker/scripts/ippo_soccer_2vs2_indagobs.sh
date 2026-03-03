#!/bin/bash
# ippo_soccer_2vs2_indagobs.sh - IPPO Training for 2vs2 Soccer (IndAgObs)
#
# @description: IPPO 2vs2 Soccer - independent per-agent PPO policies
# @env_family: multigrid
# @environments: soccer_2vs2_indagobs
# @method: IPPO
# @total_timesteps: 5000000
#
# Naming convention: {algorithm}_soccer_{NvN}_{obs_type}.sh
#   - algorithm:  ippo (Independent PPO)
#   - soccer:     the game
#   - 2vs2:       2 agents per team (Green vs Blue)
#   - indagobs:   Individual Agent Observations (7x7 egocentric view)
#
# Why IndAgObs (not TeamObs)?
#   - Each agent gets a local 7x7 view: (7,7,3) = 147 features flattened
#   - Larger view than 1v1 (3x3) because 2v2 needs spatial awareness of teammates
#   - MOSAIC_VIEW_SIZE=7 must be exported before env creation
#   - Decentralized observations force each agent to learn coordination
#     from local perception alone
#
# IPPO: each agent trains independently — no centralized critic, natural
#   for competitive self-play. With 4 agents (2v2), each agent maintains
#   its own policy and value function. No parameter sharing between agents.
#   This is the simplest multi-agent baseline and provides a clean comparison
#   against MAPPO (centralized critic) and GRU variants (recurrent memory).
#
# Environment: MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0
#   Class: SoccerGame4HIndAgObsEnv16x11N2
#   Grid:  16x11 (FIFA aspect ratio, 14x9 playable)
#   Agents: 4 (agent_0, agent_1 = Green; agent_2, agent_3 = Blue)
#   Observations: IndAgObs (7,7,3) per agent (view_size=7)
#   Actions: 8 discrete (left, right, forward, pickup, drop, toggle, done, noop)
#   Win condition: First to 2 goals
#   Max steps: 200
#   Rewards: Positive-only (zero_sum=False)
#
# Usage:
#   # Launch via GUI (XuanCe Script Form) or manually:
#   export MOSAIC_CONFIG_FILE="/path/to/config.json"
#   export MOSAIC_RUN_ID="ippo_soccer_2vs2_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash ippo_soccer_2vs2_indagobs.sh

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

TOTAL_TIMESTEPS="${IPPO_TOTAL_TIMESTEPS:-5000000}"
NUM_ENVS="${XUANCE_NUM_ENVS:-8}"
SEED="${XUANCE_SEED:-}"

# 2v2 competitive: 4 independent policies (one per agent)
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

# Critical: enable 7x7 egocentric view for IndAgObs
export MOSAIC_VIEW_SIZE="${MOSAIC_VIEW_SIZE:-7}"

# ============================================================================
# Create run-specific directory structure
# ============================================================================

# Artifact directory -- uses var/trainer/runs/{run_id} when launched from GUI
RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"
mkdir -p "$RUN_DIR/logs"
mkdir -p "$RUN_DIR/models/ippo"

# ============================================================================
# Display training configuration
# ============================================================================

echo "============================================================"
echo "  IPPO Training - 2vs2 Soccer (IndAgObs, view_size=7)"
echo "============================================================"
echo ""
echo "Run Configuration:"
echo "  Run ID:           $RUN_ID"
echo "  Base Config:      $CONFIG_FILE"
echo "  Output Directory: $RUN_DIR"
echo ""
echo "Environment:"
echo "  Family:           multigrid"
echo "  Environment:      MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0"
echo "  Class:            SoccerGame4HIndAgObsEnv16x11N2"
echo "  Grid Size:        16x11 (14x9 playable)"
echo "  Agents:           4"
echo "  Observations:     IndAgObs (7,7,3) -- egocentric 7x7 local view"
echo "  View Size:        $MOSAIC_VIEW_SIZE"
echo "  Win Condition:    First to 2 goals"
echo "  Max Steps:        200"
echo "  Rewards:          Positive-only (zero_sum=False)"
echo ""
echo "Algorithm: IPPO (Independent PPO)"
echo "  Each agent trains its own PPO independently."
echo "  No centralized critic -- natural for competitive self-play."
echo ""
echo "Training Mode:"
echo "  Mode:             competitive"
echo "  Num Policies:     4"
echo ""
echo "  Policy Structure:"
echo "    Policy 0 (Green): agent_0"
echo "    Policy 1 (Green): agent_1"
echo "    Policy 2 (Blue):  agent_2"
echo "    Policy 3 (Blue):  agent_3"
echo ""
echo "Training Parameters:"
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
# Build IPPO configuration for 2vs2 Soccer
# ============================================================================

mkdir -p "$RUN_DIR/config"
IPPO_CONFIG="$RUN_DIR/config/ippo_soccer_2vs2_indagobs_config.json"

echo "Creating IPPO configuration..."

# Build config using jq
jq --argjson steps "$TOTAL_TIMESTEPS" \
   --argjson num_envs "$NUM_ENVS" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" \
   --arg log_dir "$RUN_DIR/logs" \
   --arg model_dir "$RUN_DIR/models/ippo" \
   --arg training_mode "$TRAINING_MODE" \
   --arg seed "${SEED:-null}" '
    # Algorithm selection
    .method = "ippo" |

    # Environment configuration
    .env = "multigrid" |
    .env_id = "soccer_2vs2_indagobs" |

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
' "$CONFIG_FILE" > "$IPPO_CONFIG"

echo "Configuration saved to: $IPPO_CONFIG"
echo ""

# ============================================================================
# Run IPPO training
# ============================================================================

echo "Starting IPPO training..."
echo ""
echo "Training Mode: competitive (2vs2)"
echo "  - Green (Policy 0, 1: agent_0, agent_1) vs Blue (Policy 2, 3: agent_2, agent_3)"
echo "  - All 4 policies train independently via self-play"
echo "  - Win rate should oscillate around 50% in balanced training"
echo ""
echo "Monitor progress:"
echo "  - TensorBoard: tensorboard --logdir $RUN_DIR/tensorboard"
echo "  - Checkpoints: $RUN_DIR/checkpoints/"
echo ""
echo "------------------------------------------------------------"

python -m xuance_worker.cli --config "$IPPO_CONFIG"

# ============================================================================
# Training complete - Summary
# ============================================================================

echo ""
echo "============================================================"
echo "  IPPO 2vs2 Soccer Training Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Algorithm:       IPPO (Independent PPO)"
echo "  Environment:     MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0"
echo "  Training Mode:   competitive"
echo "  Num Policies:    4 (pi_green_0, pi_green_1, pi_blue_0, pi_blue_1)"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Checkpoints:     $RUN_DIR/checkpoints/"
echo "  TensorBoard:     $RUN_DIR/tensorboard/"
echo "  Config:          $IPPO_CONFIG"
echo ""
echo "Policy Mapping:"
echo "  Policy 0 (Green): agent_0 -> pi_green_0"
echo "  Policy 1 (Green): agent_1 -> pi_green_1"
echo "  Policy 2 (Blue):  agent_2 -> pi_blue_0"
echo "  Policy 3 (Blue):  agent_3 -> pi_blue_1"
echo ""
echo "Next Steps:"
echo "  1. Review training curves in TensorBoard"
echo "  2. Compare with MAPPO runs, deploy for RL+LLM heterogeneous evaluation"
echo "  3. Load checkpoints for evaluation or transfer learning"
echo "  4. For LLM vs RL: Load team policies, control opponents with LLM"
echo ""
echo "============================================================"
