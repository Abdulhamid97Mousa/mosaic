#!/bin/bash
# ippo_gru_soccer_2vs2_indagobs.sh - IPPO+GRU Training for 2vs2 Soccer (IndAgObs)
#
# @description: IPPO+GRU 2vs2 Soccer - recurrent independent PPO policies
# @env_family: multigrid
# @environments: soccer_2vs2_indagobs
# @method: IPPO (with GRU)
# @total_timesteps: 5000000
#
# Naming convention: {algorithm}_{rnn}_soccer_{NvN}_{obs_type}.sh
#   - algorithm:  ippo (Independent PPO)
#   - rnn:        gru (Gated Recurrent Unit)
#   - soccer:     the game
#   - 2vs2:       2 agents per team (Green vs Blue)
#   - indagobs:   Individual Agent Observations (7x7 egocentric view)
#
# Why GRU?
#   GRU provides per-agent memory for navigation under partial observability.
#   Each agent maintains independent hidden state. In a 2v2 match with (7,7,3)
#   egocentric views, agents cannot see the full field -- GRU enables temporal
#   reasoning about unseen teammates, opponents, and ball position.
#
#   Architecture: MLP(obs) -> GRU(64) -> action_head / value_head
#   The GRU output feeds directly into action and value heads (no additional
#   MLP layers after recurrence), keeping the model compact.
#
# IPPO: each agent trains independently — no centralized critic, natural
#   for competitive self-play. With 4 agents (2v2), each agent maintains
#   its own recurrent policy and value function. No parameter sharing.
#
# References:
#   - Yu et al. (2022) "The Surprising Effectiveness of PPO in Cooperative
#     Multi-Agent Games" -- NeurIPS 2022
#   - Phan et al. (2023) "Attention-Based Recurrence for Multi-Agent RL
#     under Stochastic Partial Observability" -- ICML 2023
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
#   export MOSAIC_RUN_ID="ippo_gru_soccer_2vs2_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash ippo_gru_soccer_2vs2_indagobs.sh

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

# 2v2 competitive: 4 independent recurrent policies (one per agent)
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
echo "  IPPO+GRU Training - 2vs2 Soccer (IndAgObs, view_size=7)"
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
echo "Algorithm: IPPO + GRU (Independent PPO with Recurrent Memory)"
echo "  Each agent trains its own recurrent PPO independently."
echo "  GRU hidden state enables temporal reasoning under partial observability."
echo "  Architecture: MLP(obs) -> GRU(64) -> action_head / value_head"
echo "  No centralized critic -- natural for competitive self-play."
echo ""
echo "Training Mode:"
echo "  Mode:             competitive"
echo "  Num Policies:     4"
echo ""
echo "  Policy Structure:"
echo "    Policy 0 (Green): agent_0  [GRU h_0]"
echo "    Policy 1 (Green): agent_1  [GRU h_1]"
echo "    Policy 2 (Blue):  agent_2  [GRU h_2]"
echo "    Policy 3 (Blue):  agent_3  [GRU h_3]"
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
# Build IPPO+GRU configuration for 2vs2 Soccer
# ============================================================================

mkdir -p "$RUN_DIR/config"
IPPO_CONFIG="$RUN_DIR/config/ippo_gru_soccer_2vs2_indagobs_config.json"

echo "Creating IPPO+GRU configuration..."

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

    # GRU recurrence -- override base YAML Basic_MLP
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
# Run IPPO+GRU training
# ============================================================================

echo "Starting IPPO+GRU training..."
echo ""
echo "Training Mode: competitive (2vs2)"
echo "  - Green (Policy 0, 1: agent_0, agent_1) vs Blue (Policy 2, 3: agent_2, agent_3)"
echo "  - All 4 recurrent policies train independently via self-play"
echo "  - GRU hidden states reset at episode boundaries"
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
echo "  IPPO+GRU 2vs2 Soccer Training Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Algorithm:       IPPO + GRU (Independent Recurrent PPO)"
echo "  Environment:     MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0"
echo "  Training Mode:   competitive"
echo "  Num Policies:    4 (pi_green_0, pi_green_1, pi_blue_0, pi_blue_1)"
echo "  Recurrence:      GRU (64 hidden units, 1 layer)"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Checkpoints:     $RUN_DIR/checkpoints/"
echo "  TensorBoard:     $RUN_DIR/tensorboard/"
echo "  Config:          $IPPO_CONFIG"
echo ""
echo "Policy Mapping:"
echo "  Policy 0 (Green): agent_0 -> pi_green_0  [GRU]"
echo "  Policy 1 (Green): agent_1 -> pi_green_1  [GRU]"
echo "  Policy 2 (Blue):  agent_2 -> pi_blue_0   [GRU]"
echo "  Policy 3 (Blue):  agent_3 -> pi_blue_1   [GRU]"
echo ""
echo "Next Steps:"
echo "  1. Review training curves in TensorBoard"
echo "  2. Compare GRU vs MLP IPPO, deploy with MAPPOGRUActor"
echo "  3. Load checkpoints for evaluation or transfer learning"
echo "  4. For LLM vs RL: Load team policies, control opponents with LLM"
echo ""
echo "============================================================"
