#!/bin/bash
# mappo_smac_3m.sh - MAPPO Training for SMAC 3m (3 Marines)
#
# @description: MAPPO cooperative training on SMAC 3m benchmark
# @env_family: smac
# @environments: 3m
# @method: MAPPO
# @total_timesteps: 1000000
#
# This script trains a shared MAPPO policy on SMAC's 3m map using XuanCe's
# RunnerStarCraft2. Each parallel env launches a separate SC2 process.
#
# Environment Details:
#   - Map: 3m (3 Marines vs 3 Marines)
#   - Agents: 3 (homogeneous, parameter sharing)
#   - Observations: 30-dim per agent (relative positions, health, shield)
#   - Actions: 9 discrete (noop, stop, 4 move, 3 attack)
#   - Episode limit: 60 steps
#   - Rewards: Shaped (damage dealt + kill bonus + win bonus)
#   - Difficulty: 7 (default SMAC benchmark)
#
# Training Mode:
#   - Cooperative (CTDE): centralized critic, decentralized actors
#   - Parameter sharing: single policy network for all 3 agents
#   - GRU recurrence: handles partial observability
#
# Requirements:
#   - StarCraft II headless binary: var/data/StarCraftII/
#   - Python packages: smac, pysc2, s2clientprotocol
#   - SC2PATH must be set (script auto-resolves from var/data/)
#
# Usage:
#   # Launch via GUI (XuanCe Script Form) or manually:
#   export MOSAIC_CONFIG_FILE="/path/to/config.json"
#   export MOSAIC_RUN_ID="smac_3m_001"
#   export MOSAIC_CUSTOM_SCRIPTS_DIR="/path/to/output"
#   bash mappo_smac_3m.sh

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
SMAC_MAP="${SMAC_MAP:-3m}"

# ============================================================================
# SC2 Path Resolution (three-layer pattern: env var -> var/data/ -> error)
# ============================================================================

# Resolve SC2PATH if not already set
if [ -z "$SC2PATH" ]; then
    # Check project-local var/data/StarCraftII
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
    LOCAL_SC2="$PROJECT_ROOT/var/data/StarCraftII"
    if [ -d "$LOCAL_SC2" ]; then
        export SC2PATH="$LOCAL_SC2"
    else
        echo "ERROR: SC2PATH not set and var/data/StarCraftII not found."
        echo "Download: wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip"
        echo "Install:  unzip SC2.4.10.zip -d var/data"
        exit 1
    fi
fi

echo "SC2PATH: $SC2PATH"

# Protobuf compatibility (required for pysc2)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ============================================================================
# Export environment variables for child Python processes
# ============================================================================
export GYM_GUI_FASTLANE_ONLY="${GYM_GUI_FASTLANE_ONLY:-0}"
export GYM_GUI_FASTLANE_SLOT="${GYM_GUI_FASTLANE_SLOT:-0}"
export XUANCE_RUN_ID="${XUANCE_RUN_ID:-$RUN_ID}"
export XUANCE_AGENT_ID="${XUANCE_AGENT_ID:-xuance-agent}"
export XUANCE_NUM_ENVS="$NUM_ENVS"
export XUANCE_PARALLELS="$NUM_ENVS"
export XUANCE_SEED="$SEED"
export TRACK_TENSORBOARD="${TRACK_TENSORBOARD:-1}"

# ============================================================================
# Create run-specific directory structure
# ============================================================================

RUN_DIR="${MOSAIC_RUN_DIR:-$SCRIPTS_DIR/$RUN_ID}"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/tensorboard"
mkdir -p "$RUN_DIR/logs"
mkdir -p "$RUN_DIR/models/mappo"

# ============================================================================
# Display training configuration
# ============================================================================

echo "============================================================"
echo "  MAPPO Training - SMAC $SMAC_MAP"
echo "============================================================"
echo ""
echo "Run Configuration:"
echo "  Run ID:           $RUN_ID"
echo "  Base Config:      $CONFIG_FILE"
echo "  Output Directory: $RUN_DIR"
echo ""
echo "Environment:"
echo "  Family:           SMAC (StarCraft Multi-Agent Challenge)"
echo "  Map:              $SMAC_MAP"
echo "  SC2 Path:         $SC2PATH"
echo "  Paradigm:         Cooperative (CTDE)"
echo ""
echo "Training Parameters:"
echo "  Algorithm:        MAPPO (Multi-Agent PPO)"
echo "  Total Timesteps:  $TOTAL_TIMESTEPS"
echo "  Parallel Envs:    $NUM_ENVS (each spawns a SC2 process)"
echo "  Seed:             ${SEED:-random}"
echo "  Runner:           RunnerStarCraft2"
echo "  Vectorizer:       Subproc_StarCraft2"
echo ""
echo "Network:"
echo "  Representation:   Basic_RNN (GRU, 64-dim hidden)"
echo "  Parameter Sharing: True (single policy for all agents)"
echo "  Action Masking:   True (unavailable actions masked)"
echo ""
echo "============================================================"

# ============================================================================
# Build MAPPO configuration
# ============================================================================

mkdir -p "$RUN_DIR/config"
MAPPO_CONFIG="$RUN_DIR/config/mappo_smac_${SMAC_MAP}_config.json"

echo "Creating MAPPO configuration..."

jq --argjson steps "$TOTAL_TIMESTEPS" \
   --argjson num_envs "$NUM_ENVS" \
   --arg checkpoint_dir "$RUN_DIR/checkpoints" \
   --arg tensorboard_dir "$RUN_DIR/tensorboard" \
   --arg log_dir "$RUN_DIR/logs" \
   --arg model_dir "$RUN_DIR/models/mappo" \
   --arg smac_map "$SMAC_MAP" \
   --arg seed "${SEED:-null}" '
    # Algorithm selection
    .method = "mappo" |

    # Environment configuration
    .env = "smac" |
    .env_id = $smac_map |

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
# Run MAPPO training
# ============================================================================

echo "Starting MAPPO training on SMAC $SMAC_MAP..."
echo ""
echo "Monitor progress:"
echo "  - TensorBoard: tensorboard --logdir $RUN_DIR/tensorboard"
echo "  - Checkpoints: $RUN_DIR/checkpoints/"
echo ""
echo "Note: Each parallel env spawns a SC2 process (~500MB RAM each)."
echo "      Total SC2 memory: ~$(( NUM_ENVS * 500 ))MB for $NUM_ENVS envs."
echo ""
echo "------------------------------------------------------------"

python -m xuance_worker.cli --config "$MAPPO_CONFIG"

# ============================================================================
# Training complete - Summary
# ============================================================================

echo ""
echo "============================================================"
echo "  MAPPO SMAC $SMAC_MAP Training Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Map:             $SMAC_MAP"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Checkpoints:     $RUN_DIR/checkpoints/"
echo "  TensorBoard:     $RUN_DIR/tensorboard/"
echo "  Config:          $MAPPO_CONFIG"
echo ""
echo "Next Steps:"
echo "  1. Review win rate curves in TensorBoard"
echo "  2. Load checkpoint for evaluation: --test_mode"
echo "  3. Visualize in MOSAIC GUI: select SMAC $SMAC_MAP, load policy"
echo ""
echo "============================================================"
