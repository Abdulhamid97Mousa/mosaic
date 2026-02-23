# CleanRL Custom Training Scripts

This directory contains custom training scripts for advanced training scenarios like **curriculum learning**, **multi-phase training**, and **environment progression**.

## Why Custom Scripts?

The standard CleanRL form trains on a single environment for the entire run. Custom scripts allow:

- **Curriculum Learning**: Progress through easier to harder environments
- **Multi-Phase Training**: Different hyperparameters per phase
- **Checkpoint Resume**: Continue training from previous phases
- **Custom Schedules**: Any training pattern you can script

## Environment Variables

MOSAIC passes these environment variables to your script:

| Variable | Description | Example |
|----------|-------------|---------|
| `MOSAIC_CONFIG_FILE` | Path to base JSON config | `/tmp/mosaic_config_abc123.json` |
| `MOSAIC_RUN_ID` | Unique run identifier | `cleanrl-ppo-20260201-143022` |
| `MOSAIC_CHECKPOINT_DIR` | Directory for checkpoints | `var/trainer/runs/{run_id}/checkpoints` |
| `CLEANRL_AGENT_ID` | Agent ID for multi-agent | `1` |
| `CLEANRL_NUM_ENVS` | Number of parallel envs | `4` |
| `GYM_GUI_FASTLANE_ENABLED` | FastLane streaming enabled | `1` |
| `GYM_GUI_FASTLANE_VIDEO_MODE` | Video mode (single/grid/off) | `grid` |
| `GYM_GUI_FASTLANE_GRID_LIMIT` | Max envs in grid | `4` |
| `TRACK_TENSORBOARD` | TensorBoard logging | `1` |
| `TRACK_WANDB` | WandB logging | `0` |

## Base Config Structure

The `$MOSAIC_CONFIG_FILE` contains the full worker config:

```json
{
  "algo": "ppo",
  "env_id": "MiniGrid-DoorKey-6x6-v0",
  "seed": 1,
  "total_timesteps": 500000,
  "worker_id": "1",
  "extras": {
    "cuda": true,
    "algo_params": {
      "learning_rate": 0.00025,
      "num_envs": 4,
      "num_steps": 128,
      "gamma": 0.99,
      ...
    },
    "fastlane_video_mode": "grid",
    "tensorboard_dir": "tensorboard"
  }
}
```

## Writing a Script

### Basic Template

```bash
#!/bin/bash
# my_curriculum.sh - Description of what this does
#
# @description: Brief description for the UI
# @env_family: minigrid
# @phases: 3

set -e  # Exit on error

# Read base config
CONFIG="$MOSAIC_CONFIG_FILE"
CHECKPOINT_DIR="$MOSAIC_CHECKPOINT_DIR"

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

echo "=== Phase 1: Easy ==="
# Modify config for phase 1
jq '.env_id = "MiniGrid-DoorKey-5x5-v0" | .total_timesteps = 200000' \
    "$CONFIG" > /tmp/phase1.json

python -m cleanrl_worker.cli --config /tmp/phase1.json
cp -r "var/trainer/runs/$MOSAIC_RUN_ID/checkpoints/"* "$CHECKPOINT_DIR/phase1/" 2>/dev/null || true

echo "=== Phase 2: Medium ==="
jq '.env_id = "MiniGrid-DoorKey-8x8-v0" | .total_timesteps = 200000' \
    "$CONFIG" > /tmp/phase2.json

python -m cleanrl_worker.cli --config /tmp/phase2.json

echo "=== Training Complete ==="
```

### Using jq to Modify Config

```bash
# Change environment
jq '.env_id = "NewEnv-v0"' "$CONFIG" > /tmp/modified.json

# Change timesteps
jq '.total_timesteps = 100000' "$CONFIG" > /tmp/modified.json

# Change learning rate
jq '.extras.algo_params.learning_rate = 0.0001' "$CONFIG" > /tmp/modified.json

# Multiple changes
jq '.env_id = "NewEnv-v0" | .total_timesteps = 100000 | .seed = 42' \
    "$CONFIG" > /tmp/modified.json
```

### Reading Config Values

```bash
# Get current seed
SEED=$(jq -r '.seed' "$CONFIG")

# Get learning rate
LR=$(jq -r '.extras.algo_params.learning_rate' "$CONFIG")

# Get total timesteps
STEPS=$(jq -r '.total_timesteps' "$CONFIG")

echo "Training with seed=$SEED, lr=$LR, steps=$STEPS"
```

## Script Metadata

Add metadata comments at the top of your script for the UI:

```bash
#!/bin/bash
# curriculum_doorkey.sh
#
# @description: DoorKey curriculum (5x5 → 8x8 → 16x16)
# @env_family: minigrid
# @phases: 3
# @total_timesteps: 600000
# @author: Your Name
```

The UI will parse these to display script information.

## FastLane Integration

FastLane environment variables are automatically passed to your script. Each phase of training will stream frames to the GUI if FastLane is enabled.

The script doesn't need to do anything special - just run `python -m cleanrl_worker.cli` and FastLane will work automatically.

## Examples

See the example scripts in this directory:

- `curriculum_doorkey.sh` - DoorKey progression (5x5 → 8x8 → 16x16)
- `curriculum_babyai_goto.sh` - BabyAI GoTo progression
- `curriculum_dynamic_obstacles.sh` - Dynamic obstacles with increasing difficulty

## Debugging

Run your script manually to test:

```bash
# Set required environment variables
export MOSAIC_CONFIG_FILE="/tmp/test_config.json"
export MOSAIC_RUN_ID="test-run-001"
export MOSAIC_CHECKPOINT_DIR="/tmp/checkpoints"

# Create a test config
echo '{"algo":"ppo","env_id":"CartPole-v1","seed":1,"total_timesteps":10000}' > /tmp/test_config.json

# Run your script
bash scripts/my_curriculum.sh
```
