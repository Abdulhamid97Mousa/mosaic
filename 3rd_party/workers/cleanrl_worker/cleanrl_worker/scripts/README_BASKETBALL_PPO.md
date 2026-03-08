# PPO-GRU Training for MOSAIC MultiGrid Basketball

This directory contains training scripts for PPO with GRU (recurrent policy) on MOSAIC MultiGrid Basketball solo environments.

## Overview

- **Algorithm**: PPO (Proximal Policy Optimization) with GRU for recurrent processing
- **Environments**: Basketball Solo Blue and Green agents
- **View Size**: 7 (agents see 7×7 grid around themselves)
- **Observation**: Flattened vector (7×7×3 + 4 = 151 dimensions)

## Files

- `ppo_gru_mosaic_multigrid.py` - Main PPO-GRU training script
- `train_basketball_blue.sh` - Launch training for blue agent
- `train_basketball_green.sh` - Launch training for green agent

## Quick Start

### Train Blue Agent
```bash
cd /home/hamid/Desktop/software/mosaic/3rd_party/workers/cleanrl_worker/cleanrl_worker/scripts
./train_basketball_blue.sh
```

### Train Green Agent
```bash
cd /home/hamid/Desktop/software/mosaic/3rd_party/workers/cleanrl_worker/cleanrl_worker/scripts
./train_basketball_green.sh
```

## Training Configuration

Default hyperparameters:
- **Total timesteps**: 10M
- **Parallel environments**: 8
- **Steps per rollout**: 128
- **Learning rate**: 2.5e-4 (with annealing)
- **GRU hidden size**: 128
- **GRU layers**: 1
- **Minibatches**: 4
- **Update epochs**: 4
- **Gamma**: 0.99
- **GAE lambda**: 0.95
- **Clip coefficient**: 0.1
- **Entropy coefficient**: 0.01
- **Value function coefficient**: 0.5

## Customization

You can customize training by modifying the shell scripts or running the Python script directly:

```bash
python ppo_gru_mosaic_multigrid.py \
    --env-id "MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0" \
    --view-size 7 \
    --total-timesteps 20000000 \
    --num-envs 16 \
    --learning-rate 3e-4 \
    --gru-hidden-size 256 \
    --seed 42
```

## Monitoring

Training metrics are logged to:
- **TensorBoard**: `runs/` directory
- **Weights & Biases**: If `--track` flag is enabled (requires wandb login)

View TensorBoard logs:
```bash
tensorboard --logdir runs/
```

## Checkpoints

Models are saved to:
- Blue agent: `checkpoints/basketball_solo_blue/`
- Green agent: `checkpoints/basketball_solo_green/`

Checkpoints are saved:
- Every 1M steps
- At the end of training (final model)

## Key Differences from MAPPO

This is **PPO** (single-agent), not **MAPPO** (multi-agent):
- Each agent trains independently in its solo environment
- No centralized critic or parameter sharing
- Simpler to debug and understand
- Good baseline to compare against MAPPO performance

## Architecture

**Network Structure**:
```
Input (151-dim flattened observation)
  ↓
MLP Encoder (256 → 256)
  ↓
GRU (256 → 128)
  ↓
Actor Head (128 → num_actions)
Critic Head (128 → 1)
```

**Why GRU?**
- Handles partial observability (agent only sees 7×7 window)
- Maintains memory of past observations
- Better than feedforward for sequential decision-making
- Simpler than LSTM (fewer parameters)

## Troubleshooting

### Environment not found
If you get `NameNotFound` error, the MOSAIC MultiGrid environments might not be registered. Make sure you're running from the correct Python environment:
```bash
source /home/hamid/Desktop/software/mosaic/.venv/bin/activate
```

### CUDA out of memory
Reduce `--num-envs` or `--gru-hidden-size`:
```bash
--num-envs 4 --gru-hidden-size 64
```

### Slow training
Increase `--num-envs` for more parallelism (if you have enough CPU/GPU):
```bash
--num-envs 16
```

## Expected Results

Training 10M steps should take approximately:
- **With GPU**: 2-4 hours
- **CPU only**: 8-12 hours

Performance metrics to watch:
- **Episodic return**: Should increase over time
- **Episodic length**: Should stabilize
- **Policy loss**: Should decrease initially
- **Value loss**: Should decrease and stabilize
- **Explained variance**: Should approach 1.0

## Next Steps

After training:
1. Evaluate trained agents in the GUI
2. Compare PPO vs MAPPO performance
3. Test blue vs green agent matchups
4. Experiment with different hyperparameters
