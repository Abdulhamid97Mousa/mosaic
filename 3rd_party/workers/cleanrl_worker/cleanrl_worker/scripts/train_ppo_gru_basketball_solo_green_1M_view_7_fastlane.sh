#!/bin/bash
# FASTLANE: Quick training for Basketball Solo Green Agent
# 1M timesteps for rapid testing and validation
# With invalid action masking to speed up learning

cd "$(dirname "$0")/.."

python -m cleanrl_worker.algorithms.ppo_gru \
    --env-id "MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v0" \
    --view-size 7 \
    --max-episode-steps 200 \
    --exp-name "basketball_solo_green_ppo_gru_fastlane" \
    --seed 1 \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --num-steps 300 \
    --learning-rate 1e-4 \
    --gru-hidden-size 128 \
    --gru-num-layers 1 \
    --mask-invalid-actions \
    --save-model
