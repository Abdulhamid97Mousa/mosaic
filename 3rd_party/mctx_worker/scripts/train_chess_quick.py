#!/usr/bin/env python
"""Quick chess training script that saves policies to var/trainer/runs/.

This script runs a short training session to verify the mctx_worker
produces policies in the correct location.

Usage:
    python 3rd_party/mctx_worker/scripts/train_chess_quick.py

Output:
    var/trainer/runs/chess_quick_{timestamp}/
        policies/
            policy_iter_10.pkl
            policy_iter_20.pkl
            ...
            policy_final.pkl
        tensorboard/
        analytics.json
"""

import sys
from datetime import datetime
from pathlib import Path

# Add mctx_worker to path
mctx_worker_path = Path(__file__).parent.parent
sys.path.insert(0, str(mctx_worker_path))

from mctx_worker.config import (
    MCTXWorkerConfig,
    MCTXAlgorithm,
    MCTSConfig,
    NetworkConfig,
    TrainingConfig,
)
from mctx_worker.runtime import MCTXWorkerRuntime


def main():
    """Run quick chess training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"chess_quick_{timestamp}"

    print("=" * 60)
    print("MCTX Chess Training - Quick Test")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Output: var/trainer/runs/{run_id}/")
    print()

    # Create configuration for quick training
    config = MCTXWorkerConfig(
        run_id=run_id,
        seed=42,
        env_id="chess",
        algorithm=MCTXAlgorithm.GUMBEL_MUZERO,
        max_steps=50,  # Quick training: 50 iterations
        device="gpu",

        # Smaller network for faster training
        network=NetworkConfig(
            channels=32,        # Smaller (default 128)
            num_res_blocks=2,   # Fewer blocks (default 8)
            hidden_dims=(128, 128),
        ),

        # Faster MCTS
        mcts=MCTSConfig(
            num_simulations=50,  # Fewer sims (default 800)
            max_num_considered_actions=8,
        ),

        # Training settings
        training=TrainingConfig(
            batch_size=64,
            games_per_iteration=32,
            checkpoint_interval=10,  # Save every 10 iterations
            learning_rate=1e-3,
        ),

        logging_interval=10,
        verbose=2,
    )

    print("Configuration:")
    print(f"  Network: {config.network.channels} channels, {config.network.num_res_blocks} blocks")
    print(f"  MCTS: {config.mcts.num_simulations} simulations")
    print(f"  Training: {config.training.batch_size} batch, {config.training.games_per_iteration} games/iter")
    print(f"  Max steps: {config.max_steps}")
    print()

    # Create and run runtime
    runtime = MCTXWorkerRuntime(config)

    try:
        runtime.run()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)

        # Show output location
        output_dir = Path(f"var/trainer/runs/{run_id}")
        if output_dir.exists():
            print(f"\nOutput saved to: {output_dir}")
            print("\nContents:")
            for item in sorted(output_dir.rglob("*")):
                if item.is_file():
                    rel_path = item.relative_to(output_dir)
                    size_kb = item.stat().st_size / 1024
                    print(f"  {rel_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
