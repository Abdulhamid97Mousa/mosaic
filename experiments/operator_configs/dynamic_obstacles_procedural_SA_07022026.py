"""Procedural Dynamic Obstacles - Varying Obstacle Patterns.

Demonstrates procedural mode with dynamic obstacles: each episode
generates a different obstacle configuration, testing agent robustness
to varying environment layouts with moving hazards.

In RL literature, this is procedural content generation (PCG) where the
environment structure changes each episode (Cobbe et al., ICML 2020).

Environment: MiniGrid-Dynamic-Obstacles-8x8-v0
Mode: Procedural (different seed per episode)

Note: Dynamic-Obstacles uses place_obj() for obstacle Ball placement
in _gen_grid(), so different seeds produce different obstacle positions.

Usage:
    1. Load this script in MOSAIC Operators Tab -> Script Experiments
    2. Click "Run Experiment"
    3. Observe: each episode has different obstacle placements
"""

operators = [
    {
        "id": "procedural_dynobs_random",
        "name": "Random Baseline (Procedural Dynamic Obstacles)",
        "env_name": "minigrid",
        "task": "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "max_steps": 150,
        "workers": {
            "agent": {
                "type": "baseline",
                "behavior": "random",
                "seed": 42,
            },
        }
    },
]

execution = {
    "auto_run": False,
    "num_episodes": 10,
    "seeds": range(3000, 3010),
    "environment_mode": "procedural",  # Each episode uses a different seed
}
