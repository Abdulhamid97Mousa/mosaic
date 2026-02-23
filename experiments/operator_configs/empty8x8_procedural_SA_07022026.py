"""Simple Random Baseline Test Script.

A minimal example for testing the baseline operator system.
Uses MiniGrid-Empty-8x8-v0 - a simple single-agent gridworld.

Usage:
    1. Load this script in MOSAIC Operators Tab → Script Mode
    2. Click "Validate" to check syntax
    3. Click "Apply & Configure" to launch operator
"""

operators = [
    {
        "id": "random_minigrid",
        "name": "Random Baseline (MiniGrid Empty)",
        "env_name": "minigrid",
        "task": "MiniGrid-Empty-8x8-v0",
        "max_steps": 100,  # Truncate after 100 steps to prevent infinite wandering
        "workers": {
            "agent": {
                "type": "baseline",
                "behavior": "random",
                "seed": 42,
            },
        }
    },
]

# Optional: Enable auto-execution
execution = {
    "auto_run": False,  # Set to True to run automatically
    "num_episodes": 10,
    "seeds": range(1000, 1010),
}
