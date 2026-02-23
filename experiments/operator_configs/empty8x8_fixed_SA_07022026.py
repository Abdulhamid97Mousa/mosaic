"""Fixed Layout Empty Grid - Same Layout Every Episode.

Demonstrates fixed environment mode: all episodes use the same seed for
env.reset(), producing identical grid layouts. This isolates agent
behavior variance from environment variation.

In RL literature, this corresponds to a Contextual MDP with a single
fixed context (Cobbe et al., ICML 2020; Kirk et al., 2022).

Environment: MiniGrid-Empty-8x8-v0
Mode: Fixed (same seed every reset)

Usage:
    1. Load this script in MOSAIC Operators Tab -> Script Experiments
    2. Click "Run Experiment"
    3. Observe: every episode starts with the identical grid layout
"""

operators = [
    {
        "id": "fixed_empty_random",
        "name": "Random Baseline (Fixed Empty Grid)",
        "env_name": "minigrid",
        "task": "MiniGrid-Empty-8x8-v0",
        "max_steps": 100,
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
    "seeds": range(1000, 1010),
    "environment_mode": "fixed",  # All episodes use seeds[0] for env.reset
}
