"""Procedural FourRooms - Different Layout Every Episode.

Demonstrates procedural environment mode: each episode uses a different
seed for env.reset(), producing varied agent and goal positions across
the four interconnected rooms. Tests generalization across layouts.

In RL literature, this corresponds to evaluating over a distribution
of contexts in a Contextual MDP (ProcGen-style evaluation).

Environment: MiniGrid-FourRooms-v0
Mode: Procedural (different seed per episode)

Note: FourRooms uses self.place_obj() and self.place_agent() in
_gen_grid(), so different seeds produce different agent start positions
and goal locations within the four-room structure.

Usage:
    1. Load this script in MOSAIC Operators Tab -> Script Experiments
    2. Click "Run Experiment"
    3. Observe: each episode starts with a different agent/goal placement
"""

operators = [
    {
        "id": "procedural_fourrooms_random",
        "name": "Random Baseline (Procedural FourRooms)",
        "env_name": "minigrid",
        "task": "MiniGrid-FourRooms-v0",
        "max_steps": 200,
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
    "seeds": range(2000, 2010),
    "environment_mode": "procedural",  # Each episode uses a different seed
}
