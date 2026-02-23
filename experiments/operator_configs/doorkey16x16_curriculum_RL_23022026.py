"""DoorKey 16x16 - Curriculum-Trained PPO Evaluation.

Evaluates the PPO agent trained via 4-phase curriculum learning
(5x5 → 6x6 → 8x8 → 16x16, 4M total steps) on DoorKey-16x16.

Policy: var/trainer/custom_scripts/01KJ3SB6SGRY80BW80W89AY3KQ/checkpoints/final_model.pt
Training script: cleanrl_worker/scripts/curriculum_doorkey_4m.sh

Environment: MiniGrid-DoorKey-16x16-v0
Mode: Procedural (different seed per episode to test generalization)

Usage:
    1. Load this script in MOSAIC Operators Tab -> Script Experiments
    2. Click "Run Experiment"
    3. Observe: trained agent navigating DoorKey-16x16 across 10 layouts
"""

operators = [
    {
        "id": "curriculum_ppo_doorkey16",
        "name": "Curriculum PPO (DoorKey-16x16)",
        "env_name": "minigrid",
        "task": "MiniGrid-DoorKey-16x16-v0",
        "max_steps": 200,
        "workers": {
            "agent": {
                "type": "rl",
                "algorithm": "ppo",
                "policy_path": "var/trainer/custom_scripts/01KJ3SB6SGRY80BW80W89AY3KQ/checkpoints/final_model.pt",
                "seed": 42,
            },
        },
    },
]

execution = {
    "auto_run": False,
    "num_episodes": 10,
    "seeds": list(range(5000, 5010)),
    "environment_mode": "procedural",  # Different layout each episode
}
