"""HTML documentation for the CleanRL DDPG worker."""

from __future__ import annotations

DDPG_HTML = (
    "<h3>DDPG (Deterministic Actor-Critic)</h3>"
    "<p>CleanRL's `ddpg_continuous_action.py` trains a deterministic policy with an actor, twin critics,"
    "replay buffer, and target smoothing (`tau` & `policy_frequency`). It is best suited for MuJoCo-style"
    "control (Walker2d, Hopper) where continuous actions and low-latency exploration matter.</p>"
    "<h4>Quick Start (JSON snippet)</h4>"
    "<pre><code>{\n"
    '  "algo": "ddpg_continuous_action",\n'
    '  "env_id": "Walker2d-v5",\n'
    '  "total_timesteps": 1000000,\n'
    '  "extras": {\n'
    '    "algo_params": {\n'
    '      "learning_rate": 3e-4,\n'
    '      "batch_size": 256,\n'
    '      "buffer_size": 1000000,\n'
    '      "exploration_noise": 0.1,\n'
    '      "learning_starts": 25000,\n'
    '      "policy_frequency": 2,\n'
    '      "tau": 0.005\n'
    '    }\n'
    "  }\n"
    "}</code></pre>"
    "<h4>Key Hyperparameters</h4>"
    "<ul>"
    "<li><strong>total_timesteps</strong>: 1e6 (Hopper/Walker) up to 3e6 for harder MuJoCo tasks.</li>"
    "<li><strong>learning_rate</strong>: 3e-4 with Adam for both actor and critic.</li>"
    "<li><strong>batch_size</strong>: 256, <strong>buffer_size</strong>: 1e6, <strong>gamma</strong>: 0.99.</li>"
    "<li><strong>exploration_noise</strong>: 0.1 (scale via action bounds) plus <strong>reward scaling</strong> of 0.1–10 to keep gradients stable.</li>"
    "<li><strong>tau</strong>: 0.005 smoothing; <strong>policy_frequency</strong>: 2 for delayed actor updates.</li>"
    "</ul>"
    "<p>The CleanRL script uses 2×256 ReLU layers for actor and critic; the documentation at "
    "the CleanRL documentation stresses this MLP size and reward-rescaling experiments "
    "(see the Reward Scale section). Layer norm can change the effective scale, so adjust `exploration_noise` "
    "and reward clipping accordingly.</p>"
    "<h4>Stability Tips</h4>"
    "<ul>"
    "<li>Warm up the replay buffer (`learning_starts` ≥ 25k) before taking gradient steps.</li>"
    "<li>Log `charts/episodic_return` + `charts/episodic_length` (CleanRL already writes them) and use WandB/TensorBoard.</li>"
    "<li>Hide noise during deterministic evaluation (the paper _Deep Reinforcement Learning that Matters_ urges careful seed averaging).</li>"
    "</ul>"
    "<p>Refer to the Addendum (`Deep_Reinforcement_Learning_that_Matters.pdf`) for the reproducibility checklist and the sensitivity of DDPG to hyperparameters and random seeds.</p>"
)

ALGO_DOCS = {
    "ddpg_continuous_action": DDPG_HTML,
}

__all__ = ["ALGO_DOCS", "DDPG_HTML"]
