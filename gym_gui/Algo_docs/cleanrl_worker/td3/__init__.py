"""HTML documentation for the CleanRL TD3 worker."""

from __future__ import annotations

TD3_HTML = (
    "<h3>TD3 (Twin Delayed DDPG)</h3>"
    "<p>CleanRL’s `td3_continuous_action.py` trains twin critics with delayed policy updates and clipped action noise, making it a robust deterministic baseline for MuJoCo locomotion (Walker2d, Hopper).</p>"
    "<h4>Quick Start Snippet</h4>"
    "<pre><code>{\n"
    '  "algo": "td3_continuous_action",\n'
    '  "env_id": "Walker2d-v5",\n'
    '  "extras": {\n'
    '    "algo_params": {\n'
    '      "total_timesteps": 3000000,\n'
    '      "learning_rate": 3e-4,\n'
    '      "batch_size": 256,\n'
    '      "buffer_size": 1000000,\n'
    '      "tau": 0.005,\n'
    '      "exploration_noise": 0.1,\n'
    '      "policy_noise": 0.2,\n'
    '      "noise_clip": 0.5,\n'
    '      "policy_frequency": 2\n'
    '    }\n'
    "  }\n"
    "}</code></pre>"
    "<h4>Key Parameters</h4>"
    "<ul>"
    "<li><strong>total_timesteps</strong>: 1e6 – 3e6; deterministic policies need plenty of samples.</li>"
    "<li><strong>learning_rate</strong>: 3e-4, <strong>batch_size</strong>: 256, <strong>buffer_size</strong>: 1e6.</li>"
    "<li><strong>gamma</strong>: 0.99, <strong>tau</strong>: 0.005, <strong>learning_starts</strong>: 25k.</li>"
    "<li><strong>noise</strong>: `policy_noise=0.2`, `noise_clip=0.5`, `exploration_noise=0.1`.</li>"
    "</ul>"
    "<p>The documentation emphasizes the `Deep Reinforcement Learning that Matters` results (stable `(64, 64)` nets, seed averaging, and reward scaling) because TD3’s deterministic actor is sensitive to random seeds. Track mean ± std over multiple seeds.</p>"
)

ALGO_DOCS = {
    "td3_continuous_action": TD3_HTML,
}

__all__ = ["ALGO_DOCS", "TD3_HTML"]
