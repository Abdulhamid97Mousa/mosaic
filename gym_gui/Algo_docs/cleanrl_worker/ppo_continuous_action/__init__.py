"""HTML documentation for the CleanRL PPO continuous-action worker."""

from __future__ import annotations

PPO_CONTINUOUS_ACTION_HTML = (
    "<h3>PPO Continuous Action</h3>"
    "<p>This preset runs the CleanRL <code>ppo_continuous_action.py</code> script against MuJoCo-style tasks.</p>"
    "<p>Using diagonal Gaussian policies, the worker is tuned for benchmark locomotion tasks such as Walker2d.</p>"
    "<h4>Quick Starter</h4>"
    "<pre><code>{\n"
    '  "env_id": "Walker2d-v5",\n'
    '  "algo_params": {\n'
    '    "learning_rate": 3e-4,\n'
    '    "num_envs": 1,\n'
    '    "num_steps": 2048,\n'
    '    "clip_vloss": true,\n'
    '    "capture_video": true\n'
    '  }\n'
    "}</code></pre>"
    "<p>MuJoCo environments require the <code>gymnasium[mujoco]</code> extras; install them before launching the worker.</p>"
    "<h4>Stable Options</h4>"
    "<ul>"
    "<li><strong>gamma</strong>: 0.99, <strong>gae_lambda</strong>: 0.95 for smooth advantage estimation.</li>"
    "<li><strong>norm_adv</strong>: true to keep advantages stable.</li>"
    "<li>Capture videos on the first env when experimenting with Walker2d reward shaping.</li>"
    "</ul>"
)

ALGO_DOCS = {
    "ppo_continuous_action": PPO_CONTINUOUS_ACTION_HTML,
}

__all__ = ["ALGO_DOCS", "PPO_CONTINUOUS_ACTION_HTML"]
