"""HTML documentation for the CleanRL SAC worker."""

from __future__ import annotations

SAC_HTML = (
    "<h3>SAC (Soft Actor-Critic)</h3>"
    "<p>CleanRL’s `sac_continuous_action.py` maximizes entropy as well as reward, pairing stochastic actors with twin Q-networks and automatic entropy tuning.</p>"
    "<h4>Quick Start Snippet</h4>"
    "<pre><code>{\n"
    '  "algo": "sac_continuous_action",\n'
    '  "env_id": "Walker2d-v5",\n'
    '  "extras": {\n'
    '    "algo_params": {\n'
    '      "total_timesteps": 3000000,\n'
    '      "policy_lr": 3e-4,\n'
    '      "q_lr": 1e-3,\n'
    '      "batch_size": 256,\n'
    '      "buffer_size": 1000000,\n'
    '      "tau": 0.005,\n'
    '      "alpha": 0.2,\n'
    '      "autotune": true\n'
    '    }\n'
    "  }\n"
    "}</code></pre>"
    "<h4>Key Parameters</h4>"
    "<ul>"
    "<li><strong>policy_lr</strong>: 3e-4, <strong>q_lr</strong>: 1e-3, <strong>gamma</strong>: 0.99.</li>"
    "<li><strong>batch_size</strong>: 256, <strong>buffer_size</strong>: 1e6, <strong>learning_starts</strong>: 5k.</li>"
    "<li><strong>tau</strong>: 0.005, <strong>policy_frequency</strong>: 2, <strong>target_network_frequency</strong>: 1.</li>"
    "</ul>"
    "<p>Follow the Henderson et al. reproducibility checklist (seed averaging, consistent evaluation curves, careful reward rescaling) since SAC’s soft updates can mask misconfigured hyperparameters.</p>"
)

ALGO_DOCS = {
    "sac_continuous_action": SAC_HTML,
}

__all__ = ["ALGO_DOCS", "SAC_HTML"]
