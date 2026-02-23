"""HTML documentation for the CleanRL PPG (Procgen) worker."""

from __future__ import annotations

PPG_HTML = (
    "<h3>PPG (Phasic Policy Gradient)</h3>"
    "<p>`ppg_procgen.py` trains IMPALA-style encoders with auxiliary value replay, mirroring the PPG paper and the Procgen benchmark.</p>"
    "<h4>Quick Start Snippet</h4>"
    "<pre><code>{\n"
    '  "algo": "ppg_procgen",\n'
    '  "env_id": "starpilot",\n'
    '  "total_timesteps": 25000000,\n'
    '  "extras": {\n'
    '    "algo_params": {\n'
    '      "learning_rate": 5e-4,\n'
    '      "num_envs": 64,\n'
    '      "num_steps": 256,\n'
    '      "num_minibatches": 8,\n'
    '      "clip_coef": 0.2,\n'
    '      "ent_coef": 0.01,\n'
    '      "vf_coef": 0.5,\n'
    '      "n_iteration": 32,\n'
    '      "e_auxiliary": 6,\n'
    '      "beta_clone": 1.0\n'
    '    }\n'
    "  }\n"
    "}</code></pre>"
    "<h4>Key Parameters</h4>"
    "<ul>"
    "<li><strong>total_timesteps</strong>: 25e6, <strong>gamma</strong>: 0.999, <strong>gae_lambda</strong>: 0.95.</li>"
    "<li><strong>num_envs</strong>: 64, <strong>num_steps</strong>: 256, <strong>num_minibatches</strong>: 8.</li>"
    "<li>Auxiliary loop: `n_iteration=32`, `e_policy=1`, `v_value=1`, `e_auxiliary=6`, `beta_clone=1.0`, `num_aux_rollouts=4`.</li>"
    "</ul>"
    "<p>The Henderson et al. PDF reminds us that Procgen (and any high-variance domain) requires reporting multiple seeds as well as standardized architectures; this matches the `ppg_procgen.py` emphasis on seed determinism and reward clipping.</p>"
)

ALGO_DOCS = {
    "ppg_procgen": PPG_HTML,
}

__all__ = ["ALGO_DOCS", "PPG_HTML"]
