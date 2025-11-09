"""HTML documentation for the CleanRL Rainbow Atari worker."""

from __future__ import annotations

RAINBOW_HTML = (
    "<h3>Rainbow (Atari)</h3>"
    "<p>The Rainbow implementation (`rainbow_atari.py`) fuses n-step returns, prioritized replay, noisy nets, dueling heads and a categorical distribution (51 atoms in [-10, 10]). Use it for Atari benchmarks after installing the recommended wrappers.</p>"
    "<h4>Quick Start Snippet</h4>"
    "<pre><code>{\n"
    '  "algo": "rainbow_atari",\n'
    '  "env_id": "BreakoutNoFrameskip-v4",\n'
    '  "total_timesteps": 10000000,\n'
    '  "extras": {\n'
    '    "algo_params": {\n'
    '      "learning_rate": 6.25e-5,\n'
    '      "buffer_size": 1000000,\n'
    '      "batch_size": 32,\n'
    '      "target_network_frequency": 8000,\n'
    '      "n_step": 3,\n'
    '      "n_atoms": 51,\n'
    '      "v_min": -10,\n'
    '      "v_max": 10\n'
    '    }\n'
    "  }\n"
    "}</code></pre>"
    "<h4>Key Parameters</h4>"
    "<ul>"
    "<li><strong>epsilon schedule</strong>: anneal from 1.0 to 0.01 over 10% of training.</li>"
    "<li><strong>learning rate</strong>: 6.25e-5, <strong>batch_size</strong>: 32, <strong>train_frequency</strong>: 4.</li>"
    "<li><strong>target_network_frequency</strong>: 8000, <strong>buffer_size</strong>: 1e6, <strong>n_step</strong>: 3.</li>"
    "</ul>"
    "<p>The Henderson et al. paper highlights Atari’s high variance, so fix seeds, log multiple trials, and report mean ± std or percentiles to stay reproducible.</p>"
)

ALGO_DOCS = {
    "rainbow_atari": RAINBOW_HTML,
}

__all__ = ["ALGO_DOCS", "RAINBOW_HTML"]
