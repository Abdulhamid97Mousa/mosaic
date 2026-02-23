"""HTML documentation blocks for DQN-based CleanRL algorithms."""

from __future__ import annotations

DQN_BASE_HTML = (
    "<h3>DQN (Value-Based)</h3>"
    "<p>Deep Q-Network learns an action-value function via bootstrapped TD updates and experience replay.</p>"
    "<h4>Core Hyperparameters</h4>"
    "<ul>"
    "<li><strong>Total Timesteps</strong>: 500k (classic control), 5–10M (Atari).</li>"
    "<li><strong>buffer_size</strong>: 100_000 – memory of past transitions.</li>"
    "<li><strong>batch_size</strong>: 128 (CPU) or 256 (GPU).</li>"
    "<li><strong>learning_rate</strong>: 1e-4 with Adam optimizer.</li>"
    "<li><strong>gamma</strong>: 0.99, <strong>train_frequency</strong>: 1, <strong>gradient_steps</strong>: 1.</li>"
    "<li><strong>target_network_frequency</strong>: 10_000 (updates target network every N steps).</li>"
    "</ul>"
    "<h4>Exploration Schedule</h4>"
    "<ul>"
    "<li><code>exploration_fraction</code>: 0.1 – portion of training spent linearly decaying epsilon.</li>"
    "<li><code>exploration_initial_eps</code>: 1.0, <code>exploration_final_eps</code>: 0.01 (Atari) / 0.05 (classic control).</li>"
    "</ul>"
    "<h4>Recommended Extras</h4>"
    "<ul>"
    "<li><code>algo_params</code> → { 'buffer_size': 100000, 'batch_size': 128, 'target_network_frequency': 10000 }.</li>"
    "<li><code>learning_starts</code>: 10_000 — delay training until buffer warmup.</li>"
    "<li><code>max_grad_norm</code>: 10.0, <code>clip_reward</code>=True for Atari.</li>"
    "</ul>"
    "<p>Enable <code>track_wandb</code> or <code>tensorboard_dir</code> to monitor Q-value histograms and epsilon.</p>"
)

C51_HTML = (
    "<h3>C51 (Distributional DQN)</h3>"
    "<p>Extends DQN by modeling a categorical distribution over returns (51 atoms). Requires additional parameters.</p>"
    "<ul>"
    "<li><strong>total_timesteps</strong>: 10M for Atari.</li>"
    "<li><strong>support_lower</strong>: -10, <strong>support_upper</strong>: 10, <strong>support_size</strong>: 51.</li>"
    "<li><strong>learning_rate</strong>: 1e-4, <strong>batch_size</strong>: 128, <strong>buffer_size</strong>: 100_000.</li>"
    "<li>Maintain <code>target_network_frequency</code>: 10_000 and <code>n_step_returns</code>: 3.</li>"
    "</ul>"
    "<p>Set extras accordingly: { 'support_lower': -10, 'support_upper': 10, 'support_size': 51, 'n_step_returns': 3 }.</p>"
)

ALGO_DOCS = {
    "dqn": DQN_BASE_HTML,
    "c51": C51_HTML,
}

DEFAULT_DQN_DOC = DQN_BASE_HTML

__all__ = [
    "ALGO_DOCS",
    "DEFAULT_DQN_DOC",
    "DQN_BASE_HTML",
    "C51_HTML",
]
