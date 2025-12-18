"""HTML documentation blocks for PPO-based CleanRL algorithms."""

from __future__ import annotations

PPO_BASE_HTML = (
    "<h3>PPO (Policy Gradient)</h3>"
    "<p>Proximal Policy Optimization is a trust-region policy gradient method that\n"
    "performs multiple epochs of clipping-updated gradient ascent on batches of\n"
    "advantage estimates. The CleanRL implementation follows the canonical single-file\n"
    "scripts and exposes the most common knobs via extras.</p>"
    "<h4>GUI Quick Start</h4>"
    "<pre><code>{\n"
    '  "algo": "ppo",\n'
    '  "env_id": "CartPole-v1",\n'
    '  "total_timesteps": 1000000,\n'
    '  "seed": 1,\n'
    '  "extras": {\n'
    '    "track_wandb": true,\n'
    '    "tensorboard_dir": "tensorboard",\n'
    '    "algo_params": {\n'
    '      "num_envs": 8,\n'
    '      "num_steps": 2048,\n'
    '      "learning_rate": 0.00025,\n'
    '      "gae_lambda": 0.95,\n'
    '      "clip_coef": 0.2\n'
    "    }\n"
    "  }\n"
    "}\n"
    "</code></pre>"
    "<p>Use the Agent/Worker IDs to tag runs uniquely; WANDB fields are optional but recommended when tracking is enabled.</p>"
    "<h4>Core Hyperparameters</h4>"
    "<ul>"
    "<li><strong>Total Timesteps</strong>: 1e6 for classic control, 2-10e6 for Atari.\n"
    "Timesteps correspond to the number of environment <code>step()</code> calls.</li>"
    "<li><strong>num_envs</strong> (vector environments): 8 (CPU friendly) or 16 for faster data collection.</li>"
    "<li><strong>num_steps</strong> (rollout horizon): 2048 for classic control, 128 for Atari.</li>"
    "<li><strong>learning_rate</strong>: 2.5e-4 with linear decay.</li>"
    "<li><strong>gamma</strong>: 0.99, <strong>gae_lambda</strong>: 0.95.</li>"
    "<li><strong>clip_coef</strong>: 0.2, <strong>ent_coef</strong>: 0.01 (classic control) / 0.0 (Atari).</li>"
    "</ul>"
    "<h4>Recommended Extras</h4>"
    "<ul>"
    "<li><code>algo_params</code> → { 'num_envs': 8, 'num_steps': 2048, 'learning_rate': 0.00025,\n"
    "'gae_lambda': 0.95, 'clip_coef': 0.2 }</li>"
    "<li><code>track_wandb</code> and <code>tensorboard_dir</code> to monitor returns, entropy, KL.</li>"
    "<li><code>notes</code> to capture experiment intent (e.g., sweeping learning rate).</li>"
    "</ul>"
    "<h4>Timesteps & Batch Size</h4>"
    "<p>Each update processes <code>num_envs × num_steps</code> transitions. With 8 envs and 2048 steps,\n"
    "a single update consumes 16,384 transitions. Total timesteps should be multiple of this batch size\n"
    "to avoid a fractional final batch.</p>"
    "<h4>Tips</h4>"
    "<ul>"
    "<li>Increase <code>num_envs</code> when running on powerful CPUs to keep the GPU saturated.</li>"
    "<li>Use <code>max_grad_norm</code>=0.5 (default) and monitor KL divergence to detect instability.</li>"
    "<li>For sparse rewards, add <code>reward_path</code> extras to log intermediate diagnostics.</li>"
    "</ul>"
)

PPO_CONTINUOUS_HTML = (
    "<h3>PPO (Continuous Action)</h3>"
    "<p>Designed for MuJoCo-style continuous control tasks. Uses diagonal Gaussian policy with\n"
    "state-dependent standard deviation.</p>"
    "<ul>"
    "<li><strong>Total Timesteps</strong>: 1e6 – 3e6 depending on task difficulty.</li>"
    "<li><strong>num_envs</strong>: 1 – 8 (continuous environments often have heavier simulation cost).</li>"
    "<li><strong>num_steps</strong>: 2048 (Humanoid/Bipedal) or 1024 (smaller environments).</li>"
    "<li><strong>learning_rate</strong>: 3e-4, <strong>clip_coef</strong>: 0.2, <strong>ent_coef</strong>: 0.0.</li>"
    "<li><strong>gae_lambda</strong>: 0.95, <strong>gamma</strong>: 0.99.</li>"
    "</ul>"
    "<p>Consider enabling <code>normalize_advantage</code> and <code>clip_vloss</code> extras to stabilize updates.</p>"
)

PPO_ATARI_HTML = (
    "<h3>PPO Atari</h3>"
    "<p>Uses frame stacking and Impala-like CNN encoder. Requires env wrappers that produce\n"
    "uint8 observations (CleanRL scripts handle preprocessing automatically).</p>"
    "<ul>"
    "<li><strong>Total Timesteps</strong>: 10e6 – 25e6.</li>"
    "<li><strong>num_envs</strong>: 8 (CPU) or 16 (high-end). <strong>num_steps</strong>: 128.</li>"
    "<li><strong>learning_rate</strong>: 2.5e-4 with linear decay; <strong>anneal_lr</strong> should be True.</li>"
    "<li><strong>clip_coef</strong>: 0.1, <strong>vf_coef</strong>: 0.5, <strong>ent_coef</strong>: 0.01.</li>"
    "<li>Enable <code>max_grad_norm</code>=0.5 and <code>target_kl</code>=0.01 to prevent divergence.</li>"
    "</ul>"
    "<p>Ensure the environment is wrapped with <code>cleanrl_utils.wrappers.AtariWrapper</code> for evaluation.</p>"
)

ALGO_DOCS = {
    "ppo": PPO_BASE_HTML,
    "ppo_continuous_action": PPO_CONTINUOUS_HTML,
    "ppo_atari": PPO_ATARI_HTML,
}

DEFAULT_PPO_DOC = PPO_BASE_HTML

__all__ = [
    "ALGO_DOCS",
    "DEFAULT_PPO_DOC",
    "PPO_BASE_HTML",
    "PPO_CONTINUOUS_HTML",
    "PPO_ATARI_HTML",
]
