"""Documentation for PettingZoo SISL Pursuit environment."""
from __future__ import annotations


def get_pursuit_html(env_id: str = "pursuit_v4") -> str:
    """Generate Pursuit environment HTML documentation."""
    return (
        "<h3>PettingZoo SISL: Pursuit (pursuit_v4)</h3>"
        "<p>Pursuers cooperate to catch evaders on a grid. Multiple pursuers must surround "
        "an evader to catch it. Evaders move randomly or according to defined behavior.</p>"
        "<h4>Environment Details</h4>"
        "<ul>"
        "<li><strong>Import:</strong> <code>from pettingzoo.sisl import pursuit_v4</code></li>"
        "<li><strong>Actions:</strong> Discrete</li>"
        "<li><strong>Parallel API:</strong> Yes</li>"
        "<li><strong>Agents:</strong> ['pursuer_0', ..., 'pursuer_7'] (default 8)</li>"
        "<li><strong>Action Shape:</strong> Discrete(5)</li>"
        "<li><strong>Observation Shape:</strong> (7, 7, 3) local view</li>"
        "</ul>"
        "<h4>Action Space</h4>"
        "<p>5 discrete actions: [stay, up, down, left, right]</p>"
        "<h4>Observation Space</h4>"
        "<p>7×7×3 local observation centered on the agent:</p>"
        "<ul>"
        "<li>Channel 0: Obstacles</li>"
        "<li>Channel 1: Pursuers</li>"
        "<li>Channel 2: Evaders</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li><strong>catch_reward:</strong> +5 for catching an evader</li>"
        "<li><strong>urgency_reward:</strong> Small negative reward per step to encourage quick capture</li>"
        "</ul>"
        "<h4>Key Arguments</h4>"
        "<pre><code>pursuit_v4.env(\n"
        "    n_pursuers=8,          # Number of pursuers\n"
        "    n_evaders=30,          # Number of evaders\n"
        "    obs_range=7,           # Observation range (obs_range × obs_range)\n"
        "    n_catch=2,             # Pursuers needed to catch evader\n"
        "    freeze_evaders=False,  # Freeze evaders when caught\n"
        "    catch_reward=5.0,      # Reward for catching\n"
        "    urgency_reward=0.0,    # Per-step reward (usually negative)\n"
        "    max_cycles=500         # Steps per episode\n"
        ")</code></pre>"
        "<h4>Usage</h4>"
        "<pre><code>from pettingzoo.sisl import pursuit_v4\n\n"
        "env = pursuit_v4.env(render_mode='human')\n"
        "env.reset(seed=42)\n\n"
        "for agent in env.agent_iter():\n"
        "    obs, reward, term, trunc, info = env.last()\n"
        "    action = None if term or trunc else env.action_space(agent).sample()\n"
        "    env.step(action)\n"
        "env.close()</code></pre>"
    )


PURSUIT_HTML = get_pursuit_html()

__all__ = ["PURSUIT_HTML", "get_pursuit_html"]
