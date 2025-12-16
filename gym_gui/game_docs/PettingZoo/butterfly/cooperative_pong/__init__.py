"""Documentation for PettingZoo Butterfly Cooperative Pong environment."""
from __future__ import annotations


def get_cooperative_pong_html(env_id: str = "cooperative_pong_v5") -> str:
    """Generate Cooperative Pong environment HTML documentation."""
    return (
        "<h3>PettingZoo Butterfly: Cooperative Pong (cooperative_pong_v5)</h3>"
        "<p>Two paddles work together to keep a ball in play as long as possible. "
        "The ball bounces elastically off walls and paddles. Game ends when ball goes "
        "out of bounds from left or right edge.</p>"
        "<h4>Environment Details</h4>"
        "<ul>"
        "<li><strong>Import:</strong> <code>from pettingzoo.butterfly import cooperative_pong_v5</code></li>"
        "<li><strong>Actions:</strong> Discrete</li>"
        "<li><strong>Parallel API:</strong> Yes</li>"
        "<li><strong>Manual Control:</strong> Yes (W/S for left, UP/DOWN for right)</li>"
        "<li><strong>Agents:</strong> ['paddle_0', 'paddle_1']</li>"
        "<li><strong>Action Shape:</strong> Discrete(3) - [down, stay, up]</li>"
        "<li><strong>Observation Shape:</strong> (280, 480, 3) - half screen per agent</li>"
        "<li><strong>State Shape:</strong> (560, 960, 3) - full screen</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li><strong>Per timestep:</strong> max_reward / max_cycles (default: 0.11)</li>"
        "<li><strong>Ball out of bounds:</strong> off_screen_penalty (default: -10)</li>"
        "</ul>"
        "<h4>Key Arguments</h4>"
        "<pre><code>cooperative_pong_v5.env(\n"
        "    ball_speed=9,\n"
        "    left_paddle_speed=12,\n"
        "    right_paddle_speed=12,\n"
        "    cake_paddle=True,       # Right paddle is tiered cake shape\n"
        "    max_cycles=900,\n"
        "    bounce_randomness=False, # Random angle on paddle collision\n"
        "    max_reward=100,\n"
        "    off_screen_penalty=-10\n"
        ")</code></pre>"
        "<h4>Usage</h4>"
        "<pre><code>from pettingzoo.butterfly import cooperative_pong_v5\n\n"
        "env = cooperative_pong_v5.env(render_mode='human')\n"
        "env.reset(seed=42)\n\n"
        "for agent in env.agent_iter():\n"
        "    obs, reward, term, trunc, info = env.last()\n"
        "    action = None if term or trunc else env.action_space(agent).sample()\n"
        "    env.step(action)\n"
        "env.close()</code></pre>"
    )


COOPERATIVE_PONG_HTML = get_cooperative_pong_html()

__all__ = ["COOPERATIVE_PONG_HTML", "get_cooperative_pong_html"]
