"""Documentation for PettingZoo MPE Simple environment."""
from __future__ import annotations


def get_simple_html(env_id: str = "simple_v3") -> str:
    """Generate Simple environment HTML documentation."""
    return (
        "<h3>PettingZoo MPE: Simple (simple_v3)</h3>"
        "<p>A single agent navigates to a landmark. This is primarily intended for debugging "
        "and is not a true multiagent environment.</p>"
        "<h4>Environment Details</h4>"
        "<ul>"
        "<li><strong>Import:</strong> <code>from pettingzoo.mpe import simple_v3</code></li>"
        "<li><strong>Actions:</strong> Discrete/Continuous</li>"
        "<li><strong>Parallel API:</strong> Yes</li>"
        "<li><strong>Agents:</strong> ['agent_0']</li>"
        "<li><strong>Action Shape:</strong> (5,)</li>"
        "<li><strong>Observation Shape:</strong> (4,)</li>"
        "</ul>"
        "<h4>Observation Space</h4>"
        "<p><code>[self_vel, landmark_rel_position]</code></p>"
        "<h4>Rewards</h4>"
        "<p>Agent is rewarded based on Euclidean distance to the landmark.</p>"
        "<h4>Arguments</h4>"
        "<pre><code>simple_v3.env(max_cycles=25, continuous_actions=False, dynamic_rescaling=False)</code></pre>"
        "<ul>"
        "<li><strong>max_cycles:</strong> Steps until game terminates</li>"
        "<li><strong>continuous_actions:</strong> Use continuous action space</li>"
        "<li><strong>dynamic_rescaling:</strong> Rescale agents/landmarks based on screen size</li>"
        "</ul>"
        "<h4>Usage</h4>"
        "<pre><code>from pettingzoo.mpe import simple_v3\n\n"
        "env = simple_v3.env(render_mode='human')\n"
        "env.reset(seed=42)\n\n"
        "for agent in env.agent_iter():\n"
        "    obs, reward, term, trunc, info = env.last()\n"
        "    action = None if term or trunc else env.action_space(agent).sample()\n"
        "    env.step(action)\n"
        "env.close()</code></pre>"
    )


SIMPLE_HTML = get_simple_html()

__all__ = ["SIMPLE_HTML", "get_simple_html"]
