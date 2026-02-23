"""Documentation for PettingZoo MPE Simple Adversary environment."""
from __future__ import annotations


def get_simple_adversary_html(env_id: str = "simple_adversary_v3") -> str:
    """Generate Simple Adversary environment HTML documentation."""
    return (
        "<h3>PettingZoo MPE: Simple Adversary (simple_adversary_v3)</h3>"
        "<p>1 adversary (red) vs N good agents (green) with N landmarks. One landmark is the "
        "'target' (green). Good agents are rewarded for proximity to target but penalized if "
        "the adversary gets close. The adversary must find the target without knowing which "
        "landmark it is.</p>"
        "<h4>Strategy</h4>"
        "<p>Good agents must 'split up' to cover all landmarks to deceive the adversary about "
        "which landmark is the target.</p>"
        "<h4>Environment Details</h4>"
        "<ul>"
        "<li><strong>Import:</strong> <code>from pettingzoo.mpe import simple_adversary_v3</code></li>"
        "<li><strong>Actions:</strong> Discrete/Continuous</li>"
        "<li><strong>Parallel API:</strong> Yes</li>"
        "<li><strong>Agents:</strong> ['adversary_0', 'agent_0', 'agent_1'] (N=2)</li>"
        "<li><strong>Action Shape:</strong> (5,)</li>"
        "<li><strong>Observation Shape:</strong> (8,) for agents, (10,) for adversary</li>"
        "</ul>"
        "<h4>Observation Space</h4>"
        "<ul>"
        "<li><strong>Agent:</strong> [goal_rel_position, landmark_rel_position, other_agent_rel_positions]</li>"
        "<li><strong>Adversary:</strong> [landmark_rel_position, other_agents_rel_positions]</li>"
        "</ul>"
        "<h4>Action Space</h4>"
        "<p>[no_action, move_left, move_right, move_down, move_up]</p>"
        "<h4>Rewards</h4>"
        "<p>Unscaled Euclidean distance. Good agents rewarded for proximity to target, "
        "penalized for adversary proximity. Adversary rewarded for distance to target.</p>"
        "<h4>Arguments</h4>"
        "<pre><code>simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False, dynamic_rescaling=False)</code></pre>"
        "<ul>"
        "<li><strong>N:</strong> Number of good agents and landmarks</li>"
        "<li><strong>max_cycles:</strong> Steps until game terminates</li>"
        "<li><strong>continuous_actions:</strong> Use continuous action space</li>"
        "</ul>"
        "<h4>Usage</h4>"
        "<pre><code>from pettingzoo.mpe import simple_adversary_v3\n\n"
        "env = simple_adversary_v3.env(render_mode='human')\n"
        "env.reset(seed=42)\n\n"
        "for agent in env.agent_iter():\n"
        "    obs, reward, term, trunc, info = env.last()\n"
        "    action = None if term or trunc else env.action_space(agent).sample()\n"
        "    env.step(action)\n"
        "env.close()</code></pre>"
    )


SIMPLE_ADVERSARY_HTML = get_simple_adversary_html()

__all__ = ["SIMPLE_ADVERSARY_HTML", "get_simple_adversary_html"]
