"""MultiGrid game documentation module.

gym-multigrid is a multi-agent extension of MiniGrid for training cooperative
and competitive multi-agent RL policies. All agents act simultaneously each step.

Environments:
    - MultiGrid-Soccer-v0: 4 agents (2v2 soccer), zero-sum competitive
    - MultiGrid-Collect-v0: 3 agents collecting balls, cooperative/competitive

Repository: https://github.com/ArnaudFickinger/gym-multigrid
Location: 3rd_party/gym-multigrid/
"""

from __future__ import annotations


def get_multigrid_html(env_id: str) -> str:
    """Generate MultiGrid HTML documentation for a specific variant.

    Args:
        env_id: Environment identifier (e.g., "MultiGrid-Soccer-v0")

    Returns:
        HTML string containing environment documentation.
    """
    is_soccer = "Soccer" in env_id
    is_collect = "Collect" in env_id

    if is_soccer:
        return _get_soccer_html()
    elif is_collect:
        return _get_collect_html()
    else:
        return _get_overview_html()


def _get_overview_html() -> str:
    """Return overview HTML for MultiGrid family."""
    return """
<h2>gym-multigrid</h2>

<p>
<strong>gym-multigrid</strong> is a multi-agent extension of MiniGrid designed for training
cooperative and competitive multi-agent reinforcement learning policies.
Unlike single-agent MiniGrid, all agents act <em>simultaneously</em> each step.
</p>

<h4>Key Features</h4>
<ul>
    <li><strong>Multi-Agent</strong>: 2-4 agents controlled by Operators</li>
    <li><strong>Simultaneous Actions</strong>: All agents act at once (PARALLEL paradigm)</li>
    <li><strong>Cooperative/Competitive</strong>: Team-based or individual rewards</li>
    <li><strong>Compatible with RLlib</strong>: Native multi-agent policy training</li>
</ul>

<h4>Available Environments</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Agents</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Soccer-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">4</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Zero-sum</td>
        <td style="border: 1px solid #ddd; padding: 8px;">2v2 soccer, score goals</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Collect-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Competitive</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Collect balls of your color</td>
    </tr>
</table>

<h4>Stepping Paradigm: SIMULTANEOUS</h4>
<p>
Unlike turn-based games (PettingZoo AEC), gym-multigrid uses <strong>simultaneous stepping</strong>:
</p>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
# All agents submit actions at once
actions = [agent0_action, agent1_action, agent2_action, agent3_action]
obs_list, rewards_list, done, info = env.step(actions)
</pre>

<h4>NOT for Human Control</h4>
<p>
gym-multigrid is designed for <strong>RL training</strong>, not human play.
For keyboard-controlled grid worlds, use <strong>MiniGrid</strong> instead.
</p>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ArnaudFickinger/gym-multigrid" target="_blank">GitHub Repository</a></li>
    <li>Based on: <a href="https://minigrid.farama.org/" target="_blank">MiniGrid</a></li>
</ul>
"""


def _get_soccer_html() -> str:
    """Return HTML documentation for Soccer environment."""
    return """
<h2>MultiGrid-Soccer-v0</h2>

<p>
A 4-player (2v2) soccer game where two teams compete to score goals.
Agents must pick up the ball, navigate to the opponent's goal, and drop it to score.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">15 x 10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Number of Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2 per team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1 (Red)</td><td style="border: 1px solid #ddd; padding: 8px;">Agent 0, Agent 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2 (Green)</td><td style="border: 1px solid #ddd; padding: 8px;">Agent 2, Agent 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball</td><td style="border: 1px solid #ddd; padding: 8px;">1 neutral ball</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">10,000</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (scoring penalizes opponents)</td></tr>
</table>

<h4>Observation Space</h4>
<p>
Each agent receives a partial observation of the grid:
<code>Box(low=0, high=255, shape=(view_size, view_size, 6), dtype=uint8)</code>
</p>
<p>The 6 channels encode: object type, color, state, carrying object, carrying color, direction.</p>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> - Same actions for all agents:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">STILL</td><td style="border: 1px solid #ddd; padding: 8px;">Do nothing</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Turn left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Turn right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move forward</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;">Pick up ball (or steal from agent)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;">Drop ball (scores if at goal)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;">Toggle object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;">Signal completion</td></tr>
</table>

<h4>Rewards</h4>
<p>Zero-sum rewards when a team scores:</p>
<ul>
    <li><strong>Scoring team</strong>: +1.0 (shared by both teammates)</li>
    <li><strong>Opposing team</strong>: -1.0 (shared by both opponents)</li>
</ul>

<h4>Game Mechanics</h4>
<ul>
    <li><strong>Ball Pickup</strong>: Walk to ball and use PICKUP action</li>
    <li><strong>Ball Stealing</strong>: Can PICKUP from agent carrying ball</li>
    <li><strong>Scoring</strong>: DROP ball at opponent's goal</li>
    <li><strong>Teamwork</strong>: Can DROP ball to pass to teammate, they PICKUP</li>
</ul>

<h4>Training Configurations</h4>
<p>Example Ray RLlib multi-agent config:</p>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
multi_agent:
  policies:
    team_red: [agent_0, agent_1]   # Share policy
    team_green: [agent_2, agent_3] # Share policy
  policy_mapping_fn: lambda agent_id: "team_red" if agent_id < 2 else "team_green"
</pre>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ArnaudFickinger/gym-multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_collect_html() -> str:
    """Return HTML documentation for Collect environment."""
    return """
<h2>MultiGrid-Collect-v0</h2>

<p>
A 3-player ball collection game where agents compete to pick up balls.
Each agent is assigned a color and can collect balls of any color,
but the reward structure can be competitive or cooperative.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Number of Agents</td><td style="border: 1px solid #ddd; padding: 8px;">3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agent 0</td><td style="border: 1px solid #ddd; padding: 8px;">Red</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agent 1</td><td style="border: 1px solid #ddd; padding: 8px;">Green</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agent 2</td><td style="border: 1px solid #ddd; padding: 8px;">Blue</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">5 neutral balls</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">10,000</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (collecting penalizes others)</td></tr>
</table>

<h4>Observation Space</h4>
<p>
Each agent receives a partial observation of the grid:
<code>Box(low=0, high=255, shape=(view_size, view_size, 6), dtype=uint8)</code>
</p>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> - Same as Soccer environment.</p>

<h4>Rewards</h4>
<p>Zero-sum rewards when an agent collects a ball:</p>
<ul>
    <li><strong>Collecting agent</strong>: +1.0</li>
    <li><strong>Other agents</strong>: -1.0 (split among non-collectors)</li>
</ul>

<h4>Game Mechanics</h4>
<ul>
    <li><strong>Ball Collection</strong>: Walk to ball and use PICKUP action</li>
    <li><strong>Balls disappear</strong>: Once collected, balls are removed from grid</li>
    <li><strong>Competition</strong>: Race to collect before opponents</li>
    <li><strong>Episode ends</strong>: All balls collected or max steps reached</li>
</ul>

<h4>Cooperative Variant</h4>
<p>
The environment can be configured for cooperative play where all agents
share rewards. Configure via environment kwargs:
</p>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
env = CollectGameEnv(zero_sum=False)  # Shared rewards
</pre>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ArnaudFickinger/gym-multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


# Pre-generated HTML constants for convenience
MULTIGRID_OVERVIEW_HTML = _get_overview_html()
MULTIGRID_SOCCER_HTML = _get_soccer_html()
MULTIGRID_COLLECT_HTML = _get_collect_html()

__all__ = [
    "get_multigrid_html",
    "MULTIGRID_OVERVIEW_HTML",
    "MULTIGRID_SOCCER_HTML",
    "MULTIGRID_COLLECT_HTML",
]
