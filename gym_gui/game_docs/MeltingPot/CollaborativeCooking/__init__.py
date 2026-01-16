"""Documentation for Melting Pot Collaborative Cooking substrate."""
from __future__ import annotations


COLLABORATIVE_COOKING_HTML = """
<h2>Melting Pot: Collaborative Cooking (Circuit)</h2>

<p><strong>Substrate:</strong> <code>collaborative_cooking__circuit</code></p>

<h3>Overview</h3>
<p>
Agents must cooperate to prepare dishes in a kitchen environment. Success requires
coordination, resource sharing, and task allocation among team members. This substrate
tests cooperative multi-agent behavior in a shared workspace with interdependent goals.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Number of Agents</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">2-9 players</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Pure Cooperation</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Stepping Paradigm</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Parallel (simultaneous)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Observation Space</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Dict with RGB (40×40×3) + COLLECTIVE_REWARD</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Action Space</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(8): NOOP, FORWARD, BACKWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT, INTERACT</td>
    </tr>
</table>

<h3>Game Mechanics</h3>
<p>
Players work together in a kitchen to complete cooking tasks. The environment
requires agents to:
</p>
<ul>
    <li><strong>Gather ingredients:</strong> Collect necessary items from different locations</li>
    <li><strong>Process food:</strong> Perform cooking actions in the correct sequence</li>
    <li><strong>Coordinate tasks:</strong> Divide work efficiently among team members</li>
    <li><strong>Share resources:</strong> Pass items between agents when needed</li>
</ul>

<p>
The collective reward structure encourages teamwork, as all agents benefit from
successfully completed dishes. Individual actions contribute to the team's overall success.
</p>

<h3>Research Applications</h3>
<ul>
    <li><strong>Cooperative AI:</strong> Training agents that can work effectively in teams</li>
    <li><strong>Task Allocation:</strong> Learning to divide labor optimally</li>
    <li><strong>Communication:</strong> Developing implicit coordination strategies</li>
    <li><strong>Emergent Roles:</strong> Observing specialization without explicit role assignment</li>
</ul>

<h3>Training Resources</h3>
<p>
Recommended algorithms for this cooperative substrate:
</p>
<ul>
    <li><strong>MAPPO:</strong> Multi-Agent PPO with centralized critic</li>
    <li><strong>QMIX:</strong> Value decomposition for cooperative multi-agent RL</li>
    <li><strong>CommNet:</strong> Communication-based multi-agent learning</li>
</ul>

<h3>References</h3>
<ul>
    <li><a href="https://github.com/google-deepmind/meltingpot">Melting Pot Repository</a></li>
    <li><a href="https://arxiv.org/abs/2211.13746">Paper: "Melting Pot 2.0" (arXiv 2022)</a></li>
    <li><a href="https://shimmy.farama.org/environments/meltingpot/">Shimmy Documentation</a></li>
    <li><a href="https://deepmind.google/blog/melting-pot-an-evaluation-suite-for-multi-agent-reinforcement-learning/">DeepMind Blog Post</a></li>
</ul>

<p><em>Note: Requires Linux or macOS (Windows not supported)</em></p>
"""

__all__ = ["COLLABORATIVE_COOKING_HTML"]
