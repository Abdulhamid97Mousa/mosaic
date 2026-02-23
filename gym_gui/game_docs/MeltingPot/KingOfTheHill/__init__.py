"""Documentation for Melting Pot King of the Hill substrate."""
from __future__ import annotations


KING_OF_THE_HILL_HTML = """
<h2>Melting Pot: King of the Hill (Repeated)</h2>

<p><strong>Substrate:</strong> <code>king_of_the_hill__repeated</code></p>

<h3>Overview</h3>
<p>
Agents compete to occupy a central hill area for as long as possible. Tests competitive
strategies, timing, area control, and the ability to dislodge opponents from advantageous positions.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Number of Agents</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Up to 16 players</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Competitive (Area Control)</td>
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
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(8): movement + interaction actions</td>
    </tr>
</table>

<h3>Game Mechanics</h3>
<p>
Classic competitive multiplayer scenario where:
</p>
<ul>
    <li><strong>Hill Control:</strong> Reward accumulates for each timestep an agent occupies the hill</li>
    <li><strong>Displacement:</strong> Agents can push others off the hill through physical contact</li>
    <li><strong>Strategic Positioning:</strong> Approach angles and timing matter for successful takeovers</li>
    <li><strong>Risk Assessment:</strong> Balance between holding position and defending against challengers</li>
</ul>

<p>
The environment rewards sustained control rather than brief occupancy, encouraging
agents to develop both offensive (taking the hill) and defensive (holding it) strategies.
</p>

<h3>Competitive Dynamics</h3>
<ul>
    <li><strong>Crowding:</strong> Multiple agents competing simultaneously</li>
    <li><strong>Momentum:</strong> Physical dynamics affect displacement success</li>
    <li><strong>Cooldown Periods:</strong> Strategic retreats and re-engagements</li>
    <li><strong>Winner-Take-All:</strong> Only the occupying agent receives rewards</li>
</ul>

<h3>Research Applications</h3>
<ul>
    <li><strong>Competitive MARL:</strong> Pure competitive multi-agent scenarios</li>
    <li><strong>Physical Interactions:</strong> Learning spatial coordination in crowded environments</li>
    <li><strong>Population Training:</strong> Diverse opponent strategies through self-play</li>
    <li><strong>Nash Equilibria:</strong> Analyzing equilibrium strategies in repeated competition</li>
</ul>

<h3>References</h3>
<ul>
    <li><a href="https://github.com/google-deepmind/meltingpot">Melting Pot Repository</a></li>
    <li><a href="https://arxiv.org/abs/2211.13746">Paper: "Melting Pot 2.0"</a></li>
</ul>

<p><em>Note: Requires Linux or macOS (Windows not supported)</em></p>
"""

__all__ = ["KING_OF_THE_HILL_HTML"]
