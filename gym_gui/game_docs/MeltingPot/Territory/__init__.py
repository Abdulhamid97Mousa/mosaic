"""Documentation for Melting Pot Territory substrate."""
from __future__ import annotations


TERRITORY_HTML = """
<h2>Melting Pot: Territory (Rooms)</h2>

<p><strong>Substrate:</strong> <code>territory__rooms</code></p>

<h3>Overview</h3>
<p>
Agents compete to claim and hold territory in connected rooms. Tests strategic positioning,
resource control, and competitive multi-agent dynamics in spatial environments.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Number of Agents</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Up to 8 players</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Competitive</td>
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
Territorial control game where agents:
</p>
<ul>
    <li><strong>Claim Territory:</strong> Occupy and control specific rooms or areas</li>
    <li><strong>Defend Resources:</strong> Prevent opponents from accessing claimed spaces</li>
    <li><strong>Strategic Movement:</strong> Plan routes to maximize territorial control</li>
    <li><strong>Timing:</strong> Choose when to attack, defend, or expand</li>
</ul>

<p>
Rewards are based on the amount of territory controlled over time, creating
incentives for both aggressive expansion and defensive positioning.
</p>

<h3>Strategic Elements</h3>
<ul>
    <li><strong>Area Control:</strong> Maximize owned territory while minimizing losses</li>
    <li><strong>Chokepoints:</strong> Control key passages between rooms</li>
    <li><strong>Resource Distribution:</strong> Balance between expansion and defense</li>
    <li><strong>Opponent Modeling:</strong> Predict and counter enemy strategies</li>
</ul>

<h3>Research Applications</h3>
<ul>
    <li><strong>Competitive MARL:</strong> Zero-sum or near-zero-sum scenarios</li>
    <li><strong>Strategic Reasoning:</strong> Long-term planning in adversarial settings</li>
    <li><strong>Self-Play Training:</strong> Iterative policy improvement</li>
    <li><strong>AlphaZero-style Methods:</strong> Search + learning approaches</li>
</ul>

<h3>References</h3>
<ul>
    <li><a href="https://github.com/google-deepmind/meltingpot">Melting Pot Repository</a></li>
    <li><a href="https://arxiv.org/abs/2211.13746">Paper: "Melting Pot 2.0"</a></li>
</ul>

<p><em>Note: Requires Linux or macOS (Windows not supported)</em></p>
"""

__all__ = ["TERRITORY_HTML"]
