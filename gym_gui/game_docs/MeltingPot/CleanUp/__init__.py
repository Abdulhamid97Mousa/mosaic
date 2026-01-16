"""Documentation for Melting Pot Clean Up substrate."""
from __future__ import annotations


CLEAN_UP_HTML = """
<h2>Melting Pot: Clean Up (Repeated)</h2>

<p><strong>Substrate:</strong> <code>clean_up__repeated</code></p>

<h3>Overview</h3>
<p>
A public goods game where agents harvest apples but must also clean pollution.
Tests cooperation in maintaining shared resources versus free-riding incentives.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Number of Agents</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Up to 7 players</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative (Public Goods Dilemma)</td>
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
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(8): move, zap (for defense), clean (pollution removal)</td>
    </tr>
</table>

<h3>Game Mechanics</h3>
<p>
Public goods dilemma with environmental management:
</p>
<ul>
    <li><strong>Apple Harvesting:</strong> Collect apples from the river for immediate reward</li>
    <li><strong>Pollution Accumulation:</strong> Harvesting generates pollution that blocks apple spawning</li>
    <li><strong>Cleaning Action:</strong> Use the cleaning beam to remove pollution (costs time, no direct reward)</li>
    <li><strong>Collective Benefit:</strong> Everyone benefits from a clean environment, but cleaning is individually costly</li>
</ul>

<p>
The key tension: agents can free-ride by only harvesting while others clean, but if
no one cleans, the apple supply eventually depletes for everyone.
</p>

<h3>Social Dynamics</h3>
<ul>
    <li><strong>Free-Riding Problem:</strong> Temptation to harvest without contributing to maintenance</li>
    <li><strong>Conditional Cooperation:</strong> Agents may clean only if others also clean</li>
    <li><strong>Punishment:</strong> Using the zapper to discourage free-riders</li>
    <li><strong>Social Norms:</strong> Emergence of fairness expectations</li>
</ul>

<h3>Controls</h3>
<p><strong>Keyboard:</strong> W/A/S/D (movement), Q/E (turn), 1 (zapper), 2 (cleaning beam), TAB (switch players)</p>

<h3>Research Applications</h3>
<ul>
    <li><strong>Public Goods Games:</strong> Classic economic scenario in spatial setting</li>
    <li><strong>Conditional Cooperation:</strong> Learning reciprocal maintenance strategies</li>
    <li><strong>Social Sanctioning:</strong> Using punishment to enforce norms</li>
    <li><strong>Collective Action:</strong> Coordinating to maintain shared resources</li>
</ul>

<h3>References</h3>
<ul>
    <li><a href="https://github.com/google-deepmind/meltingpot">Melting Pot Repository</a></li>
    <li><a href="https://arxiv.org/abs/2211.13746">Paper: "Melting Pot 2.0"</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Public_good_(economics)">Public Goods (Wikipedia)</a></li>
</ul>

<p><em>Note: Requires Linux or macOS (Windows not supported)</em></p>
"""

__all__ = ["CLEAN_UP_HTML"]
