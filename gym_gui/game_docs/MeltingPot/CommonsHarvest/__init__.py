"""Documentation for Melting Pot Commons Harvest substrate."""
from __future__ import annotations


COMMONS_HARVEST_HTML = """
<h2>Melting Pot: Commons Harvest (Open)</h2>

<p><strong>Substrate:</strong> <code>commons_harvest__open</code></p>

<h3>Overview</h3>
<p>
A tragedy of the commons scenario where agents harvest from a shared renewable resource.
Over-harvesting depletes the resource permanently, testing sustainable cooperation and
long-term strategic thinking.
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
        <td style="border: 1px solid #ddd; padding: 8px;">Mixed-Motive (Social Dilemma)</td>
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
The classic "tragedy of the commons" dilemma where:
</p>
<ul>
    <li><strong>Shared Resource:</strong> Apples regrow at a sustainable rate if not over-harvested</li>
    <li><strong>Individual Incentive:</strong> Each agent benefits immediately from harvesting</li>
    <li><strong>Collective Risk:</strong> Over-harvesting reduces future availability for all</li>
    <li><strong>Long-term Strategy:</strong> Cooperation leads to sustained rewards</li>
</ul>

<p>
This substrate tests whether agents can learn to resist short-term gains in favor
of long-term collective benefit, a fundamental challenge in multi-agent cooperation.
</p>

<h3>Social Dynamics</h3>
<ul>
    <li><strong>Free-Riding:</strong> Some agents may harvest while others conserve</li>
    <li><strong>Punishment:</strong> Agents may develop strategies to discourage over-harvesting</li>
    <li><strong>Norms:</strong> Emergence of implicit harvesting rules</li>
    <li><strong>Sustainability:</strong> Balancing individual and collective interests</li>
</ul>

<h3>Research Applications</h3>
<ul>
    <li><strong>Social Dilemmas:</strong> Studying cooperation vs. defection dynamics</li>
    <li><strong>Resource Management:</strong> Learning sustainable harvesting policies</li>
    <li><strong>Policy Gradient Methods:</strong> Training with delayed consequences</li>
    <li><strong>Game Theory:</strong> Testing Nash equilibrium predictions</li>
</ul>

<h3>References</h3>
<ul>
    <li><a href="https://github.com/google-deepmind/meltingpot">Melting Pot Repository</a></li>
    <li><a href="https://arxiv.org/abs/2211.13746">Paper: "Melting Pot 2.0"</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Tragedy_of_the_commons">Tragedy of the Commons (Wikipedia)</a></li>
</ul>

<p><em>Note: Requires Linux or macOS (Windows not supported)</em></p>
"""

__all__ = ["COMMONS_HARVEST_HTML"]
