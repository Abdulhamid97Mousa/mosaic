"""Documentation for Melting Pot Allelopathic Harvest substrate."""
from __future__ import annotations


ALLELOPATHIC_HARVEST_HTML = """
<h2>Melting Pot: Allelopathic Harvest (Open)</h2>

<p><strong>Substrate:</strong> <code>allelopathic_harvest__open</code></p>

<h3>Overview</h3>
<p>
Agents harvest resources but leave behind chemicals that damage others' future harvests.
Tests competitive resource gathering with negative externalities and strategic pollution placement.
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
        <td style="border: 1px solid #ddd; padding: 8px;">Competitive (with Negative Externalities)</td>
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
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(8): movement + harvesting actions</td>
    </tr>
</table>

<h3>Game Mechanics</h3>
<p>
Biological allegory where:
</p>
<ul>
    <li><strong>Resource Harvest:</strong> Collecting resources gives immediate rewards</li>
    <li><strong>Chemical Trail:</strong> Harvesting leaves behind inhibitory chemicals</li>
    <li><strong>Reduced Yields:</strong> Contaminated areas yield fewer resources for all agents</li>
    <li><strong>Spatial Strategy:</strong> Where you harvest affects both you and competitors</li>
</ul>

<p>
The name comes from allelopathy in biology, where plants release chemicals that
inhibit the growth of competing plants. Agents must balance personal gain against
the long-term degradation of the resource base.
</p>

<h3>Strategic Elements</h3>
<ul>
    <li><strong>Territorial Denial:</strong> Polluting areas to deny access to competitors</li>
    <li><strong>Clean Foraging:</strong> Finding unpolluted resource patches</li>
    <li><strong>Pollution Management:</strong> Avoiding your own chemical trails</li>
    <li><strong>Competitive Inhibition:</strong> Strategically degrading opponent resources</li>
</ul>

<h3>Research Applications</h3>
<ul>
    <li><strong>Negative Externalities:</strong> Learning to account for pollution and degradation</li>
    <li><strong>Competitive Dynamics:</strong> Strategies that harm opponents while benefiting self</li>
    <li><strong>Resource Economics:</strong> Short-term exploitation vs. long-term sustainability</li>
    <li><strong>Spatial Reasoning:</strong> Planning movement to avoid contaminated areas</li>
</ul>

<h3>References</h3>
<ul>
    <li><a href="https://github.com/google-deepmind/meltingpot">Melting Pot Repository</a></li>
    <li><a href="https://arxiv.org/abs/2211.13746">Paper: "Melting Pot 2.0"</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Allelopathy">Allelopathy (Wikipedia)</a></li>
</ul>

<p><em>Note: Requires Linux or macOS (Windows not supported)</em></p>
"""

__all__ = ["ALLELOPATHIC_HARVEST_HTML"]
