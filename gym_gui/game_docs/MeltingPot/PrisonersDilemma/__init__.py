"""Documentation for Melting Pot Prisoners Dilemma substrate."""
from __future__ import annotations


PRISONERS_DILEMMA_HTML = """
<h2>Melting Pot: Prisoners Dilemma in the Matrix (Repeated)</h2>

<p><strong>Substrate:</strong> <code>prisoners_dilemma_in_the_matrix__repeated</code></p>

<h3>Overview</h3>
<p>
The classic game theory scenario in a spatial environment. Agents choose between
cooperation and defection across repeated interactions, testing reciprocity and trust-building.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Number of Agents</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">2 players</td>
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
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(8): movement + cooperation/defection actions</td>
    </tr>
</table>

<h3>Game Theory Background</h3>
<p>
The Prisoner's Dilemma payoff matrix:
</p>
<ul>
    <li><strong>Both Cooperate (C,C):</strong> Moderate reward for both (e.g., 3,3)</li>
    <li><strong>Both Defect (D,D):</strong> Low reward for both (e.g., 1,1)</li>
    <li><strong>One Defects (D,C):</strong> High reward for defector, zero for cooperator (e.g., 5,0)</li>
</ul>

<p>
In the repeated version, agents can develop strategies like tit-for-tat, establishing
trust through consistent reciprocity, or exploiting overly cooperative opponents.
</p>

<h3>Strategic Concepts</h3>
<ul>
    <li><strong>Tit-for-Tat:</strong> Copy opponent's previous move</li>
    <li><strong>Grim Trigger:</strong> Cooperate until first defection, then defect forever</li>
    <li><strong>Win-Stay-Lose-Shift:</strong> Repeat if rewarded, change if not</li>
    <li><strong>Trust Building:</strong> Establish cooperation patterns</li>
</ul>

<h3>Research Applications</h3>
<ul>
    <li><strong>Game Theory:</strong> Testing classic strategies in spatial settings</li>
    <li><strong>Reciprocity:</strong> Learning to respond to partner behavior</li>
    <li><strong>Opponent Modeling:</strong> Predicting partner strategies</li>
    <li><strong>Theory of Mind:</strong> Understanding partner intentions</li>
</ul>

<h3>References</h3>
<ul>
    <li><a href="https://github.com/google-deepmind/meltingpot">Melting Pot Repository</a></li>
    <li><a href="https://arxiv.org/abs/2211.13746">Paper: "Melting Pot 2.0"</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Prisoner%27s_dilemma">Prisoner's Dilemma (Wikipedia)</a></li>
    <li><a href="https://plato.stanford.edu/entries/prisoner-dilemma/">Stanford Encyclopedia of Philosophy</a></li>
</ul>

<p><em>Note: Requires Linux or macOS (Windows not supported)</em></p>
"""

__all__ = ["PRISONERS_DILEMMA_HTML"]
