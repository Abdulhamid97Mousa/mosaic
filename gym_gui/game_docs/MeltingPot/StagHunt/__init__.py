"""Documentation for Melting Pot Stag Hunt substrate."""
from __future__ import annotations


STAG_HUNT_HTML = """
<h2>Melting Pot: Stag Hunt in the Matrix (Repeated)</h2>

<p><strong>Substrate:</strong> <code>stag_hunt_in_the_matrix__repeated</code></p>

<h3>Overview</h3>
<p>
A coordination game where mutual cooperation yields the best outcome, but solo play is safer.
Tests trust, coordination, and risk assessment in multi-agent settings.
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
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative (Coordination Game)</td>
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
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(8): hunt stag (risky) or hare (safe)</td>
    </tr>
</table>

<h3>Game Theory Background</h3>
<p>
The Stag Hunt payoff matrix:
</p>
<ul>
    <li><strong>Both Hunt Stag (S,S):</strong> Highest reward (e.g., 4,4) - requires coordination</li>
    <li><strong>Both Hunt Hare (H,H):</strong> Moderate reward (e.g., 2,2) - safe but suboptimal</li>
    <li><strong>Asymmetric (S,H):</strong> Stag hunter gets nothing (0), hare hunter gets 2</li>
</ul>

<p>
The dilemma: cooperating (stag) gives the best outcome if your partner also cooperates,
but hunting hare is safer. This tests whether agents can build trust to achieve
the collectively optimal solution.
</p>

<h3>Strategic Considerations</h3>
<ul>
    <li><strong>Risk vs. Reward:</strong> High payoff requires trusting your partner</li>
    <li><strong>Nash Equilibria:</strong> Both (S,S) and (H,H) are stable equilibria</li>
    <li><strong>Payoff Dominance:</strong> (S,S) is better for both, but riskier</li>
    <li><strong>Risk Dominance:</strong> (H,H) is safer against uncertain partners</li>
</ul>

<h3>Research Applications</h3>
<ul>
    <li><strong>Coordination:</strong> Learning to synchronize actions</li>
    <li><strong>Trust Building:</strong> Establishing reliable partnerships</li>
    <li><strong>Communication:</strong> Implicit signaling through actions</li>
    <li><strong>Social Norms:</strong> Emergence of coordination conventions</li>
</ul>

<h3>References</h3>
<ul>
    <li><a href="https://github.com/google-deepmind/meltingpot">Melting Pot Repository</a></li>
    <li><a href="https://arxiv.org/abs/2211.13746">Paper: "Melting Pot 2.0"</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Stag_hunt">Stag Hunt (Wikipedia)</a></li>
    <li><a href="https://plato.stanford.edu/entries/game-theory/">Stanford Encyclopedia: Game Theory</a></li>
</ul>

<p><em>Note: Requires Linux or macOS (Windows not supported)</em></p>
"""

__all__ = ["STAG_HUNT_HTML"]
