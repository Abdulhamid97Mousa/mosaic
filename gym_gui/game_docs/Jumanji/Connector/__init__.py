"""Documentation for Jumanji Connector environment.

Connector is a puzzle where the agent must connect pairs of endpoints
without crossing paths, similar to Flow Free or Numberlink puzzles.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_connector_html(env_id: str = "Connector-v2") -> str:
    """Generate Connector HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Connector (also known as Numberlink or Flow Free) is a puzzle where
the agent must draw paths to connect pairs of matching endpoints on
a grid without any paths crossing each other.
</p>

<h4>Objective</h4>
<ul>
    <li>Connect all pairs of matching endpoints</li>
    <li>Paths cannot cross or overlap</li>
    <li>Fill the entire grid with paths</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>grid</strong>: Current state with endpoints and paths</li>
    <li><strong>current_pair</strong>: The pair being connected</li>
    <li><strong>action_mask</strong>: Valid next cells for path</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete</code> - Select next cell to add to current path</p>

<h4>Rewards</h4>
<ul>
    <li><strong>+1</strong>: Successfully connecting a pair</li>
    <li><strong>Bonus</strong>: Completing all connections</li>
    <li><strong>-1</strong>: Invalid move (crossing paths)</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Termination (Win)</strong>: All pairs connected</li>
    <li><strong>Termination (Lose)</strong>: No valid moves remaining</li>
</ul>

<h4>Strategies</h4>
<ul>
    <li>Start with pairs that have limited path options</li>
    <li>Avoid creating isolated regions</li>
    <li>Think ahead to prevent blocking future paths</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Numberlink" target="_blank">Numberlink (Wikipedia)</a></li>
</ul>
"""


CONNECTOR_HTML = get_connector_html()

__all__ = ["get_connector_html", "CONNECTOR_HTML"]
