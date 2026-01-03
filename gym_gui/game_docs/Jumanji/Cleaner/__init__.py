"""Documentation for Jumanji Cleaner environment.

Cleaner is a grid-based navigation task where an agent must clean
all dirty cells in the environment.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_cleaner_html(env_id: str = "Cleaner-v0") -> str:
    """Generate Cleaner HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Cleaner is a grid-world navigation environment where the agent must
visit and clean all dirty cells in the grid. The agent can move in
four cardinal directions.
</p>

<h4>Objective</h4>
<ul>
    <li>Navigate the grid to reach all dirty cells</li>
    <li>Clean cells by visiting them</li>
    <li>Minimize the number of steps taken</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>grid</strong>: 2D grid showing clean/dirty cells</li>
    <li><strong>agent_position</strong>: Current agent location</li>
    <li><strong>action_mask</strong>: Valid movement actions</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(4)</code> - Movement directions:</p>
<ul>
    <li><strong>0</strong>: Up</li>
    <li><strong>1</strong>: Down</li>
    <li><strong>2</strong>: Left</li>
    <li><strong>3</strong>: Right</li>
</ul>

<h4>Rewards</h4>
<ul>
    <li><strong>+1</strong>: Cleaning a dirty cell</li>
    <li><strong>0</strong>: Moving to already clean cell</li>
    <li><strong>Bonus</strong>: Completing the task quickly</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Termination</strong>: All cells cleaned</li>
    <li><strong>Truncation</strong>: Maximum steps reached</li>
</ul>

<h4>Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Up Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Up</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Down Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Down</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Left Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Right Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Right</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
</ul>
"""


CLEANER_HTML = get_cleaner_html()

__all__ = ["get_cleaner_html", "CLEANER_HTML"]
