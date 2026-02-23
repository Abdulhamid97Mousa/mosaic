"""Documentation for Jumanji Sokoban environment.

Sokoban is a classic puzzle game where the player pushes boxes
onto target locations in a warehouse.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_sokoban_html(env_id: str = "Sokoban-v0") -> str:
    """Generate Sokoban HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Sokoban (Japanese for "warehouse keeper") is a classic puzzle game
where the player pushes boxes around a warehouse, trying to put them
on designated storage locations. The challenge is that boxes can only
be pushed, not pulled.
</p>

<h4>Game Rules</h4>
<ul>
    <li>Push boxes onto all target locations</li>
    <li>Boxes can only be pushed, not pulled</li>
    <li>Only one box can be pushed at a time</li>
    <li>Avoid pushing boxes into corners or against walls</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>grid</strong>: Warehouse layout (walls, floor, targets)</li>
    <li><strong>player_position</strong>: Warehouse keeper location</li>
    <li><strong>box_positions</strong>: Location of each box</li>
    <li><strong>target_positions</strong>: Goal locations for boxes</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(4)</code> - Movement directions:</p>
<ul>
    <li><strong>0</strong>: Up (push box if in front)</li>
    <li><strong>1</strong>: Right</li>
    <li><strong>2</strong>: Down</li>
    <li><strong>3</strong>: Left</li>
</ul>

<h4>Rewards</h4>
<ul>
    <li><strong>+1</strong>: Pushing box onto target</li>
    <li><strong>-1</strong>: Pushing box off target</li>
    <li><strong>+10</strong>: Solving the puzzle</li>
    <li><strong>-0.01</strong>: Step penalty</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Win</strong>: All boxes on targets</li>
    <li><strong>Lose</strong>: Box stuck in corner (deadlock)</li>
    <li><strong>Truncation</strong>: Maximum steps exceeded</li>
</ul>

<h4>Deadlock Detection</h4>
<p>Common deadlock situations:</p>
<ul>
    <li><strong>Corner deadlock</strong>: Box in corner with no targets</li>
    <li><strong>Wall deadlock</strong>: Box against wall, can't reach target</li>
    <li><strong>Freeze deadlock</strong>: Two boxes blocking each other</li>
</ul>

<h4>Solution Strategies</h4>
<ul>
    <li>Work backwards from goal state</li>
    <li>Identify boxes that must be pushed last</li>
    <li>Avoid creating deadlocks</li>
    <li>Use macro-actions (push box to target)</li>
</ul>

<h4>Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W / Up Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move/Push Up (action 0)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D / Right Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move/Push Right (action 1)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">S / Down Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move/Push Down (action 2)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A / Left Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move/Push Left (action 3)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Sokoban" target="_blank">Sokoban (Wikipedia)</a></li>
</ul>
"""


SOKOBAN_HTML = get_sokoban_html()

__all__ = ["get_sokoban_html", "SOKOBAN_HTML"]
