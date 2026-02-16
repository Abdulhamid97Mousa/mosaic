"""Documentation for Jumanji Maze environment.

Maze is a classic navigation puzzle where the agent must find
a path from start to goal through a maze.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_maze_html(env_id: str = "Maze-v0") -> str:
    """Generate Maze HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Maze is a classic grid-world navigation task. The agent must navigate
from a starting position to a goal position while avoiding walls.
The maze is procedurally generated for each episode.
</p>

<h4>Objective</h4>
<ul>
    <li>Navigate from start position to goal</li>
    <li>Avoid walls (impassable cells)</li>
    <li>Find the shortest path</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>walls</strong>: 2D grid indicating wall positions</li>
    <li><strong>agent_position</strong>: Current agent location</li>
    <li><strong>goal_position</strong>: Target location</li>
    <li><strong>action_mask</strong>: Valid movement directions</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(4)</code> - Movement directions:</p>
<ul>
    <li><strong>0</strong>: Up</li>
    <li><strong>1</strong>: Right</li>
    <li><strong>2</strong>: Down</li>
    <li><strong>3</strong>: Left</li>
</ul>

<h4>Rewards</h4>
<ul>
    <li><strong>+1</strong>: Reaching the goal</li>
    <li><strong>-0.01</strong>: Step penalty (encourages efficiency)</li>
    <li><strong>0</strong>: Hitting a wall (no movement)</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Termination</strong>: Goal reached</li>
    <li><strong>Truncation</strong>: Maximum steps exceeded</li>
</ul>

<h4>Maze Generation</h4>
<p>Mazes are procedurally generated using algorithms like:</p>
<ul>
    <li>Recursive backtracking</li>
    <li>Prim's algorithm</li>
    <li>Kruskal's algorithm</li>
</ul>

<h4>Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W / Up Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Up (action 0)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D / Right Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Right (action 1)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">S / Down Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Down (action 2)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A / Left Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Left (action 3)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Maze_generation_algorithm" target="_blank">Maze Generation (Wikipedia)</a></li>
</ul>
"""


MAZE_HTML = get_maze_html()

__all__ = ["get_maze_html", "MAZE_HTML"]
