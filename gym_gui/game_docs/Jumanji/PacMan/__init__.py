"""Documentation for Jumanji PacMan environment.

PacMan is the classic arcade game where the player navigates a maze,
eating pellets while avoiding ghosts.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_pacman_html(env_id: str = "PacMan-v1") -> str:
    """Generate PacMan HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Pac-Man is the iconic 1980 arcade game. The player controls Pac-Man
through a maze, eating pellets while avoiding four ghosts. Power pellets
temporarily allow Pac-Man to eat the ghosts.
</p>

<h4>Game Rules</h4>
<ul>
    <li>Eat all pellets to complete the level</li>
    <li>Avoid ghosts (costs a life)</li>
    <li>Power pellets make ghosts vulnerable</li>
    <li>Bonus fruits appear for extra points</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>maze</strong>: Grid with walls, pellets, power pellets</li>
    <li><strong>pacman_position</strong>: Player location</li>
    <li><strong>ghost_positions</strong>: Location of each ghost</li>
    <li><strong>ghost_states</strong>: Normal, frightened, or eaten</li>
    <li><strong>power_pellet_timer</strong>: Remaining frighten time</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(5)</code> - Movement directions:</p>
<ul>
    <li><strong>0</strong>: No-op (stay)</li>
    <li><strong>1</strong>: Up</li>
    <li><strong>2</strong>: Right</li>
    <li><strong>3</strong>: Down</li>
    <li><strong>4</strong>: Left</li>
</ul>

<h4>Scoring</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Item</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Points</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Pellet</td><td style="border: 1px solid #ddd; padding: 8px;">10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Power Pellet</td><td style="border: 1px solid #ddd; padding: 8px;">50</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ghost (1st)</td><td style="border: 1px solid #ddd; padding: 8px;">200</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ghost (2nd)</td><td style="border: 1px solid #ddd; padding: 8px;">400</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ghost (3rd)</td><td style="border: 1px solid #ddd; padding: 8px;">800</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ghost (4th)</td><td style="border: 1px solid #ddd; padding: 8px;">1600</td></tr>
</table>

<h4>Ghost Behaviors</h4>
<ul>
    <li><strong>Blinky (Red)</strong>: Chases directly</li>
    <li><strong>Pinky (Pink)</strong>: Ambushes ahead</li>
    <li><strong>Inky (Cyan)</strong>: Unpredictable</li>
    <li><strong>Clyde (Orange)</strong>: Random/shy</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Win</strong>: All pellets eaten</li>
    <li><strong>Lose</strong>: All lives lost</li>
</ul>

<h4>Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W / Up Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Up (action 1)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D / Right Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Right (action 2)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">S / Down Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Down (action 3)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A / Left Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move Left (action 4)</td></tr>
</table>
<p><em>Note: Action 0 (No-op/Stay) is not mapped to a key.</em></p>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Pac-Man" target="_blank">Pac-Man (Wikipedia)</a></li>
</ul>
"""


PACMAN_HTML = get_pacman_html()

__all__ = ["get_pacman_html", "PACMAN_HTML"]
