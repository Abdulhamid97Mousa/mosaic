"""Documentation for MiniHack navigation environments."""

from .controls import MINIHACK_CONTROLS_HTML

MINIHACK_ROOM_HTML = """
<h3>MiniHack - Room</h3>
<p>
  Navigate through simple rooms to reach the goal (staircase down). This is
  the simplest MiniHack environment, ideal for testing basic movement and
  goal-oriented navigation.
</p>

<h4>Objective</h4>
<p>Find and reach the <code>&gt;</code> (downstairs) symbol to complete the level.</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-Room-5x5-v0</code></td><td>5x5 empty room</td></tr>
  <tr><td><code>MiniHack-Room-15x15-v0</code></td><td>15x15 empty room</td></tr>
  <tr><td><code>MiniHack-Room-Random-5x5-v0</code></td><td>Random 5x5 room layout</td></tr>
  <tr><td><code>MiniHack-Room-Random-15x15-v0</code></td><td>Random 15x15 room layout</td></tr>
  <tr><td><code>MiniHack-Room-Monster-5x5-v0</code></td><td>5x5 room with monster</td></tr>
  <tr><td><code>MiniHack-Room-Monster-15x15-v0</code></td><td>15x15 room with monster</td></tr>
  <tr><td><code>MiniHack-Room-Trap-5x5-v0</code></td><td>5x5 room with traps</td></tr>
  <tr><td><code>MiniHack-Room-Trap-15x15-v0</code></td><td>15x15 room with traps</td></tr>
  <tr><td><code>MiniHack-Room-Ultimate-5x5-v0</code></td><td>5x5 room with monster, trap, and lava</td></tr>
  <tr><td><code>MiniHack-Room-Ultimate-15x15-v0</code></td><td>15x15 room with monster, trap, and lava</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> - 8 compass directions for movement</p>

<h4>Observation Space</h4>
<p>Dictionary with keys: <code>glyphs</code>, <code>chars</code>, <code>colors</code>, <code>blstats</code>, <code>message</code></p>
<p>Default crop size: 9x9 agent-centered view</p>
""" + MINIHACK_CONTROLS_HTML


MINIHACK_CORRIDOR_HTML = """
<h3>MiniHack - Corridor</h3>
<p>
  Navigate through corridors of varying complexity. Corridors may have turns,
  dead ends, or require opening doors. Tests pathfinding and exploration skills.
</p>

<h4>Objective</h4>
<p>Navigate through the corridor to reach the goal (staircase).</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-Corridor-R2-v0</code></td><td>2 rooms connected</td></tr>
  <tr><td><code>MiniHack-Corridor-R3-v0</code></td><td>3 rooms connected</td></tr>
  <tr><td><code>MiniHack-Corridor-R5-v0</code></td><td>5 rooms connected</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(11)</code> - Movement + OPEN, KICK, SEARCH</p>

<h4>Key Skills</h4>
<ul>
  <li>Navigation through corridors</li>
  <li>Opening doors (press <b>o</b>)</li>
  <li>Searching for secret passages (<b>s</b>)</li>
</ul>
""" + MINIHACK_CONTROLS_HTML


MINIHACK_MAZEWALK_HTML = """
<h3>MiniHack - MazeWalk</h3>
<p>
  Navigate through procedurally generated mazes. The agent must find the optimal
  path through the maze to reach the goal. Tests spatial reasoning and exploration.
</p>

<h4>Objective</h4>
<p>Find your way through the maze to reach the staircase.</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-MazeWalk-9x9-v0</code></td><td>9x9 maze</td></tr>
  <tr><td><code>MiniHack-MazeWalk-15x15-v0</code></td><td>15x15 maze</td></tr>
  <tr><td><code>MiniHack-MazeWalk-45x19-v0</code></td><td>Large rectangular maze</td></tr>
  <tr><td><code>MiniHack-MazeWalk-Mapped-9x9-v0</code></td><td>9x9 maze (fully visible)</td></tr>
  <tr><td><code>MiniHack-MazeWalk-Mapped-15x15-v0</code></td><td>15x15 maze (fully visible)</td></tr>
  <tr><td><code>MiniHack-MazeWalk-Mapped-45x19-v0</code></td><td>Large maze (fully visible)</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> - 8 compass directions</p>

<h4>Challenge</h4>
<p>Unmapped variants require the agent to explore and remember visited locations.</p>
""" + MINIHACK_CONTROLS_HTML


MINIHACK_RIVER_HTML = """
<h3>MiniHack - River</h3>
<p>
  Cross a river using various methods. The agent must find objects (wands, rings)
  to safely cross the water, as stepping directly into water is lethal.
</p>

<h4>Objective</h4>
<p>Cross the river to reach the goal on the other side.</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-River-v0</code></td><td>Basic river crossing</td></tr>
  <tr><td><code>MiniHack-River-Narrow-v0</code></td><td>Narrow river (easier)</td></tr>
  <tr><td><code>MiniHack-River-Monster-v0</code></td><td>River with monsters</td></tr>
  <tr><td><code>MiniHack-River-Lava-v0</code></td><td>Lava river (more dangerous)</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(23)</code> - Movement + item interactions</p>

<h4>Key Skills</h4>
<ul>
  <li>Finding and picking up items</li>
  <li>Using levitation (wand, ring, boots)</li>
  <li>Zapping wands with <b>z</b></li>
  <li>Wearing rings with <b>P</b> (put on)</li>
</ul>

<h4>Danger</h4>
<p><strong>Warning:</strong> Stepping into water <code>}</code> without levitation is instant death!</p>
""" + MINIHACK_CONTROLS_HTML


__all__ = [
    "MINIHACK_ROOM_HTML",
    "MINIHACK_CORRIDOR_HTML",
    "MINIHACK_MAZEWALK_HTML",
    "MINIHACK_RIVER_HTML",
]
