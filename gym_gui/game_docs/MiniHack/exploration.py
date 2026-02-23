"""Documentation for MiniHack exploration and memory environments."""

from .controls import MINIHACK_CONTROLS_HTML

MINIHACK_EXPLORE_MAZE_HTML = """
<h3>MiniHack - Explore Maze</h3>
<p>
  Explore procedurally generated mazes with partial observability. The agent
  must systematically explore to find the goal while building a mental map
  of the environment. Tests memory and exploration strategies.
</p>

<h4>Objective</h4>
<p>Explore the maze and find the staircase to the next level.</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-ExploreMaze-Easy-v0</code></td><td>Small maze, simple layout</td></tr>
  <tr><td><code>MiniHack-ExploreMaze-Hard-v0</code></td><td>Large maze, complex layout</td></tr>
  <tr><td><code>MiniHack-ExploreMaze-Mapped-Easy-v0</code></td><td>Small maze, full visibility</td></tr>
  <tr><td><code>MiniHack-ExploreMaze-Mapped-Hard-v0</code></td><td>Large maze, full visibility</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> - 8 compass directions</p>

<h4>Observation</h4>
<p>Agent-centered crop (default 9x9). Unmapped variants only show visited areas.</p>

<h4>Key Challenges</h4>
<ul>
  <li>Partial observability - can only see nearby tiles</li>
  <li>Memory required - remember visited locations</li>
  <li>Efficient exploration - avoid revisiting areas</li>
</ul>
""" + MINIHACK_CONTROLS_HTML


MINIHACK_HIDENSEEK_HTML = """
<h3>MiniHack - Hide and Seek</h3>
<p>
  Find a hidden goal in a complex environment. The goal location is randomized
  and may be behind doors, in corridors, or hidden in rooms. Tests systematic
  search and exploration under uncertainty.
</p>

<h4>Objective</h4>
<p>Search the level to find the hidden goal.</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-HideNSeek-v0</code></td><td>Basic hide and seek</td></tr>
  <tr><td><code>MiniHack-HideNSeek-Big-v0</code></td><td>Larger search area</td></tr>
  <tr><td><code>MiniHack-HideNSeek-Lava-v0</code></td><td>With lava obstacles</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(11)</code> - Movement + OPEN, KICK, SEARCH</p>

<h4>Key Skills</h4>
<ul>
  <li>Systematic room-by-room search</li>
  <li>Opening doors (<b>o</b>) to explore new areas</li>
  <li>Searching (<b>s</b>) for secret doors</li>
  <li>Memory of explored areas</li>
</ul>

<h4>Tips</h4>
<ul>
  <li>Check every room thoroughly</li>
  <li>Secret doors may hide the goal</li>
  <li>Use search command near walls</li>
</ul>
""" + MINIHACK_CONTROLS_HTML


MINIHACK_MEMENTO_HTML = """
<h3>MiniHack - Memento</h3>
<p>
  Memory-based navigation task inspired by the movie "Memento". The agent
  must remember information from earlier in the episode to make correct
  decisions later. Tests long-term memory and information retention.
</p>

<h4>Objective</h4>
<p>Use remembered information to navigate to the correct goal.</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-Memento-Short-F2-v0</code></td><td>Short memory, 2 floors</td></tr>
  <tr><td><code>MiniHack-Memento-Short-F4-v0</code></td><td>Short memory, 4 floors</td></tr>
  <tr><td><code>MiniHack-Memento-F2-v0</code></td><td>Standard memory, 2 floors</td></tr>
  <tr><td><code>MiniHack-Memento-F4-v0</code></td><td>Standard memory, 4 floors</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> - 8 compass directions</p>

<h4>Key Challenges</h4>
<ul>
  <li><strong>Information gathering:</strong> Observe clues early in episode</li>
  <li><strong>Memory retention:</strong> Remember clues across many timesteps</li>
  <li><strong>Decision making:</strong> Use remembered info to choose path</li>
</ul>

<h4>Why "Memento"?</h4>
<p>
  Like the protagonist in the film who has short-term memory loss, the agent
  must rely on information gathered earlier that may no longer be visible.
  This tests whether RL agents can maintain and use long-term memory.
</p>
""" + MINIHACK_CONTROLS_HTML


__all__ = [
    "MINIHACK_EXPLORE_MAZE_HTML",
    "MINIHACK_HIDENSEEK_HTML",
    "MINIHACK_MEMENTO_HTML",
]
