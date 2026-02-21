"""INI MultiGrid game documentation module.

INI MultiGrid is a cooperative multi-agent extension of MiniGrid by INI
(github.com/ini/multigrid). It provides cooperative exploration environments
with simultaneous stepping.

Package location: 3rd_party/multigrid-ini/
Repository: https://github.com/ini/multigrid
API: Gymnasium (env.reset(seed=N), 5-tuple step returns)

Environments:
    - MultiGrid-Empty-{5x5,6x6,8x8,16x16}-v0: Training grounds
    - MultiGrid-Empty-Random-{5x5,6x6}-v0: Random spawn variants
    - MultiGrid-RedBlueDoors-{6x6,8x8}-v0: Sequential door puzzle
    - MultiGrid-LockedHallway-{2,4,6}Rooms-v0: Multi-room navigation
    - MultiGrid-BlockedUnlockPickup-v0: Complex pickup with obstacles
    - MultiGrid-Playground-v0: Diverse object sandbox
"""

from __future__ import annotations


def get_ini_multigrid_html(env_id: str) -> str:
    """Generate INI MultiGrid HTML documentation for a specific variant.

    Args:
        env_id: Environment identifier (e.g., "MultiGrid-Empty-8x8-v0")

    Returns:
        HTML string containing environment documentation.
    """
    if "RedBlueDoors" in env_id:
        return _get_redbluedoors_html(env_id)
    elif "Empty" in env_id:
        return _get_empty_html(env_id)
    elif "LockedHallway" in env_id:
        return _get_lockedhallway_html(env_id)
    elif "BlockedUnlockPickup" in env_id:
        return _get_blockedunlockpickup_html()
    elif "Playground" in env_id:
        return _get_playground_html()
    else:
        return _get_overview_html()


def _get_overview_html() -> str:
    """Return overview HTML for INI MultiGrid family."""
    return """
<h2>INI MultiGrid</h2>

<p>
<strong>INI MultiGrid</strong> is a cooperative multi-agent extension of MiniGrid
by INI. All agents act simultaneously each step. Focused on cooperative
exploration, puzzle-solving, and navigation tasks.
</p>

<h4>Available Environments</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Agents</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Empty-*-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Empty grid training ground</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-RedBlueDoors-*-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Sequential door-opening puzzle</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-LockedHallway-*-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Multi-room navigation with keys</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-BlockedUnlockPickup-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Complex pickup with obstacles</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Playground-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Diverse object sandbox</td>
    </tr>
</table>

<h4>API</h4>
<p>
Uses <strong>Gymnasium</strong> API:
<code>env.reset(seed=42)</code> returns <code>(obs, info)</code>.
<code>env.step(actions)</code> returns <code>(obs, rewards, terminated, truncated, info)</code>.
</p>

<h4>Action Space (7 actions)</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #e3f2fd;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A or Left</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D or Right</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W or Up</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q or <em>(no key)</em></td><td style="border: 1px solid #ddd; padding: 8px;">DONE (idle)</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td></tr>
</table>
<p style="background-color: #ffecb3; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>Note:</strong> INI MultiGrid has no STILL action. When no key is pressed,
<code>DONE (6)</code> is sent as idle. This differs from MOSAIC MultiGrid which uses <code>NOOP (0)</code>.
</p>

<h4>Source</h4>
<ul>
    <li><code>3rd_party/multigrid-ini/</code></li>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_redbluedoors_html(env_id: str) -> str:
    """Return HTML documentation for RedBlueDoors environment."""
    if "6x6" in env_id:
        size = "6x6"
        max_steps = 720
    else:
        size = "8x8"
        max_steps = 1280
    return f"""
<h2>MultiGrid-RedBlueDoors-{size}-v0</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> INI MultiGrid (Gymnasium) -- <code>env.reset(seed=N)</code>.
<a href="https://github.com/ini/multigrid" target="_blank">GitHub</a>
</p>

<p>
A cooperative multi-agent puzzle where agents must open doors in sequence:
first the <strong>red door</strong>, then the <strong>blue door</strong>.
Opening them in the wrong order results in failure.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">{size}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1-4 (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Task</td><td style="border: 1px solid #ddd; padding: 8px;">Open red door, then blue door</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>{max_steps}</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Reward</td><td style="border: 1px solid #ddd; padding: 8px;">1 - 0.9 * (step_count / max_steps)</td></tr>
</table>

<h4>Success/Failure Conditions</h4>
<ul>
    <li><strong>Success</strong>: Any agent opens blue door while red door is already open</li>
    <li><strong>Failure</strong>: Any agent opens blue door while red door is still closed</li>
    <li><strong>Truncation</strong>: Maximum steps reached</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_empty_html(env_id: str) -> str:
    """Return HTML documentation for Empty environment."""
    if "5x5" in env_id:
        size, grid_size, max_steps = "5x5", "5", 100
    elif "6x6" in env_id:
        size, grid_size, max_steps = "6x6", "6", 144
    elif "16x16" in env_id:
        size, grid_size, max_steps = "16x16", "16", 1024
    else:
        size, grid_size, max_steps = "8x8", "8", 256

    is_random = "Random" in env_id
    spawn_mode = "random positions" if is_random else "fixed corner position"

    return f"""
<h2>MultiGrid-Empty-{size}-v0</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> INI MultiGrid (Gymnasium) -- <code>env.reset(seed=N)</code>.
<a href="https://github.com/ini/multigrid" target="_blank">GitHub</a>
</p>

<p>
An empty {size} grid environment with no obstacles.
Agents spawn at {spawn_mode} and must navigate to the green goal square.
The simplest MultiGrid environment, ideal for testing basic navigation.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">{grid_size} x {grid_size}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1-4 (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Spawn</td><td style="border: 1px solid #ddd; padding: 8px;">{spawn_mode.capitalize()}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Task</td><td style="border: 1px solid #ddd; padding: 8px;">Navigate to green goal square</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>{max_steps}</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Reward</td><td style="border: 1px solid #ddd; padding: 8px;">1 - 0.9 * (step_count / max_steps)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_lockedhallway_html(env_id: str) -> str:
    """Return HTML documentation for LockedHallway environment."""
    if "2Rooms" in env_id:
        num_rooms = 2
        max_steps = 400
    elif "4Rooms" in env_id:
        num_rooms = 4
        max_steps = 800
    else:
        num_rooms = 6
        max_steps = 1200

    return f"""
<h2>MultiGrid-LockedHallway-{num_rooms}Rooms-v0</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> INI MultiGrid (Gymnasium) -- <code>env.reset(seed=N)</code>.
<a href="https://github.com/ini/multigrid" target="_blank">GitHub</a>
</p>

<p>
A multi-room navigation puzzle with {num_rooms} rooms connected by locked doors.
Agents must find keys to unlock doors and navigate through all rooms to reach the goal.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Number of Rooms</td><td style="border: 1px solid #ddd; padding: 8px;">{num_rooms}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1-4 (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Keys</td><td style="border: 1px solid #ddd; padding: 8px;">{num_rooms - 1} (one per locked door)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Task</td><td style="border: 1px solid #ddd; padding: 8px;">Unlock all doors and reach goal</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>{max_steps}</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Reward</td><td style="border: 1px solid #ddd; padding: 8px;">1 - 0.9 * (step_count / max_steps)</td></tr>
</table>

<h4>Key Mechanics</h4>
<ul>
    <li><strong>Key Collection</strong>: Use <code>PICKUP</code> to collect keys</li>
    <li><strong>Door Unlocking</strong>: Use <code>TOGGLE</code> on locked door while carrying matching key</li>
    <li><strong>Sequential Progress</strong>: Must unlock doors in order to progress</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_blockedunlockpickup_html() -> str:
    """Return HTML documentation for BlockedUnlockPickup environment."""
    return """
<h2>MultiGrid-BlockedUnlockPickup-v0</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> INI MultiGrid (Gymnasium) -- <code>env.reset(seed=N)</code>.
<a href="https://github.com/ini/multigrid" target="_blank">GitHub</a>
</p>

<p>
The objective is to <strong>pick up a box</strong> in another room,
behind a <strong>locked door</strong> that is also <strong>blocked by a ball</strong>.
Agents must learn: move ball, pick up key, open door, pick up box.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Layout</td><td style="border: 1px solid #ddd; padding: 8px;">2 rooms (1 row x 2 columns)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Room Size</td><td style="border: 1px solid #ddd; padding: 8px;">6x6 (default)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1+ (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Mission</td><td style="border: 1px solid #ddd; padding: 8px;">"pick up the {{color}} box"</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>576</strong></td></tr>
</table>

<h4>Task Sequence</h4>
<ol>
    <li><strong>Move the Ball</strong>: Push or pick up the ball blocking the door</li>
    <li><strong>Pick Up Key</strong>: Use <code>PICKUP</code> action on the key</li>
    <li><strong>Unlock Door</strong>: Face door and use <code>TOGGLE</code> while carrying key</li>
    <li><strong>Enter Right Room</strong>: Move through the open door</li>
    <li><strong>Pick Up Box</strong>: Use <code>PICKUP</code> on the target box -- Success!</li>
</ol>

<h4>Action Space</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Use</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Turn left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Turn right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move forward</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;">Pick up key, ball, or box</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;">Drop carried object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;">Unlock door (with key)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;">Idle (no action)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_playground_html() -> str:
    """Return HTML documentation for Playground environment."""
    return """
<h2>MultiGrid-Playground-v0</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> INI MultiGrid (Gymnasium) -- <code>env.reset(seed=N)</code>.
<a href="https://github.com/ini/multigrid" target="_blank">GitHub</a>
</p>

<p>
A diverse environment filled with various objects and interactive elements.
Acts as a sandbox for testing agent behaviors with multiple object types,
colors, and interaction mechanics all in one place.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1-4 (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Objects</td><td style="border: 1px solid #ddd; padding: 8px;">Keys, doors, boxes, balls, walls, goals</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Purpose</td><td style="border: 1px solid #ddd; padding: 8px;">Exploratory sandbox</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>100</strong></td></tr>
</table>

<h4>Use Cases</h4>
<ul>
    <li><strong>Behavior Testing</strong>: See how agents interact with diverse objects</li>
    <li><strong>Transfer Learning</strong>: Pre-train on varied interactions</li>
    <li><strong>Emergent Behaviors</strong>: Observe unexpected agent strategies</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


# Pre-generated HTML constants for convenience
INI_MULTIGRID_OVERVIEW_HTML = _get_overview_html()

__all__ = [
    "get_ini_multigrid_html",
    "INI_MULTIGRID_OVERVIEW_HTML",
]
