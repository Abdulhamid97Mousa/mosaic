"""Shared HTML fragments for RWARE (Robotic Warehouse) game documentation."""

# Consistent inline CSS matching MOSAIC MultiGrid style
_TH = 'style="border: 1px solid #ddd; padding: 8px;"'
_TD = 'style="border: 1px solid #ddd; padding: 8px;"'
_TABLE = 'style="width:100%; border-collapse: collapse; margin: 10px 0;"'
_HDR_ROW = 'style="background-color: #f0f0f0;"'
_KBD_HDR_ROW = 'style="background-color: #fff3e0;"'

ACTIONS_HTML = f"""
<h3>Action Space</h3>
<p><code>Discrete(5)</code> -- same for all agents:</p>
<table {_TABLE}>
    <tr {_HDR_ROW}>
        <th {_TH}>ID</th>
        <th {_TH}>Action</th>
        <th {_TH}>Description</th>
    </tr>
    <tr><td {_TD}>0</td><td {_TD}>NOOP</td><td {_TD}>Do nothing (idle)</td></tr>
    <tr><td {_TD}>1</td><td {_TD}>FORWARD</td><td {_TD}>Move forward one cell in current direction</td></tr>
    <tr><td {_TD}>2</td><td {_TD}>LEFT</td><td {_TD}>Rotate 90 degrees counter-clockwise</td></tr>
    <tr><td {_TD}>3</td><td {_TD}>RIGHT</td><td {_TD}>Rotate 90 degrees clockwise</td></tr>
    <tr><td {_TD}>4</td><td {_TD}>TOGGLE_LOAD</td><td {_TD}>Pick up shelf (if adjacent) or put down carried shelf</td></tr>
</table>
"""

OBSERVATIONS_HTML = f"""
<h3>Observations</h3>
<p>Each agent receives a local observation based on its sensor range (default: 1 cell).
Observation types are configurable via the config panel:</p>
<table {_TABLE}>
    <tr {_HDR_ROW}>
        <th {_TH}>Type</th>
        <th {_TH}>Shape</th>
        <th {_TH}>Description</th>
    </tr>
    <tr><td {_TD}><b>Flattened</b> (default)</td><td {_TD}>1D vector</td><td {_TD}>Self-info + sensor grid data concatenated</td></tr>
    <tr><td {_TD}>Dict</td><td {_TD}>Nested dict</td><td {_TD}>Self location, direction, carrying status, sensor readings</td></tr>
    <tr><td {_TD}>Image</td><td {_TD}>Multi-channel grid</td><td {_TD}>Channels: shelves, requests, agents, goals, accessible cells</td></tr>
    <tr><td {_TD}>Image+Dict</td><td {_TD}>Combined</td><td {_TD}>Image and dictionary observations together</td></tr>
</table>
<p>Self info includes: position (x,y), carrying shelf flag, direction (one-hot), on highway flag.</p>
"""

REWARDS_HTML = f"""
<h3>Reward Types</h3>
<table {_TABLE}>
    <tr {_HDR_ROW}>
        <th {_TH}>Type</th>
        <th {_TH}>On Delivery</th>
        <th {_TH}>Description</th>
    </tr>
    <tr>
        <td {_TD}><b>Individual</b> (default)</td>
        <td {_TD}><strong>+1.0</strong> delivering agent only</td>
        <td {_TD}>Only the agent that completes the delivery is rewarded</td>
    </tr>
    <tr>
        <td {_TD}>Global</td>
        <td {_TD}><strong>+1.0</strong> all agents</td>
        <td {_TD}>All agents receive reward when any agent delivers</td>
    </tr>
    <tr>
        <td {_TD}>Two-Stage</td>
        <td {_TD}><strong>+0.5</strong> delivery + <strong>+0.5</strong> return</td>
        <td {_TD}>Split reward: half for delivery, half when shelf returned</td>
    </tr>
</table>
<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>Note:</strong> Rewards are sparse -- agents only receive reward on successful delivery events.
</p>
"""

MECHANICS_HTML = """
<h3>Environment Mechanics</h3>
<ul>
    <li>Robots navigate a grid warehouse with shelves arranged in columns</li>
    <li>Requested shelves (highlighted) must be picked up and delivered to goal locations</li>
    <li>Unloaded agents can move beneath shelves; loaded agents must use corridors (highways)</li>
    <li><strong>Collision resolution:</strong> when multiple agents target the same cell, priority
        goes to blocking agents; otherwise arbitrary selection</li>
    <li><strong>Communication:</strong> optional message bits allow agents to broadcast binary signals
        (configurable: 0-8 bits per agent, 0 = silent)</li>
</ul>
"""

KEYBOARD_HTML = f"""
<h3>Keyboard Controls</h3>
<table {_TABLE}>
    <tr {_KBD_HDR_ROW}>
        <th {_TH}>Key</th>
        <th {_TH}>Action</th>
        <th {_TH}>ID</th>
        <th {_TH}>Warehouse Use</th>
    </tr>
    <tr><td {_TD}><em>(no key)</em></td><td {_TD}>NOOP</td><td {_TD}><strong>0</strong></td><td {_TD}>Idle (do nothing)</td></tr>
    <tr><td {_TD}>W or Up</td><td {_TD}>FORWARD</td><td {_TD}><strong>1</strong></td><td {_TD}>Move forward one cell</td></tr>
    <tr><td {_TD}>A or Left</td><td {_TD}>LEFT</td><td {_TD}><strong>2</strong></td><td {_TD}>Rotate counter-clockwise</td></tr>
    <tr><td {_TD}>D or Right</td><td {_TD}>RIGHT</td><td {_TD}><strong>3</strong></td><td {_TD}>Rotate clockwise</td></tr>
    <tr><td {_TD}>Space, E, or Enter</td><td {_TD}>TOGGLE_LOAD</td><td {_TD}><strong>4</strong></td><td {_TD}>Pick up / put down shelf</td></tr>
</table>
<p style="background-color: #ffecb3; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>Note:</strong> RWARE uses <code>NOOP (0)</code> as the idle action when no key is pressed.
Multi-keyboard support is available for simultaneous multi-agent human play.
</p>
"""

REFERENCE_HTML = """
<h3>References</h3>
<ul>
    <li>Papoudakis et al. (2021). "Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks"</li>
    <li>Repository: <a href="https://github.com/uoe-agents/robotic-warehouse">github.com/uoe-agents/robotic-warehouse</a></li>
    <li>Source: <code>3rd_party/robotic-warehouse/</code></li>
</ul>
"""
