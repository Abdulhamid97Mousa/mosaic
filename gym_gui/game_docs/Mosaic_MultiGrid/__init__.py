"""MOSAIC MultiGrid game documentation module.

MOSAIC MultiGrid is our custom multi-agent grid-world package built on
Gymnasium.  It provides competitive team-based environments (Soccer, Collect)
with simultaneous stepping, partial observability, and multi-keyboard support.

Package location: 3rd_party/mosaic_multigrid/
PyPI: mosaic-multigrid
API: Gymnasium (env.reset(seed=N), 5-tuple step returns)

Environments:
    Deprecated (v1.0.2):
        - MosaicMultiGrid-Soccer-v0: 2v2 soccer, 15x10 grid
        - MosaicMultiGrid-Collect-v0: 3-agent individual competition
        - MosaicMultiGrid-Collect-2vs2-v0: 2v2 team collection

    IndAgObs (v3.0.0+) -- RECOMMENDED:
        - MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0: 2v2 soccer, 16x11 FIFA grid
        - MosaicMultiGrid-Collect-IndAgObs-v0: 3-agent with natural termination
        - MosaicMultiGrid-Collect-2vs2-IndAgObs-v0: 2v2 with natural termination

    TeamObs (v3.0.0+) -- SMAC-style teammate awareness:
        - MosaicMultiGrid-Soccer-2vs2-TeamObs-v0: IndAgObs + teammate features
        - MosaicMultiGrid-Collect-2vs2-TeamObs-v0: IndAgObs + teammate features
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Soccer
# ---------------------------------------------------------------------------

def _get_soccer_base_html() -> str:
    """Soccer v0 (deprecated) documentation."""
    return """
<h2>MosaicMultiGrid-Soccer-v0</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Status:</strong> Deprecated -- use <code>MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0</code> instead.
</p>

<p>
A 4-player (2v2) soccer game where two teams compete to score goals.
Agents pick up the ball, navigate to the opponent's goal, and DROP to score.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">15 x 10 (13 x 8 playable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2 per team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 0, 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 2, 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball</td><td style="border: 1px solid #ddd; padding: 8px;">1 wildcard ball</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Goals</td><td style="border: 1px solid #ddd; padding: 8px;">(1,5) and (13,5)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3 (9% of grid)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">10,000</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">None (runs until max_steps)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes</td></tr>
</table>

<h4>Known Issues (Fixed in IndAgObs)</h4>
<ul>
    <li>Ball disappears after goal -- no respawn</li>
    <li>No natural termination -- episode always runs 10,000 steps</li>
    <li>Adjacent-only passing -- DROP passes to the 1-cell in front only</li>
    <li>No steal cooldown -- agents can ping-pong steal indefinitely</li>
</ul>
"""


def _get_soccer_enhanced_html() -> str:
    """Soccer IndAgObs (recommended) documentation."""
    return """
<h2>MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>RECOMMENDED</strong> for RL training and human play.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
</p>

<p>
Enhanced 2v2 soccer with <strong>teleport passing</strong>, ball respawn,
first-to-2-goals termination, and dual steal cooldown.
FIFA-ratio grid (16x11) with meaningful partial observability (3x3 view).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">16 x 11 (14 x 9 playable, FIFA 1.54 ratio)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2 per team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 0, 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 2, 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball</td><td style="border: 1px solid #ddd; padding: 8px;">1 wildcard ball (respawns after goal)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Goals</td><td style="border: 1px solid #ddd; padding: 8px;">(1,5) and (14,5) -- vertical center</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3 (9 cells, meaningful fog-of-war)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">200</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Win Condition</td><td style="border: 1px solid #ddd; padding: 8px;">First to 2 goals</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Steal Cooldown</td><td style="border: 1px solid #ddd; padding: 8px;">10 steps (both stealer and victim)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (+1 scoring team, -1 opponents)</td></tr>
</table>

<h4>Observation Space</h4>
<p>
Each agent receives a partial observation:
<code>Box(low=0, high=255, shape=(3, 3, 3), dtype=uint8)</code>
</p>
<p>The 3 channels encode: object type, color, state.</p>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> -- same for all agents:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Soccer Use</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">STILL</td><td style="border: 1px solid #ddd; padding: 8px;">Idle (stand still)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate counter-clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move in facing direction</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;">Grab ball / steal from opponent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;">Score at goal / teleport pass / drop</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;">Toggle object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;">Signal completion</td></tr>
</table>

<h4>Game Mechanics</h4>

<h5>Ball Pickup &amp; Stealing</h5>
<ul>
    <li><strong>Pickup</strong>: Face ball on ground and use PICKUP</li>
    <li><strong>Steal</strong>: Face an opponent carrying the ball and PICKUP to steal it</li>
    <li><strong>Cooldown</strong>: After stealing, both stealer and victim cannot steal for 10 steps
        (prevents ping-pong stealing exploits)</li>
</ul>

<h5>DROP Action -- Priority Chain</h5>
<p>When carrying the ball, DROP follows this priority:</p>
<ol>
    <li><strong>Score</strong>: If facing the opponent's goal, the ball scores (+1 team reward).
        Ball respawns at a random location.</li>
    <li><strong>Teleport Pass</strong>: Ball <em>instantly teleports</em> to a random eligible teammate
        anywhere on the grid (teammate must not already be carrying). This is a "blind pass"
        under partial observability -- the passer cannot see where their teammate is.</li>
    <li><strong>Ground Drop</strong>: If no teammate is available, the ball drops onto the empty cell
        in front (fallback).</li>
</ol>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>Why teleport passing?</strong> With 3x3 view on a 16x11 grid, agents can only see 9 of 126 playable cells.
Adjacent passing (old behavior) was nearly useless -- teammates had to be in the same 3x3 window.
Teleport passing creates meaningful attack/defense dynamics: pass to redistribute the ball
while opponents must mark both players. Combined with stealing, this creates a non-transitive
strategy space (pass beats solo-carry, marking beats pass, solo-carry beats marking).
</p>

<h5>Scoring &amp; Termination</h5>
<ul>
    <li>DROP ball at opponent's goal: +1 to scoring team, -1 to opponents (zero-sum)</li>
    <li>Ball respawns at random position after each goal</li>
    <li>First team to score <strong>2 goals wins</strong> -- all agents terminated</li>
    <li>If no team reaches 2 goals by step 200, episode truncates</li>
</ul>

<h4>Rewards</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Event</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Scoring Team</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Opponent Team</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Goal scored</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>+1.0</strong> (both teammates)</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>-1.0</strong> (both opponents)</td>
    </tr>
</table>

<h4>Keyboard Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #fff3e0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Soccer Use</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><em>(no key)</em></td><td style="border: 1px solid #ddd; padding: 8px;">STILL</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Idle (stand still)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A or Left</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Rotate counter-clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D or Right</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Rotate clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W or Up</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Move in facing direction</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Grab ball / steal from opponent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Score / teleport pass / drop</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Toggle object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>7</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Signal completion</td></tr>
</table>
<p style="background-color: #ffecb3; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>Note:</strong> MOSAIC MultiGrid uses <code>STILL (0)</code> as the idle action when no key is pressed.
This is different from INI MultiGrid which uses <code>DONE (6)</code>.
</p>

<h4>Improvements over Deprecated Soccer-v0</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Feature</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Soccer-v0 (Deprecated)</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Soccer-2vs2-IndAgObs-v0</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid</td><td style="border: 1px solid #ddd; padding: 8px;">15x10</td><td style="border: 1px solid #ddd; padding: 8px;">16x11 (FIFA ratio)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Passing</td><td style="border: 1px solid #ddd; padding: 8px;">1-cell adjacency (useless)</td><td style="border: 1px solid #ddd; padding: 8px;">Teleport to teammate</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball after goal</td><td style="border: 1px solid #ddd; padding: 8px;">Disappears (bug)</td><td style="border: 1px solid #ddd; padding: 8px;">Respawns at random</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">Never (10k steps)</td><td style="border: 1px solid #ddd; padding: 8px;">First to 2 goals</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Steal cooldown</td><td style="border: 1px solid #ddd; padding: 8px;">None</td><td style="border: 1px solid #ddd; padding: 8px;">10 steps (dual)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Avg episode</td><td style="border: 1px solid #ddd; padding: 8px;">10,000 steps</td><td style="border: 1px solid #ddd; padding: 8px;">~200 steps</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Training speed</td><td style="border: 1px solid #ddd; padding: 8px;">Baseline</td><td style="border: 1px solid #ddd; padding: 8px;">~50x faster</td></tr>
</table>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>Class: <code>SoccerGame4HIndAgObsEnv16x11N2</code></li>
    <li>See: <code>SOCCER_IMPROVEMENTS.md</code> for full technical details</li>
</ul>
"""


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def _get_collect_base_html() -> str:
    """Collect v0 (deprecated 3-agent) documentation."""
    return """
<h2>MosaicMultiGrid-Collect-v0</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Status:</strong> Deprecated -- use <code>MosaicMultiGrid-Collect-IndAgObs-v0</code> instead.
</p>

<p>
A 3-player ball collection game where each agent is their own team.
Agents compete to pick up wildcard balls. Ball is consumed on pickup (not carried).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">3 (each on their own team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">5 wildcard (index=0)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">10,000</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">None (bug: runs until max_steps)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes</td></tr>
</table>

<h4>Known Issue</h4>
<p>
Episode does <strong>not terminate</strong> when all balls are collected.
Agents wander aimlessly for ~9,500 remaining steps with zero reward.
Fixed in IndAgObs variant.
</p>
"""


def _get_collect_enhanced_html() -> str:
    """Collect IndAgObs (recommended) documentation."""
    return """
<h2>MosaicMultiGrid-Collect-IndAgObs-v0</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>RECOMMENDED</strong> for RL training.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
</p>

<p>
Enhanced 3-agent individual competition with <strong>natural termination</strong>
when all balls are collected. 35x faster training than the deprecated variant.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10 (8 x 8 playable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">3 (each on their own team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">5 wildcard (index=0, any agent can collect)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3 (9% of grid)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">300</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">When all 5 balls collected</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (+1 collector, -1 others)</td></tr>
</table>

<h4>Game Mechanics</h4>
<ul>
    <li><strong>Pickup</strong>: Face a ball and use PICKUP -- ball is consumed (not carried)</li>
    <li><strong>Team matching</strong>: Wildcard balls (index=0) can be collected by anyone;
        team-indexed balls can only be collected by matching team</li>
    <li><strong>Zero-sum reward</strong>: Collector gets +1, all other agents get -1</li>
    <li><strong>Termination</strong>: Episode ends when all balls are picked up</li>
</ul>

<h4>Rewards</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Event</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Collecting Agent</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Other Agents</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Ball pickup</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>+1.0</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>-1.0</strong> each</td>
    </tr>
</table>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>Class: <code>CollectGame3HIndAgObsEnv10x10N3</code></li>
    <li>See: <code>COLLECT_IMPROVEMENTS.md</code> for full technical details</li>
</ul>
"""


def _get_collect2vs2_base_html() -> str:
    """Collect2vs2 v0 (deprecated) documentation."""
    return """
<h2>MosaicMultiGrid-Collect-2vs2-v0</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Status:</strong> Deprecated -- use <code>MosaicMultiGrid-Collect-2vs2-IndAgObs-v0</code> instead.
</p>

<p>
A 4-agent (2v2) team ball collection game. Green team (Agents 0, 1) vs Red team (Agents 2, 3).
7 wildcard balls (odd number prevents draws).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2v2 teams)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1 (Green)</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 0, 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2 (Red)</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 2, 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">7 wildcard (odd = no draws)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">10,000</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">None (runs until max_steps)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes</td></tr>
</table>
"""


def _get_collect2vs2_enhanced_html() -> str:
    """Collect2vs2 IndAgObs (recommended) documentation."""
    return """
<h2>MosaicMultiGrid-Collect-2vs2-IndAgObs-v0</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>RECOMMENDED</strong> for RL training.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
</p>

<p>
Enhanced 2v2 team ball collection with <strong>natural termination</strong>.
Green team (Agents 0, 1) vs Red team (Agents 2, 3) compete to collect
7 wildcard balls. Odd number guarantees a winner (no draws).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10 (8 x 8 playable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2v2 teams)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1 (Green)</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 0, 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2 (Red)</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 2, 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">7 wildcard (odd = guaranteed winner)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">400</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">When all 7 balls collected</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (+1 team, -1 opponents)</td></tr>
</table>

<h4>Game Mechanics</h4>
<ul>
    <li><strong>Pickup</strong>: Face a ball and use PICKUP -- ball consumed on pickup</li>
    <li><strong>Team reward</strong>: Both teammates get +1 when either collects; both opponents get -1</li>
    <li><strong>Guaranteed winner</strong>: 7 balls means one team gets at least 4, other at most 3</li>
    <li><strong>Termination</strong>: Episode ends immediately when all 7 balls collected</li>
</ul>

<h4>Strategy</h4>
<ul>
    <li><strong>Role splitting</strong>: One agent collects, one blocks opponents</li>
    <li><strong>Map coverage</strong>: Split search areas (left/right) to avoid chasing same ball</li>
    <li><strong>MAPPO</strong>: Learns team coordination ("don't both chase same ball")</li>
</ul>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>Class: <code>CollectGame4HIndAgObsEnv10x10N2</code></li>
    <li>See: <code>COLLECT_IMPROVEMENTS.md</code> for full technical details</li>
</ul>
"""


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def get_mosaic_multigrid_html(env_id: str) -> str:
    """Generate MOSAIC MultiGrid HTML documentation for a specific variant.

    Args:
        env_id: Environment identifier (e.g., "MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0")

    Returns:
        HTML string containing environment documentation.
    """
    _is_modern = ("IndAgObs" in env_id or "TeamObs" in env_id
                   or "Enhanced" in env_id)
    if "Soccer" in env_id:
        if _is_modern:
            return _get_soccer_enhanced_html()
        return _get_soccer_base_html()
    elif "Collect-2vs2" in env_id or "Collect2vs2" in env_id:
        if _is_modern:
            return _get_collect2vs2_enhanced_html()
        return _get_collect2vs2_base_html()
    elif "Collect-1vs1" in env_id:
        return _get_collect2vs2_enhanced_html()  # 1vs1 shares collect team docs
    elif "Collect" in env_id:
        if _is_modern:
            return _get_collect_enhanced_html()
        return _get_collect_base_html()
    else:
        return _get_overview_html()


def _get_overview_html() -> str:
    """Return overview HTML for MOSAIC MultiGrid family."""
    return """
<h2>MOSAIC MultiGrid</h2>

<p>
<strong>MOSAIC MultiGrid</strong> is a competitive multi-agent grid-world package
built on Gymnasium. It provides team-based environments (Soccer, Collect) with
simultaneous stepping, partial observability (3x3 view), and multi-keyboard support.
</p>

<h4>Available Environments</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Agents</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Status</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">4 (2v2)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Zero-sum soccer</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Collect-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Individual competition</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Collect-2vs2-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">4 (2v2)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Team collection</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">2 (1v1)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Zero-sum soccer</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Collect-1vs1-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">2 (1v1)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Team collection</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
</table>

<h4>API</h4>
<p>
Uses <strong>Gymnasium</strong> API:
<code>env.reset(seed=42)</code> returns <code>(obs, info)</code>.
<code>env.step(actions)</code> returns <code>(obs, rewards, terminated, truncated, info)</code>.
</p>

<h4>Source</h4>
<p><code>3rd_party/mosaic_multigrid/</code></p>
"""


# Pre-generated HTML constants for convenience
MOSAIC_SOCCER_HTML = _get_soccer_enhanced_html()
MOSAIC_SOCCER_BASE_HTML = _get_soccer_base_html()
MOSAIC_COLLECT_HTML = _get_collect_enhanced_html()
MOSAIC_COLLECT_BASE_HTML = _get_collect_base_html()
MOSAIC_COLLECT2VS2_HTML = _get_collect2vs2_enhanced_html()
MOSAIC_COLLECT2VS2_BASE_HTML = _get_collect2vs2_base_html()

__all__ = [
    "get_mosaic_multigrid_html",
    "MOSAIC_SOCCER_HTML",
    "MOSAIC_SOCCER_BASE_HTML",
    "MOSAIC_COLLECT_HTML",
    "MOSAIC_COLLECT_BASE_HTML",
    "MOSAIC_COLLECT2VS2_HTML",
    "MOSAIC_COLLECT2VS2_BASE_HTML",
]
