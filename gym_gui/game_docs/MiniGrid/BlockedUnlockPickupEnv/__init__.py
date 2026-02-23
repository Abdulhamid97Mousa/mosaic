"""Documentation for MiniGrid Blocked Unlock Pickup environment."""

MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML = """
<h2>Blocked Unlock Pickup</h2>
<p>The agent has to pick up a box which is placed in another room, behind a locked door. The door is also blocked by a ball which the agent has to move before it can unlock the door. This environment requires the agent to learn a complex sequence: move the ball, pick up the key, open the door, and pick up the object in the other room.</p>

<h3>Mission</h3>
<p><em>"pick up the {color} {type}"</em></p>
<ul>
    <li><strong>{color}</strong>: red, green, blue, purple, yellow, or grey</li>
    <li><strong>{type}</strong>: box or key</li>
</ul>

<h3>Action Space</h3>
<p><strong>Discrete(7)</strong> - Full action set:</p>
<ul>
    <li><strong>0</strong>: Turn left</li>
    <li><strong>1</strong>: Turn right</li>
    <li><strong>2</strong>: Move forward</li>
    <li><strong>3</strong>: Pick up an object</li>
    <li><strong>4</strong>: Drop (unused)</li>
    <li><strong>5</strong>: Toggle (unused)</li>
    <li><strong>6</strong>: Done (unused)</li>
</ul>

<h3>Observation Space</h3>
<p>A dictionary containing:</p>
<ul>
    <li><strong>direction</strong>: Discrete(4) - Agent's current direction (0=right, 1=down, 2=left, 3=up)</li>
    <li><strong>image</strong>: Box(0, 255, (7, 7, 3), uint8) - Partially observable 7×7 grid view centered on agent</li>
    <li><strong>mission</strong>: String describing the goal with color and object type</li>
</ul>

<h3>Observation Encoding</h3>
<p>Each tile is encoded as a 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)</p>
<ul>
    <li><strong>OBJECT_IDX</strong>: Type of object (wall, ball, door, key, box, etc.)</li>
    <li><strong>COLOR_IDX</strong>: Color of the object</li>
    <li><strong>STATE</strong>: Door state (0=open, 1=closed, 2=locked)</li>
</ul>

<h3>Rewards</h3>
<ul>
    <li><strong>Success</strong>: 1 - 0.9 × (step_count / max_steps)</li>
    <li><strong>Failure</strong>: 0</li>
</ul>

<h3>Termination</h3>
<p>The episode ends when:</p>
<ul>
    <li>The agent picks up the correct box (success)</li>
    <li>Maximum steps reached (timeout)</li>
</ul>

<h3>Notes</h3>
<p>This environment can be solved without relying on language. It tests the agent's ability to plan and execute a multi-step strategy involving object manipulation and navigation.</p>

<h3>Reference</h3>
<p><a href="https://minigrid.farama.org/environments/minigrid/BlockedUnlockPickupEnv/" target="_blank">MiniGrid Blocked Unlock Pickup Documentation</a></p>
"""

__all__ = ["MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML"]
