"""Documentation for MiniGrid Multi Room environments."""

MINIGRID_MULTIROOM_HTML = """
<h2>Multi Room</h2>
<p>This environment has a series of connected rooms with doors that must be opened in order to get to the next room. The final room has the green goal square the agent must reach. This environment is extremely difficult to solve using RL alone. However, by gradually increasing the number of rooms and building a curriculum, the environment can be solved.</p>

<h3>Variants</h3>
<ul>
    <li><strong>MiniGrid-MultiRoom-N2-S4-v0</strong>: 2 rooms in a 4×4 grid (two small rooms)</li>
    <li><strong>MiniGrid-MultiRoom-N4-S5-v0</strong>: 4 rooms in a 5×5 grid (four rooms)</li>
    <li><strong>MiniGrid-MultiRoom-N6-v0</strong>: 6 rooms (six rooms, standard configuration)</li>
</ul>

<h3>Mission</h3>
<p><em>"traverse the rooms to get to the goal"</em></p>

<h3>Action Space</h3>
<p><strong>Discrete(7)</strong> - Full action set:</p>
<ul>
    <li><strong>0</strong>: Turn left</li>
    <li><strong>1</strong>: Turn right</li>
    <li><strong>2</strong>: Move forward</li>
    <li><strong>3</strong>: Pick up (unused)</li>
    <li><strong>4</strong>: Drop (unused)</li>
    <li><strong>5</strong>: Toggle/activate an object (used to open doors)</li>
    <li><strong>6</strong>: Done (unused)</li>
</ul>

<h3>Observation Space</h3>
<p>A dictionary containing:</p>
<ul>
    <li><strong>direction</strong>: Discrete(4) - Agent's current direction (0=right, 1=down, 2=left, 3=up)</li>
    <li><strong>image</strong>: Box(0, 255, (7, 7, 3), uint8) - Partially observable 7×7 grid view centered on agent</li>
    <li><strong>mission</strong>: String describing the goal</li>
</ul>

<h3>Observation Encoding</h3>
<p>Each tile is encoded as a 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)</p>
<ul>
    <li><strong>OBJECT_IDX</strong>: Type of object (wall, door, goal, etc.)</li>
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
    <li>The agent reaches the green goal square in the final room</li>
    <li>Maximum steps reached (timeout)</li>
</ul>

<h3>Notes</h3>
<p>This environment tests navigation through multiple rooms and door manipulation. The curriculum approach (starting with N2-S4 and progressing to N6) is recommended for training. The toggle action (5) is essential for opening doors between rooms.</p>

<h3>Reference</h3>
<p><a href="https://minigrid.farama.org/environments/minigrid/MultiRoomEnv/" target="_blank">MiniGrid Multi Room Documentation</a></p>
"""

__all__ = ["MINIGRID_MULTIROOM_HTML"]
