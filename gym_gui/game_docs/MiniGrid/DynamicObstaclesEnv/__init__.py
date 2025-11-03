"""Documentation for MiniGrid Dynamic Obstacles environments."""

MINIGRID_DYNAMIC_OBSTACLES_HTML = """
<h2>Dynamic Obstacles</h2>
<p>This environment is an empty room with moving obstacles. The goal of the agent is to reach the green goal square without colliding with any obstacle. A large penalty is subtracted if the agent collides with an obstacle and the episode finishes.</p>

<h3>Variants</h3>
<ul>
    <li><strong>MiniGrid-Dynamic-Obstacles-5x5-v0</strong>: 5×5 grid with moving obstacles</li>
    <li><strong>MiniGrid-Dynamic-Obstacles-Random-5x5-v0</strong>: 5×5 grid with randomized obstacles</li>
    <li><strong>MiniGrid-Dynamic-Obstacles-6x6-v0</strong>: 6×6 grid with moving obstacles</li>
    <li><strong>MiniGrid-Dynamic-Obstacles-Random-6x6-v0</strong>: 6×6 grid with randomized obstacles</li>
    <li><strong>MiniGrid-Dynamic-Obstacles-8x8-v0</strong>: 8×8 grid with moving obstacles</li>
    <li><strong>MiniGrid-Dynamic-Obstacles-16x16-v0</strong>: 16×16 grid with moving obstacles</li>
</ul>

<h3>Mission</h3>
<p><em>"get to the green goal square"</em></p>

<h3>Action Space</h3>
<p><strong>Discrete(3)</strong> - Only movement actions are used:</p>
<ul>
    <li><strong>0</strong>: Turn left</li>
    <li><strong>1</strong>: Turn right</li>
    <li><strong>2</strong>: Move forward</li>
</ul>

<h3>Observation Space</h3>
<p>A dictionary containing:</p>
<ul>
    <li><strong>direction</strong>: Discrete(4) - Agent's current direction (0=right, 1=down, 2=left, 3=up)</li>
    <li><strong>image</strong>: Box(0, 255, (7, 7, 3), uint8) - Partially observable 7×7 grid view centered on agent</li>
    <li><strong>mission</strong>: String describing the goal</li>
</ul>

<h3>Rewards</h3>
<ul>
    <li><strong>Success</strong>: 1 - 0.9 × (step_count / max_steps)</li>
    <li><strong>Collision</strong>: -1 penalty</li>
    <li><strong>Failure</strong>: 0</li>
</ul>

<h3>Termination</h3>
<p>The episode ends when:</p>
<ul>
    <li>The agent reaches the green goal square</li>
    <li>The agent collides with an obstacle (receives -1 penalty)</li>
    <li>Maximum steps reached (timeout)</li>
</ul>

<h3>Reference</h3>
<p><a href="https://minigrid.farama.org/environments/minigrid/DynamicObstaclesEnv/" target="_blank">MiniGrid Dynamic Obstacles Documentation</a></p>
"""

__all__ = ["MINIGRID_DYNAMIC_OBSTACLES_HTML"]
