"""Documentation for MiniGrid Obstructed Maze Full variant."""
from __future__ import annotations

MINIGRID_OBSTRUCTED_MAZE_FULL_HTML = (
    "<h2>Obstructed Maze - Full (v1)</h2>"
    "<p>The complete 3×3 maze configuration with maximum complexity. A blue ball is hidden in one of 4 corners, with locked doors, obstructed doors, and keys hidden in boxes. This is one of the most challenging MiniGrid environments.</p>"
    
    "<h3>Environment ID</h3>"
    "<p><strong>MiniGrid-ObstructedMaze-Full-v1</strong></p>"
    
    "<h3>Environment Details</h3>"
    "<ul>"
    "<li><strong>Grid Size</strong>: 3×3 maze layout</li>"
    "<li><strong>Complexity</strong>: Maximum - requires multi-step planning and reasoning</li>"
    "<li><strong>Puzzle Elements</strong>: Locked doors, obstructed doors, keys in boxes, hidden target</li>"
    "<li><strong>Target Location</strong>: Blue ball randomly placed in one of 4 corners</li>"
    "</ul>"
    
    "<h3>Mission</h3>"
    "<p>Find and reach the hidden blue ball</p>"
    
    "<h3>Action Space</h3>"
    "<p><strong>Discrete(7)</strong> - All actions required for solving:</p>"
    "<ul>"
    "<li><strong>0</strong>: Turn left</li>"
    "<li><strong>1</strong>: Turn right</li>"
    "<li><strong>2</strong>: Move forward</li>"
    "<li><strong>3</strong>: Pick up an object (keys, balls)</li>"
    "<li><strong>4</strong>: Drop an object</li>"
    "<li><strong>5</strong>: Toggle/activate an object (open boxes, unlock doors)</li>"
    "<li><strong>6</strong>: Done</li>"
    "</ul>"
    
    "<h3>Observation Space</h3>"
    "<p>A dictionary containing:</p>"
    "<ul>"
    "<li><strong>direction</strong>: Discrete(4) - Agent's current direction (0=right, 1=down, 2=left, 3=up)</li>"
    "<li><strong>image</strong>: Box(0, 255, (7, 7, 3), uint8) - Partially observable 7×7 grid view</li>"
    "<li><strong>mission</strong>: MissionSpace with blue ball target</li>"
    "</ul>"
    
    "<h3>Rewards</h3>"
    "<ul>"
    "<li><strong>Success</strong>: 1 - 0.9 × (step_count / max_steps)</li>"
    "<li><strong>Failure</strong>: 0</li>"
    "</ul>"
    
    "<h3>Termination</h3>"
    "<p>The episode ends when:</p>"
    "<ul>"
    "<li>The agent reaches the hidden blue ball (success)</li>"
    "<li>Maximum steps reached (timeout)</li>"
    "</ul>"
    
    "<h3>Required Strategy</h3>"
    "<p>To successfully solve this environment, the agent must:</p>"
    "<ol>"
    "<li>Explore the 3×3 maze to locate boxes containing keys</li>"
    "<li>Open boxes by toggling them to retrieve keys</li>"
    "<li>Move obstructing balls out of doorways</li>"
    "<li>Use keys to unlock doors blocking access to other rooms</li>"
    "<li>Systematically search all 4 corners to find the blue ball</li>"
    "<li>Navigate to the blue ball's location</li>"
    "</ol>"
    
    "<h3>Notes</h3>"
    "<p>This is an extremely challenging environment that requires sophisticated planning, memory, and multi-step reasoning. The agent must coordinate complex sequences of actions while dealing with partial observability. Success typically requires advanced exploration strategies and hierarchical planning.</p>"
    
    "<h3>Reference</h3>"
    "<p><a href=\"https://minigrid.farama.org/environments/minigrid/ObstructedMazeEnv/\" target=\"_blank\">MiniGrid Obstructed Maze Documentation</a></p>"
)

__all__ = ["MINIGRID_OBSTRUCTED_MAZE_FULL_HTML"]
