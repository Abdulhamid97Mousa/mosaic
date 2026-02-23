"""Documentation for MiniGrid Obstructed Maze 1Dlhb variant."""
from __future__ import annotations

MINIGRID_OBSTRUCTED_MAZE_1DLHB_HTML = (
    "<h2>Obstructed Maze - 1Dlhb (v1)</h2>"
    "<p>A 2×1 maze configuration variant with obstructed doors and hidden keys. This is a simplified version of the full obstructed maze that focuses on the core puzzle mechanics.</p>"
    
    "<h3>Environment ID</h3>"
    "<p><strong>MiniGrid-ObstructedMaze-1Dlhb-v1</strong></p>"
    
    "<h3>Environment Details</h3>"
    "<ul>"
    "<li><strong>Grid Size</strong>: 2×1 maze layout</li>"
    "<li><strong>Complexity</strong>: Reduced compared to Full variant</li>"
    "<li><strong>Puzzle Elements</strong>: Obstructed doors, hidden keys in boxes</li>"
    "</ul>"
    
    "<h3>Mission</h3>"
    "<p>Find and reach the hidden blue ball</p>"
    
    "<h3>Action Space</h3>"
    "<p><strong>Discrete(7)</strong> - Full action set required:</p>"
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
    
    "<h3>Strategy</h3>"
    "<p>This simplified variant is useful for initial training and testing puzzle-solving mechanics before attempting the full 3×3 maze version.</p>"
    
    "<h3>Reference</h3>"
    "<p><a href=\"https://minigrid.farama.org/environments/minigrid/ObstructedMazeEnv/\" target=\"_blank\">MiniGrid Obstructed Maze Documentation</a></p>"
)

__all__ = ["MINIGRID_OBSTRUCTED_MAZE_1DLHB_HTML"]
