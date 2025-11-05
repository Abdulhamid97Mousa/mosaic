"""Documentation for MiniGrid Obstructed Maze environments."""
from __future__ import annotations


def get_obstructed_maze_html(env_id: str) -> str:
    """Generate Obstructed Maze HTML documentation for a specific variant."""
    config = "3×3 maze"
    desc = "complex multi-step puzzle"
    
    if "1Dlhb" in env_id:
        config = "2×1 maze configuration"
        desc = "simplified puzzle with obstructed doors and hidden keys"
    else:  # Full
        config = "3×3 maze"
        desc = "full complexity puzzle with blue ball hidden in one of 4 corners, locked doors, obstructed doors, and keys in boxes"
    
    return (
        f"<h2>{env_id}</h2>"
        f"<p>A blue ball is hidden in a <strong>{config}</strong> with locked doors. This {desc} tests the agent's ability to solve "
        "complex multi-step puzzles involving object manipulation, navigation, and sequential reasoning.</p>"
        "<h4>Available Variants</h4>"
        "<ul>"
        "<li><strong>MiniGrid-ObstructedMaze-1Dlhb-v1</strong>: 2×1 maze (simplified)</li>"
        "<li><strong>MiniGrid-ObstructedMaze-Full-v1</strong>: 3×3 maze (full complexity)</li>"
        "</ul>"
    
    "<h3>Mission</h3>"
    "<p>Find and reach the hidden blue ball</p>"
    
    "<h3>Action Space</h3>"
    "<p><strong>Discrete(7)</strong> - Full action set required for puzzle solving:</p>"
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
    "<li><strong>image</strong>: Box(0, 255, (7, 7, 3), uint8) - Partially observable 7×7 grid view centered on agent</li>"
    "<li><strong>mission</strong>: MissionSpace with blue ball target</li>"
    "</ul>"
    
    "<h3>Observation Encoding</h3>"
    "<p>Each tile is encoded as a 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)</p>"
    "<ul>"
    "<li><strong>OBJECT_IDX</strong>: Type of object (wall, door, ball, box, key, goal, etc.)</li>"
    "<li><strong>COLOR_IDX</strong>: Color of the object</li>"
    "<li><strong>STATE</strong>: Door state (0=open, 1=closed, 2=locked) or box state</li>"
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
    
    "<h3>Strategy Requirements</h3>"
    "<p>To solve this environment, the agent must:</p>"
    "<ol>"
    "<li>Find and open boxes to retrieve keys</li>"
    "<li>Move obstructing balls out of the way</li>"
    "<li>Use keys to unlock doors</li>"
    "<li>Navigate through the maze to find the target blue ball</li>"
    "</ol>"
    
    "<h3>Notes</h3>"
    "<p>This is a highly complex environment that requires careful planning and multi-step reasoning. The agent must coordinate object manipulation (picking up keys from boxes, moving balls) with navigation and door unlocking.</p>"
    
    "<h3>Reference</h3>"
    "<p><a href=\"https://minigrid.farama.org/environments/minigrid/ObstructedMazeEnv/\" target=\"_blank\">MiniGrid Obstructed Maze Documentation</a></p>"
    )


# For backward compatibility
MINIGRID_OBSTRUCTED_MAZE_HTML = get_obstructed_maze_html("MiniGrid-ObstructedMaze-Full-v1")

__all__ = ["MINIGRID_OBSTRUCTED_MAZE_HTML", "get_obstructed_maze_html"]
