"""Documentation for MiniGrid Multi Room environments."""
from __future__ import annotations


def get_multiroom_html(env_id: str) -> str:
    """Generate MultiRoom HTML documentation for a specific variant."""
    num_rooms = "6"
    grid_size = ""
    desc = "standard configuration"
    
    if "N2-S4" in env_id:
        num_rooms = "2"
        grid_size = " in a 4×4 grid"
        desc = "simple two-room layout"
    elif "N4-S5" in env_id:
        num_rooms = "4"
        grid_size = " in a 5×5 grid"
        desc = "intermediate four-room challenge"
    else:  # N6
        num_rooms = "6"
        grid_size = ""
        desc = "standard six-room configuration"
    
    return (
        f"<h2>{env_id}</h2>"
        f"<p>This environment has <strong>{num_rooms} connected rooms{grid_size}</strong> with doors that must be opened to reach the next room. "
        f"The final room contains the green goal square. This {desc} is extremely difficult to solve using RL alone, "
        "but can be mastered by gradually increasing the number of rooms and building a curriculum.</p>"
        "<h4>Available Variants</h4>"
        "<ul>"
        "<li><strong>MiniGrid-MultiRoom-N2-S4-v0</strong>: 2 rooms in a 4×4 grid</li>"
        "<li><strong>MiniGrid-MultiRoom-N4-S5-v0</strong>: 4 rooms in a 5×5 grid</li>"
        "<li><strong>MiniGrid-MultiRoom-N6-v0</strong>: 6 rooms (standard)</li>"
        "</ul>"
    
    "<h3>Mission</h3>"
    "<p><em>\"traverse the rooms to get to the goal\"</em></p>"
    
    "<h3>Action Space</h3>"
    "<p><strong>Discrete(7)</strong> - Full action set:</p>"
    "<ul>"
    "<li><strong>0</strong>: Turn left</li>"
    "<li><strong>1</strong>: Turn right</li>"
    "<li><strong>2</strong>: Move forward</li>"
    "<li><strong>3</strong>: Pick up (unused)</li>"
    "<li><strong>4</strong>: Drop (unused)</li>"
    "<li><strong>5</strong>: Toggle/activate an object (used to open doors)</li>"
    "<li><strong>6</strong>: Done (unused)</li>"
    "</ul>"
    
    "<h3>Observation Space</h3>"
    "<p>A dictionary containing:</p>"
    "<ul>"
    "<li><strong>direction</strong>: Discrete(4) - Agent's current direction (0=right, 1=down, 2=left, 3=up)</li>"
    "<li><strong>image</strong>: Box(0, 255, (7, 7, 3), uint8) - Partially observable 7×7 grid view centered on agent</li>"
    "<li><strong>mission</strong>: String describing the goal</li>"
    "</ul>"
    
    "<h3>Observation Encoding</h3>"
    "<p>Each tile is encoded as a 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)</p>"
    "<ul>"
    "<li><strong>OBJECT_IDX</strong>: Type of object (wall, door, goal, etc.)</li>"
    "<li><strong>COLOR_IDX</strong>: Color of the object</li>"
    "<li><strong>STATE</strong>: Door state (0=open, 1=closed, 2=locked)</li>"
    "</ul>"
    
    "<h3>Rewards</h3>"
    "<ul>"
    "<li><strong>Success</strong>: 1 - 0.9 × (step_count / max_steps)</li>"
    "<li><strong>Failure</strong>: 0</li>"
    "</ul>"
    
    "<h3>Termination</h3>"
    "<p>The episode ends when:</p>"
    "<ul>"
    "<li>The agent reaches the green goal square in the final room</li>"
    "<li>Maximum steps reached (timeout)</li>"
    "</ul>"
    
    "<h3>Notes</h3>"
    "<p>This environment tests navigation through multiple rooms and door manipulation. The curriculum approach (starting with N2-S4 and progressing to N6) is recommended for training. The toggle action (5) is essential for opening doors between rooms.</p>"
    
    "<h3>Reference</h3>"
    "<p><a href=\"https://minigrid.farama.org/environments/minigrid/MultiRoomEnv/\" target=\"_blank\">MiniGrid Multi Room Documentation</a></p>"
    )


# For backward compatibility
MINIGRID_MULTIROOM_HTML = get_multiroom_html("MiniGrid-MultiRoom-N6-v0")

__all__ = ["MINIGRID_MULTIROOM_HTML", "get_multiroom_html"]
