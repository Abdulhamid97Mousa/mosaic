"""MiniGrid Crossing environment documentation."""


def get_crossing_html(env_id: str) -> str:
    """Generate HTML documentation for a specific Crossing variant.
    
    Args:
        env_id: The environment ID (e.g., "MiniGrid-LavaCrossingS9N1-v0")
        
    Returns:
        HTML documentation string for the specified variant
    """
    # Determine obstacle type
    if "Lava" in env_id:
        obstacle = "lava"
        mission = "avoid the lava and get to the green goal square"
        description_detail = (
            "The agent has to reach the green goal square on the other corner of the room while "
            "avoiding rivers of <strong>deadly lava</strong> which terminate the episode in failure. "
            "Each lava stream runs across the room either horizontally or vertically, and has a single "
            "crossing point which can be safely used. A path to the goal is guaranteed to exist. "
            "This environment is useful for studying <strong>safety and safe exploration</strong>."
        )
        obstacle_desc = "Lava rivers with safe crossing points"
    else:
        obstacle = "walls"
        mission = "find the opening and get to the green goal square"
        description_detail = (
            "Similar to the LavaCrossing environment, the agent has to reach the green goal square "
            "on the other corner of the room, however lava is replaced by <strong>walls</strong>. "
            "This MDP is therefore much easier and may be useful for quickly testing your algorithms."
        )
        obstacle_desc = "Wall barriers with openings"
    
    # Extract size and crossings from env_id
    if "S9N1" in env_id:
        size = "9×9"
        crossings = "1 valid crossing"
    elif "S9N2" in env_id:
        size = "9×9"
        crossings = "2 valid crossings"
    elif "S9N3" in env_id:
        size = "9×9"
        crossings = "3 valid crossings"
    elif "S11N5" in env_id:
        size = "11×11"
        crossings = "5 valid crossings"
    else:
        size = "variable"
        crossings = "variable crossings"
    
    return (
        f"<h2>{env_id}</h2>"
        "<h3>Description</h3>"
        f"<p>{description_detail}</p>"
        f"<p><strong>Grid Size:</strong> {size}<br>"
        f"<strong>Obstacles:</strong> {obstacle_desc}<br>"
        f"<strong>Crossings:</strong> {crossings} from start to goal</p>"
        "<h3>Mission</h3>"
        f'<p>"{mission}"</p>'
        "<h3>Action Space</h3>"
        "<p><strong>Discrete(7)</strong> - Seven possible actions:</p>"
        "<ul>"
        "<li><strong>0:</strong> Turn left</li>"
        "<li><strong>1:</strong> Turn right</li>"
        "<li><strong>2:</strong> Move forward</li>"
        "<li><strong>3:</strong> Pickup (unused)</li>"
        "<li><strong>4:</strong> Drop (unused)</li>"
        "<li><strong>5:</strong> Toggle (unused)</li>"
        "<li><strong>6:</strong> Done (unused)</li>"
        "</ul>"
        "<h3>Observation Space</h3>"
        "<p><strong>Dict</strong> containing:</p>"
        "<ul>"
        "<li><strong>direction:</strong> Discrete(4) - Agent's facing direction</li>"
        "<li><strong>image:</strong> Box(0, 255, (7, 7, 3), uint8) - Partial view of the grid</li>"
        "<li><strong>mission:</strong> Mission string describing the task</li>"
        "</ul>"
        "<h3>Observation Encoding</h3>"
        "<p>Each tile is encoded as a 3-dimensional tuple: <code>(OBJECT_IDX, COLOR_IDX, STATE)</code></p>"
        "<ul>"
        "<li><code>OBJECT_TO_IDX</code> and <code>COLOR_TO_IDX</code> mappings are in <code>minigrid/core/constants.py</code></li>"
        "<li><code>STATE</code> refers to door state: 0=open, 1=closed, 2=locked</li>"
        "</ul>"
        "<h3>Rewards</h3>"
        "<p>A reward of <code>1 - 0.9 * (step_count / max_steps)</code> is given for success, and <code>0</code> for failure.</p>"
        "<h3>Termination</h3>"
        "<p>The episode ends if any one of the following conditions is met:</p>"
        "<ul>"
        "<li>The agent reaches the goal (success)</li>"
        "<li>The agent falls into lava (failure, only in LavaCrossing variants)</li>"
        "<li>Timeout - maximum steps reached</li>"
        "</ul>"
        "<h4>Available Variants</h4>"
        "<p><strong>Lava Crossing:</strong></p>"
        "<ul>"
        "<li>MiniGrid-LavaCrossingS9N1-v0 (9×9, 1 crossing)</li>"
        "<li>MiniGrid-LavaCrossingS9N2-v0 (9×9, 2 crossings)</li>"
        "<li>MiniGrid-LavaCrossingS9N3-v0 (9×9, 3 crossings)</li>"
        "<li>MiniGrid-LavaCrossingS11N5-v0 (11×11, 5 crossings)</li>"
        "</ul>"
        "<p><strong>Simple Crossing:</strong></p>"
        "<ul>"
        "<li>MiniGrid-SimpleCrossingS9N1-v0 (9×9, 1 crossing)</li>"
        "<li>MiniGrid-SimpleCrossingS9N2-v0 (9×9, 2 crossings)</li>"
        "<li>MiniGrid-SimpleCrossingS9N3-v0 (9×9, 3 crossings)</li>"
        "<li>MiniGrid-SimpleCrossingS11N5-v0 (11×11, 5 crossings)</li>"
        "</ul>"
        "<h3>Reference</h3>"
        '<p><a href="https://minigrid.farama.org/environments/minigrid/CrossingEnv/">MiniGrid Crossing Documentation</a></p>'
    )


# For backward compatibility - default to a representative variant
MINIGRID_CROSSING_HTML = get_crossing_html("MiniGrid-LavaCrossingS9N1-v0")

__all__ = ["MINIGRID_CROSSING_HTML", "get_crossing_html"]
