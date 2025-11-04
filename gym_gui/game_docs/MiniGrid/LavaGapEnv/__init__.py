"""Documentation for MiniGrid Lava Gap environments."""
from __future__ import annotations


def get_lavagap_html(env_id: str) -> str:
    """Generate Lava Gap HTML documentation for a specific variant."""
    size = "7×7"
    desc = "standard benchmark"
    
    if "S5" in env_id:
        size = "5×5"
        desc = "compact challenge"
    elif "S6" in env_id:
        size = "6×6"
        desc = "intermediate difficulty"
    else:  # S7
        size = "7×7"
        desc = "standard benchmark"
    
    return (
        f"<h2>{env_id}</h2>"
        f"<p>The agent must reach the green goal square at the opposite corner of a <strong>{size}</strong> room, passing through a narrow gap in a vertical strip of deadly lava. "
        f"Touching the lava terminates the episode with zero reward. This {desc} is useful for studying safety and safe exploration.</p>"
        "<h4>Available Variants</h4>"
        "<ul>"
        "<li><strong>MiniGrid-LavaGapS5-v0</strong>: 5×5 grid with narrow lava gap</li>"
        "<li><strong>MiniGrid-LavaGapS6-v0</strong>: 6×6 grid with narrow lava gap</li>"
        "<li><strong>MiniGrid-LavaGapS7-v0</strong>: 7×7 grid with narrow lava gap (standard)</li>"
        "</ul>"
    "<h4>Mission</h4>"
    "<p>Depending on the obstacle_type parameter:</p>"
    "<ul>"
    "<li><strong>Lava:</strong> \"avoid the lava and get to the green goal square\"</li>"
    "<li><strong>Otherwise:</strong> \"find the opening and get to the green goal square\"</li>"
    "</ul>"
    "<h4>Observation</h4>"
    "<p>Dict observation with <code>image</code> (RGB, 7×7×3), <code>direction</code> (0=right,1=down,2=left,3=up), and <code>mission</code> string. "
    "Each tile is encoded as a 3-dimensional tuple: <code>(OBJECT_IDX, COLOR_IDX, STATE)</code>.</p>"
    "<p><code>OBJECT_TO_IDX</code> and <code>COLOR_TO_IDX</code> mappings can be found in <code>minigrid/core/constants.py</code>. "
    "STATE refers to the door state with 0=open, 1=closed, 2=locked.</p>"
    "<h4>Action space (Discrete(7))</h4>"
    "<ul>"
    "<li>0 → turn left</li>"
    "<li>1 → turn right</li>"
    "<li>2 → move forward</li>"
    "<li>3 → pick up (unused in this environment)</li>"
    "<li>4 → drop (unused)</li>"
    "<li>5 → toggle (unused)</li>"
    "<li>6 → done (no-op)</li>"
    "</ul>"
    "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space/G (pick up), H (drop), E/Enter (toggle), Q (done).</p>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code> scaled by the GUI multiplier (default ×10)</li>"
    "<li>Lava contact → 0 (episode terminates)</li>"
    "<li>Otherwise → 0</li>"
    "</ul>"
    "<h4>Episode end</h4>"
    "<ul>"
    "<li>Termination: agent reaches the goal or falls into lava</li>"
    "<li>Truncation: max_episode_steps timeout</li>"
    "</ul>"
    "<p>See the docs: <a href=\"https://minigrid.farama.org/environments/minigrid/LavaGapEnv/\">MiniGrid Lava Gap</a></p>"
    )


# For backward compatibility
MINIGRID_LAVAGAP_HTML = get_lavagap_html("MiniGrid-LavaGapS7-v0")

__all__ = ["MINIGRID_LAVAGAP_HTML", "get_lavagap_html"]
