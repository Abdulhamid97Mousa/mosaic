"""Documentation for MiniGrid DoorKey environments."""
from __future__ import annotations


def get_doorkey_html(env_id: str) -> str:
    """Generate DoorKey HTML documentation for a specific variant."""
    # Extract grid size from env_id (e.g., "MiniGrid-DoorKey-5x5-v0" -> "5×5")
    size = "8×8"  # default
    if "5x5" in env_id:
        size = "5×5"
        description = "This tiny map is perfect for fast curriculum starts and initial learning."
    elif "6x6" in env_id:
        size = "6×6"
        description = "This intermediate-sized map provides a balance between simplicity and challenge."
    elif "16x16" in env_id:
        size = "16×16"
        description = "This long-horizon environment provides a sparse reward challenge."
    else:  # 8x8
        size = "8×8"
        description = "This standard benchmark is a classic test-bed for curiosity and curriculum learning."
    
    return (
        f"<h2>{env_id}</h2>"
        f"<p>Collect the yellow key, unlock a door, and reach the goal in a <strong>{size}</strong> room split by a wall. {description}</p>"
        "<h4>Observation</h4>"
        "<p>Dict observation with <code>image</code> (RGB, 7×7×3), <code>direction</code>, and <code>mission</code> string (e.g., 'use the key to open the door and then get to the goal'). Tile encodings follow MiniGrid's <code>OBJECT_TO_IDX</code> / <code>COLOR_TO_IDX</code> tables.</p>"
    "<h4>Action space (Discrete(7))</h4>"
    "<ul>"
    "<li>0 → turn left</li>"
    "<li>1 → turn right</li>"
    "<li>2 → move forward</li>"
    "<li>3 → pick up an object (key)</li>"
    "<li>4 → drop (unused)</li>"
    "<li>5 → toggle / open the door</li>"
    "<li>6 → done (no-op)</li>"
    "</ul>"
    "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space/G (pick up), H (drop), E/Enter (toggle), Q (done).</p>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li>Reach goal → <code>1 - 0.9 × (step_count / max_steps)</code> scaled by the GUI multiplier (default ×10)</li>"
    "<li>Otherwise → 0</li>"
    "</ul>"
    "<h4>Episode end</h4>"
    "<ul>"
    "<li>Termination: goal reached</li>"
    "<li>Truncation: default max steps 100</li>"
    "</ul>"
    "<h4>Available Variants</h4>"
    "<ul>"
    "<li><strong>MiniGrid-DoorKey-5x5-v0</strong>: 5×5 grid - tiny map for fast curriculum starts</li>"
    "<li><strong>MiniGrid-DoorKey-6x6-v0</strong>: 6×6 grid - intermediate difficulty</li>"
    "<li><strong>MiniGrid-DoorKey-8x8-v0</strong>: 8×8 grid - standard benchmark</li>"
    "<li><strong>MiniGrid-DoorKey-16x16-v0</strong>: 16×16 grid - long-horizon, sparse reward challenge</li>"
    "</ul>"
    "<p>See the docs: <a href=\"https://minigrid.farama.org/environments/minigrid/doorkey/\">MiniGrid DoorKey</a></p>"
    )


# For backward compatibility - generic HTML
MINIGRID_DOORKEY_HTML = get_doorkey_html("MiniGrid-DoorKey-8x8-v0")

__all__ = ["MINIGRID_DOORKEY_HTML", "get_doorkey_html"]
