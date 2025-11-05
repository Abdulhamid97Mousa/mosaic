"""Documentation for MiniGrid Empty Room environments."""
from __future__ import annotations


def get_empty_html(env_id: str) -> str:
    """Generate Empty Room HTML documentation for a specific variant."""
    size = "8×8"
    start_type = "fixed start"
    
    if "5x5" in env_id:
        size = "5×5"
        desc = "Tiny room"
    elif "6x6" in env_id:
        size = "6×6"
        desc = "Small room"
    elif "16x16" in env_id:
        size = "16×16"
        desc = "Large room for exploration experiments"
    else:  # 8x8
        size = "8×8"
        desc = "Medium room"
    
    if "Random" in env_id:
        start_type = "random start"
        desc += " with randomized agent starting position each episode"
    else:
        desc += " with fixed starting position"
    
    return (
        f"<h2>{env_id}</h2>"
        f"<p>{desc}. Empty room with a green goal tile — no obstacles. Ideal for debugging observation pipelines, testing sparse reward handling, and validating agent navigation.</p>"
        "<h4>Available Variants</h4>"
        "<ul>"
        "<li><strong>MiniGrid-Empty-5x5-v0</strong>: 5×5 grid (fixed start)</li>"
        "<li><strong>MiniGrid-Empty-Random-5x5-v0</strong>: 5×5 grid (random start)</li>"
        "<li><strong>MiniGrid-Empty-6x6-v0</strong>: 6×6 grid (fixed start)</li>"
        "<li><strong>MiniGrid-Empty-Random-6x6-v0</strong>: 6×6 grid (random start)</li>"
        "<li><strong>MiniGrid-Empty-8x8-v0</strong>: 8×8 grid (fixed start)</li>"
        "<li><strong>MiniGrid-Empty-16x16-v0</strong>: 16×16 grid (fixed start)</li>"
        "</ul>"
    "<h4>Observation</h4>"
    "<p>Dict observation with <code>image</code> (RGB, 7×7×3), <code>direction</code> (0=right,1=down,2=left,3=up) and <code>mission</code> text. The GUI flattens this to a uint8 vector while keeping the RGB frame for rendering.</p>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code> (scaled by reward multiplier, default ×10)</li>"
    "<li>Otherwise → 0</li>"
    "</ul>"
    "<h4>Episode end</h4>"
    "<ul>"
    "<li>Termination: agent reaches the goal tile</li>"
    "<li>Truncation: max_episode_steps timeout</li>"
    "</ul>"
    "<h4>Action space (Discrete(7))</h4>"
    "<ul>"
    "<li>0 → turn left</li>"
    "<li>1 → turn right</li>"
    "<li>2 → move forward</li>"
    "<li>3 → pick up (unused)</li>"
    "<li>4 → drop (unused)</li>"
    "<li>5 → toggle / interact (unused)</li>"
    "<li>6 → done (no-op)</li>"
    "</ul>"
    "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space/G (pick up), H (drop), E/Enter (toggle), Q (done).</p>"
    "<p>See the docs: <a href=\"https://minigrid.farama.org/environments/minigrid/EmptyEnv/\">MiniGrid Empty Room</a></p>"
    )


# For backward compatibility
MINIGRID_EMPTY_HTML = get_empty_html("MiniGrid-Empty-8x8-v0")

__all__ = ["MINIGRID_EMPTY_HTML", "get_empty_html"]
