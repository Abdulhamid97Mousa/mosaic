"""Documentation for BabyAI PickupLoc environment."""
from __future__ import annotations


def get_pickup_loc_html(env_id: str = "BabyAI-PickupLoc-v0") -> str:
    """Generate PickupLoc HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Pick up a specific object identified by its location. Single-room setting "
        "testing pickup and location-understanding competencies.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"pick up the {color} {type}\"</em> — colors: red, green, blue, purple, yellow, grey; "
        "types: ball, box, key</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-PickupLoc-v0</strong></li>"
        "</ul>"
        "<h4>Observation</h4>"
        "<p>Dict with <code>image</code> (7×7×3 uint8), <code>direction</code> (0-3), and <code>mission</code> text.</p>"
        "<h4>Action Space (Discrete(7))</h4>"
        "<ul>"
        "<li>0 → turn left</li>"
        "<li>1 → turn right</li>"
        "<li>2 → move forward</li>"
        "<li>3 → pick up object</li>"
        "<li>4 → drop</li>"
        "<li>5 → toggle</li>"
        "<li>6 → done</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code></li>"
        "<li>Failure → 0</li>"
        "</ul>"
        "<h4>Episode End</h4>"
        "<ul>"
        "<li>Termination: agent picks up the object</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/PickupLoc/\">BabyAI PickupLoc</a></p>"
    )


BABYAI_PICKUP_LOC_HTML = get_pickup_loc_html()

__all__ = ["BABYAI_PICKUP_LOC_HTML", "get_pickup_loc_html"]
