"""Documentation for BabyAI UnlockPickup environment."""
from __future__ import annotations


def get_unlock_pickup_html(env_id: str = "BabyAI-UnlockPickup-v0") -> str:
    """Generate UnlockPickup HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Unlock a door and retrieve a box from another room. "
        "Combines unlocking mechanics with object retrieval.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"pick up the {color} box\"</em> — colors: red, green, blue, purple, yellow, grey</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-UnlockPickup-v0</strong></li>"
        "<li><strong>BabyAI-UnlockPickupDist-v0</strong>: Distance-based variant</li>"
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
        "<li>5 → toggle (unlock/open)</li>"
        "<li>6 → done</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code></li>"
        "<li>Failure → 0</li>"
        "</ul>"
        "<h4>Episode End</h4>"
        "<ul>"
        "<li>Termination: agent picks up the box</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/UnlockPickup/\">BabyAI UnlockPickup</a></p>"
    )


BABYAI_UNLOCK_PICKUP_HTML = get_unlock_pickup_html()

__all__ = ["BABYAI_UNLOCK_PICKUP_HTML", "get_unlock_pickup_html"]
