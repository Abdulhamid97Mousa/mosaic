"""Documentation for BabyAI OpenRedDoor environment."""
from __future__ import annotations


def get_open_reddoor_html(env_id: str = "BabyAI-OpenRedDoor-v0") -> str:
    """Generate OpenRedDoor HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Locate and open a red door in the current room. Door is always unlocked. "
        "Deliberately simplified debugging environment.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"open the red door\"</em></p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-OpenRedDoor-v0</strong></li>"
        "</ul>"
        "<h4>Observation</h4>"
        "<p>Dict with <code>image</code> (7×7×3 uint8), <code>direction</code> (0-3), and <code>mission</code> text.</p>"
        "<h4>Action Space (Discrete(7))</h4>"
        "<ul>"
        "<li>0 → turn left</li>"
        "<li>1 → turn right</li>"
        "<li>2 → move forward</li>"
        "<li>3 → pick up (unused)</li>"
        "<li>4 → drop (unused)</li>"
        "<li>5 → toggle (open door)</li>"
        "<li>6 → done (unused)</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code></li>"
        "<li>Failure → 0</li>"
        "</ul>"
        "<h4>Episode End</h4>"
        "<ul>"
        "<li>Termination: agent opens the red door</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/OpenRedDoor/\">BabyAI OpenRedDoor</a></p>"
    )


BABYAI_OPEN_REDDOOR_HTML = get_open_reddoor_html()

__all__ = ["BABYAI_OPEN_REDDOOR_HTML", "get_open_reddoor_html"]
