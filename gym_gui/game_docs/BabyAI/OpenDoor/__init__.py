"""Documentation for BabyAI OpenDoor environment."""
from __future__ import annotations


def get_open_door_html(env_id: str = "BabyAI-OpenDoor-v0") -> str:
    """Generate OpenDoor HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Navigate to and open a door in the current room. Door identified by color, always unlocked. "
        "Foundational BabyAI task for basic door interaction.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"open the {color} door\"</em> — colors: red, green, blue, purple, yellow, grey</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-OpenDoor-v0</strong></li>"
        "<li><strong>BabyAI-OpenDoorDebug-v0</strong>: Debug version</li>"
        "<li><strong>BabyAI-OpenDoorColor-v0</strong>: Color identification focus</li>"
        "<li><strong>BabyAI-OpenDoorLoc-v0</strong>: Location-based identification</li>"
        "</ul>"
        "<h4>Observation</h4>"
        "<p>Dict with <code>image</code> (7×7×3 uint8), <code>direction</code> (0-3), and <code>mission</code> text.</p>"
        "<h4>Action Space (Discrete(7))</h4>"
        "<ul>"
        "<li>0 → turn left</li>"
        "<li>1 → turn right</li>"
        "<li>2 → move forward</li>"
        "<li>3 → pick up</li>"
        "<li>4 → drop</li>"
        "<li>5 → toggle (open door)</li>"
        "<li>6 → done</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code></li>"
        "<li>Failure → 0</li>"
        "</ul>"
        "<h4>Episode End</h4>"
        "<ul>"
        "<li>Termination: agent opens the door</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/OpenDoor/\">BabyAI OpenDoor</a></p>"
    )


BABYAI_OPEN_DOOR_HTML = get_open_door_html()

__all__ = ["BABYAI_OPEN_DOOR_HTML", "get_open_door_html"]
