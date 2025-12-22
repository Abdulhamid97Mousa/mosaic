"""Documentation for BabyAI ActionObjDoor environment."""
from __future__ import annotations


def get_action_objdoor_html(env_id: str = "BabyAI-ActionObjDoor-v0") -> str:
    """Generate ActionObjDoor HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Complete one of three types of instructions in a single room: retrieve a colored object, "
        "navigate to an object/door, or unlock a door.</p>"
        "<h4>Mission Types</h4>"
        "<ul>"
        "<li><em>\"pick up the {color} {type}\"</em></li>"
        "<li><em>\"go to the {color} {type}\"</em></li>"
        "<li><em>\"open a {color} door\"</em></li>"
        "</ul>"
        "<p>Colors: red, green, blue, purple, yellow, grey; Types: ball, box, door, key</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-ActionObjDoor-v0</strong></li>"
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
        "<li>Termination: agent completes the instruction</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/ActionObjDoor/\">BabyAI ActionObjDoor</a></p>"
    )


BABYAI_ACTION_OBJDOOR_HTML = get_action_objdoor_html()

__all__ = ["BABYAI_ACTION_OBJDOOR_HTML", "get_action_objdoor_html"]
