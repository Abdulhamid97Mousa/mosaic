"""Documentation for BabyAI GoToDoor environment."""
from __future__ import annotations


def get_goto_door_html(env_id: str = "BabyAI-GoToDoor-v0") -> str:
    """Generate GoToDoor HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Navigate to a specific colored door in the current room. Straightforward task "
        "with no distractors. Foundational navigation task in BabyAI curriculum.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"go to the {color} door\"</em> — colors: red, green, blue, purple, yellow, grey</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-GoToDoor-v0</strong></li>"
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
        "<li>Termination: agent reaches the door</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/GoToDoor/\">BabyAI GoToDoor</a></p>"
    )


BABYAI_GOTO_DOOR_HTML = get_goto_door_html()

__all__ = ["BABYAI_GOTO_DOOR_HTML", "get_goto_door_html"]
