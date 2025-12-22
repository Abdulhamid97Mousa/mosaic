"""Documentation for BabyAI GoTo environment."""
from __future__ import annotations


def get_goto_html(env_id: str = "BabyAI-GoTo-v0") -> str:
    """Generate GoTo HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Navigate to a specified object that may be in a different room. Multi-room navigation "
        "with distractors. Requires navigation planning and object identification.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"go to a/the {color} {type}\"</em> — colors: red, green, blue, purple, yellow, grey; "
        "types: ball, box, key</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-GoTo-v0</strong></li>"
        "</ul>"
        "<h4>Observation</h4>"
        "<p>Dict with <code>image</code> (7×7×3 uint8), <code>direction</code> (0-3), and <code>mission</code> text. "
        "Each tile encoded as (OBJECT_IDX, COLOR_IDX, STATE).</p>"
        "<h4>Action Space (Discrete(7))</h4>"
        "<ul>"
        "<li>0 → turn left</li>"
        "<li>1 → turn right</li>"
        "<li>2 → move forward</li>"
        "<li>3 → pick up object</li>"
        "<li>4 → drop</li>"
        "<li>5 → toggle (open doors)</li>"
        "<li>6 → done</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code></li>"
        "<li>Failure → 0</li>"
        "</ul>"
        "<h4>Episode End</h4>"
        "<ul>"
        "<li>Termination: agent reaches the target object</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/GoTo/\">BabyAI GoTo</a></p>"
    )


BABYAI_GOTO_HTML = get_goto_html()

__all__ = ["BABYAI_GOTO_HTML", "get_goto_html"]
