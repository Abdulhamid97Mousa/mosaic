"""Documentation for BabyAI GoToLocal environment."""
from __future__ import annotations


def get_goto_local_html(env_id: str = "BabyAI-GoToLocal-v0") -> str:
    """Generate GoToLocal HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Navigate to a specified object in a single room with distractors. No doors. "
        "Difficulty scales with room size and number of distractors (S{X}N{Y} naming).</p>"
        "<h4>Mission</h4>"
        "<p><em>\"go to the {color} {type}\"</em> — colors: red, green, blue, purple, yellow, grey; "
        "types: ball, box, key</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-GoToLocal-v0</strong></li>"
        "<li><strong>BabyAI-GoToLocalS5N2-v0</strong>: 5×5 room, 2 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS6N2-v0</strong>: 6×6 room, 2 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS6N3-v0</strong>: 6×6 room, 3 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS6N4-v0</strong>: 6×6 room, 4 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS7N4-v0</strong>: 7×7 room, 4 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS7N5-v0</strong>: 7×7 room, 5 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS8N2-v0</strong>: 8×8 room, 2 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS8N3-v0</strong>: 8×8 room, 3 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS8N4-v0</strong>: 8×8 room, 4 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS8N5-v0</strong>: 8×8 room, 5 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS8N6-v0</strong>: 8×8 room, 6 distractors</li>"
        "<li><strong>BabyAI-GoToLocalS8N7-v0</strong>: 8×8 room, 7 distractors</li>"
        "</ul>"
        "<h4>Observation</h4>"
        "<p>Dict with <code>image</code> (7×7×3 uint8), <code>direction</code> (0-3), and <code>mission</code> text.</p>"
        "<h4>Action Space (Discrete(7))</h4>"
        "<ul>"
        "<li>0 → turn left</li>"
        "<li>1 → turn right</li>"
        "<li>2 → move forward</li>"
        "<li>3 → pick up object</li>"
        "<li>4 → drop (unused)</li>"
        "<li>5 → toggle (unused)</li>"
        "<li>6 → done (unused)</li>"
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
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/GoToLocal/\">BabyAI GoToLocal</a></p>"
    )


BABYAI_GOTO_LOCAL_HTML = get_goto_local_html()

__all__ = ["BABYAI_GOTO_LOCAL_HTML", "get_goto_local_html"]
