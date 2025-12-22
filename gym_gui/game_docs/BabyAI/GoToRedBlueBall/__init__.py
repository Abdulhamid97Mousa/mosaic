"""Documentation for BabyAI GoToRedBlueBall environment."""
from __future__ import annotations


def get_goto_redblueball_html(env_id: str = "BabyAI-GoToRedBlueBall-v0") -> str:
    """Generate GoToRedBlueBall HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Navigate to either a red or blue ball. Exactly one red or blue ball exists, "
        "with distractors guaranteed not to be red or blue balls. No language comprehension required.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"go to the {color} ball\"</em> — color is red or blue</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-GoToRedBlueBall-v0</strong></li>"
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
        "<li>Termination: agent reaches the target ball</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/GoToRedBlueBall/\">BabyAI GoToRedBlueBall</a></p>"
    )


BABYAI_GOTO_REDBLUEBALL_HTML = get_goto_redblueball_html()

__all__ = ["BABYAI_GOTO_REDBLUEBALL_HTML", "get_goto_redblueball_html"]
