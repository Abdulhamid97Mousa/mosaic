"""Documentation for BabyAI MoveTwoAcrossS8N9 environment."""
from __future__ import annotations


def get_movetwoacross_html(env_id: str = "BabyAI-MoveTwoAcrossS8N9-v0") -> str:
    """Generate MoveTwoAcrossS8N9 HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Move two objects across the environment. Multi-object manipulation task "
        "requiring planning and coordination.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"put the {color} {type} next to the {color} {type}\"</em> (multiple times)</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-MoveTwoAcrossS5N2-v0</strong>: Size 5, 2 objects</li>"
        "<li><strong>BabyAI-MoveTwoAcrossS8N9-v0</strong>: Size 8, 9 objects</li>"
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
        "<li>Termination: task completed</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), H (drop), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/\">BabyAI MoveTwoAcross</a></p>"
    )


BABYAI_MOVETWOACROSS_HTML = get_movetwoacross_html()

__all__ = ["BABYAI_MOVETWOACROSS_HTML", "get_movetwoacross_html"]
