"""Documentation for BabyAI Unlock environment."""
from __future__ import annotations


def get_unlock_html(env_id: str = "BabyAI-Unlock-v0") -> str:
    """Generate Unlock HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Unlock a door to complete the task. Combines maze navigation, door opening, "
        "and unlocking mechanics without requiring object unblocking.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"open the {color} door\"</em> — colors: red, green, blue, purple, yellow, grey</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-Unlock-v0</strong></li>"
        "</ul>"
        "<h4>Observation</h4>"
        "<p>Dict with <code>image</code> (7×7×3 uint8), <code>direction</code> (0-3), and <code>mission</code> text. "
        "Door state: 0=open, 1=closed, 2=locked.</p>"
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
        "<li>Termination: agent opens the correct door</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/Unlock/\">BabyAI Unlock</a></p>"
    )


BABYAI_UNLOCK_HTML = get_unlock_html()

__all__ = ["BABYAI_UNLOCK_HTML", "get_unlock_html"]
