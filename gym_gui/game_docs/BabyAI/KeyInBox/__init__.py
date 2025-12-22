"""Documentation for BabyAI KeyInBox environment."""
from __future__ import annotations


def get_key_inbox_html(env_id: str = "BabyAI-KeyInBox-v0") -> str:
    """Generate KeyInBox HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Unlock a door where the key is hidden inside a box in the current room. "
        "Combines object manipulation and door interaction. Note: the BabyAI bot cannot solve this level.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"open the door\"</em></p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-KeyInBox-v0</strong></li>"
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
        "<li>5 → toggle (open box/unlock door)</li>"
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
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/KeyInBox/\">BabyAI KeyInBox</a></p>"
    )


BABYAI_KEY_INBOX_HTML = get_key_inbox_html()

__all__ = ["BABYAI_KEY_INBOX_HTML", "get_key_inbox_html"]
