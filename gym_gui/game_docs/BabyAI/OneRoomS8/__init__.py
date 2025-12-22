"""Documentation for BabyAI OneRoomS8 environment."""
from __future__ import annotations


def get_oneroom_html(env_id: str = "BabyAI-OneRoomS8-v0") -> str:
    """Generate OneRoomS8 HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Pick up the ball in a single room. Among the simplest BabyAI tasks, "
        "designed as a foundational environment for RL research.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"pick up the ball\"</em></p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-OneRoomS8-v0</strong>: Size 8</li>"
        "<li><strong>BabyAI-OneRoomS12-v0</strong>: Size 12</li>"
        "<li><strong>BabyAI-OneRoomS16-v0</strong>: Size 16</li>"
        "<li><strong>BabyAI-OneRoomS20-v0</strong>: Size 20</li>"
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
        "<li>Termination: agent picks up the ball</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/OneRoomS8/\">BabyAI OneRoomS8</a></p>"
    )


BABYAI_ONEROOM_HTML = get_oneroom_html()

__all__ = ["BABYAI_ONEROOM_HTML", "get_oneroom_html"]
