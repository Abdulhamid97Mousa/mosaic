"""Documentation for BabyAI KeyCorridor environment."""
from __future__ import annotations


def get_keycorridor_html(env_id: str = "BabyAI-KeyCorridorS3R1-v0") -> str:
    """Generate KeyCorridor HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Retrieve a ball behind a locked door. The key is randomly placed in one of several rooms. "
        "Multi-step puzzle: locate key, then access goal.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"pick up the ball\"</em></p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-KeyCorridorS3R1-v0</strong>: Size 3, 1 room</li>"
        "<li><strong>BabyAI-KeyCorridorS3R2-v0</strong>: Size 3, 2 rooms</li>"
        "<li><strong>BabyAI-KeyCorridorS3R3-v0</strong>: Size 3, 3 rooms</li>"
        "<li><strong>BabyAI-KeyCorridorS4R3-v0</strong>: Size 4, 3 rooms</li>"
        "<li><strong>BabyAI-KeyCorridorS5R3-v0</strong>: Size 5, 3 rooms</li>"
        "<li><strong>BabyAI-KeyCorridorS6R3-v0</strong>: Size 6, 3 rooms</li>"
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
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/KeyCorridor/\">BabyAI KeyCorridor</a></p>"
    )


BABYAI_KEYCORRIDOR_HTML = get_keycorridor_html()

__all__ = ["BABYAI_KEYCORRIDOR_HTML", "get_keycorridor_html"]
