"""Documentation for BabyAI OpenTwoDoors environment."""
from __future__ import annotations


def get_open_twodoors_html(env_id: str = "BabyAI-OpenTwoDoors-v0") -> str:
    """Generate OpenTwoDoors HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Open two doors in sequence. Doors face opposite directions, so agent cannot see "
        "if second door is open. Requires memory-based or recurrent policies.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"open the {color} door, then open the {color} door\"</em> — colors: red, green, blue, purple, yellow, grey</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-OpenTwoDoors-v0</strong></li>"
        "<li><strong>BabyAI-OpenTwoDoorsDebug-v0</strong>: Debug version</li>"
        "<li><strong>BabyAI-OpenRedBlueDoors-v0</strong>: Red and blue doors only</li>"
        "<li><strong>BabyAI-OpenRedBlueDoorsDebug-v0</strong>: Debug version</li>"
        "</ul>"
        "<h4>Observation</h4>"
        "<p>Dict with <code>image</code> (7×7×3 uint8), <code>direction</code> (0-3), and <code>mission</code> text.</p>"
        "<h4>Action Space (Discrete(7))</h4>"
        "<ul>"
        "<li>0 → turn left</li>"
        "<li>1 → turn right</li>"
        "<li>2 → move forward</li>"
        "<li>3 → pick up</li>"
        "<li>4 → drop</li>"
        "<li>5 → toggle (open door)</li>"
        "<li>6 → done</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code></li>"
        "<li>Failure → 0</li>"
        "</ul>"
        "<h4>Episode End</h4>"
        "<ul>"
        "<li>Termination: agent opens a door (either one)</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/OpenTwoDoors/\">BabyAI OpenTwoDoors</a></p>"
    )


BABYAI_OPEN_TWODOORS_HTML = get_open_twodoors_html()

__all__ = ["BABYAI_OPEN_TWODOORS_HTML", "get_open_twodoors_html"]
