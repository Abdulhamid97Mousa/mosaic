"""Documentation for BabyAI OpenDoorsOrder environment."""
from __future__ import annotations


def get_open_doorsorder_html(env_id: str = "BabyAI-OpenDoorsOrderN2-v0") -> str:
    """Generate OpenDoorsOrder HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Open one or two doors in a specified sequence. Agent must understand and execute "
        "ordering instructions. Tests multi-step instruction following.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"open the {color} door, then open the {color} door\"</em> or "
        "<em>\"open the {color} door after you open the {color} door\"</em></p>"
        "<p>Colors: red, green, blue, purple, yellow, grey</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-OpenDoorsOrderN2-v0</strong>: 2 doors</li>"
        "<li><strong>BabyAI-OpenDoorsOrderN4-v0</strong>: 4 doors</li>"
        "<li><strong>BabyAI-OpenDoorsOrderN2Debug-v0</strong>: Debug, 2 doors</li>"
        "<li><strong>BabyAI-OpenDoorsOrderN4Debug-v0</strong>: Debug, 4 doors</li>"
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
        "<li>Termination: agent completes the door sequence</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/OpenDoorsOrder/\">BabyAI OpenDoorsOrder</a></p>"
    )


BABYAI_OPEN_DOORSORDER_HTML = get_open_doorsorder_html()

__all__ = ["BABYAI_OPEN_DOORSORDER_HTML", "get_open_doorsorder_html"]
