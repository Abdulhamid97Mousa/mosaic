"""Documentation for BabyAI FindObjS5 environment."""
from __future__ import annotations


def get_findobj_html(env_id: str = "BabyAI-FindObjS5-v0") -> str:
    """Generate FindObjS5 HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Locate and pick up a randomly placed object. Requires systematic exploration "
        "due to randomized object placement.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"pick up the {type}\"</em> — types: ball, box, key</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-FindObjS5-v0</strong>: Size 5</li>"
        "<li><strong>BabyAI-FindObjS6-v0</strong>: Size 6</li>"
        "<li><strong>BabyAI-FindObjS7-v0</strong>: Size 7</li>"
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
        "<li>Termination: agent picks up the object</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/FindObjS5/\">BabyAI FindObjS5</a></p>"
    )


BABYAI_FINDOBJ_HTML = get_findobj_html()

__all__ = ["BABYAI_FINDOBJ_HTML", "get_findobj_html"]
