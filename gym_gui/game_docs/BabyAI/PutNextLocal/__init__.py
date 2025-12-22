"""Documentation for BabyAI PutNextLocal environment."""
from __future__ import annotations


def get_putnext_local_html(env_id: str = "BabyAI-PutNextLocal-v0") -> str:
    """Generate PutNextLocal HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Place one object adjacent to another in a single room. No doors or distractors. "
        "Foundational task for object manipulation.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"put the {color} {type} next to the {color} {type}\"</em> — colors: red, green, blue, purple, yellow, grey; "
        "types: ball, box, key</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-PutNextLocal-v0</strong></li>"
        "<li><strong>BabyAI-PutNextLocalS5N3-v0</strong>: Size 5, 3 objects</li>"
        "<li><strong>BabyAI-PutNextLocalS6N4-v0</strong>: Size 6, 4 objects</li>"
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
        "<li>Termination: object placed next to target</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), H (drop), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/PutNextLocal/\">BabyAI PutNextLocal</a></p>"
    )


BABYAI_PUTNEXT_LOCAL_HTML = get_putnext_local_html()

__all__ = ["BABYAI_PUTNEXT_LOCAL_HTML", "get_putnext_local_html"]
