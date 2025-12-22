"""Documentation for BabyAI SynthLoc environment."""
from __future__ import annotations


def get_synthloc_html(env_id: str = "BabyAI-SynthLoc-v0") -> str:
    """Generate SynthLoc HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Like Synth, but with location-based language (in front, behind, left, right). "
        "No implicit unlocking. Tests spatial reasoning with natural language.</p>"
        "<h4>Mission Types</h4>"
        "<ul>"
        "<li><em>\"go to the {color} {type} {location}\"</em></li>"
        "<li><em>\"pick up a/the {color} {type} {location}\"</em></li>"
        "<li><em>\"open the {color} door {location}\"</em></li>"
        "<li><em>\"put the {color} {type} {location} next to the {color} {type} {location}\"</em></li>"
        "</ul>"
        "<p>Locations: in front of you, behind you, on your left, on your right</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-SynthLoc-v0</strong></li>"
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
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/SynthLoc/\">BabyAI SynthLoc</a></p>"
    )


BABYAI_SYNTHLOC_HTML = get_synthloc_html()

__all__ = ["BABYAI_SYNTHLOC_HTML", "get_synthloc_html"]
