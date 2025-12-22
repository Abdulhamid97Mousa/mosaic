"""Documentation for BabyAI GoToSeq environment."""
from __future__ import annotations


def get_goto_seq_html(env_id: str = "BabyAI-GoToSeq-v0") -> str:
    """Generate GoToSeq HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Sequencing of go-to-object commands. Complete multiple navigation objectives in sequence. "
        "No locked rooms or blocking obstacles. Tests memory and planning.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"go to a/the {color} {type} and go to a/the {color} {type}, then go to a/the {color} {type} and go to a/the {color} {type}\"</em></p>"
        "<p>Colors: red, green, blue, purple, yellow, grey; Types: ball, box, key</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-GoToSeq-v0</strong></li>"
        "<li><strong>BabyAI-GoToSeqS5R2-v0</strong>: Size 5, 2 rooms</li>"
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
        "<li>Termination: agent completes all sequential objectives</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/GoToSeq/\">BabyAI GoToSeq</a></p>"
    )


BABYAI_GOTO_SEQ_HTML = get_goto_seq_html()

__all__ = ["BABYAI_GOTO_SEQ_HTML", "get_goto_seq_html"]
