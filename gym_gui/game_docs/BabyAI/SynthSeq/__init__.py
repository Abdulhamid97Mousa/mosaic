"""Documentation for BabyAI SynthSeq environment."""
from __future__ import annotations


def get_synthseq_html(env_id: str = "BabyAI-SynthSeq-v0") -> str:
    """Generate SynthSeq HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Extends SynthLoc with sequential command execution. Combines GoToSeq-style sequencing "
        "with spatial language. No implicit unlocking.</p>"
        "<h4>Mission Structure</h4>"
        "<p>Actions connected with 'and', 'then', or 'after you':</p>"
        "<ul>"
        "<li><em>\"go to the green key and put the box next to the yellow ball\"</em></li>"
        "<li><em>\"pick up the red ball then go to the blue door\"</em></li>"
        "</ul>"
        "<h4>Competencies</h4>"
        "<p>Maze navigation, unblocking, unlocking, navigation, pickup, placement, door opening, "
        "spatial reasoning, sequential execution</p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-SynthSeq-v0</strong></li>"
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
        "<li>Termination: all sequential tasks completed</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), H (drop), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/SynthSeq/\">BabyAI SynthSeq</a></p>"
    )


BABYAI_SYNTHSEQ_HTML = get_synthseq_html()

__all__ = ["BABYAI_SYNTHSEQ_HTML", "get_synthseq_html"]
