"""Documentation for BabyAI MiniBossLevel environment."""
from __future__ import annotations


def get_minibosslevel_html(env_id: str = "BabyAI-MiniBossLevel-v0") -> str:
    """Generate MiniBossLevel HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Comprehensive challenge combining all BabyAI competencies. Union of all task types "
        "in a moderately-sized room with lower probability of locked rooms than BossLevel.</p>"
        "<h4>Mission Types</h4>"
        "<ul>"
        "<li><strong>Action:</strong> <em>\"go to the red ball\"</em>, <em>\"pick up a green key\"</em></li>"
        "<li><strong>And:</strong> Two actions combined</li>"
        "<li><strong>Sequence:</strong> Tasks chained with 'then' or 'after you'</li>"
        "</ul>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-MiniBossLevel-v0</strong></li>"
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
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/MiniBossLevel/\">BabyAI MiniBossLevel</a></p>"
    )


BABYAI_MINIBOSSLEVEL_HTML = get_minibosslevel_html()

__all__ = ["BABYAI_MINIBOSSLEVEL_HTML", "get_minibosslevel_html"]
