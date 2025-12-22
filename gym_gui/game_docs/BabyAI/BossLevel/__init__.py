"""Documentation for BabyAI BossLevel environment."""
from __future__ import annotations


def get_bosslevel_html(env_id: str = "BabyAI-BossLevel-v0") -> str:
    """Generate BossLevel HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Superset of all BabyAI levels. Highest difficulty in the BabyAI curriculum. "
        "Combines all competencies into complex multi-step tasks.</p>"
        "<h4>Mission Structure</h4>"
        "<ul>"
        "<li><strong>Action:</strong> Single tasks (go, pick up, open, put next)</li>"
        "<li><strong>And:</strong> Two actions joined with 'and'</li>"
        "<li><strong>Sequence:</strong> Missions joined with 'then' or 'after you'</li>"
        "</ul>"
        "<p>Example: <em>\"open a red door and go to the ball on your left after you put the grey ball next to a door\"</em></p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-BossLevel-v0</strong></li>"
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
        "<li>Termination: all tasks completed</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), H (drop), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/BossLevel/\">BabyAI BossLevel</a></p>"
    )


BABYAI_BOSSLEVEL_HTML = get_bosslevel_html()

__all__ = ["BABYAI_BOSSLEVEL_HTML", "get_bosslevel_html"]
