"""Documentation for BabyAI GoToRedBallNoDists environment."""
from __future__ import annotations


def get_goto_redball_nodists_html(env_id: str = "BabyAI-GoToRedBallNoDists-v0") -> str:
    """Generate GoToRedBallNoDists HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Navigate a single room to reach a red ball. No distractors present — the simplest "
        "GoTo variant for basic navigation learning.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"go to the red ball\"</em></p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-GoToRedBallNoDists-v0</strong></li>"
        "</ul>"
        "<h4>Observation</h4>"
        "<p>Dict with <code>image</code> (7×7×3 uint8), <code>direction</code> (0-3), and <code>mission</code> text. "
        "Each tile encoded as (OBJECT_IDX, COLOR_IDX, STATE).</p>"
        "<h4>Action Space (Discrete(7))</h4>"
        "<ul>"
        "<li>0 → turn left</li>"
        "<li>1 → turn right</li>"
        "<li>2 → move forward</li>"
        "<li>3 → pick up object</li>"
        "<li>4 → drop (unused)</li>"
        "<li>5 → toggle (unused)</li>"
        "<li>6 → done (unused)</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code></li>"
        "<li>Failure → 0</li>"
        "</ul>"
        "<h4>Episode End</h4>"
        "<ul>"
        "<li>Termination: agent reaches the red ball</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/GoToRedBallNoDists/\">BabyAI GoToRedBallNoDists</a></p>"
    )


BABYAI_GOTO_REDBALL_NODISTS_HTML = get_goto_redball_nodists_html()

__all__ = ["BABYAI_GOTO_REDBALL_NODISTS_HTML", "get_goto_redball_nodists_html"]
