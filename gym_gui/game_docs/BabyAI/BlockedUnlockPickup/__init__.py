"""Documentation for BabyAI BlockedUnlockPickup environment."""
from __future__ import annotations


def get_blocked_unlock_pickup_html(env_id: str = "BabyAI-BlockedUnlockPickup-v0") -> str:
    """Generate BlockedUnlockPickup HTML documentation."""
    return (
        f"<h2>{env_id}</h2>"
        "<p>Unlock a door blocked by a ball, then navigate to another room to pick up a box. "
        "Multi-step challenge combining blocking puzzle and object retrieval.</p>"
        "<h4>Mission</h4>"
        "<p><em>\"pick up the box\"</em></p>"
        "<h4>Registered Variants</h4>"
        "<ul>"
        "<li><strong>BabyAI-BlockedUnlockPickup-v0</strong></li>"
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
        "<li>5 → toggle (unlock/open)</li>"
        "<li>6 → done</li>"
        "</ul>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li>Success → <code>1 - 0.9 * (step_count / max_steps)</code></li>"
        "<li>Failure → 0</li>"
        "</ul>"
        "<h4>Episode End</h4>"
        "<ul>"
        "<li>Termination: agent picks up the box</li>"
        "<li>Truncation: max_steps timeout</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space (pick up), H (drop), E/Enter (toggle).</p>"
        "<p>See docs: <a href=\"https://minigrid.farama.org/environments/babyai/BlockedUnlockPickup/\">BabyAI BlockedUnlockPickup</a></p>"
    )


BABYAI_BLOCKED_UNLOCK_PICKUP_HTML = get_blocked_unlock_pickup_html()

__all__ = ["BABYAI_BLOCKED_UNLOCK_PICKUP_HTML", "get_blocked_unlock_pickup_html"]
