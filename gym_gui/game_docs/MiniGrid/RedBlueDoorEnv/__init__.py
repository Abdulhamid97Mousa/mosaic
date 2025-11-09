"""Documentation for MiniGrid RedBlueDoors environments."""
from __future__ import annotations


def get_redbluedoors_html(env_id: str) -> str:
    """Generate RedBlueDoors HTML documentation for a specific variant.

    Variants:
    - MiniGrid-RedBlueDoors-6x6-v0
    - MiniGrid-RedBlueDoors-8x8-v0
    """
    size = "8×8"
    if "6x6" in env_id:
        size = "6×6"
        size_desc = "Compact map with short horizons, good for debugging and quick experiments."
    else:
        size = "8×8"
        size_desc = "Standard benchmark size offering moderate exploration requirements."

    return (
        f"<h2>{env_id}</h2>"
        f"<p>In RedBlueDoors, the agent must guess which color door to open. One door leads to the goal; the other blocks progress. "
        f"This variant uses a <strong>{size}</strong> grid. {size_desc}</p>"
        "<h4>Observation</h4>"
        "<p>Dict observation with <code>image</code> (RGB, 7×7×3 by default), <code>direction</code> (0–3), and <code>mission</code> text. "
        "The GUI flattens the RGB observation and appends the agent's direction by default.</p>"
        "<h4>Action space (Discrete(7))</h4>"
        "<ul>"
        "<li>0 → turn left</li>"
        "<li>1 → turn right</li>"
        "<li>2 → move forward</li>"
        "<li>3 → pick up object</li>"
        "<li>4 → drop object</li>"
        "<li>5 → toggle (open/close door)</li>"
        "<li>6 → done (no-op)</li>"
        "</ul>"
        "<p><strong>Keyboard:</strong> ←/A, →/D, ↑/W, Space/G (pick up), H (drop), E/Enter (toggle), Q (done).</p>"
        "<h4>Rewards</h4>"
        "<ul>"
        "<li>Reach goal → <code>1 - 0.9 × (step_count / max_steps)</code> (scaled by the GUI multiplier; default ×10)</li>"
        "<li>Otherwise → 0</li>"
        "</ul>"
        "<h4>Episode end</h4>"
        "<ul>"
        "<li>Termination: goal reached via the correct colored door</li>"
        "<li>Truncation: environment's default max steps</li>"
        "</ul>"
        "<p>Reference: <a href=\"https://minigrid.farama.org/environments/minigrid/redbluedoors/\">MiniGrid RedBlueDoors</a></p>"
    )


# Generic default HTML for listing pages
MINIGRID_REDBLUEDOORS_HTML = get_redbluedoors_html("MiniGrid-RedBlueDoors-8x8-v0")

__all__ = ["get_redbluedoors_html", "MINIGRID_REDBLUEDOORS_HTML"]
