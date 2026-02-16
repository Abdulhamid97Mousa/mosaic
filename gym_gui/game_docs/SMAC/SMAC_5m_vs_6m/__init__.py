"""Documentation for SMAC v1: 5m_vs_6m (5 Marines vs 6 Marines -- asymmetric)."""
from __future__ import annotations

from gym_gui.game_docs.SMAC._shared import (
    SMAC_CTDE_HTML,
    SMAC_OBS_HTML,
    SMAC_ACTIONS_HTML,
    SMAC_REWARD_HTML,
)


def get_smac_5m_vs_6m_html() -> str:
    """Generate HTML documentation for the 5m_vs_6m map."""
    return f"""
<h2>SMAC v1: 5m_vs_6m</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Asymmetric Cooperative Micromanagement (Hard)</strong> --
Control 5 allied Marines against 6 enemy Marines.  The numerical disadvantage
requires focus fire and retreat tactics to win.
</p>

<h4>Map Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Map Name</td><td style="border: 1px solid #ddd; padding: 8px;"><code>5m_vs_6m</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Allies</td><td style="border: 1px solid #ddd; padding: 8px;">5 Marines (5 agents)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Enemies</td><td style="border: 1px solid #ddd; padding: 8px;">6 Marines (6 units)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Hard (asymmetric)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode Limit</td><td style="border: 1px solid #ddd; padding: 8px;">70 steps</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous (all agents act in parallel)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">PyGame 2D top-down (terrain, unit circles, health arcs, HUD)</td></tr>
</table>

<p><b>Notes:</b> Asymmetric: fewer allies than enemies.  Requires focused fire
(all agents attacking the same enemy) and retreat tactics (kiting low-health
Marines out of combat).  A key test for coordination algorithms.</p>

{SMAC_CTDE_HTML}
{SMAC_OBS_HTML}
{SMAC_ACTIONS_HTML}
{SMAC_REWARD_HTML}

<h4>References</h4>
<ul>
    <li>Samvelyan et al. (2019). "The StarCraft Multi-Agent Challenge"</li>
    <li>Repository: <a href="https://github.com/oxwhirl/smac">github.com/oxwhirl/smac</a></li>
</ul>
"""


SMAC_5M_VS_6M_HTML = get_smac_5m_vs_6m_html()

__all__ = ["SMAC_5M_VS_6M_HTML", "get_smac_5m_vs_6m_html"]
