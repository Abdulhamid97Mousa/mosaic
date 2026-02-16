"""Documentation for SMAC v1: 2s3z (2 Stalkers + 3 Zealots vs same)."""
from __future__ import annotations

from gym_gui.game_docs.SMAC._shared import (
    SMAC_CTDE_HTML,
    SMAC_OBS_HTML,
    SMAC_ACTIONS_HTML,
    SMAC_REWARD_HTML,
)


def get_smac_2s3z_html() -> str:
    """Generate HTML documentation for the 2s3z map."""
    return f"""
<h2>SMAC v1: 2s3z</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Cooperative Multi-Agent Micromanagement</strong> --
Control 2 Stalkers + 3 Zealots to defeat an identical enemy composition.
</p>

<h4>Map Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Map Name</td><td style="border: 1px solid #ddd; padding: 8px;"><code>2s3z</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Allies</td><td style="border: 1px solid #ddd; padding: 8px;">2 Stalkers + 3 Zealots (5 agents)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Enemies</td><td style="border: 1px solid #ddd; padding: 8px;">2 Stalkers + 3 Zealots (5 units)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Easy</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode Limit</td><td style="border: 1px solid #ddd; padding: 8px;">120 steps</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous (all agents act in parallel)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">PyGame 2D top-down (terrain, unit circles, health arcs, HUD)</td></tr>
</table>

<p><b>Notes:</b> Mixed composition: ranged Stalkers and melee Zealots require
different micro strategies.  Stalkers should kite (attack then retreat) while
Zealots engage in close combat.</p>

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


SMAC_2S3Z_HTML = get_smac_2s3z_html()

__all__ = ["SMAC_2S3Z_HTML", "get_smac_2s3z_html"]
