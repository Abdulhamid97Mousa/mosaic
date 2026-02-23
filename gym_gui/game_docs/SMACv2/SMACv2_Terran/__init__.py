"""Documentation for SMACv2: 10gen_terran (Procedural Terran compositions)."""
from __future__ import annotations

from gym_gui.game_docs.SMACv2._shared import (
    SMAC_CTDE_HTML,
    SMAC_OBS_HTML,
    SMAC_ACTIONS_HTML,
    SMAC_REWARD_HTML,
)


def get_smacv2_terran_html() -> str:
    """Generate HTML documentation for the 10gen_terran map."""
    return f"""
<h2>SMACv2: 10gen_terran</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Procedural Multi-Agent Micromanagement</strong> --
Team compositions vary every episode with random Terran units, forcing agents
to generalise rather than memorise fixed strategies.
</p>

<h4>Map Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Map Name</td><td style="border: 1px solid #ddd; padding: 8px;"><code>10gen_terran</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Race</td><td style="border: 1px solid #ddd; padding: 8px;">Terran</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Unit Pool</td><td style="border: 1px solid #ddd; padding: 8px;">Marines, Marauders, Medivacs</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents per Episode</td><td style="border: 1px solid #ddd; padding: 8px;">~10 (varies by procedural generation)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Generation</td><td style="border: 1px solid #ddd; padding: 8px;">Procedural (new composition each reset)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">PyGame 2D top-down (terrain, unit circles, health arcs, HUD)</td></tr>
</table>

<p><b>Key Difference from SMAC v1:</b> Agent count and observation/action shapes may change
between episodes.  The adapter re-queries <code>get_env_info()</code> after each reset.</p>

<p><b>Notes:</b> Bio-ball composition with healing support.
Marines provide ranged DPS, Marauders slow enemies, and Medivacs heal biological units.</p>

{SMAC_CTDE_HTML}
{SMAC_OBS_HTML}
{SMAC_ACTIONS_HTML}
{SMAC_REWARD_HTML}

<h4>References</h4>
<ul>
    <li>Ellis et al. (2023). "SMACv2: An Improved Benchmark for Cooperative MARL"</li>
    <li>Repository: <a href="https://github.com/oxwhirl/smacv2">github.com/oxwhirl/smacv2</a></li>
</ul>
"""


SMACV2_TERRAN_HTML = get_smacv2_terran_html()

__all__ = ["SMACV2_TERRAN_HTML", "get_smacv2_terran_html"]
