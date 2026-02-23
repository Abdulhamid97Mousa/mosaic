"""Documentation for SMAC v1: MMM2 (1 Medivac + 2 Marauders + 7 Marines -- Super Hard)."""
from __future__ import annotations

from gym_gui.game_docs.SMAC._shared import (
    SMAC_CTDE_HTML,
    SMAC_OBS_HTML,
    SMAC_ACTIONS_HTML,
    SMAC_REWARD_HTML,
)


def get_smac_mmm2_html() -> str:
    """Generate HTML documentation for the MMM2 map."""
    return f"""
<h2>SMAC v1: MMM2</h2>

<p style="background-color: #fce4ec; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Heterogeneous Cooperative Micromanagement (Super Hard)</strong> --
Control 1 Medivac + 2 Marauders + 7 Marines against a larger enemy force.
The benchmark ceiling scenario for cooperative MARL.
</p>

<h4>Map Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Map Name</td><td style="border: 1px solid #ddd; padding: 8px;"><code>MMM2</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Allies</td><td style="border: 1px solid #ddd; padding: 8px;">1 Medivac + 2 Marauders + 7 Marines (10 agents)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Enemies</td><td style="border: 1px solid #ddd; padding: 8px;">1 Medivac + 3 Marauders + 8 Marines (12 units)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Super Hard</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode Limit</td><td style="border: 1px solid #ddd; padding: 8px;">180 steps</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous (all agents act in parallel)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">PyGame 2D top-down (terrain, unit circles, health arcs, HUD)</td></tr>
</table>

<h4>Unit Roles</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 6px;">Unit</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Role</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Micro Strategy</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Medivac</td><td style="border: 1px solid #ddd; padding: 6px;">Healer (no attack)</td><td style="border: 1px solid #ddd; padding: 6px;">Position near damaged units, avoid enemy fire</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Marauders</td><td style="border: 1px solid #ddd; padding: 6px;">Tanky ranged DPS</td><td style="border: 1px solid #ddd; padding: 6px;">Absorb damage in front line, slow enemies</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Marines</td><td style="border: 1px solid #ddd; padding: 6px;">Fragile ranged DPS</td><td style="border: 1px solid #ddd; padding: 6px;">Focus fire, retreat when low HP for Medivac healing</td></tr>
</table>

<p><b>Notes:</b> Complex heterogeneous composition.  The Medivac must learn to
heal efficiently; Marines and Marauders must coordinate focus fire differently.
This is the benchmark ceiling scenario -- many algorithms struggle here.</p>

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


SMAC_MMM2_HTML = get_smac_mmm2_html()

__all__ = ["SMAC_MMM2_HTML", "get_smac_mmm2_html"]
