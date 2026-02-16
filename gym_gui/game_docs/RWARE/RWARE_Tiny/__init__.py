"""Game documentation for RWARE Tiny warehouse variants."""

from gym_gui.game_docs.RWARE._shared import (
    ACTIONS_HTML,
    KEYBOARD_HTML,
    MECHANICS_HTML,
    OBSERVATIONS_HTML,
    REFERENCE_HTML,
    REWARDS_HTML,
)

_TH = 'style="border: 1px solid #ddd; padding: 8px;"'
_TD = 'style="border: 1px solid #ddd; padding: 8px;"'
_TABLE = 'style="width:100%; border-collapse: collapse; margin: 10px 0;"'
_HDR = 'style="background-color: #f0f0f0;"'

_TINY_INTRO = f"""
<h2>Robotic Warehouse - Tiny</h2>
<p>A small warehouse with <b>1x3 shelf layout</b> (1 row, 3 columns of shelves).
Grid size is approximately 10x11 cells. Good for quick experiments and debugging.</p>
"""


def _variant_table(n_agents: int, difficulty: str) -> str:
    return f"""
<h4>Variant Details</h4>
<table {_TABLE}>
    <tr {_HDR}>
        <th {_TH}>Property</th>
        <th {_TH}>Value</th>
    </tr>
    <tr><td {_TD}>Warehouse Size</td><td {_TD}>Tiny (1x3 shelves)</td></tr>
    <tr><td {_TD}>Agents</td><td {_TD}>{n_agents}</td></tr>
    <tr><td {_TD}>Difficulty</td><td {_TD}>{difficulty}</td></tr>
    <tr><td {_TD}>Max Steps</td><td {_TD}>500 (configurable)</td></tr>
    <tr><td {_TD}>Stepping</td><td {_TD}>Simultaneous (all agents act together)</td></tr>
</table>
"""


RWARE_TINY_2AG_HTML = (
    _TINY_INTRO + _variant_table(2, "Normal")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

RWARE_TINY_4AG_HTML = (
    _TINY_INTRO + _variant_table(4, "Normal")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

__all__ = ["RWARE_TINY_2AG_HTML", "RWARE_TINY_4AG_HTML"]
