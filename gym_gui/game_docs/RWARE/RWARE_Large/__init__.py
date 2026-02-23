"""Game documentation for RWARE Large warehouse variants."""

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

_LARGE_INTRO = """
<h2>Robotic Warehouse - Large</h2>
<p>A large warehouse with <b>3x5 shelf layout</b> (3 rows, 5 columns of shelves).
The most challenging standard size -- requires strong coordination between agents.</p>
"""


def _variant_table(n_agents: int, difficulty: str) -> str:
    diff_note = ""
    if difficulty == "Hard":
        diff_note = " (0.5x request queue -- fewer delivery opportunities)"
    return f"""
<h4>Variant Details</h4>
<table {_TABLE}>
    <tr {_HDR}>
        <th {_TH}>Property</th>
        <th {_TH}>Value</th>
    </tr>
    <tr><td {_TD}>Warehouse Size</td><td {_TD}>Large (3x5 shelves)</td></tr>
    <tr><td {_TD}>Agents</td><td {_TD}>{n_agents}</td></tr>
    <tr><td {_TD}>Difficulty</td><td {_TD}>{difficulty}{diff_note}</td></tr>
    <tr><td {_TD}>Max Steps</td><td {_TD}>500 (configurable)</td></tr>
    <tr><td {_TD}>Stepping</td><td {_TD}>Simultaneous (all agents act together)</td></tr>
</table>
"""


RWARE_LARGE_4AG_HTML = (
    _LARGE_INTRO + _variant_table(4, "Normal")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

RWARE_LARGE_4AG_HARD_HTML = (
    _LARGE_INTRO + _variant_table(4, "Hard")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

RWARE_LARGE_8AG_HTML = (
    _LARGE_INTRO + _variant_table(8, "Normal")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

RWARE_LARGE_8AG_HARD_HTML = (
    _LARGE_INTRO + _variant_table(8, "Hard")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

__all__ = [
    "RWARE_LARGE_4AG_HTML",
    "RWARE_LARGE_4AG_HARD_HTML",
    "RWARE_LARGE_8AG_HTML",
    "RWARE_LARGE_8AG_HARD_HTML",
]
