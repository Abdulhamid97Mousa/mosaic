"""Game documentation for RWARE Medium warehouse variants."""

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

_MEDIUM_INTRO = """
<h2>Robotic Warehouse - Medium</h2>
<p>A medium warehouse with <b>2x5 shelf layout</b> (2 rows, 5 columns of shelves).
The standard benchmark size used in the RWARE paper.</p>
"""


def _variant_table(n_agents: int, difficulty: str) -> str:
    diff_note = ""
    if difficulty == "Easy":
        diff_note = " (2x request queue -- more delivery opportunities)"
    elif difficulty == "Hard":
        diff_note = " (0.5x request queue -- fewer delivery opportunities)"
    return f"""
<h4>Variant Details</h4>
<table {_TABLE}>
    <tr {_HDR}>
        <th {_TH}>Property</th>
        <th {_TH}>Value</th>
    </tr>
    <tr><td {_TD}>Warehouse Size</td><td {_TD}>Medium (2x5 shelves)</td></tr>
    <tr><td {_TD}>Agents</td><td {_TD}>{n_agents}</td></tr>
    <tr><td {_TD}>Difficulty</td><td {_TD}>{difficulty}{diff_note}</td></tr>
    <tr><td {_TD}>Max Steps</td><td {_TD}>500 (configurable)</td></tr>
    <tr><td {_TD}>Stepping</td><td {_TD}>Simultaneous (all agents act together)</td></tr>
</table>
"""


RWARE_MEDIUM_2AG_HTML = (
    _MEDIUM_INTRO + _variant_table(2, "Normal")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

RWARE_MEDIUM_4AG_HTML = (
    _MEDIUM_INTRO + _variant_table(4, "Normal")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

RWARE_MEDIUM_4AG_EASY_HTML = (
    _MEDIUM_INTRO + _variant_table(4, "Easy")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

RWARE_MEDIUM_4AG_HARD_HTML = (
    _MEDIUM_INTRO + _variant_table(4, "Hard")
    + ACTIONS_HTML + OBSERVATIONS_HTML + REWARDS_HTML
    + MECHANICS_HTML + KEYBOARD_HTML + REFERENCE_HTML
)

__all__ = [
    "RWARE_MEDIUM_2AG_HTML",
    "RWARE_MEDIUM_4AG_HTML",
    "RWARE_MEDIUM_4AG_EASY_HTML",
    "RWARE_MEDIUM_4AG_HARD_HTML",
]
