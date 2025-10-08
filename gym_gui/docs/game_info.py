"""Canonical help text (HTML) for toy-text environments displayed in the Game Info panel.

Each entry is an HTML string with a short description, rewards, episode end conditions,
keyboard mappings, and a link to the authoritative Gymnasium docs.
"""
from __future__ import annotations

from typing import Dict

from gym_gui.core.enums import GameId


TAXI_HTML = (
    "<h3>Taxi-v3</h3>"
    "<p>Pickup and drop off passengers on a small grid with walls.</p>"
    "<h4>Starting state</h4>"
    "<p>The initial state is sampled uniformly from positions where the passenger is not in the taxi and not at their destination. There are 300 possible initial states.</p>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li>-1 per step (time penalty)</li>"
    "<li>+20 for successfully delivering the passenger</li>"
    "<li>-10 for illegal pickup or drop-off</li>"
    "</ul>"
    "<h4>Episode end</h4>"
    "<ul>"
    "<li>Termination: passenger successfully dropped off</li>"
    "<li>Truncation (with time_limit): episode length 200</li>"
    "</ul>"
    "<h4>Info returned</h4>"
    "<p><code>step()</code> and <code>reset()</code> return an <code>info</code> dict that may include:</p>"
    "<ul>"
    "<li><code>p</code> — transition probability</li>"
    "<li><code>action_mask</code> — boolean mask indicating which actions will change state</li>"
    "</ul>"
    "<h4>Keyboard (human controls)</h4>"
    "<p>Map the taxi action space to keys for human play:</p>"
    "<ul>"
    "<li>South (0) → Down arrow</li>"
    "<li>East (1) → Right arrow</li>"
    "<li>North (2) → Up arrow</li>"
    "<li>West (3) → Left arrow</li>"
    "<li>Pickup → Spacebar</li>"
    "<li>Dropoff → E</li>"
    "</ul>"
    "<h4>Arguments / Config</h4>"
    "<p><code>is_raining</code>: when True, movement succeeds with 80% probability; else 10% slip left/right.</p>"
    "<p><code>fickle_passenger</code>: when True, passenger may change destination on first pickup with 30% chance.</p>"
    "<p>See the docs: <a href=\"https://gymnasium.farama.org/environments/toy_text/taxi/\">Taxi (Gymnasium)</a></p>"
)


FROZEN_HTML = (
    "<h3>FrozenLake-v1</h3>"
    "<p>A grid world where the agent must reach the goal while avoiding holes.</p>"
    "<h4>Starting state</h4>"
    "<p>The episode starts at the upper-left corner (state 0).</p>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li>Reach goal → +1</li>"
    "<li>Reach hole → 0</li>"
    "<li>Each frozen tile otherwise → 0</li>"
    "</ul>"
    "<h4>Episode end</h4>"
    "<ul>"
    "<li>Termination: fall into a hole, or reach the goal</li>"
    "<li>Truncation (with time_limit): 100 steps for 4x4, 200 for 8x8</li>"
    "</ul>"
    "<h4>Keyboard (human controls)</h4>"
    "<ul>"
    "<li>Up → Up arrow</li>"
    "<li>Right → Right arrow</li>"
    "<li>Down → Down arrow</li>"
    "<li>Left → Left arrow</li>"
    "</ul>"
    "<h4>Arguments / Config</h4>"
    "<p><code>is_slippery</code>: when True, movement is stochastic (slips to perpendicular directions); when False movement is deterministic.</p>"
    "<p>See the docs: <a href=\"https://gymnasium.farama.org/environments/toy_text/frozen_lake/\">FrozenLake (Gymnasium)</a></p>"
)


CLIFF_HTML = (
    "<h3>CliffWalking-v1</h3>"
    "<p>Navigate from start to goal along a cliff; stepping off the cliff is heavily penalized.</p>"
    "<h4>Starting state</h4>"
    "<p>Player starts in state 36 (row 3, column 0) by default.</p>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li>-1 per step</li>"
    "<li>-100 for stepping into the cliff squares</li>"
    "</ul>"
    "<h4>Episode end</h4>"
    "<ul>"
    "<li>Termination: reaching the goal state (state 47)</li>"
    "</ul>"
    "<h4>Keyboard (human controls)</h4>"
    "<ul>"
    "<li>Up → Up arrow</li>"
    "<li>Right → Right arrow</li>"
    "<li>Down → Down arrow</li>"
    "<li>Left → Left arrow</li>"
    "</ul>"
    "<p>See the docs: <a href=\"https://gymnasium.farama.org/environments/toy_text/cliff_walking/\">CliffWalking (Gymnasium)</a></p>"
)


GAME_INFO: Dict[GameId, str] = {
    GameId.TAXI: TAXI_HTML,
    GameId.FROZEN_LAKE: FROZEN_HTML,
    GameId.CLIFF_WALKING: CLIFF_HTML,
}


def get_game_info(game_id: GameId) -> str:
    """Return HTML help text for the given game_id.

    If no entry exists, an empty string is returned.
    """
    return GAME_INFO.get(game_id, "")
