"""Documentation for OpenSpiel Checkers environment.

OpenSpiel is a collection of games from Google DeepMind for research in
reinforcement learning, search, and game theory. Shimmy provides PettingZoo-
compatible wrappers for OpenSpiel games.

Repository: https://github.com/google-deepmind/open_spiel
Shimmy: https://shimmy.farama.org/environments/open_spiel/
"""

from __future__ import annotations

CHECKERS_HTML = (
    "<h3>OpenSpiel: Checkers (English Draughts)</h3>"
    "<p>Classic 8x8 checkers (English draughts) via Google DeepMind's OpenSpiel library, "
    "wrapped with Shimmy for PettingZoo compatibility.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> AEC (turn-based)</li>"
    "<li><strong>Players:</strong> 2 (player_0 = Black, player_1 = White)</li>"
    "<li><strong>Board:</strong> 8x8 grid (only dark squares used)</li>"
    "<li><strong>Pieces:</strong> 12 per player at start</li>"
    "<li><strong>Observation:</strong> Information state tensor</li>"
    "<li><strong>Actions:</strong> Move indices (variable based on legal moves)</li>"
    "</ul>"
    "<h4>Game Rules</h4>"
    "<ul>"
    "<li>Pieces move diagonally forward on dark squares</li>"
    "<li>Capture by jumping over opponent's piece</li>"
    "<li>Multiple jumps allowed in single turn</li>"
    "<li>Reaching opposite end promotes piece to King</li>"
    "<li>Kings can move/capture diagonally in any direction</li>"
    "<li>Win by capturing all opponent pieces or blocking all moves</li>"
    "</ul>"
    "<h4>Installation</h4>"
    "<pre><code>pip install open-spiel shimmy[openspiel]</code></pre>"
    "<p>Or with MOSAIC:</p>"
    "<pre><code>pip install -e '.[openspiel]'</code></pre>"
    "<h4>Make</h4>"
    "<pre><code>from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0\n\n"
    "env = OpenSpielCompatibilityV0(\n"
    "    game_name='checkers',\n"
    "    render_mode='rgb_array'\n"
    ")\n"
    "env.reset()</code></pre>"
    "<h4>Usage Example</h4>"
    "<pre><code>env.reset()\n\n"
    "for agent in env.agent_iter():\n"
    "    obs, reward, term, trunc, info = env.last()\n\n"
    "    if term or trunc:\n"
    "        action = None\n"
    "    else:\n"
    "        # action_mask is in info for OpenSpiel\n"
    "        mask = info.get('action_mask')\n"
    "        legal = [i for i, v in enumerate(mask) if v]\n"
    "        action = legal[0]  # or use policy\n\n"
    "    env.step(action)\n"
    "env.close()</code></pre>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li><strong>Win:</strong> +1</li>"
    "<li><strong>Loss:</strong> -1</li>"
    "<li><strong>Draw:</strong> 0</li>"
    "</ul>"
    "<h4>Piece Values</h4>"
    "<table style='border-collapse: collapse; margin: 10px 0;'>"
    "<tr><th style='border: 1px solid #ccc; padding: 5px;'>Value</th>"
    "<th style='border: 1px solid #ccc; padding: 5px;'>Piece</th></tr>"
    "<tr><td style='border: 1px solid #ccc; padding: 5px;'>0</td>"
    "<td style='border: 1px solid #ccc; padding: 5px;'>Empty</td></tr>"
    "<tr><td style='border: 1px solid #ccc; padding: 5px;'>1</td>"
    "<td style='border: 1px solid #ccc; padding: 5px;'>Black piece</td></tr>"
    "<tr><td style='border: 1px solid #ccc; padding: 5px;'>2</td>"
    "<td style='border: 1px solid #ccc; padding: 5px;'>Black king</td></tr>"
    "<tr><td style='border: 1px solid #ccc; padding: 5px;'>3</td>"
    "<td style='border: 1px solid #ccc; padding: 5px;'>White piece</td></tr>"
    "<tr><td style='border: 1px solid #ccc; padding: 5px;'>4</td>"
    "<td style='border: 1px solid #ccc; padding: 5px;'>White king</td></tr>"
    "</table>"
    "<h4>References</h4>"
    "<ul>"
    "<li><a href='https://github.com/google-deepmind/open_spiel'>OpenSpiel GitHub</a></li>"
    "<li><a href='https://shimmy.farama.org/environments/open_spiel/'>Shimmy OpenSpiel Docs</a></li>"
    "</ul>"
    "<h4>Citation</h4>"
    "<pre><code>@article{lanctot2019openspiel,\n"
    "  title={OpenSpiel: A General Framework for Reinforcement Learning in Games},\n"
    "  author={Lanctot, Marc and Lockhart, Edward and Lespiau, Jean-Baptiste and others},\n"
    "  journal={arXiv preprint arXiv:1908.09453},\n"
    "  year={2019}\n"
    "}</code></pre>"
)


def get_checkers_html() -> str:
    """Return HTML documentation for Checkers environment."""
    return CHECKERS_HTML


__all__ = ["CHECKERS_HTML", "get_checkers_html"]
