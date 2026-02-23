"""Documentation for OpenSpiel board game environments.

OpenSpiel is a collection of games from Google DeepMind for research in
reinforcement learning, search, and game theory. It provides implementations
of many well-known games including Chess, Go, Checkers, Poker variants, etc.

Shimmy provides PettingZoo-compatible wrappers for OpenSpiel games, allowing
them to be used with standard multi-agent RL frameworks.

Repository: https://github.com/google-deepmind/open_spiel
Shimmy: https://shimmy.farama.org/environments/open_spiel/

Currently supported in MOSAIC:
- Checkers (English draughts)
"""

from __future__ import annotations

from .checkers import CHECKERS_HTML, get_checkers_html

# Family-level documentation
OPENSPIEL_FAMILY_HTML = (
    "<h3>OpenSpiel Environments</h3>"
    "<p>OpenSpiel is a collection of games from Google DeepMind for research in "
    "reinforcement learning, search, and game theory. It provides efficient implementations "
    "of many well-known games.</p>"
    "<h4>Installation</h4>"
    "<pre><code>pip install open-spiel shimmy[openspiel]</code></pre>"
    "<p>Or with MOSAIC:</p>"
    "<pre><code>pip install -e '.[openspiel]'</code></pre>"
    "<h4>Available Games in MOSAIC</h4>"
    "<ul>"
    "<li><strong>Checkers</strong> - Classic 8x8 English draughts</li>"
    "</ul>"
    "<h4>Other OpenSpiel Games (not yet in MOSAIC)</h4>"
    "<ul>"
    "<li>Backgammon, Breakthrough, Chess, Connect Four, Go</li>"
    "<li>Hex, Othello, Tic-Tac-Toe, Y</li>"
    "<li>Poker variants (Leduc, Texas Hold'em, Kuhn)</li>"
    "<li>And 70+ more games</li>"
    "</ul>"
    "<h4>Key Features</h4>"
    "<ul>"
    "<li>Perfect and imperfect information games</li>"
    "<li>Simultaneous and sequential move games</li>"
    "<li>Cooperative, competitive, and general-sum games</li>"
    "<li>Efficient C++ implementations with Python bindings</li>"
    "<li>PettingZoo compatibility via Shimmy wrapper</li>"
    "</ul>"
    "<h4>Usage with Shimmy</h4>"
    "<pre><code>from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0\n\n"
    "# Create any OpenSpiel game as PettingZoo environment\n"
    "env = OpenSpielCompatibilityV0(game_name='checkers')\n"
    "env.reset()\n\n"
    "for agent in env.agent_iter():\n"
    "    obs, reward, term, trunc, info = env.last()\n"
    "    action = ...  # Your policy\n"
    "    env.step(action)</code></pre>"
    "<h4>Citation</h4>"
    "<pre><code>@article{lanctot2019openspiel,\n"
    "  title={OpenSpiel: A General Framework for Reinforcement Learning in Games},\n"
    "  author={Lanctot, Marc and Lockhart, Edward and Lespiau, Jean-Baptiste and others},\n"
    "  journal={arXiv preprint arXiv:1908.09453},\n"
    "  year={2019}\n"
    "}</code></pre>"
    "<p>Docs: <a href='https://github.com/google-deepmind/open_spiel'>OpenSpiel GitHub</a></p>"
)

__all__ = [
    "OPENSPIEL_FAMILY_HTML",
    "CHECKERS_HTML",
    "get_checkers_html",
]
