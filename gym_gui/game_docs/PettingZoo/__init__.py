"""Documentation for PettingZoo multi-agent environments.

This module provides HTML documentation strings for PettingZoo environments
organized by family (Classic, MPE, SISL, Butterfly, Atari).

PettingZoo environments support two API types:
- AEC (Agent Environment Cycle): Turn-based games where agents act sequentially
- Parallel: Simultaneous action games where agents act at the same time
"""

from __future__ import annotations

from .classic.chess import CHESS_HTML, get_chess_html
from .classic.connect_four import CONNECT_FOUR_HTML, get_connect_four_html
from .classic.go import GO_HTML, get_go_html
from .classic.tictactoe import TICTACTOE_HTML, get_tictactoe_html

# ═══════════════════════════════════════════════════════════════════════════════
# Classic Family - Board Games and Card Games (AEC API)
# ═══════════════════════════════════════════════════════════════════════════════

CLASSIC_FAMILY_HTML = (
    "<h3>PettingZoo Classic Environments</h3>"
    "<p>Classic environments are implementations of popular turn-based human games, "
    "mostly competitive in nature.</p>"
    "<h4>Installation</h4>"
    "<pre><code>pip install 'pettingzoo[classic]'</code></pre>"
    "<h4>Available Environments</h4>"
    "<ul>"
    "<li><strong>Chess</strong> - Standard chess with AlphaZero-style observations</li>"
    "<li><strong>Connect Four</strong> - Classic 4-in-a-row game</li>"
    "<li><strong>Gin Rummy</strong> - Card game for two players</li>"
    "<li><strong>Go</strong> - Ancient board game (9x9, 13x13, 19x19)</li>"
    "<li><strong>Hanabi</strong> - Cooperative card game</li>"
    "<li><strong>Leduc Hold'em</strong> - Simplified poker variant</li>"
    "<li><strong>Rock Paper Scissors</strong> - Classic hand game</li>"
    "<li><strong>Texas Hold'em</strong> - Popular poker variant</li>"
    "<li><strong>Texas Hold'em No Limit</strong> - No-limit betting variant</li>"
    "<li><strong>Tic-Tac-Toe</strong> - Simple 3x3 grid game</li>"
    "</ul>"
    "<h4>Key Features</h4>"
    "<ul>"
    "<li>No environment arguments required (simple instantiation)</li>"
    "<li>Terminal rendering via text output</li>"
    "<li>Rewards at game end: +1 win, -1 loss, 0 draw</li>"
    "<li>Action masking for legal moves (<code>observation['action_mask']</code>)</li>"
    "<li>Illegal moves terminate game with -1 penalty</li>"
    "</ul>"
    "<h4>Usage Example</h4>"
    "<pre><code>from pettingzoo.classic import connect_four_v3\n\n"
    "env = connect_four_v3.env(render_mode='human')\n"
    "env.reset(seed=42)\n\n"
    "for agent in env.agent_iter():\n"
    "    observation, reward, termination, truncation, info = env.last()\n\n"
    "    if termination or truncation:\n"
    "        action = None\n"
    "    else:\n"
    "        mask = observation['action_mask']\n"
    "        action = env.action_space(agent).sample(mask)\n\n"
    "    env.step(action)\n"
    "env.close()</code></pre>"
    "<h4>Citation (RLCard based environments)</h4>"
    "<pre><code>@article{zha2019rlcard,\n"
    "  title={RLCard: A Toolkit for Reinforcement Learning in Card Games},\n"
    "  author={Zha, Daochen and Lai, Kwei-Herng and others},\n"
    "  journal={arXiv preprint arXiv:1910.04376},\n"
    "  year={2019}\n"
    "}</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/classic/'>PettingZoo Classic</a></p>"
)



TIC_TAC_TOE_HTML = (
    "<h3>PettingZoo: Tic-Tac-Toe</h3>"
    "<p>Classic 3x3 Tic-Tac-Toe game. Simple environment ideal for testing "
    "multi-agent algorithms and human vs agent play.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> AEC (turn-based)</li>"
    "<li><strong>Players:</strong> 2 (player_0, player_1)</li>"
    "<li><strong>Observation:</strong> 3x3x2 board representation</li>"
    "<li><strong>Actions:</strong> 9 (cell indices 0-8)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.classic import tictactoe_v3\nenv = tictactoe_v3.env()</code></pre>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li>+1 for three in a row</li>"
    "<li>-1 for opponent's three in a row</li>"
    "<li>0 for draw</li>"
    "</ul>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/classic/tictactoe/'>PettingZoo Tic-Tac-Toe</a></p>"
)


HANABI_HTML = (
    "<h3>PettingZoo: Hanabi</h3>"
    "<p>Cooperative card game where players must give clues to help each other play cards "
    "in the correct order. Players can see others' cards but not their own.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> AEC (turn-based)</li>"
    "<li><strong>Players:</strong> 2-5 (configurable)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Actions:</strong> Play card, discard card, or give clue</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.classic import hanabi_v5\nenv = hanabi_v5.env(players=2)</code></pre>"
    "<h4>Configuration</h4>"
    "<ul>"
    "<li><code>players</code>: Number of players (2-5)</li>"
    "<li><code>colors</code>: Number of card colors (default 5)</li>"
    "<li><code>ranks</code>: Number of card ranks (default 5)</li>"
    "</ul>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/classic/hanabi/'>PettingZoo Hanabi</a></p>"
)

TEXAS_HOLDEM_HTML = (
    "<h3>PettingZoo: Texas Hold'em</h3>"
    "<p>Limit Texas Hold'em poker. Players bet with fixed bet sizes during "
    "pre-flop, flop, turn, and river rounds.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> AEC (turn-based)</li>"
    "<li><strong>Players:</strong> 2 (configurable up to 10)</li>"
    "<li><strong>Actions:</strong> Fold, Call, Raise</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.classic import texas_holdem_v4\nenv = texas_holdem_v4.env(num_players=2)</code></pre>"
    "<h4>Observation</h4>"
    "<p>72-dimensional vector encoding:</p>"
    "<ul>"
    "<li>Hole cards (52 one-hot)</li>"
    "<li>Community cards (52 one-hot)</li>"
    "<li>Betting history</li>"
    "<li>Pot size</li>"
    "</ul>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/classic/texas_holdem/'>PettingZoo Texas Hold'em</a></p>"
)

RPS_HTML = (
    "<h3>PettingZoo: Rock Paper Scissors</h3>"
    "<p>Classic Rock Paper Scissors game. Both players select simultaneously, "
    "though implemented as AEC with hidden actions until reveal.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> AEC</li>"
    "<li><strong>Players:</strong> 2</li>"
    "<li><strong>Actions:</strong> 3 (Rock=0, Paper=1, Scissors=2)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.classic import rps_v2\nenv = rps_v2.env()</code></pre>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li>+1 for winning</li>"
    "<li>-1 for losing</li>"
    "<li>0 for tie</li>"
    "</ul>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/classic/rps/'>PettingZoo RPS</a></p>"
)

# ═══════════════════════════════════════════════════════════════════════════════
# MPE Family - Multi-Particle Environments (Parallel API)
# ═══════════════════════════════════════════════════════════════════════════════

SIMPLE_SPREAD_HTML = (
    "<h3>PettingZoo: Simple Spread</h3>"
    "<p>Cooperative environment where N agents must spread out to cover N landmarks. "
    "Agents are rewarded based on proximity to landmarks and penalized for collisions.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> N (default 3)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Observation:</strong> Continuous (positions, velocities)</li>"
    "<li><strong>Actions:</strong> Discrete (5) or Continuous (2D)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.mpe import simple_spread_v3\nenv = simple_spread_v3.parallel_env(N=3)</code></pre>"
    "<h4>Rewards</h4>"
    "<p>Shared reward based on:</p>"
    "<ul>"
    "<li>Negative distance to nearest uncovered landmark</li>"
    "<li>Collision penalty between agents</li>"
    "</ul>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/mpe/simple_spread/'>PettingZoo Simple Spread</a></p>"
)

SIMPLE_TAG_HTML = (
    "<h3>PettingZoo: Simple Tag</h3>"
    "<p>Predator-prey pursuit game. Predators (adversaries) try to catch the prey agent "
    "while prey tries to escape. Mixed cooperative-competitive.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> N predators + 1 prey</li>"
    "<li><strong>Game Type:</strong> Mixed (cooperative predators vs prey)</li>"
    "<li><strong>Observation:</strong> Continuous (positions, velocities)</li>"
    "<li><strong>Actions:</strong> Discrete (5) or Continuous (2D)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.mpe import simple_tag_v3\nenv = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3)</code></pre>"
    "<h4>Rewards</h4>"
    "<ul>"
    "<li>Predators: +10 for catching prey</li>"
    "<li>Prey: -10 when caught, otherwise 0</li>"
    "</ul>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/mpe/simple_tag/'>PettingZoo Simple Tag</a></p>"
)

SIMPLE_ADVERSARY_HTML = (
    "<h3>PettingZoo: Simple Adversary</h3>"
    "<p>One adversary tries to reach a target landmark while N agents try to block it. "
    "The agents don't know which landmark is the target.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> N agents + 1 adversary</li>"
    "<li><strong>Game Type:</strong> Competitive</li>"
    "<li><strong>Actions:</strong> Discrete (5) or Continuous (2D)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.mpe import simple_adversary_v3\nenv = simple_adversary_v3.parallel_env(N=2)</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/mpe/simple_adversary/'>PettingZoo Simple Adversary</a></p>"
)

SIMPLE_SPEAKER_LISTENER_HTML = (
    "<h3>PettingZoo: Simple Speaker Listener</h3>"
    "<p>Cooperative communication environment. A speaker agent observes the target landmark "
    "and must communicate to a listener agent which landmark to navigate to.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> 2 (speaker + listener)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.mpe import simple_speaker_listener_v4\nenv = simple_speaker_listener_v4.parallel_env()</code></pre>"
    "<h4>Communication</h4>"
    "<p>Speaker has discrete communication actions (N symbols). Listener observes speaker's "
    "message and must infer target landmark.</p>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/mpe/simple_speaker_listener/'>PettingZoo Speaker Listener</a></p>"
)

# ═══════════════════════════════════════════════════════════════════════════════
# SISL Family - Stanford Intelligent Systems Lab (Parallel API)
# ═══════════════════════════════════════════════════════════════════════════════

MULTIWALKER_HTML = (
    "<h3>PettingZoo: Multiwalker</h3>"
    "<p>Cooperative continuous control environment where multiple bipedal walkers must "
    "coordinate to carry a package across terrain without dropping it.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> 3 (default)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Observation:</strong> Continuous (31-dim per agent)</li>"
    "<li><strong>Actions:</strong> Continuous (4-dim per agent)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.sisl import multiwalker_v9\nenv = multiwalker_v9.parallel_env(n_walkers=3)</code></pre>"
    "<h4>Rewards</h4>"
    "<p>Shared reward for:</p>"
    "<ul>"
    "<li>Moving package forward</li>"
    "<li>Penalty for dropping package</li>"
    "<li>Penalty for falling</li>"
    "</ul>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/sisl/multiwalker/'>PettingZoo Multiwalker</a></p>"
)

WATERWORLD_HTML = (
    "<h3>PettingZoo: Waterworld</h3>"
    "<p>Cooperative continuous control where agents navigate a 2D world to capture food "
    "while avoiding poison. Agents have limited sensing range.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> 2 (default)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Observation:</strong> Continuous (sensor readings)</li>"
    "<li><strong>Actions:</strong> Continuous (2D thrust)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.sisl import waterworld_v4\nenv = waterworld_v4.parallel_env(n_pursuers=2)</code></pre>"
    "<h4>Configuration</h4>"
    "<ul>"
    "<li><code>n_pursuers</code>: Number of cooperative agents</li>"
    "<li><code>n_evaders</code>: Number of food items</li>"
    "<li><code>n_poisons</code>: Number of poison items</li>"
    "</ul>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/sisl/waterworld/'>PettingZoo Waterworld</a></p>"
)

PURSUIT_HTML = (
    "<h3>PettingZoo: Pursuit</h3>"
    "<p>Grid-based cooperative pursuit game where multiple predators try to surround "
    "and capture evader agents on a discrete grid.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> 8 predators (default)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Observation:</strong> Local grid view (7x7 default)</li>"
    "<li><strong>Actions:</strong> Discrete (5 - N/S/E/W/stay)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.sisl import pursuit_v4\nenv = pursuit_v4.parallel_env(n_pursuers=8, n_evaders=30)</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/sisl/pursuit/'>PettingZoo Pursuit</a></p>"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Butterfly Family - Visual Cooperative Games (Parallel API)
# ═══════════════════════════════════════════════════════════════════════════════

KNIGHTS_ARCHERS_ZOMBIES_HTML = (
    "<h3>PettingZoo: Knights Archers Zombies</h3>"
    "<p>Tower defense-style cooperative game. Knights and archers must defend against "
    "waves of zombies. Knights fight in melee, archers attack from range.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> 2 knights + 2 archers (default)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Observation:</strong> RGB image (512x512x3)</li>"
    "<li><strong>Actions:</strong> Discrete (6 per agent)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.butterfly import knights_archers_zombies_v10\n"
    "env = knights_archers_zombies_v10.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/'>"
    "PettingZoo KAZ</a></p>"
)

PISTONBALL_HTML = (
    "<h3>PettingZoo: Pistonball</h3>"
    "<p>Cooperative environment where pistons must coordinate to push a ball to the left "
    "side of the screen. Classic benchmark for multi-agent cooperation.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> 20 (default)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Observation:</strong> RGB image (local view)</li>"
    "<li><strong>Actions:</strong> Discrete (3 - up/down/stay)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.butterfly import pistonball_v6\nenv = pistonball_v6.parallel_env(n_pistons=20)</code></pre>"
    "<h4>Rewards</h4>"
    "<p>All agents share reward based on ball's horizontal progress toward goal.</p>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/butterfly/pistonball/'>PettingZoo Pistonball</a></p>"
)

COOPERATIVE_PONG_HTML = (
    "<h3>PettingZoo: Cooperative Pong</h3>"
    "<p>Cooperative variant of Pong where two paddles on the same side must work together "
    "to return the ball. Tests coordination between agents.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Agents:</strong> 2 (left paddle, right paddle)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Observation:</strong> RGB image</li>"
    "<li><strong>Actions:</strong> Discrete (3 - up/down/stay)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.butterfly import cooperative_pong_v5\nenv = cooperative_pong_v5.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/butterfly/cooperative_pong/'>"
    "PettingZoo Cooperative Pong</a></p>"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Atari Family - 2-Player Atari Games (Parallel API)
# ═══════════════════════════════════════════════════════════════════════════════

PONG_HTML = (
    "<h3>PettingZoo: Pong (Atari)</h3>"
    "<p>Classic two-player Pong. First player to reach 21 points wins. "
    "Competitive environment for testing self-play algorithms.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Players:</strong> 2 (first_0, second_0)</li>"
    "<li><strong>Game Type:</strong> Competitive</li>"
    "<li><strong>Observation:</strong> RGB image (210x160x3)</li>"
    "<li><strong>Actions:</strong> Discrete (18 Atari actions)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.atari import pong_v3\nenv = pong_v3.parallel_env()</code></pre>"
    "<h4>Note</h4>"
    "<p>Requires AutoROM license acceptance for Atari ROMs.</p>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/atari/pong/'>PettingZoo Pong</a></p>"
)

BOXING_HTML = (
    "<h3>PettingZoo: Boxing (Atari)</h3>"
    "<p>Two-player boxing game. Score points by landing punches on your opponent. "
    "Each successful hit scores one point.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Players:</strong> 2</li>"
    "<li><strong>Game Type:</strong> Competitive</li>"
    "<li><strong>Observation:</strong> RGB image (210x160x3)</li>"
    "<li><strong>Actions:</strong> Discrete (18 Atari actions)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.atari import boxing_v2\nenv = boxing_v2.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/atari/boxing/'>PettingZoo Boxing</a></p>"
)

TENNIS_HTML = (
    "<h3>PettingZoo: Tennis (Atari)</h3>"
    "<p>Two-player tennis with full court gameplay. Players control their racket position "
    "to return the ball over the net.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Players:</strong> 2</li>"
    "<li><strong>Game Type:</strong> Competitive</li>"
    "<li><strong>Observation:</strong> RGB image (210x160x3)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.atari import tennis_v3\nenv = tennis_v3.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/atari/tennis/'>PettingZoo Tennis</a></p>"
)

SPACE_INVADERS_HTML = (
    "<h3>PettingZoo: Space Invaders (Atari)</h3>"
    "<p>Cooperative two-player Space Invaders. Both players defend against alien invasion. "
    "Work together to clear waves of enemies.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Players:</strong> 2</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Observation:</strong> RGB image (210x160x3)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.atari import space_invaders_v2\nenv = space_invaders_v2.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/atari/space_invaders/'>"
    "PettingZoo Space Invaders</a></p>"
)

ICE_HOCKEY_HTML = (
    "<h3>PettingZoo: Ice Hockey (Atari)</h3>"
    "<p>Fast-paced two-player ice hockey. Control your team's players to score goals "
    "against your opponent.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Players:</strong> 2</li>"
    "<li><strong>Game Type:</strong> Competitive</li>"
    "<li><strong>Observation:</strong> RGB image (210x160x3)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.atari import ice_hockey_v2\nenv = ice_hockey_v2.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/atari/ice_hockey/'>"
    "PettingZoo Ice Hockey</a></p>"
)

MARIO_BROS_HTML = (
    "<h3>PettingZoo: Mario Bros (Atari)</h3>"
    "<p>Original Mario Bros cooperative game. Two players work together to clear "
    "enemies from the sewers by hitting platforms from below.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Players:</strong> 2 (Mario and Luigi)</li>"
    "<li><strong>Game Type:</strong> Cooperative</li>"
    "<li><strong>Observation:</strong> RGB image (210x160x3)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.atari import mario_bros_v3\nenv = mario_bros_v3.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/atari/mario_bros/'>"
    "PettingZoo Mario Bros</a></p>"
)

WARLORDS_HTML = (
    "<h3>PettingZoo: Warlords (Atari)</h3>"
    "<p>Medieval castle defense game. Protect your castle wall while trying to destroy "
    "your opponent's. Up to 4 players supported.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Players:</strong> 4 (configurable)</li>"
    "<li><strong>Game Type:</strong> Competitive</li>"
    "<li><strong>Observation:</strong> RGB image (210x160x3)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.atari import warlords_v3\nenv = warlords_v3.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/atari/warlords/'>"
    "PettingZoo Warlords</a></p>"
)

COMBAT_TANK_HTML = (
    "<h3>PettingZoo: Combat Tank (Atari)</h3>"
    "<p>Two-player tank battle in an arena with obstacles. Destroy your opponent's tank "
    "while avoiding their fire.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Players:</strong> 2</li>"
    "<li><strong>Game Type:</strong> Competitive</li>"
    "<li><strong>Observation:</strong> RGB image (210x160x3)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.atari import combat_tank_v2\nenv = combat_tank_v2.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/atari/combat_tank/'>"
    "PettingZoo Combat Tank</a></p>"
)

COMBAT_PLANE_HTML = (
    "<h3>PettingZoo: Combat Plane (Atari)</h3>"
    "<p>Two-player aerial dogfight with biplanes. Shoot down your opponent while "
    "avoiding their attacks.</p>"
    "<h4>Environment Details</h4>"
    "<ul>"
    "<li><strong>API Type:</strong> Parallel (simultaneous)</li>"
    "<li><strong>Players:</strong> 2</li>"
    "<li><strong>Game Type:</strong> Competitive</li>"
    "<li><strong>Observation:</strong> RGB image (210x160x3)</li>"
    "</ul>"
    "<h4>Make</h4>"
    "<pre><code>from pettingzoo.atari import combat_plane_v2\nenv = combat_plane_v2.parallel_env()</code></pre>"
    "<p>Docs: <a href='https://pettingzoo.farama.org/environments/atari/combat_plane/'>"
    "PettingZoo Combat Plane</a></p>"
)

__all__ = [
    # Classic
    "CLASSIC_FAMILY_HTML",
    "CHESS_HTML",
    "get_chess_html",
    "CONNECT_FOUR_HTML",
    "get_connect_four_html",
    "TIC_TAC_TOE_HTML",
    "TICTACTOE_HTML",
    "get_tictactoe_html",
    "GO_HTML",
    "get_go_html",
    "HANABI_HTML",
    "TEXAS_HOLDEM_HTML",
    "RPS_HTML",
    # MPE
    "SIMPLE_SPREAD_HTML",
    "SIMPLE_TAG_HTML",
    "SIMPLE_ADVERSARY_HTML",
    "SIMPLE_SPEAKER_LISTENER_HTML",
    # SISL
    "MULTIWALKER_HTML",
    "WATERWORLD_HTML",
    "PURSUIT_HTML",
    # Butterfly
    "KNIGHTS_ARCHERS_ZOMBIES_HTML",
    "PISTONBALL_HTML",
    "COOPERATIVE_PONG_HTML",
    # Atari
    "PONG_HTML",
    "BOXING_HTML",
    "TENNIS_HTML",
    "SPACE_INVADERS_HTML",
    "ICE_HOCKEY_HTML",
    "MARIO_BROS_HTML",
    "WARLORDS_HTML",
    "COMBAT_TANK_HTML",
    "COMBAT_PLANE_HTML",
]
