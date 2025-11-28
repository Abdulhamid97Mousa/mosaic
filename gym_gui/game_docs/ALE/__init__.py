"""Documentation for ALE (Atari Learning Environment) games.

Currently provides a shared Adventure blurb used for both Adventure-v4 and
ALE/Adventure-v5 variants. Expand this module with additional ALE titles as
they are added.
"""

from __future__ import annotations

ADVENTURE_HTML = (
    "<h3>ALE: Adventure</h3>"
    "<p>Adventure is a classic Atari 2600 title where you explore mazes, find the"
    " enchanted chalice, and return it to the golden castle while avoiding or"
    " defeating dragons. You can carry items like keys, a sword, a bridge, or a"
    " magnet to solve puzzles and progress.</p>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(18)</code> with the following semantics:</p>"
    "<ul>"
    "<li>0 → NOOP</li>"
    "<li>1 → FIRE</li>"
    "<li>2 → UP</li>"
    "<li>3 → RIGHT</li>"
    "<li>4 → LEFT</li>"
    "<li>5 → DOWN</li>"
    "<li>6 → UPRIGHT</li>"
    "<li>7 → UPLEFT</li>"
    "<li>8 → DOWNRIGHT</li>"
    "<li>9 → DOWNLEFT</li>"
    "<li>10 → UPFIRE</li>"
    "<li>11 → RIGHTFIRE</li>"
    "<li>12 → LEFTFIRE</li>"
    "<li>13 → DOWNFIRE</li>"
    "<li>14 → UPRIGHTFIRE</li>"
    "<li>15 → UPLEFTFIRE</li>"
    "<li>16 → DOWNRIGHTFIRE</li>"
    "<li>17 → DOWNLEFTFIRE</li>"
    "</ul>"
    "<h4>Observation Space</h4>"
    "<p>Default <code>obs_type=\"rgb\"</code>: <code>Box(0, 255, (210, 160, 3), uint8)</code>.<br/>"
    "Optionally, <code>obs_type=\"ram\"</code> → <code>Box(0, 255, (128,), uint8)</code> or"
    " <code>obs_type=\"grayscale\"</code> → <code>Box(0, 255, (210, 160), uint8)</code>.</p>"
    "<h4>Variants</h4>"
    "<ul>"
    "<li>Adventure-v4 (classic v4 ruleset)</li>"
    "<li>ALE/Adventure-v5 (ALE namespaced v5 ruleset)</li>"
    "</ul>"
    "<h4>Config</h4>"
    "<p>ALE supports <code>mode</code> and <code>difficulty</code> keyword arguments (defaults 0)."
    " Refer to the upstream docs for the full matrix.</p>"
    "<p>Docs: <a href=\"https://ale.farama.org/environments/\">ALE Environments</a></p>"
)

AIR_RAID_HTML = (
    "<h3>ALE: AirRaid</h3>"
    "<p>You control a ship that can move sideways to protect two buildings from"
    " flying saucers that attempt to drop bombs on them.</p>"
    "<h4>Make</h4>"
    "<pre><code>gymnasium.make(\"ALE/AirRaid-v5\")</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(6)</code></p>"
    "<ul>"
    "<li>0 → NOOP</li>"
    "<li>1 → FIRE</li>"
    "<li>2 → RIGHT</li>"
    "<li>3 → LEFT</li>"
    "<li>4 → RIGHTFIRE</li>"
    "<li>5 → LEFTFIRE</li>"
    "</ul>"
    "<p>To enable all 18 Atari actions, pass <code>full_action_space=True</code>"
    " when creating the environment.</p>"
    "<h4>Observation Space</h4>"
    "<p>Default <code>obs_type=\"rgb\"</code>: <code>Box(0, 255, (210, 160, 3), uint8)</code>.</p>"
    "<p>Other types:</p>"
    "<ul>"
    "<li><code>obs_type=\"ram\"</code> → <code>Box(0, 255, (128,), uint8)</code></li>"
    "<li><code>obs_type=\"grayscale\"</code> → <code>Box(0, 255, (210, 160), uint8)</code></li>"
    "</ul>"
    "<h4>Variants</h4>"
    "<table>"
    "<thead><tr><th>Env-id</th><th>obs_type</th><th>frameskip</th><th>repeat_action_probability</th></tr></thead>"
    "<tbody>"
    "<tr><td>AirRaid-v4</td><td>rgb</td><td>(2, 5)</td><td>0.00</td></tr>"
    "<tr><td>ALE/AirRaid-v5</td><td>rgb</td><td>4</td><td>0.25</td></tr>"
    "</tbody>"
    "</table>"
    "<h4>Difficulty and modes</h4>"
    "<p>Flavours can be selected via <code>difficulty</code> and <code>mode</code>"
    " kwargs. Available modes are <code>[1, ..., 8]</code> (default 1) and"
    " available difficulties are <code>[0]</code> (default 0).</p>"
    "<h4>Docs</h4>"
    "<p><a href=\"https://ale.farama.org/environments/air_raid/\">Farama: AirRaid</a></p>"
)

ASSAULT_HTML = (
    "<h3>ALE: Assault</h3>"
    "<p>You control a vehicle that can move sideways while a mothership deploys"
    " drones. Destroy enemies and dodge their attacks.</p>"
    "<h4>Make</h4>"
    "<pre><code>gymnasium.make(\"ALE/Assault-v5\")</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(7)</code></p>"
    "<ul>"
    "<li>0 → NOOP</li>"
    "<li>1 → FIRE</li>"
    "<li>2 → UP</li>"
    "<li>3 → RIGHT</li>"
    "<li>4 → LEFT</li>"
    "<li>5 → RIGHTFIRE</li>"
    "<li>6 → LEFTFIRE</li>"
    "</ul>"
    "<p>To enable all 18 Atari actions, pass <code>full_action_space=True</code>"
    " when creating the environment.</p>"
    "<h4>Observation Space</h4>"
    "<p>Default <code>obs_type=\"rgb\"</code>: <code>Box(0, 255, (210, 160, 3), uint8)</code>.</p>"
    "<p>Other types:</p>"
    "<ul>"
    "<li><code>obs_type=\"ram\"</code> → <code>Box(0, 255, (128,), uint8)</code></li>"
    "<li><code>obs_type=\"grayscale\"</code> → <code>Box(0, 255, (210, 160), uint8)</code></li>"
    "</ul>"
    "<h4>Variants</h4>"
    "<table>"
    "<thead><tr><th>Env-id</th><th>obs_type</th><th>frameskip</th><th>repeat_action_probability</th></tr></thead>"
    "<tbody>"
    "<tr><td>Assault-v4</td><td>rgb</td><td>(2, 5)</td><td>0.00</td></tr>"
    "<tr><td>ALE/Assault-v5</td><td>rgb</td><td>4</td><td>0.25</td></tr>"
    "</tbody>"
    "</table>"
    "<h4>Difficulty and modes</h4>"
    "<p>Flavours can be selected via <code>difficulty</code> and <code>mode</code>"
    " kwargs. For Assault, available modes are <code>[0]</code> (default 0) and"
    " available difficulties are <code>[0]</code> (default 0).</p>"
    "<h4>Docs</h4>"
    "<p><a href=\"https://ale.farama.org/environments/assault/\">Farama: Assault</a></p>"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Player Atari Games (PettingZoo Compatible)
# ═══════════════════════════════════════════════════════════════════════════════

PONG_2P_HTML = (
    "<h3>ALE: Pong (2-Player)</h3>"
    "<p>Classic two-player Pong. First to 21 points wins. Available as single-player "
    "ALE environment or two-player PettingZoo environment.</p>"
    "<h4>Make (Single-Player)</h4>"
    "<pre><code>gymnasium.make(\"ALE/Pong-v5\")</code></pre>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import pong_v3\nenv = pong_v3.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(6)</code> - NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE</p>"
    "<h4>Observation Space</h4>"
    "<p><code>Box(0, 255, (210, 160, 3), uint8)</code></p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/pong/'>ALE: Pong</a></p>"
)

BOXING_2P_HTML = (
    "<h3>ALE: Boxing (2-Player)</h3>"
    "<p>Two-player boxing match. Score points by landing punches. Each hit scores "
    "one point. Match ends after time limit.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import boxing_v2\nenv = boxing_v2.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(18)</code> - Full Atari action set</p>"
    "<h4>Controls</h4>"
    "<ul>"
    "<li>Move with joystick directions</li>"
    "<li>Punch with FIRE + direction</li>"
    "</ul>"
    "<p>Docs: <a href='https://ale.farama.org/environments/boxing/'>ALE: Boxing</a></p>"
)

TENNIS_2P_HTML = (
    "<h3>ALE: Tennis (2-Player)</h3>"
    "<p>Full-court tennis game for two players. Standard tennis scoring with "
    "serve, volley, and ground stroke mechanics.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import tennis_v3\nenv = tennis_v3.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(18)</code> - Full Atari action set</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/tennis/'>ALE: Tennis</a></p>"
)

SPACE_INVADERS_2P_HTML = (
    "<h3>ALE: Space Invaders (2-Player)</h3>"
    "<p>Cooperative two-player Space Invaders. Both players defend against alien "
    "invasion. Alternating or simultaneous play modes.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import space_invaders_v2\nenv = space_invaders_v2.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(6)</code> - NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE</p>"
    "<h4>Game Type</h4>"
    "<p><strong>Cooperative</strong> - Both players share score and lives.</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/space_invaders/'>ALE: Space Invaders</a></p>"
)

ICE_HOCKEY_2P_HTML = (
    "<h3>ALE: Ice Hockey (2-Player)</h3>"
    "<p>Fast-paced two-player ice hockey. Control your team to score goals. "
    "Features checking, passing, and shooting mechanics.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import ice_hockey_v2\nenv = ice_hockey_v2.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(18)</code> - Full Atari action set</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/ice_hockey/'>ALE: Ice Hockey</a></p>"
)

DOUBLE_DUNK_2P_HTML = (
    "<h3>ALE: Double Dunk (2-Player)</h3>"
    "<p>Two-on-two basketball game featuring dunking mechanics. Fast-paced "
    "gameplay with shooting, passing, and stealing.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import double_dunk_v3\nenv = double_dunk_v3.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(18)</code> - Full Atari action set</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/double_dunk/'>ALE: Double Dunk</a></p>"
)

WARLORDS_2P_HTML = (
    "<h3>ALE: Warlords (2-4 Player)</h3>"
    "<p>Medieval castle defense for up to 4 players. Each player protects their "
    "castle wall while trying to break through opponents' defenses.</p>"
    "<h4>Make (Multi-Player)</h4>"
    "<pre><code>from pettingzoo.atari import warlords_v3\nenv = warlords_v3.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(6)</code> - NOOP, FIRE, UP, DOWN, UPFIRE, DOWNFIRE</p>"
    "<h4>Players</h4>"
    "<p>Supports 2-4 players in competitive free-for-all mode.</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/warlords/'>ALE: Warlords</a></p>"
)

MARIO_BROS_2P_HTML = (
    "<h3>ALE: Mario Bros (2-Player)</h3>"
    "<p>The original Mario Bros arcade game. Mario and Luigi cooperate to clear "
    "enemies from the sewers by hitting platforms from below.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import mario_bros_v3\nenv = mario_bros_v3.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(18)</code> - Full Atari action set</p>"
    "<h4>Game Type</h4>"
    "<p><strong>Cooperative</strong> - Both players work together.</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/mario_bros/'>ALE: Mario Bros</a></p>"
)

COMBAT_PLANE_2P_HTML = (
    "<h3>ALE: Combat Plane (2-Player)</h3>"
    "<p>Two-player aerial dogfight with World War I biplanes. Engage in air combat "
    "with various game modes including invisible planes.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import combat_plane_v2\nenv = combat_plane_v2.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(18)</code> - Full Atari action set including thrust and fire</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/combat/'>ALE: Combat</a></p>"
)

COMBAT_TANK_2P_HTML = (
    "<h3>ALE: Combat Tank (2-Player)</h3>"
    "<p>Two-player tank battle in maze-like arenas. Destroy your opponent's tank "
    "while navigating obstacles. Multiple arena configurations.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import combat_tank_v2\nenv = combat_tank_v2.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(18)</code> - Full Atari action set</p>"
    "<h4>Arena Types</h4>"
    "<ul>"
    "<li>Open battlefield</li>"
    "<li>Maze with walls</li>"
    "<li>Invisible tanks mode</li>"
    "</ul>"
    "<p>Docs: <a href='https://ale.farama.org/environments/combat/'>ALE: Combat</a></p>"
)

JOUST_2P_HTML = (
    "<h3>ALE: Joust (2-Player)</h3>"
    "<p>Aerial combat on flying ostriches. Can be played cooperatively or "
    "competitively. Defeat enemies by colliding from above.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import joust_v3\nenv = joust_v3.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(18)</code> - Full Atari action set with flap controls</p>"
    "<h4>Game Types</h4>"
    "<ul>"
    "<li><strong>Cooperative:</strong> Work together against enemies</li>"
    "<li><strong>Competitive:</strong> Battle each other for points</li>"
    "</ul>"
    "<p>Docs: <a href='https://ale.farama.org/environments/joust/'>ALE: Joust</a></p>"
)

WIZARD_OF_WOR_2P_HTML = (
    "<h3>ALE: Wizard of Wor (2-Player)</h3>"
    "<p>Dungeon crawler shooter for two players. Navigate mazes, destroy monsters, "
    "and battle the Wizard. Can be cooperative or competitive.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import wizard_of_wor_v3\nenv = wizard_of_wor_v3.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(10)</code> - Movement in 4 directions plus fire</p>"
    "<h4>Game Types</h4>"
    "<ul>"
    "<li><strong>Cooperative:</strong> Clear dungeons together</li>"
    "<li><strong>Competitive:</strong> Score more than your partner</li>"
    "</ul>"
    "<p>Docs: <a href='https://ale.farama.org/environments/wizard_of_wor/'>ALE: Wizard of Wor</a></p>"
)

SURROUND_2P_HTML = (
    "<h3>ALE: Surround (2-Player)</h3>"
    "<p>Tron-like light cycle game. Players leave trails behind them and must "
    "avoid crashing into walls, trails, or each other. Last player standing wins.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import surround_v2\nenv = surround_v2.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(5)</code> - NOOP, UP, DOWN, LEFT, RIGHT</p>"
    "<h4>Strategy</h4>"
    "<p>Trap your opponent while avoiding being trapped yourself.</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/surround/'>ALE: Surround</a></p>"
)

OTHELLO_2P_HTML = (
    "<h3>ALE: Othello/Reversi (2-Player)</h3>"
    "<p>Classic Reversi board game. Place pieces to flip your opponent's discs. "
    "Player with most discs at the end wins.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import othello_v3\nenv = othello_v3.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete(65)</code> - 64 board positions + pass</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/othello/'>ALE: Othello</a></p>"
)

VIDEO_CHECKERS_2P_HTML = (
    "<h3>ALE: Video Checkers (2-Player)</h3>"
    "<p>Classic checkers/draughts board game. Move diagonally, jump opponents' "
    "pieces to capture them. King pieces can move backwards.</p>"
    "<h4>Make (Two-Player)</h4>"
    "<pre><code>from pettingzoo.atari import video_checkers_v4\nenv = video_checkers_v4.parallel_env()</code></pre>"
    "<h4>Action Space</h4>"
    "<p><code>Discrete()</code> - Valid moves vary by game state</p>"
    "<p>Docs: <a href='https://ale.farama.org/environments/video_checkers/'>ALE: Video Checkers</a></p>"
)

__all__ = [
    # Single-player classics
    "ADVENTURE_HTML",
    "AIR_RAID_HTML",
    "ASSAULT_HTML",
    # Multi-player games
    "PONG_2P_HTML",
    "BOXING_2P_HTML",
    "TENNIS_2P_HTML",
    "SPACE_INVADERS_2P_HTML",
    "ICE_HOCKEY_2P_HTML",
    "DOUBLE_DUNK_2P_HTML",
    "WARLORDS_2P_HTML",
    "MARIO_BROS_2P_HTML",
    "COMBAT_PLANE_2P_HTML",
    "COMBAT_TANK_2P_HTML",
    "JOUST_2P_HTML",
    "WIZARD_OF_WOR_2P_HTML",
    "SURROUND_2P_HTML",
    "OTHELLO_2P_HTML",
    "VIDEO_CHECKERS_2P_HTML",
]
