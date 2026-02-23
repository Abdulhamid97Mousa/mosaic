"""Documentation for ALE (Atari Learning Environment) games.

Provides documentation for both single-player ALE games and multi-player
PettingZoo Atari environments.
"""

from __future__ import annotations

# Multi-Player Atari Games (PettingZoo Compatible) - Imported from directories
from .Boxing import BOXING_HTML
from .CombatPlane import COMBAT_PLANE_HTML
from .CombatTank import COMBAT_TANK_HTML
from .DoubleDunk import DOUBLE_DUNK_HTML
from .EntombedCompetitive import ENTOMBED_COMPETITIVE_HTML
from .EntombedCooperative import ENTOMBED_COOPERATIVE_HTML
from .FlagCapture import FLAG_CAPTURE_HTML
from .IceHockey import ICE_HOCKEY_HTML
from .Joust import JOUST_HTML
from .MarioBros import MARIO_BROS_HTML
from .MazeCraze import MAZE_CRAZE_HTML
from .Othello import OTHELLO_HTML
from .Pong import PONG_HTML
from .Quadrapong import QUADRAPONG_HTML
from .SpaceInvaders import SPACE_INVADERS_HTML
from .SpaceWar import SPACE_WAR_HTML
from .Surround import SURROUND_HTML
from .Tennis import TENNIS_HTML
from .VideoCheckers import VIDEO_CHECKERS_HTML
from .Warlords import WARLORDS_HTML
from .WizardOfWor import WIZARD_OF_WOR_HTML

# ═══════════════════════════════════════════════════════════════════════════════
# Single-Player ALE Games (Inline Documentation)
# ═══════════════════════════════════════════════════════════════════════════════

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

__all__ = [
    # Single-player ALE games
    "ADVENTURE_HTML",
    "AIR_RAID_HTML",
    "ASSAULT_HTML",
    # Multi-player games - 2-player competitive
    "BOXING_HTML",
    "COMBAT_PLANE_HTML",
    "COMBAT_TANK_HTML",
    "DOUBLE_DUNK_HTML",
    "ENTOMBED_COMPETITIVE_HTML",
    "FLAG_CAPTURE_HTML",
    "ICE_HOCKEY_HTML",
    "MAZE_CRAZE_HTML",
    "OTHELLO_HTML",
    "PONG_HTML",
    "SPACE_WAR_HTML",
    "SURROUND_HTML",
    "TENNIS_HTML",
    "VIDEO_CHECKERS_HTML",
    # Multi-player games - mixed/cooperative
    "ENTOMBED_COOPERATIVE_HTML",
    "JOUST_HTML",
    "MARIO_BROS_HTML",
    "SPACE_INVADERS_HTML",
    "WIZARD_OF_WOR_HTML",
    # Multi-player games - 4-player
    "QUADRAPONG_HTML",
    "WARLORDS_HTML",
]
