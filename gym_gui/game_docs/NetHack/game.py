"""NetHack full game documentation."""

from __future__ import annotations

NETHACK_FULL_HTML = """
<h2>NetHack Learning Environment (NLE)</h2>

<p>
NetHack is a classic roguelike dungeon crawler first released in 1987. The NetHack Learning
Environment (NLE) provides a gym-compatible interface to the full game, making it one of
the most challenging benchmarks for reinforcement learning research.
</p>

<h4>Why NetHack is Hard</h4>
<ul>
    <li><strong>Procedural Generation</strong>: Every game generates unique dungeons</li>
    <li><strong>Permadeath</strong>: One life per episode, death is permanent</li>
    <li><strong>Sparse Rewards</strong>: Only score/death provide clear signals</li>
    <li><strong>Long Horizon</strong>: Winning requires ~100,000+ steps</li>
    <li><strong>Complex State Space</strong>: Inventory, status effects, dungeon features</li>
    <li><strong>Large Action Space</strong>: 100+ possible actions</li>
    <li><strong>Hidden Information</strong>: Item identification, trap detection</li>
</ul>

<h4>Observation Space</h4>
<p>NLE provides multiple observation channels:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Shape</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">glyphs</td><td style="border: 1px solid #ddd; padding: 8px;">(21, 79)</td><td style="border: 1px solid #ddd; padding: 8px;">Map glyph IDs (5991 unique)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">chars</td><td style="border: 1px solid #ddd; padding: 8px;">(21, 79)</td><td style="border: 1px solid #ddd; padding: 8px;">ASCII characters</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">colors</td><td style="border: 1px solid #ddd; padding: 8px;">(21, 79)</td><td style="border: 1px solid #ddd; padding: 8px;">Terminal colors (0-15)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">specials</td><td style="border: 1px solid #ddd; padding: 8px;">(21, 79)</td><td style="border: 1px solid #ddd; padding: 8px;">Special attributes</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">blstats</td><td style="border: 1px solid #ddd; padding: 8px;">(25,)</td><td style="border: 1px solid #ddd; padding: 8px;">Bottom-line stats (HP, AC, etc.)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">message</td><td style="border: 1px solid #ddd; padding: 8px;">(256,)</td><td style="border: 1px solid #ddd; padding: 8px;">Game messages</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">inv_*</td><td style="border: 1px solid #ddd; padding: 8px;">various</td><td style="border: 1px solid #ddd; padding: 8px;">Inventory information</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">tty_chars</td><td style="border: 1px solid #ddd; padding: 8px;">(24, 80)</td><td style="border: 1px solid #ddd; padding: 8px;">Full terminal display</td></tr>
</table>

<h4>Action Space</h4>
<p>
NLE uses a discrete action space with 93 core actions (movement, items, combat, etc.)
plus additional menu/text input actions. Common actions include:
</p>
<ul>
    <li><strong>Movement (8 dirs)</strong>: y, k, u, h, l, b, j, n</li>
    <li><strong>Items</strong>: pickup, drop, eat, quaff, read, wear, wield, zap</li>
    <li><strong>Interaction</strong>: open, close, kick, search, apply</li>
    <li><strong>Combat</strong>: fire, throw, force-fight</li>
    <li><strong>Navigation</strong>: go up (<), go down (>), wait (.)</li>
</ul>

<h4>NLE Task Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Task</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Objective</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">NetHackScore</td><td style="border: 1px solid #ddd; padding: 8px;">Maximize in-game score</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">NetHackStaircase</td><td style="border: 1px solid #ddd; padding: 8px;">Reach stairs on level 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">NetHackStaircasePet</td><td style="border: 1px solid #ddd; padding: 8px;">Reach stairs with pet alive</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">NetHackOracle</td><td style="border: 1px solid #ddd; padding: 8px;">Find the Oracle (dungeon level 5-9)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">NetHackGold</td><td style="border: 1px solid #ddd; padding: 8px;">Collect gold</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">NetHackEat</td><td style="border: 1px solid #ddd; padding: 8px;">Eat food and survive</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">NetHackScout</td><td style="border: 1px solid #ddd; padding: 8px;">Explore dungeon levels</td></tr>
</table>

<h4>Dungeon Structure</h4>
<p>The dungeon contains 50+ levels across multiple branches:</p>
<ul>
    <li><strong>Main Dungeon</strong>: Levels 1-29, getting progressively harder</li>
    <li><strong>Gnomish Mines</strong>: Branch with Mine's End treasure</li>
    <li><strong>Sokoban</strong>: Puzzle levels with guaranteed prizes</li>
    <li><strong>Oracle Level</strong>: Information about the game</li>
    <li><strong>Quest</strong>: Role-specific quest for the Quest Artifact</li>
    <li><strong>Gehennom</strong>: Hell levels with the Amulet of Yendor</li>
    <li><strong>Elemental Planes</strong>: Final challenge before ascension</li>
</ul>

<h4>Character Roles</h4>
<p>NLE supports all 13 NetHack roles:</p>
<p style="font-family: monospace;">
Archeologist, Barbarian, Caveman, Healer, Knight, Monk, Priest, Ranger,
Rogue, Samurai, Tourist, Valkyrie, Wizard
</p>

<h4>Map Symbols</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Symbol</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Meaning</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">@</td><td style="border: 1px solid #ddd; padding: 8px;">Player (or human monster)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">.</td><td style="border: 1px solid #ddd; padding: 8px;">Floor</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">#</td><td style="border: 1px solid #ddd; padding: 8px;">Corridor</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">-</td><td style="border: 1px solid #ddd; padding: 8px;">Wall (horizontal)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">|</td><td style="border: 1px solid #ddd; padding: 8px;">Wall (vertical)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">+</td><td style="border: 1px solid #ddd; padding: 8px;">Closed door</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">></td><td style="border: 1px solid #ddd; padding: 8px;">Stairs down</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><</td><td style="border: 1px solid #ddd; padding: 8px;">Stairs up</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">$</td><td style="border: 1px solid #ddd; padding: 8px;">Gold</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">%</td><td style="border: 1px solid #ddd; padding: 8px;">Food</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">!</td><td style="border: 1px solid #ddd; padding: 8px;">Potion</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">?</td><td style="border: 1px solid #ddd; padding: 8px;">Scroll</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">/</td><td style="border: 1px solid #ddd; padding: 8px;">Wand</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">[</td><td style="border: 1px solid #ddd; padding: 8px;">Armor</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">)</td><td style="border: 1px solid #ddd; padding: 8px;">Weapon</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A-Z, a-z</td><td style="border: 1px solid #ddd; padding: 8px;">Monsters</td></tr>
</table>

<h4>Benchmark Results</h4>
<p>NetHack remains largely unsolved by RL agents:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Agent</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Score Task</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Oracle Task</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Human Expert</td><td style="border: 1px solid #ddd; padding: 8px;">~100,000+</td><td style="border: 1px solid #ddd; padding: 8px;">100%</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">IMPALA</td><td style="border: 1px solid #ddd; padding: 8px;">~700</td><td style="border: 1px solid #ddd; padding: 8px;">~0%</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">RND</td><td style="border: 1px solid #ddd; padding: 8px;">~800</td><td style="border: 1px solid #ddd; padding: 8px;">~0%</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Random</td><td style="border: 1px solid #ddd; padding: 8px;">~200</td><td style="border: 1px solid #ddd; padding: 8px;">0%</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/facebookresearch/nle" target="_blank">NLE GitHub Repository</a></li>
    <li><a href="https://nethackwiki.com" target="_blank">NetHack Wiki</a></li>
    <li>Paper: Küttler et al. (2020). The NetHack Learning Environment. NeurIPS 2020.</li>
</ul>
"""

__all__ = ["NETHACK_FULL_HTML"]
