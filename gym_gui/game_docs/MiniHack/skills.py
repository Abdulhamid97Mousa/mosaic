"""Documentation for MiniHack skill acquisition environments."""

from .controls import MINIHACK_CONTROLS_HTML

MINIHACK_SKILLS_SIMPLE_HTML = """
<h3>MiniHack - Simple Skills</h3>
<p>
  Learn basic NetHack skills like picking up items, wielding weapons, wearing armor,
  eating food, and using simple objects. These environments isolate individual
  game mechanics for focused learning.
</p>

<h4>Objective</h4>
<p>Complete the specified skill task (varies by environment).</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Skill</th><th>Description</th></tr>
  <tr><td><code>MiniHack-Eat-v0</code></td><td>Eating</td><td>Find and eat food to survive</td></tr>
  <tr><td><code>MiniHack-Pray-v0</code></td><td>Praying</td><td>Pray to the gods for help</td></tr>
  <tr><td><code>MiniHack-Sink-v0</code></td><td>Sink interaction</td><td>Interact with a sink</td></tr>
  <tr><td><code>MiniHack-Wear-v0</code></td><td>Wearing armor</td><td>Pick up and wear armor</td></tr>
  <tr><td><code>MiniHack-Wield-v0</code></td><td>Wielding weapons</td><td>Pick up and wield a weapon</td></tr>
  <tr><td><code>MiniHack-Zap-v0</code></td><td>Zapping wands</td><td>Use a wand on a target</td></tr>
  <tr><td><code>MiniHack-Read-v0</code></td><td>Reading scrolls</td><td>Read a scroll</td></tr>
  <tr><td><code>MiniHack-Quaff-v0</code></td><td>Drinking potions</td><td>Quaff a potion</td></tr>
  <tr><td><code>MiniHack-PutOn-v0</code></td><td>Wearing accessories</td><td>Put on a ring or amulet</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(~23)</code> - Movement + relevant skill actions</p>

<h4>Key Commands</h4>
<ul>
  <li><b>,</b> - Pick up items</li>
  <li><b>e</b> - Eat food</li>
  <li><b>w</b> - Wield weapon</li>
  <li><b>W</b> - Wear armor</li>
  <li><b>P</b> - Put on ring/amulet</li>
  <li><b>r</b> - Read scroll</li>
  <li><b>q</b> - Quaff potion</li>
  <li><b>z</b> - Zap wand</li>
</ul>
""" + MINIHACK_CONTROLS_HTML


MINIHACK_SKILLS_LAVA_HTML = """
<h3>MiniHack - Lava Crossing</h3>
<p>
  Cross a lava field using levitation items. The agent must find and use
  the correct items (wand of fire, ring of levitation, boots of levitation)
  to safely cross the deadly lava.
</p>

<h4>Objective</h4>
<p>Cross the lava field to reach the goal without dying.</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-LavaCross-Levitate-Ring-v0</code></td><td>Use ring of levitation</td></tr>
  <tr><td><code>MiniHack-LavaCross-Levitate-Potion-v0</code></td><td>Use potion of levitation</td></tr>
  <tr><td><code>MiniHack-LavaCross-Levitate-v0</code></td><td>Any levitation method</td></tr>
  <tr><td><code>MiniHack-LavaCross-v0</code></td><td>General lava crossing</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(~23)</code> - Movement + item interactions</p>

<h4>Key Skills</h4>
<ul>
  <li>Identifying useful items</li>
  <li>Using levitation to fly over lava</li>
  <li>Timing - levitation wears off!</li>
</ul>

<h4>Danger</h4>
<p><strong>Warning:</strong> Stepping into lava <code>}</code> without levitation is instant death!</p>
""" + MINIHACK_CONTROLS_HTML


MINIHACK_SKILLS_WOD_HTML = """
<h3>MiniHack - Wand of Death</h3>
<p>
  Use the Wand of Death to defeat monsters blocking your path. This teaches
  the agent to use powerful magical items strategically. The wand must be
  aimed correctly to hit the target.
</p>

<h4>Objective</h4>
<p>Defeat the monster using the Wand of Death and reach the goal.</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-WoD-Easy-v0</code></td><td>Single monster, clear line of sight</td></tr>
  <tr><td><code>MiniHack-WoD-Medium-v0</code></td><td>Monster in corridor</td></tr>
  <tr><td><code>MiniHack-WoD-Hard-v0</code></td><td>Multiple monsters or obstacles</td></tr>
  <tr><td><code>MiniHack-WoD-Pro-v0</code></td><td>Complex layout requiring planning</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(~75)</code> - Full action set for combat</p>

<h4>Key Skills</h4>
<ul>
  <li>Finding and picking up the wand</li>
  <li>Aiming the wand (<b>z</b> then direction)</li>
  <li>Understanding line-of-sight mechanics</li>
</ul>

<h4>Combat Commands</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>z</b></td><td>Zap wand (then choose direction)</td></tr>
  <tr><td><b>f</b></td><td>Fire from quiver</td></tr>
  <tr><td><b>t</b></td><td>Throw item</td></tr>
</table>
""" + MINIHACK_CONTROLS_HTML


MINIHACK_SKILLS_QUEST_HTML = """
<h3>MiniHack - Quest</h3>
<p>
  Multi-step quests requiring combinations of skills. The agent must navigate,
  find items, defeat monsters, and solve puzzles to complete the quest.
  These are the most challenging MiniHack environments.
</p>

<h4>Objective</h4>
<p>Complete all quest objectives and reach the final goal.</p>

<h4>Variants</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Environment</th><th>Description</th></tr>
  <tr><td><code>MiniHack-Quest-Easy-v0</code></td><td>Simple multi-room quest</td></tr>
  <tr><td><code>MiniHack-Quest-Medium-v0</code></td><td>Quest with combat and items</td></tr>
  <tr><td><code>MiniHack-Quest-Hard-v0</code></td><td>Complex multi-objective quest</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(~75)</code> - Full NetHack action set</p>

<h4>Required Skills</h4>
<ul>
  <li>Navigation and exploration</li>
  <li>Combat (melee and ranged)</li>
  <li>Item identification and usage</li>
  <li>Resource management (HP, items)</li>
  <li>Multi-step planning</li>
</ul>

<h4>Tips</h4>
<ul>
  <li>Explore carefully - don't rush into danger</li>
  <li>Pick up useful items along the way</li>
  <li>Use ranged attacks when possible</li>
  <li>Retreat and heal if low on health</li>
</ul>
""" + MINIHACK_CONTROLS_HTML


__all__ = [
    "MINIHACK_SKILLS_SIMPLE_HTML",
    "MINIHACK_SKILLS_LAVA_HTML",
    "MINIHACK_SKILLS_WOD_HTML",
    "MINIHACK_SKILLS_QUEST_HTML",
]
