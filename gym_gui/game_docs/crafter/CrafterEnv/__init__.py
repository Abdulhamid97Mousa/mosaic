"""Documentation for Crafter environments.

Crafter is an open world survival game benchmark that evaluates a wide range
of agent capabilities within a single environment. Agents learn from visual
inputs and aim to unlock 22 semantically meaningful achievements.

Paper: "Benchmarking the Spectrum of Agent Capabilities" (Hafner, 2022, ICLR 2022)
Repository: https://github.com/danijar/crafter
"""

from __future__ import annotations


def get_crafter_html(env_id: str) -> str:
    """Generate Crafter HTML documentation for a specific variant.

    Args:
        env_id: Environment identifier (e.g., "CrafterReward-v1")

    Returns:
        HTML string containing environment documentation.
    """
    reward_enabled = "CrafterReward" in env_id  # Check for CrafterReward, not just Reward

    return f"""
<h2>{env_id}</h2>

<p>
Crafter is an open-world survival game benchmark for reinforcement learning research.
It features procedurally generated 2D worlds with forests, lakes, mountains, and caves.
The player must forage for food and water, find shelter to sleep, defend against monsters,
collect materials, and build tools.
</p>

<h4>Environment Variants</h4>
<ul>
    <li><code>CrafterReward-v1</code> - With reward signals (+1 per achievement, health-based rewards)</li>
    <li><code>CrafterNoReward-v1</code> - Reward-free variant for unsupervised learning</li>
</ul>

<h4>Observation Space</h4>
<p>
RGB images of shape <code>(64, 64, 3)</code> with values in <code>[0, 255]</code>.
The image shows:
</p>
<ul>
    <li><strong>Top portion (7x9 grid)</strong>: Local top-down view around the player</li>
    <li><strong>Bottom portion (2x9 grid)</strong>: Inventory display showing health, food, water, rest, materials, and tools</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(17)</code> - Flat categorical action space:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Requirement</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">noop</td><td style="border: 1px solid #ddd; padding: 8px;">Always</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">move_left</td><td style="border: 1px solid #ddd; padding: 8px;">Flat ground left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">move_right</td><td style="border: 1px solid #ddd; padding: 8px;">Flat ground right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">move_up</td><td style="border: 1px solid #ddd; padding: 8px;">Flat ground above</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">move_down</td><td style="border: 1px solid #ddd; padding: 8px;">Flat ground below</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">do</td><td style="border: 1px solid #ddd; padding: 8px;">Facing creature/material</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">sleep</td><td style="border: 1px solid #ddd; padding: 8px;">Energy below max</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">place_stone</td><td style="border: 1px solid #ddd; padding: 8px;">Stone in inventory</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">8</td><td style="border: 1px solid #ddd; padding: 8px;">place_table</td><td style="border: 1px solid #ddd; padding: 8px;">Wood in inventory</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">9</td><td style="border: 1px solid #ddd; padding: 8px;">place_furnace</td><td style="border: 1px solid #ddd; padding: 8px;">Stone in inventory</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">10</td><td style="border: 1px solid #ddd; padding: 8px;">place_plant</td><td style="border: 1px solid #ddd; padding: 8px;">Sapling in inventory</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">11</td><td style="border: 1px solid #ddd; padding: 8px;">make_wood_pickaxe</td><td style="border: 1px solid #ddd; padding: 8px;">Near table; wood</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">12</td><td style="border: 1px solid #ddd; padding: 8px;">make_stone_pickaxe</td><td style="border: 1px solid #ddd; padding: 8px;">Near table; wood, stone</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">13</td><td style="border: 1px solid #ddd; padding: 8px;">make_iron_pickaxe</td><td style="border: 1px solid #ddd; padding: 8px;">Near table+furnace; wood, coal, iron</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">14</td><td style="border: 1px solid #ddd; padding: 8px;">make_wood_sword</td><td style="border: 1px solid #ddd; padding: 8px;">Near table; wood</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">15</td><td style="border: 1px solid #ddd; padding: 8px;">make_stone_sword</td><td style="border: 1px solid #ddd; padding: 8px;">Near table; wood, stone</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">16</td><td style="border: 1px solid #ddd; padding: 8px;">make_iron_sword</td><td style="border: 1px solid #ddd; padding: 8px;">Near table+furnace; wood, coal, iron</td></tr>
</table>

<h4>Rewards{"" if reward_enabled else " (Disabled)"}</h4>
{"<p>Reward signal consists of two components:</p>" if reward_enabled else "<p>This variant has rewards disabled for unsupervised learning.</p>"}
{"<ul><li><strong>+1.0</strong>: Unlocking an achievement for the first time in the episode</li><li><strong>-0.1</strong>: Per health point lost</li><li><strong>+0.1</strong>: Per health point regenerated</li></ul><p>Maximum episode reward: 22 (all achievements unlocked)</p>" if reward_enabled else ""}

<h4>22 Achievements</h4>
<p>Agents are evaluated by success rates on semantically meaningful achievements:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Category</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Achievements</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Collect</td><td style="border: 1px solid #ddd; padding: 8px;">coal, diamond, drink, iron, sapling, stone, wood</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Defeat</td><td style="border: 1px solid #ddd; padding: 8px;">skeleton, zombie</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Eat</td><td style="border: 1px solid #ddd; padding: 8px;">cow, plant</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Make Pickaxes</td><td style="border: 1px solid #ddd; padding: 8px;">wood_pickaxe, stone_pickaxe, iron_pickaxe</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Make Swords</td><td style="border: 1px solid #ddd; padding: 8px;">wood_sword, stone_sword, iron_sword</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Place</td><td style="border: 1px solid #ddd; padding: 8px;">furnace, plant, stone, table</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Other</td><td style="border: 1px solid #ddd; padding: 8px;">wake_up</td></tr>
</table>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: Player health reaches 0 (hunger, thirst, tiredness, monster attack, lava)</li>
    <li><strong>Truncation</strong>: Episode reaches 10,000 steps</li>
</ul>

<h4>World Generation</h4>
<p>
Each episode generates a unique 64x64 world using OpenSimplex noise:
</p>
<ul>
    <li><strong>Terrain</strong>: Grasslands, lakes (with shores), mountains (with caves)</li>
    <li><strong>Resources</strong>: Trees, coal, iron, diamonds in appropriate biomes</li>
    <li><strong>Creatures</strong>: Cows and zombies in grasslands, skeletons in caves</li>
    <li><strong>Day/Night Cycle</strong>: Restricted view at night, more zombies spawn</li>
</ul>

<h4>Keyboard Controls (Human Play)</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W/Up, A/Left, S/Down, D/Right</td><td style="border: 1px solid #ddd; padding: 8px;">Movement</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space</td><td style="border: 1px solid #ddd; padding: 8px;">Interact (do)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">R</td><td style="border: 1px solid #ddd; padding: 8px;">Sleep</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1-4</td><td style="border: 1px solid #ddd; padding: 8px;">Place (stone/table/furnace/plant)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q, E, F</td><td style="border: 1px solid #ddd; padding: 8px;">Make pickaxes (wood/stone/iron)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Z, X, C</td><td style="border: 1px solid #ddd; padding: 8px;">Make swords (wood/stone/iron)</td></tr>
</table>

<h4>Benchmark Scores</h4>
<p>Crafter score = geometric mean of success rates for all 22 achievements (1M step budget):</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Method</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Score (%)</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Human Experts</td><td style="border: 1px solid #ddd; padding: 8px;">50.5</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">DreamerV2</td><td style="border: 1px solid #ddd; padding: 8px;">10.0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">PPO</td><td style="border: 1px solid #ddd; padding: 8px;">4.6</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rainbow</td><td style="border: 1px solid #ddd; padding: 8px;">4.3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Random</td><td style="border: 1px solid #ddd; padding: 8px;">1.6</td></tr>
</table>

<h4>Research Challenges</h4>
<ul>
    <li><strong>Exploration</strong>: Wide and deep exploration via technology tree</li>
    <li><strong>Generalization</strong>: Procedural generation forces robust behaviors</li>
    <li><strong>Credit Assignment</strong>: Sparse rewards, long-term dependencies</li>
    <li><strong>Memory</strong>: Partial observability requires remembering map locations</li>
    <li><strong>Representation</strong>: Learning from high-dimensional images</li>
    <li><strong>Survival</strong>: Constant pressure to maintain health, food, water, rest</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/danijar/crafter" target="_blank">GitHub Repository</a></li>
    <li><a href="https://danijar.com/crafter" target="_blank">Project Website</a></li>
    <li>Paper: Hafner, D. (2022). Benchmarking the Spectrum of Agent Capabilities. ICLR 2022.</li>
</ul>
"""


# Backward compatibility constant
CRAFTER_HTML = get_crafter_html("CrafterReward-v1")
CRAFTER_REWARD_HTML = get_crafter_html("CrafterReward-v1")
CRAFTER_NO_REWARD_HTML = get_crafter_html("CrafterNoReward-v1")

__all__ = [
    "get_crafter_html",
    "CRAFTER_HTML",
    "CRAFTER_REWARD_HTML",
    "CRAFTER_NO_REWARD_HTML",
]
