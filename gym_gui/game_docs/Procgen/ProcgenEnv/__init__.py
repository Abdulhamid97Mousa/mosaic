"""Documentation for Procgen environments.

Procgen provides 16 procedurally-generated game-like environments designed
to measure sample efficiency and generalization in reinforcement learning.

Paper: "Leveraging Procedural Generation to Benchmark Reinforcement Learning"
       (Cobbe et al., 2019)
Repository: https://github.com/openai/procgen
"""

from __future__ import annotations

# Environment descriptions for each Procgen game
PROCGEN_ENV_DESCRIPTIONS = {
    "bigfish": {
        "name": "BigFish",
        "description": "Eat smaller fish to grow larger. Avoid larger fish. An aquatic food chain game.",
        "objective": "Eat fish smaller than you to grow; avoid being eaten by larger fish.",
        "strategy": "Start small, hunt appropriately sized prey, grow progressively.",
    },
    "bossfight": {
        "name": "BossFight",
        "description": "Defeat waves of boss starships by dodging projectiles and shooting back.",
        "objective": "Destroy the boss while dodging its attacks.",
        "strategy": "Learn boss attack patterns, position for safe shots.",
    },
    "caveflyer": {
        "name": "CaveFlyer",
        "description": "Navigate through cave systems while destroying targets and avoiding obstacles.",
        "objective": "Fly through caves, destroy targets, reach the goal.",
        "strategy": "Careful navigation, precision shooting.",
    },
    "chaser": {
        "name": "Chaser",
        "description": "MsPacman-inspired maze game. Eat all pellets while avoiding enemies.",
        "objective": "Collect all pellets while avoiding or eating enemies.",
        "strategy": "Plan escape routes, use power-ups strategically.",
    },
    "climber": {
        "name": "Climber",
        "description": "Platformer where you climb upward collecting stars and avoiding hazards.",
        "objective": "Climb as high as possible while collecting stars.",
        "strategy": "Time jumps carefully, plan ascent routes.",
    },
    "coinrun": {
        "name": "CoinRun",
        "description": "Classic platformer. Reach the coin at the end of procedural levels.",
        "objective": "Navigate obstacles to reach the coin at level end.",
        "strategy": "Master jumping mechanics, learn enemy patterns.",
    },
    "dodgeball": {
        "name": "Dodgeball",
        "description": "Avoid and throw balls in a multi-agent dodgeball arena.",
        "objective": "Hit opponents with balls while avoiding being hit.",
        "strategy": "Keep moving, time throws, anticipate opponents.",
    },
    "fruitbot": {
        "name": "FruitBot",
        "description": "Collect fruit while avoiding non-fruit items moving on conveyors.",
        "objective": "Collect only fruit items, avoid everything else.",
        "strategy": "Pattern recognition, quick decision making.",
    },
    "heist": {
        "name": "Heist",
        "description": "Collect keys to unlock doors and steal the gem in maze-like levels.",
        "objective": "Navigate the maze, collect keys, reach the gem.",
        "strategy": "Memorize key locations, plan efficient routes.",
    },
    "jumper": {
        "name": "Jumper",
        "description": "Open-world platformer with exploration and collection objectives.",
        "objective": "Explore and collect items across open platform levels.",
        "strategy": "Explore systematically, master movement.",
    },
    "leaper": {
        "name": "Leaper",
        "description": "Frogger-inspired game. Cross traffic and rivers to reach the goal.",
        "objective": "Cross hazardous lanes safely to reach the other side.",
        "strategy": "Time crossings, use safe zones.",
    },
    "maze": {
        "name": "Maze",
        "description": "Navigate procedurally-generated mazes to find the cheese.",
        "objective": "Find the path through the maze to the goal.",
        "strategy": "Systematic exploration, memory of visited areas.",
    },
    "miner": {
        "name": "Miner",
        "description": "BoulderDash-inspired digging game. Collect gems while avoiding falling rocks.",
        "objective": "Dig for gems while managing falling boulder physics.",
        "strategy": "Plan dig paths, avoid trapping yourself.",
    },
    "ninja": {
        "name": "Ninja",
        "description": "Platformer with bombs. Navigate levels using ninja abilities.",
        "objective": "Use ninja skills to navigate dangerous platform levels.",
        "strategy": "Master bomb timing, precise platforming.",
    },
    "plunder": {
        "name": "Plunder",
        "description": "Destroy enemy pirate ships in nautical combat.",
        "objective": "Sink enemy ships while protecting your own.",
        "strategy": "Aim carefully, manage positioning.",
    },
    "starpilot": {
        "name": "StarPilot",
        "description": "Side-scrolling space shooter. Destroy enemies and avoid obstacles.",
        "objective": "Shoot enemies while navigating space hazards.",
        "strategy": "Constant movement, pattern recognition.",
    },
}


def get_procgen_html(env_name: str) -> str:
    """Generate Procgen HTML documentation for a specific environment.

    Args:
        env_name: Environment name (e.g., "coinrun", "starpilot")

    Returns:
        HTML string containing environment documentation.
    """
    env_info = PROCGEN_ENV_DESCRIPTIONS.get(env_name, {
        "name": env_name.title(),
        "description": f"Procgen {env_name} environment.",
        "objective": "Complete the level objectives.",
        "strategy": "Learn from experience.",
    })

    return f"""
<h2>Procgen: {env_info["name"]}</h2>

<p>
{env_info["description"]}
</p>

<h4>Objective</h4>
<p>{env_info["objective"]}</p>

<h4>Strategy</h4>
<p>{env_info["strategy"]}</p>

<h4>Observation Space</h4>
<p>
RGB images of shape <code>(64, 64, 3)</code> with values in <code>[0, 255]</code>.
All Procgen environments provide the same observation format, making them ideal
for testing generalization across diverse visual domains.
</p>

<h4>Action Space</h4>
<p><code>Discrete(15)</code> - Button combinations:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Keys</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">down_left</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT+DOWN</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">left</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">up_left</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT+UP</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">down</td><td style="border: 1px solid #ddd; padding: 8px;">DOWN</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">noop</td><td style="border: 1px solid #ddd; padding: 8px;">(none)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">up</td><td style="border: 1px solid #ddd; padding: 8px;">UP</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">down_right</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT+DOWN</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">right</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">8</td><td style="border: 1px solid #ddd; padding: 8px;">up_right</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT+UP</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">9</td><td style="border: 1px solid #ddd; padding: 8px;">action_d</td><td style="border: 1px solid #ddd; padding: 8px;">D (fire/interact)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">10</td><td style="border: 1px solid #ddd; padding: 8px;">action_a</td><td style="border: 1px solid #ddd; padding: 8px;">A (secondary)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">11</td><td style="border: 1px solid #ddd; padding: 8px;">action_w</td><td style="border: 1px solid #ddd; padding: 8px;">W (tertiary)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">12</td><td style="border: 1px solid #ddd; padding: 8px;">action_s</td><td style="border: 1px solid #ddd; padding: 8px;">S (quaternary)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">13</td><td style="border: 1px solid #ddd; padding: 8px;">action_q</td><td style="border: 1px solid #ddd; padding: 8px;">Q (special 1)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">14</td><td style="border: 1px solid #ddd; padding: 8px;">action_e</td><td style="border: 1px solid #ddd; padding: 8px;">E (special 2)</td></tr>
</table>
<p><em>Note: Not all actions are meaningful in every environment. Each game uses a subset.</em></p>

<h4>Level Generation</h4>
<p>Key configuration parameters:</p>
<ul>
    <li><code>num_levels=0</code>: Unlimited procedural levels (tests generalization)</li>
    <li><code>num_levels=N</code>: Train on N fixed levels (tests memorization)</li>
    <li><code>start_level=S</code>: Start from seed S for reproducibility</li>
    <li><code>distribution_mode</code>: "easy", "hard", "extreme", or "memory"</li>
</ul>

<h4>Difficulty Modes</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Mode</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">easy</td><td style="border: 1px solid #ddd; padding: 8px;">Shorter episodes, simpler layouts</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">hard</td><td style="border: 1px solid #ddd; padding: 8px;">Standard difficulty (recommended)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">extreme</td><td style="border: 1px solid #ddd; padding: 8px;">Maximum difficulty settings</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">memory</td><td style="border: 1px solid #ddd; padding: 8px;">Tests memory with specific configurations</td></tr>
</table>

<h4>Benchmark Protocol</h4>
<p>Standard evaluation setup for generalization testing:</p>
<ul>
    <li><strong>Training</strong>: 200 fixed levels (<code>num_levels=200</code>)</li>
    <li><strong>Testing</strong>: Full procedural distribution (<code>num_levels=0</code>)</li>
    <li><strong>Budget</strong>: 25M environment steps (200M frames with frame skip)</li>
    <li><strong>Metric</strong>: Mean test episode return</li>
</ul>

<h4>Research Challenges</h4>
<ul>
    <li><strong>Generalization</strong>: Train-test distribution shift from level generation</li>
    <li><strong>Sample Efficiency</strong>: Learning with limited training levels</li>
    <li><strong>Procedural Diversity</strong>: Robust policies across level variations</li>
    <li><strong>Visual Learning</strong>: Consistent 64x64 RGB across diverse games</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/openai/procgen" target="_blank">GitHub Repository</a></li>
    <li><a href="https://openai.com/index/procgen-benchmark/" target="_blank">OpenAI Blog Post</a></li>
    <li>Paper: Cobbe et al. (2019). Leveraging Procedural Generation to Benchmark Reinforcement Learning.</li>
</ul>
"""


# Generate HTML for all 16 environments
PROCGEN_BIGFISH_HTML = get_procgen_html("bigfish")
PROCGEN_BOSSFIGHT_HTML = get_procgen_html("bossfight")
PROCGEN_CAVEFLYER_HTML = get_procgen_html("caveflyer")
PROCGEN_CHASER_HTML = get_procgen_html("chaser")
PROCGEN_CLIMBER_HTML = get_procgen_html("climber")
PROCGEN_COINRUN_HTML = get_procgen_html("coinrun")
PROCGEN_DODGEBALL_HTML = get_procgen_html("dodgeball")
PROCGEN_FRUITBOT_HTML = get_procgen_html("fruitbot")
PROCGEN_HEIST_HTML = get_procgen_html("heist")
PROCGEN_JUMPER_HTML = get_procgen_html("jumper")
PROCGEN_LEAPER_HTML = get_procgen_html("leaper")
PROCGEN_MAZE_HTML = get_procgen_html("maze")
PROCGEN_MINER_HTML = get_procgen_html("miner")
PROCGEN_NINJA_HTML = get_procgen_html("ninja")
PROCGEN_PLUNDER_HTML = get_procgen_html("plunder")
PROCGEN_STARPILOT_HTML = get_procgen_html("starpilot")

# Mapping from environment names to HTML
PROCGEN_HTML_MAP = {
    "bigfish": PROCGEN_BIGFISH_HTML,
    "bossfight": PROCGEN_BOSSFIGHT_HTML,
    "caveflyer": PROCGEN_CAVEFLYER_HTML,
    "chaser": PROCGEN_CHASER_HTML,
    "climber": PROCGEN_CLIMBER_HTML,
    "coinrun": PROCGEN_COINRUN_HTML,
    "dodgeball": PROCGEN_DODGEBALL_HTML,
    "fruitbot": PROCGEN_FRUITBOT_HTML,
    "heist": PROCGEN_HEIST_HTML,
    "jumper": PROCGEN_JUMPER_HTML,
    "leaper": PROCGEN_LEAPER_HTML,
    "maze": PROCGEN_MAZE_HTML,
    "miner": PROCGEN_MINER_HTML,
    "ninja": PROCGEN_NINJA_HTML,
    "plunder": PROCGEN_PLUNDER_HTML,
    "starpilot": PROCGEN_STARPILOT_HTML,
}

__all__ = [
    "get_procgen_html",
    "PROCGEN_ENV_DESCRIPTIONS",
    "PROCGEN_HTML_MAP",
    "PROCGEN_BIGFISH_HTML",
    "PROCGEN_BOSSFIGHT_HTML",
    "PROCGEN_CAVEFLYER_HTML",
    "PROCGEN_CHASER_HTML",
    "PROCGEN_CLIMBER_HTML",
    "PROCGEN_COINRUN_HTML",
    "PROCGEN_DODGEBALL_HTML",
    "PROCGEN_FRUITBOT_HTML",
    "PROCGEN_HEIST_HTML",
    "PROCGEN_JUMPER_HTML",
    "PROCGEN_LEAPER_HTML",
    "PROCGEN_MAZE_HTML",
    "PROCGEN_MINER_HTML",
    "PROCGEN_NINJA_HTML",
    "PROCGEN_PLUNDER_HTML",
    "PROCGEN_STARPILOT_HTML",
]
