"""TextWorld game documentation module.

TextWorld is a Microsoft Research sandbox for training RL agents on text-based games.
It generates and simulates text-based adventure games for training agents on
language understanding and sequential decision making.

Repository: https://github.com/microsoft/TextWorld
Project: https://www.microsoft.com/en-us/research/project/textworld/

Key Features:
- Generates procedural text-based adventure games
- Trains agents for language understanding + sequential decision making
- Supports Python 3.9-3.12 on Linux/macOS only
- Provides both game generation and playing APIs
- Includes various difficulty levels and game types

Installation:
    pip install textworld
    # or
    pip install -r requirements/textworld.txt
"""
from __future__ import annotations


def get_textworld_html(env_id: str = "TextWorld") -> str:
    """Generate TextWorld HTML documentation for a specific variant."""
    return f"""
<h2>{env_id}</h2>
<p>TextWorld is a sandbox for training reinforcement learning agents on text-based games.
It generates and simulates text-based adventure games for research in language understanding
and sequential decision making.</p>

<h4>Key Features</h4>
<ul>
    <li><strong>Procedural Game Generation:</strong> Generate unlimited unique text-based games</li>
    <li><strong>Language Understanding:</strong> Train agents to interpret natural language descriptions</li>
    <li><strong>Sequential Decision Making:</strong> Complex multi-step reasoning required</li>
    <li><strong>Customizable Difficulty:</strong> Control game complexity and length</li>
    <li><strong>Gymnasium Integration:</strong> Standard RL environment interface</li>
</ul>

<h4>Observation Space</h4>
<p>Text-based observations containing:</p>
<ul>
    <li><code>description</code>: Current room/location description</li>
    <li><code>inventory</code>: List of items the agent is carrying</li>
    <li><code>feedback</code>: Result of the last action taken</li>
    <li><code>admissible_commands</code>: List of valid commands (optional)</li>
</ul>

<h4>Action Space</h4>
<p>Text commands as strings. Examples:</p>
<ul>
    <li><code>go north</code>, <code>go south</code>, <code>go east</code>, <code>go west</code></li>
    <li><code>take [object]</code>, <code>drop [object]</code></li>
    <li><code>open [container]</code>, <code>close [container]</code></li>
    <li><code>examine [object]</code>, <code>look</code></li>
    <li><code>inventory</code></li>
</ul>

<h4>Rewards</h4>
<ul>
    <li>Task completion rewards based on game objectives</li>
    <li>Intermediate rewards for progress (optional)</li>
    <li>Configurable reward shaping</li>
</ul>

<h4>Game Types</h4>
<ul>
    <li><strong>Treasure Hunter:</strong> Find and collect specific items</li>
    <li><strong>Coin Collector:</strong> Collect coins scattered across rooms</li>
    <li><strong>Cooking:</strong> Follow recipes to prepare meals</li>
    <li><strong>Custom:</strong> Generate games with specific properties</li>
</ul>

<h4>System Requirements</h4>
<ul>
    <li>Python 3.9-3.12</li>
    <li>Linux or macOS only (Windows not supported)</li>
</ul>

<h4>Documentation</h4>
<ul>
    <li><a href="https://github.com/microsoft/TextWorld">GitHub Repository</a></li>
    <li><a href="https://www.microsoft.com/en-us/research/project/textworld/">Microsoft Research Project Page</a></li>
    <li><a href="https://textworld.readthedocs.io/">TextWorld Documentation</a></li>
</ul>
"""


# Default HTML for backward compatibility
TEXTWORLD_HTML = get_textworld_html()

# Specific game type documentation
TEXTWORLD_TREASURE_HUNTER_HTML = get_textworld_html("TextWorld-TreasureHunter")
TEXTWORLD_COIN_COLLECTOR_HTML = get_textworld_html("TextWorld-CoinCollector")
TEXTWORLD_COOKING_HTML = get_textworld_html("TextWorld-Cooking")
TEXTWORLD_CUSTOM_HTML = get_textworld_html("TextWorld-Custom")

__all__ = [
    "TEXTWORLD_HTML",
    "TEXTWORLD_TREASURE_HUNTER_HTML",
    "TEXTWORLD_COIN_COLLECTOR_HTML",
    "TEXTWORLD_COOKING_HTML",
    "TEXTWORLD_CUSTOM_HTML",
    "get_textworld_html",
]
