"""Documentation for Jumanji Game2048 environment.

2048 is a single-player sliding block puzzle game where players combine
numbered tiles to reach the 2048 tile.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_game2048_html(env_id: str = "Game2048-v1") -> str:
    """Generate Game2048 HTML documentation.

    Args:
        env_id: Environment identifier (e.g., "Game2048-v1")

    Returns:
        HTML string containing environment documentation.
    """
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
2048 is a single-player sliding block puzzle game. The objective is to slide
numbered tiles on a 4x4 grid to combine them and create a tile with the number 2048.
</p>

<h4>Game Rules</h4>
<ul>
    <li>The game starts with two tiles (usually 2s or 4s) placed randomly</li>
    <li>Slide all tiles in one direction (up, down, left, right)</li>
    <li>When two tiles with the same number collide, they merge into one with double the value</li>
    <li>After each move, a new tile (2 or 4) appears in a random empty cell</li>
    <li>The game ends when no valid moves remain (grid full, no adjacent matching tiles)</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>board</strong>: <code>Box(0, 2^17, shape=(4, 4), dtype=int32)</code> - The 4x4 game grid with tile values (0 = empty, 2, 4, 8, ..., 131072)</li>
    <li><strong>action_mask</strong>: <code>Box(0, 1, shape=(4,), dtype=bool)</code> - Valid actions mask</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(4)</code> - Four cardinal directions:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">Up</td><td style="border: 1px solid #ddd; padding: 8px;">W / Up Arrow</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">Right</td><td style="border: 1px solid #ddd; padding: 8px;">D / Right Arrow</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">Down</td><td style="border: 1px solid #ddd; padding: 8px;">S / Down Arrow</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">Left</td><td style="border: 1px solid #ddd; padding: 8px;">A / Left Arrow</td></tr>
</table>

<h4>Rewards</h4>
<p>The reward is the sum of merged tile values in each step:</p>
<ul>
    <li>Merging two 2s gives reward = 4</li>
    <li>Merging two 4s gives reward = 8</li>
    <li>Merging two 1024s gives reward = 2048</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: No valid moves remaining (board full with no adjacent matching tiles)</li>
    <li><strong>Truncation</strong>: Optional step limit reached</li>
</ul>

<h4>Strategies</h4>
<ul>
    <li><strong>Corner Strategy</strong>: Keep the highest tile in a corner</li>
    <li><strong>Snake Pattern</strong>: Arrange tiles in a snake-like pattern</li>
    <li><strong>Monotonic Rows/Columns</strong>: Keep values increasing/decreasing along rows/columns</li>
</ul>

<h4>Keyboard Controls (Human Play)</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W / Up Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Slide Up</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">S / Down Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Slide Down</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A / Left Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Slide Left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D / Right Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Slide Right</td></tr>
</table>

<h4>JAX Features</h4>
<ul>
    <li><strong>JIT Compilation</strong>: Environment step is JIT-compiled for fast execution</li>
    <li><strong>Vectorization</strong>: Can run multiple environments in parallel with vmap</li>
    <li><strong>Hardware Acceleration</strong>: Runs on CPU, GPU, or TPU</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://play2048.co/" target="_blank">Original 2048 Game</a></li>
</ul>
"""


GAME2048_HTML = get_game2048_html()

__all__ = [
    "get_game2048_html",
    "GAME2048_HTML",
]
