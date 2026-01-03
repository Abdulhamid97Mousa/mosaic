"""Documentation for Jumanji Minesweeper environment.

Minesweeper is a single-player puzzle game where the player must reveal
all safe cells without detonating any mines.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_minesweeper_html(env_id: str = "Minesweeper-v0") -> str:
    """Generate Minesweeper HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Minesweeper is a classic single-player puzzle game. The goal is to reveal all safe
cells on a grid without clicking on any hidden mines. Numbers indicate how many
adjacent cells contain mines.
</p>

<h4>Game Rules</h4>
<ul>
    <li>The grid contains hidden mines and safe cells</li>
    <li>Click on a cell to reveal it</li>
    <li>If a mine is revealed, the game ends</li>
    <li>If a safe cell is revealed, it shows the count of adjacent mines (0-8)</li>
    <li>Cells with 0 adjacent mines automatically reveal neighboring cells</li>
    <li>Win by revealing all safe cells</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>board</strong>: <code>Box(shape=(H, W), dtype=int32)</code> - The game grid (-1 = hidden, 0-8 = revealed with mine count, 9 = mine)</li>
    <li><strong>action_mask</strong>: <code>Box(shape=(H*W,), dtype=bool)</code> - Valid actions (unrevealed cells)</li>
    <li><strong>num_mines</strong>: <code>int</code> - Total number of mines in the grid</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(H * W)</code> - Each action corresponds to revealing a cell:</p>
<ul>
    <li>Action index = row * width + column</li>
    <li>Only unrevealed cells are valid actions</li>
</ul>

<h4>Rewards</h4>
<ul>
    <li><strong>+1</strong>: Successfully revealing a safe cell</li>
    <li><strong>-1</strong>: Revealing a mine (game over)</li>
    <li><strong>+10</strong>: Winning by revealing all safe cells</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination (Loss)</strong>: A mine is revealed</li>
    <li><strong>Termination (Win)</strong>: All safe cells are revealed</li>
</ul>

<h4>Grid Configurations</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Difficulty</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Grid Size</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Mines</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Beginner</td><td style="border: 1px solid #ddd; padding: 8px;">8x8</td><td style="border: 1px solid #ddd; padding: 8px;">10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Intermediate</td><td style="border: 1px solid #ddd; padding: 8px;">16x16</td><td style="border: 1px solid #ddd; padding: 8px;">40</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Expert</td><td style="border: 1px solid #ddd; padding: 8px;">16x30</td><td style="border: 1px solid #ddd; padding: 8px;">99</td></tr>
</table>

<h4>Controls</h4>
<p>
Minesweeper uses a grid-based action space that is best suited for mouse/touch
interaction. Each cell can be selected by its index.
</p>

<h4>Strategies</h4>
<ul>
    <li><strong>Corner Start</strong>: Start from corners to maximize information gain</li>
    <li><strong>Number Logic</strong>: Use revealed numbers to deduce safe/mine cells</li>
    <li><strong>Pattern Recognition</strong>: Learn common mine patterns (1-2-1, 1-2-2-1)</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://minesweeper.online/" target="_blank">Online Minesweeper</a></li>
</ul>
"""


MINESWEEPER_HTML = get_minesweeper_html()

__all__ = [
    "get_minesweeper_html",
    "MINESWEEPER_HTML",
]
