"""Documentation for Jumanji Tetris environment.

Tetris is the classic falling block puzzle game where pieces must be
placed to complete horizontal lines.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_tetris_html(env_id: str = "Tetris-v0") -> str:
    """Generate Tetris HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Tetris is the iconic puzzle video game where players manipulate falling
tetrominoes to create complete horizontal lines. When a line is completed,
it disappears, and pieces above fall down.
</p>

<h4>Game Rules</h4>
<ul>
    <li>Tetrominoes (7 shapes) fall from the top of the board</li>
    <li>Player can rotate and move pieces horizontally</li>
    <li>Complete horizontal lines are cleared</li>
    <li>Game ends when pieces stack to the top</li>
</ul>

<h4>Tetrominoes</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Piece</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Shape</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Color</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">I</td><td style="border: 1px solid #ddd; padding: 8px;">Four in a row</td><td style="border: 1px solid #ddd; padding: 8px;">Cyan</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">O</td><td style="border: 1px solid #ddd; padding: 8px;">2x2 Square</td><td style="border: 1px solid #ddd; padding: 8px;">Yellow</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">T</td><td style="border: 1px solid #ddd; padding: 8px;">T-shape</td><td style="border: 1px solid #ddd; padding: 8px;">Purple</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">S</td><td style="border: 1px solid #ddd; padding: 8px;">S-shape (zigzag)</td><td style="border: 1px solid #ddd; padding: 8px;">Green</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Z</td><td style="border: 1px solid #ddd; padding: 8px;">Z-shape (zigzag)</td><td style="border: 1px solid #ddd; padding: 8px;">Red</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">J</td><td style="border: 1px solid #ddd; padding: 8px;">J-shape</td><td style="border: 1px solid #ddd; padding: 8px;">Blue</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">L</td><td style="border: 1px solid #ddd; padding: 8px;">L-shape</td><td style="border: 1px solid #ddd; padding: 8px;">Orange</td></tr>
</table>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>board</strong>: 2D grid showing placed blocks</li>
    <li><strong>current_piece</strong>: The falling tetromino</li>
    <li><strong>next_piece</strong>: Preview of next tetromino</li>
    <li><strong>position</strong>: Current piece position</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete</code> - Control the falling piece:</p>
<ul>
    <li><strong>Left/Right</strong>: Move horizontally</li>
    <li><strong>Rotate CW/CCW</strong>: Rotate piece</li>
    <li><strong>Soft Drop</strong>: Move down faster</li>
    <li><strong>Hard Drop</strong>: Instantly place piece</li>
</ul>

<h4>Scoring</h4>
<ul>
    <li><strong>Single</strong>: 1 line cleared</li>
    <li><strong>Double</strong>: 2 lines cleared</li>
    <li><strong>Triple</strong>: 3 lines cleared</li>
    <li><strong>Tetris</strong>: 4 lines cleared (bonus!)</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: Blocks reach the top (game over)</li>
</ul>

<h4>Strategies</h4>
<ul>
    <li><strong>Keep flat</strong>: Avoid creating holes</li>
    <li><strong>Reserve I-piece slot</strong>: Keep column open for Tetris</li>
    <li><strong>T-spin</strong>: Advanced rotation technique</li>
    <li><strong>Combo</strong>: Clear lines consecutively</li>
</ul>

<h4>Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Left/Right Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move piece</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Up Arrow / Z</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Down Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Soft drop</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space</td><td style="border: 1px solid #ddd; padding: 8px;">Hard drop</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://tetris.com/" target="_blank">Official Tetris</a></li>
</ul>
"""


TETRIS_HTML = get_tetris_html()

__all__ = [
    "get_tetris_html",
    "TETRIS_HTML",
]
