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
    <li><strong>grid</strong>: <code>Box(shape=(num_rows, num_cols), dtype=int32)</code> - Binary grid of placed blocks (0=empty, 1=filled)</li>
    <li><strong>tetromino</strong>: <code>Box(shape=(4, 4), dtype=int32)</code> - Current piece shape</li>
    <li><strong>action_mask</strong>: <code>Box(shape=(4, num_cols), dtype=bool)</code> - Valid [rotation, column] placements</li>
    <li><strong>step_count</strong>: Scalar step counter</li>
</ul>

<h4>Action Space</h4>
<p><code>MultiDiscrete([4, num_cols])</code> - Each action is a placement decision:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Dimension</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Range</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Meaning</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0: Rotation</td><td style="border: 1px solid #ddd; padding: 8px;">0-3</td><td style="border: 1px solid #ddd; padding: 8px;">0=0deg, 1=90deg, 2=180deg, 3=270deg</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1: Column</td><td style="border: 1px solid #ddd; padding: 8px;">0 to num_cols-1</td><td style="border: 1px solid #ddd; padding: 8px;">Horizontal position for piece placement</td></tr>
</table>
<p><em>Note: Unlike classic Tetris, Jumanji Tetris places pieces instantly
rather than dropping them in real-time. You choose rotation + column,
then the piece is placed immediately.</em></p>

<h4>Scoring</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Lines Cleared</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Points</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1 (Single)</td><td style="border: 1px solid #ddd; padding: 8px;">40</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2 (Double)</td><td style="border: 1px solid #ddd; padding: 8px;">100</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3 (Triple)</td><td style="border: 1px solid #ddd; padding: 8px;">300</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4 (Tetris)</td><td style="border: 1px solid #ddd; padding: 8px;">1200</td></tr>
</table>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: No valid placements remain (action_mask is all False)</li>
</ul>

<h4>Strategies</h4>
<ul>
    <li><strong>Keep flat</strong>: Avoid creating holes</li>
    <li><strong>Reserve I-piece slot</strong>: Keep column open for Tetris</li>
    <li><strong>T-spin</strong>: Advanced rotation technique</li>
    <li><strong>Combo</strong>: Clear lines consecutively</li>
</ul>

<h4>Controls</h4>
<p>Use cursor keys to adjust rotation and column, then press Space to place:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A / Left Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move cursor left (column - 1)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D / Right Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Move cursor right (column + 1)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W / Up Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate clockwise (+90deg)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">S / Down Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate counter-clockwise (-90deg)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space / Enter</td><td style="border: 1px solid #ddd; padding: 8px;">Place piece at current rotation + column</td></tr>
</table>
<p><em>The status bar shows your current cursor position. After placing, the
cursor resets to column center with 0-degree rotation.</em></p>

<h4>Mouse Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Input</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Left Click on column</td><td style="border: 1px solid #ddd; padding: 8px;">Place piece at that column (with current rotation)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scroll Up</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate piece clockwise (+90deg)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scroll Down</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate piece counter-clockwise (-90deg)</td></tr>
</table>
<p><em>Scroll to choose rotation, then click the target column. Both keyboard
and mouse input work simultaneously.</em></p>

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
