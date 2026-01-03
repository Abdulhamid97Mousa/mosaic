"""Documentation for Jumanji RubiksCube environment.

Rubik's Cube is a 3D combination puzzle where the goal is to return
a scrambled cube to a state where each face has a single color.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_rubiks_cube_html(env_id: str = "RubiksCube-v0") -> str:
    """Generate RubiksCube HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
The Rubik's Cube is a 3D combination puzzle invented in 1974 by Erno Rubik.
The goal is to return a scrambled 3x3x3 cube to a state where each of the
six faces shows a single solid color.
</p>

<h4>Cube Structure</h4>
<ul>
    <li><strong>6 Faces</strong>: Front (F), Back (B), Up (U), Down (D), Left (L), Right (R)</li>
    <li><strong>54 Stickers</strong>: 9 stickers per face (3x3)</li>
    <li><strong>6 Colors</strong>: White, Yellow, Red, Orange, Blue, Green</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>cube</strong>: <code>Box(shape=(6, 3, 3), dtype=int32)</code> - The cube state (6 faces, each 3x3, values 0-5 for colors)</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(12)</code> - Face rotations (6 faces x 2 directions):</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Move</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">R</td><td style="border: 1px solid #ddd; padding: 8px;">Right face clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">R</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">R'</td><td style="border: 1px solid #ddd; padding: 8px;">Right face counter-clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">T</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">L</td><td style="border: 1px solid #ddd; padding: 8px;">Left face clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">L</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">L'</td><td style="border: 1px solid #ddd; padding: 8px;">Left face counter-clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">K</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">U</td><td style="border: 1px solid #ddd; padding: 8px;">Up face clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">U</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">U'</td><td style="border: 1px solid #ddd; padding: 8px;">Up face counter-clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">Y</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">D</td><td style="border: 1px solid #ddd; padding: 8px;">Down face clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">D</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">D'</td><td style="border: 1px solid #ddd; padding: 8px;">Down face counter-clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">E</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">8</td><td style="border: 1px solid #ddd; padding: 8px;">F</td><td style="border: 1px solid #ddd; padding: 8px;">Front face clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">F</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">9</td><td style="border: 1px solid #ddd; padding: 8px;">F'</td><td style="border: 1px solid #ddd; padding: 8px;">Front face counter-clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">G</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">10</td><td style="border: 1px solid #ddd; padding: 8px;">B</td><td style="border: 1px solid #ddd; padding: 8px;">Back face clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">B</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">11</td><td style="border: 1px solid #ddd; padding: 8px;">B'</td><td style="border: 1px solid #ddd; padding: 8px;">Back face counter-clockwise</td><td style="border: 1px solid #ddd; padding: 8px;">N</td></tr>
</table>

<h4>Rewards</h4>
<ul>
    <li><strong>Dense</strong>: Reward based on number of correctly placed stickers</li>
    <li><strong>Sparse</strong>: +1 only when cube is solved</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: Cube is solved (all faces single color)</li>
    <li><strong>Truncation</strong>: Maximum steps reached</li>
</ul>

<h4>Solving Methods</h4>
<ul>
    <li><strong>Layer-by-Layer</strong>: Solve one layer at a time (beginner method)</li>
    <li><strong>CFOP</strong>: Cross, F2L, OLL, PLL (speedcubing method)</li>
    <li><strong>Kociemba</strong>: God's algorithm finds solutions in 20 moves or less</li>
</ul>

<h4>Complexity</h4>
<ul>
    <li><strong>Permutations</strong>: 43,252,003,274,489,856,000 possible states</li>
    <li><strong>God's Number</strong>: Any cube can be solved in 20 moves or fewer</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://ruwix.com/the-rubiks-cube/" target="_blank">Rubik's Cube Tutorial</a></li>
</ul>
"""


RUBIKS_CUBE_HTML = get_rubiks_cube_html()

__all__ = [
    "get_rubiks_cube_html",
    "RUBIKS_CUBE_HTML",
]
