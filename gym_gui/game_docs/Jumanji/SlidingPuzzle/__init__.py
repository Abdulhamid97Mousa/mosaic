"""Documentation for Jumanji SlidingTilePuzzle environment.

The Sliding Tile Puzzle (N-Puzzle) is a classic puzzle where numbered tiles
must be arranged in order by sliding them into an empty space.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_sliding_puzzle_html(env_id: str = "SlidingTilePuzzle-v0") -> str:
    """Generate SlidingTilePuzzle HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
The Sliding Tile Puzzle (also known as N-Puzzle, 15-Puzzle, or 8-Puzzle) is a
classic combinatorial puzzle. Tiles are arranged in a grid with one empty space,
and the goal is to arrange the tiles in numerical order by sliding them.
</p>

<h4>Game Rules</h4>
<ul>
    <li>Tiles can only slide into the empty space</li>
    <li>Only tiles adjacent to the empty space can move</li>
    <li>The goal is to arrange tiles in numerical order (1, 2, 3, ... with empty in bottom-right)</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>puzzle</strong>: <code>Box(shape=(N, N), dtype=int32)</code> - The puzzle grid (0 = empty, 1 to N*N-1 for numbered tiles)</li>
    <li><strong>empty_position</strong>: <code>Tuple[int, int]</code> - Row, column of empty space</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(4)</code> - Slide direction (moves the empty space):</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Effect</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">Up</td><td style="border: 1px solid #ddd; padding: 8px;">Slide tile above empty space down</td><td style="border: 1px solid #ddd; padding: 8px;">W / Up Arrow</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">Right</td><td style="border: 1px solid #ddd; padding: 8px;">Slide tile right of empty space left</td><td style="border: 1px solid #ddd; padding: 8px;">D / Right Arrow</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">Down</td><td style="border: 1px solid #ddd; padding: 8px;">Slide tile below empty space up</td><td style="border: 1px solid #ddd; padding: 8px;">S / Down Arrow</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">Left</td><td style="border: 1px solid #ddd; padding: 8px;">Slide tile left of empty space right</td><td style="border: 1px solid #ddd; padding: 8px;">A / Left Arrow</td></tr>
</table>

<h4>Puzzle Sizes</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Name</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Size</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Tiles</th>
        <th style="border: 1px solid #ddd; padding: 8px;">States</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">8-Puzzle</td><td style="border: 1px solid #ddd; padding: 8px;">3x3</td><td style="border: 1px solid #ddd; padding: 8px;">8</td><td style="border: 1px solid #ddd; padding: 8px;">181,440</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">15-Puzzle</td><td style="border: 1px solid #ddd; padding: 8px;">4x4</td><td style="border: 1px solid #ddd; padding: 8px;">15</td><td style="border: 1px solid #ddd; padding: 8px;">~10^13</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">24-Puzzle</td><td style="border: 1px solid #ddd; padding: 8px;">5x5</td><td style="border: 1px solid #ddd; padding: 8px;">24</td><td style="border: 1px solid #ddd; padding: 8px;">~10^25</td></tr>
</table>

<h4>Rewards</h4>
<ul>
    <li><strong>-1</strong>: Per step (encourages optimal solutions)</li>
    <li><strong>+100</strong>: Solving the puzzle</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: Puzzle is solved (tiles in order)</li>
    <li><strong>Truncation</strong>: Maximum steps reached</li>
</ul>

<h4>Heuristics</h4>
<ul>
    <li><strong>Manhattan Distance</strong>: Sum of distances of each tile from its goal position</li>
    <li><strong>Misplaced Tiles</strong>: Count of tiles not in correct position</li>
    <li><strong>Linear Conflict</strong>: Manhattan + penalty for tiles blocking each other</li>
</ul>

<h4>Solvability</h4>
<p>
Not all random configurations are solvable. A puzzle is solvable if:
</p>
<ul>
    <li>For odd-width puzzles: number of inversions is even</li>
    <li>For even-width puzzles: (inversions + blank row from bottom) is odd</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/" target="_blank">Solvability Check</a></li>
</ul>
"""


SLIDING_PUZZLE_HTML = get_sliding_puzzle_html()

__all__ = [
    "get_sliding_puzzle_html",
    "SLIDING_PUZZLE_HTML",
]
