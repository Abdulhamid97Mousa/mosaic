"""Documentation for Jumanji Sudoku environment.

Sudoku is a logic-based number-placement puzzle where the goal is to fill
a 9x9 grid such that each row, column, and 3x3 box contains digits 1-9.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_sudoku_html(env_id: str = "Sudoku-v0") -> str:
    """Generate Sudoku HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Sudoku is a logic-based, combinatorial number-placement puzzle. The objective
is to fill a 9x9 grid with digits so that each column, each row, and each of
the nine 3x3 subgrids (boxes) contains all of the digits from 1 to 9.
</p>

<h4>Game Rules</h4>
<ul>
    <li>Fill empty cells with digits 1-9</li>
    <li>Each row must contain digits 1-9 exactly once</li>
    <li>Each column must contain digits 1-9 exactly once</li>
    <li>Each 3x3 box must contain digits 1-9 exactly once</li>
    <li>Pre-filled cells (clues) cannot be changed</li>
</ul>

<h4>Observation Space</h4>
<p>
<code>Dict</code> with:
</p>
<ul>
    <li><strong>board</strong>: <code>Box(shape=(9, 9), dtype=int32)</code> - The Sudoku grid (0 = empty, 1-9 = filled)</li>
    <li><strong>action_mask</strong>: <code>Box(shape=(9*9*9,), dtype=bool)</code> - Valid actions</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(9 * 9 * 9 = 729)</code> - Place a digit in a cell:</p>
<ul>
    <li>Action = row * 81 + col * 9 + (digit - 1)</li>
    <li>Each action places digit (1-9) in cell (row, col)</li>
</ul>

<h4>Difficulty Levels</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Difficulty</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Clues Given</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Empty Cells</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Easy</td><td style="border: 1px solid #ddd; padding: 8px;">36-45</td><td style="border: 1px solid #ddd; padding: 8px;">36-45</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Medium</td><td style="border: 1px solid #ddd; padding: 8px;">27-35</td><td style="border: 1px solid #ddd; padding: 8px;">46-54</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Hard</td><td style="border: 1px solid #ddd; padding: 8px;">22-26</td><td style="border: 1px solid #ddd; padding: 8px;">55-59</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Expert</td><td style="border: 1px solid #ddd; padding: 8px;">17-21</td><td style="border: 1px solid #ddd; padding: 8px;">60-64</td></tr>
</table>
<p><em>Note: Minimum 17 clues are required for a unique solution.</em></p>

<h4>Rewards</h4>
<ul>
    <li><strong>+1</strong>: Placing a correct digit</li>
    <li><strong>-1</strong>: Placing an incorrect digit (violates constraints)</li>
    <li><strong>+10</strong>: Completing the puzzle correctly</li>
</ul>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination (Win)</strong>: All cells filled correctly</li>
    <li><strong>Termination (Loss)</strong>: Invalid placement (optional strict mode)</li>
    <li><strong>Truncation</strong>: Maximum steps reached</li>
</ul>

<h4>Solving Techniques</h4>
<ul>
    <li><strong>Naked Singles</strong>: Cell has only one possible value</li>
    <li><strong>Hidden Singles</strong>: Digit can only go in one cell in row/col/box</li>
    <li><strong>Naked Pairs/Triples</strong>: Groups of cells with same candidates</li>
    <li><strong>X-Wing</strong>: Pattern across rows and columns</li>
    <li><strong>Backtracking</strong>: Try values and backtrack on contradictions</li>
</ul>

<h4>Controls</h4>
<p>
Sudoku uses a complex action space (cell + digit selection) that is best suited
for mouse/touch interaction with a number pad, or can use keyboard number keys
after selecting a cell.
</p>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://sudoku.com/" target="_blank">Online Sudoku</a></li>
</ul>
"""


SUDOKU_HTML = get_sudoku_html()

__all__ = [
    "get_sudoku_html",
    "SUDOKU_HTML",
]
