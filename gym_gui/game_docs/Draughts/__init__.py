"""Documentation for Draughts/Checkers environment variants.

MOSAIC includes custom implementations of three official draughts variants:
- American Checkers (8x8, English draughts)
- Russian Checkers (8x8, backward captures + flying kings)
- International Draughts (10x10, 20 pieces per side)

Each variant follows its official competition rules precisely.

References:
- American: https://en.wikipedia.org/wiki/English_draughts
- Russian: https://en.wikipedia.org/wiki/Russian_draughts
- International: https://en.wikipedia.org/wiki/International_draughts
"""

from __future__ import annotations


AMERICAN_CHECKERS_HTML = """
<h2>American Checkers (English Draughts)</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px;">
Part of the <strong>MOSAIC Draughts</strong> family. Custom implementation following official American Checkers rules.
</p>

<h3>Description</h3>
<p>Classic 8×8 checkers as played in the United States and United Kingdom. The simplest of the three draughts variants.</p>

<h3>Board & Setup</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Board size</td>
    <td style="border: 1px solid #ddd; padding: 8px;">8×8 (only dark squares used)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Pieces per player</td>
    <td style="border: 1px solid #ddd; padding: 8px;">12</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Starting rows</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Black: rows 0-2, White: rows 5-7</td>
  </tr>
</table>

<h3>Observation Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>observation</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Dict</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Contains 'board' (8×8 int array), 'current_player' (0 or 1), 'action_mask' (legal moves)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>board</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(8, 8)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">0=Empty, 1=Black piece, 2=Black king, 3=White piece, 4=White king</td>
  </tr>
</table>

<h3>Action Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #fff3e0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Type</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Discrete</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Discrete(N)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">N varies by position (filtered by action_mask). Actions represent from→to square moves.</td>
  </tr>
</table>

<h3>Game Rules (American Variant)</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Rule</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Regular piece movement</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Forward diagonal only, 1 square</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Capturing</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Jump over opponent's piece diagonally (forward only for regular pieces)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Multi-jump</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Multiple captures allowed in single turn if available</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Promotion</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Piece reaching opposite end becomes King</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">King movement</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Forward OR backward diagonal, 1 square</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">King capturing</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Can capture forward or backward</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><strong>NO flying kings</strong></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Kings cannot move multiple squares (unlike Russian/International)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Mandatory capture</td>
    <td style="border: 1px solid #ddd; padding: 8px;">If capture available, must capture (cannot make non-capture move)</td>
  </tr>
</table>

<h3>Rewards & Episode End</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Condition</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Reward</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Win (capture all opponent pieces or block all moves)</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>+1</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Loss</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>-1</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Draw (50-move rule or repetition)</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>0</code></td>
  </tr>
</table>

<h3>References</h3>
<ul>
<li><strong>Official Rules:</strong> <a href="https://en.wikipedia.org/wiki/English_draughts">English Draughts (Wikipedia)</a></li>
<li><strong>Implementation:</strong> <code>gym_gui/core/adapters/draughts.py</code></li>
</ul>
"""


RUSSIAN_CHECKERS_HTML = """
<h2>Russian Checkers</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px;">
Part of the <strong>MOSAIC Draughts</strong> family. Custom implementation following official Russian Checkers rules.
</p>

<h3>Description</h3>
<p>8×8 checkers variant popular in Russia and former Soviet states. Key differences: backward captures allowed and flying kings.</p>

<h3>Board & Setup</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Board size</td>
    <td style="border: 1px solid #ddd; padding: 8px;">8×8 (only dark squares used)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Pieces per player</td>
    <td style="border: 1px solid #ddd; padding: 8px;">12</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Starting rows</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Black: rows 0-2, White: rows 5-7</td>
  </tr>
</table>

<h3>Observation Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>observation</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Dict</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Contains 'board' (8×8 int array), 'current_player' (0 or 1), 'action_mask' (legal moves)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>board</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(8, 8)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">0=Empty, 1=Black piece, 2=Black king, 3=White piece, 4=White king</td>
  </tr>
</table>

<h3>Action Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #fff3e0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Type</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Discrete</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Discrete(N)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">N varies by position (filtered by action_mask). Actions represent from→to square moves.</td>
  </tr>
</table>

<h3>Game Rules (Russian Variant)</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Rule</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Regular piece movement</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Forward diagonal only, 1 square</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Backward captures</strong></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Regular pieces CAN capture backward (unlike American)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Multi-jump</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Multiple captures allowed in single turn if available</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Promotion</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Piece reaching opposite end becomes King</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Flying kings</strong></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Kings can move ANY number of squares diagonally (major difference from American)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">King capturing</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Can capture along any diagonal path, landing anywhere after the captured piece</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Mandatory capture</td>
    <td style="border: 1px solid #ddd; padding: 8px;">If capture available, must capture (cannot make non-capture move)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Maximum capture rule</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Must choose move that captures the most pieces</td>
  </tr>
</table>

<h3>Rewards & Episode End</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Condition</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Reward</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Win (capture all opponent pieces or block all moves)</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>+1</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Loss</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>-1</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Draw (repetition or stalemate)</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>0</code></td>
  </tr>
</table>

<h3>References</h3>
<ul>
<li><strong>Official Rules:</strong> <a href="https://en.wikipedia.org/wiki/Russian_draughts">Russian Checkers (Wikipedia)</a></li>
<li><strong>Implementation:</strong> <code>gym_gui/core/adapters/draughts.py</code></li>
</ul>
"""


INTERNATIONAL_DRAUGHTS_HTML = """
<h2>International Draughts (Polish Draughts)</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px;">
Part of the <strong>MOSAIC Draughts</strong> family. Custom implementation following official International Draughts rules.
</p>

<h3>Description</h3>
<p>The most complex draughts variant, played on a 10×10 board with 20 pieces per side. Standard for international competitions.</p>

<h3>Board & Setup</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Board size</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><strong>10×10</strong> (only dark squares used, 50 playable squares)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Pieces per player</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><strong>20</strong> (double American/Russian)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Starting rows</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Black: rows 0-3, White: rows 6-9</td>
  </tr>
</table>

<h3>Observation Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>observation</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Dict</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Contains 'board' (10×10 int array), 'current_player' (0 or 1), 'action_mask' (legal moves)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>board</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(10, 10)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">0=Empty, 1=Black piece, 2=Black king, 3=White piece, 4=White king</td>
  </tr>
</table>

<h3>Action Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #fff3e0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Type</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Discrete</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Discrete(N)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">N varies by position (filtered by action_mask). Larger action space due to 10×10 board.</td>
  </tr>
</table>

<h3>Game Rules (International Variant)</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Rule</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Regular piece movement</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Forward diagonal only, 1 square</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Backward captures</strong></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Regular pieces CAN capture backward (same as Russian)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Multi-jump</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Multiple captures allowed in single turn if available</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Promotion</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Piece reaching opposite end becomes King</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Flying kings</strong></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Kings can move ANY number of squares diagonally (same as Russian)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">King capturing</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Can capture along any diagonal path, landing anywhere after the captured piece</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Mandatory capture</td>
    <td style="border: 1px solid #ddd; padding: 8px;">If capture available, must capture (cannot make non-capture move)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Maximum capture rule</strong></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Must choose move that captures the most pieces (strictly enforced)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">King priority rule</td>
    <td style="border: 1px solid #ddd; padding: 8px;">If equal captures, prefer capturing kings over regular pieces</td>
  </tr>
</table>

<h3>Rewards & Episode End</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Condition</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Reward</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Win (capture all opponent pieces or block all moves)</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>+1</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Loss</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>-1</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Draw (repetition or stalemate)</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>0</code></td>
  </tr>
</table>

<h3>Key Differences from 8×8 Variants</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #e3f2fd;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Aspect</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">International</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">American/Russian</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Board size</td>
    <td style="border: 1px solid #ddd; padding: 8px;">10×10 (50 squares)</td>
    <td style="border: 1px solid #ddd; padding: 8px;">8×8 (32 squares)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Pieces per side</td>
    <td style="border: 1px solid #ddd; padding: 8px;">20</td>
    <td style="border: 1px solid #ddd; padding: 8px;">12</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Game length</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Typically longer</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Shorter</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Strategy depth</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Most complex</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Less complex</td>
  </tr>
</table>

<h3>References</h3>
<ul>
<li><strong>Official Rules:</strong> <a href="https://en.wikipedia.org/wiki/International_draughts">International Draughts (Wikipedia)</a></li>
<li><strong>World Championship:</strong> <a href="https://fmjd.org/">FMJD (Fédération Mondiale du Jeu de Dames)</a></li>
<li><strong>Implementation:</strong> <code>gym_gui/core/adapters/draughts.py</code></li>
</ul>
"""


__all__ = [
    "AMERICAN_CHECKERS_HTML",
    "RUSSIAN_CHECKERS_HTML",
    "INTERNATIONAL_DRAUGHTS_HTML",
]
