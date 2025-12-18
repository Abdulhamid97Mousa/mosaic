"""NetHack keyboard controls documentation."""

from __future__ import annotations

NETHACK_CONTROLS_HTML = """
<h4>Movement Keys</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Direction</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">y / 7</td><td style="border: 1px solid #ddd; padding: 8px;">Up-Left (Northwest)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">k / 8</td><td style="border: 1px solid #ddd; padding: 8px;">Up (North)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">u / 9</td><td style="border: 1px solid #ddd; padding: 8px;">Up-Right (Northeast)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">h / 4</td><td style="border: 1px solid #ddd; padding: 8px;">Left (West)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">. / 5</td><td style="border: 1px solid #ddd; padding: 8px;">Wait (rest one turn)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">l / 6</td><td style="border: 1px solid #ddd; padding: 8px;">Right (East)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">b / 1</td><td style="border: 1px solid #ddd; padding: 8px;">Down-Left (Southwest)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">j / 2</td><td style="border: 1px solid #ddd; padding: 8px;">Down (South)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">n / 3</td><td style="border: 1px solid #ddd; padding: 8px;">Down-Right (Southeast)</td></tr>
</table>

<h4>Common Actions</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">,</td><td style="border: 1px solid #ddd; padding: 8px;">Pick up item</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">d</td><td style="border: 1px solid #ddd; padding: 8px;">Drop item</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">i</td><td style="border: 1px solid #ddd; padding: 8px;">Inventory</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">e</td><td style="border: 1px solid #ddd; padding: 8px;">Eat</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">q</td><td style="border: 1px solid #ddd; padding: 8px;">Quaff (drink)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">r</td><td style="border: 1px solid #ddd; padding: 8px;">Read (scroll/spellbook)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">w</td><td style="border: 1px solid #ddd; padding: 8px;">Wield weapon</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W</td><td style="border: 1px solid #ddd; padding: 8px;">Wear armor</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">T</td><td style="border: 1px solid #ddd; padding: 8px;">Take off armor</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">P</td><td style="border: 1px solid #ddd; padding: 8px;">Put on ring/amulet</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">R</td><td style="border: 1px solid #ddd; padding: 8px;">Remove ring/amulet</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">z</td><td style="border: 1px solid #ddd; padding: 8px;">Zap wand</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">a</td><td style="border: 1px solid #ddd; padding: 8px;">Apply tool</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">t</td><td style="border: 1px solid #ddd; padding: 8px;">Throw</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">f</td><td style="border: 1px solid #ddd; padding: 8px;">Fire from quiver</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">s</td><td style="border: 1px solid #ddd; padding: 8px;">Search (for traps/doors)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">o</td><td style="border: 1px solid #ddd; padding: 8px;">Open door</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">c</td><td style="border: 1px solid #ddd; padding: 8px;">Close door</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">></td><td style="border: 1px solid #ddd; padding: 8px;">Go down stairs</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><</td><td style="border: 1px solid #ddd; padding: 8px;">Go up stairs</td></tr>
</table>

<h4>Combat</h4>
<p>Move into a monster to attack it with your wielded weapon. Use <code>F</code> followed by a direction for a forced attack.</p>
"""

__all__ = ["NETHACK_CONTROLS_HTML"]
