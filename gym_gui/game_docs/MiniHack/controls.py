"""Control documentation for MiniHack Human Control Mode."""

MINIHACK_CONTROLS_HTML = """
<h4>Human Control Mode</h4>
<p>
  MiniHack uses NetHack-style keyboard commands. In Human Control Mode,
  keyboard input is captured by the Qt window and sent to the environment.
</p>

<h5>Movement (8 Compass Directions)</h5>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th><th>Description</th></tr>
  <tr><td><b>y</b> / <b>7</b></td><td>NW</td><td>Move northwest (up-left)</td></tr>
  <tr><td><b>k</b> / <b>8</b> / <b>↑</b></td><td>N</td><td>Move north (up)</td></tr>
  <tr><td><b>u</b> / <b>9</b></td><td>NE</td><td>Move northeast (up-right)</td></tr>
  <tr><td><b>h</b> / <b>4</b> / <b>←</b></td><td>W</td><td>Move west (left)</td></tr>
  <tr><td><b>.</b> / <b>5</b></td><td>WAIT</td><td>Wait one turn</td></tr>
  <tr><td><b>l</b> / <b>6</b> / <b>→</b></td><td>E</td><td>Move east (right)</td></tr>
  <tr><td><b>b</b> / <b>1</b></td><td>SW</td><td>Move southwest (down-left)</td></tr>
  <tr><td><b>j</b> / <b>2</b> / <b>↓</b></td><td>S</td><td>Move south (down)</td></tr>
  <tr><td><b>n</b> / <b>3</b></td><td>SE</td><td>Move southeast (down-right)</td></tr>
</table>

<h5>Common Actions</h5>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th><th>Description</th></tr>
  <tr><td><b>o</b></td><td>OPEN</td><td>Open a door</td></tr>
  <tr><td><b>c</b></td><td>CLOSE</td><td>Close a door</td></tr>
  <tr><td><b>s</b></td><td>SEARCH</td><td>Search for traps/secret doors</td></tr>
  <tr><td><b>Ctrl+D</b></td><td>KICK</td><td>Kick something (door, monster)</td></tr>
  <tr><td><b>e</b></td><td>EAT</td><td>Eat something</td></tr>
  <tr><td><b>,</b></td><td>PICKUP</td><td>Pick up items</td></tr>
  <tr><td><b>d</b></td><td>DROP</td><td>Drop an item</td></tr>
  <tr><td><b>i</b></td><td>INVENTORY</td><td>Show inventory</td></tr>
  <tr><td><b>w</b></td><td>WIELD</td><td>Wield a weapon</td></tr>
  <tr><td><b>W</b></td><td>WEAR</td><td>Wear armor</td></tr>
  <tr><td><b>r</b></td><td>READ</td><td>Read scroll/book</td></tr>
  <tr><td><b>q</b></td><td>QUAFF</td><td>Drink a potion</td></tr>
  <tr><td><b>z</b></td><td>ZAP</td><td>Zap a wand</td></tr>
  <tr><td><b>Z</b></td><td>CAST</td><td>Cast a spell</td></tr>
  <tr><td><b>a</b></td><td>APPLY</td><td>Use a tool</td></tr>
  <tr><td><b>t</b></td><td>THROW</td><td>Throw something</td></tr>
  <tr><td><b>f</b></td><td>FIRE</td><td>Fire from quiver</td></tr>
  <tr><td><b>&lt;</b></td><td>UP</td><td>Go up stairs</td></tr>
  <tr><td><b>&gt;</b></td><td>DOWN</td><td>Go down stairs</td></tr>
</table>

<h5>Tips</h5>
<ul>
  <li>Use <b>vi-keys</b> (hjkl/yubn) for 8-directional movement</li>
  <li>Arrow keys only provide 4-directional movement</li>
  <li>Numpad works if available on your keyboard</li>
  <li>Press <b>ESC</b> to cancel current action</li>
</ul>
"""

__all__ = ["MINIHACK_CONTROLS_HTML"]
