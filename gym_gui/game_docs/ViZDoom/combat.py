"""Documentation snippets for combat-heavy ViZDoom scenarios."""

from .controls import VIZDOOM_CONTROLS_HTML

VIZDOOM_DEFEND_THE_CENTER_HTML = """
<h3>ViZDoom – Defend The Center</h3>
<p>
  Stand in place and rotate to neutralize waves of approaching monsters. Only
  rotation and shooting are allowed, which keeps the action space compact while
  still rewarding precise aim and rapid situational awareness.
</p>
<h4>Keyboard Controls (3 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>Space</b> / <b>Ctrl</b></td><td>ATTACK (fire weapon)</td></tr>
  <tr><td><b>A</b> / <b>←</b></td><td>TURN_LEFT</td></tr>
  <tr><td><b>D</b> / <b>→</b></td><td>TURN_RIGHT</td></tr>
</table>
<p><i>Note: No movement keys (W/S) - you are stationary, rotation only.</i></p>
""" + VIZDOOM_CONTROLS_HTML

VIZDOOM_DEFEND_THE_LINE_HTML = """
<h3>ViZDoom – Defend The Line</h3>
<p>
  Similar to Defend The Center but enemies attack from a straight corridor. The
  player rotates in place to cover the approach vector, highlighting reward
  shaping, HUD visibility, and hybrid control experiments.
</p>
<h4>Keyboard Controls (3 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>Space</b> / <b>Ctrl</b></td><td>ATTACK (fire weapon)</td></tr>
  <tr><td><b>A</b> / <b>←</b></td><td>TURN_LEFT</td></tr>
  <tr><td><b>D</b> / <b>→</b></td><td>TURN_RIGHT</td></tr>
</table>
<p><i>Note: No movement keys (W/S) - you are stationary, rotation only.</i></p>
""" + VIZDOOM_CONTROLS_HTML

VIZDOOM_DEADLY_CORRIDOR_HTML = """
<h3>ViZDoom – Deadly Corridor</h3>
<p>
  Navigate a long hallway filled with enemies. The scenario enables full
  movement (forward, strafe, rotate) plus shooting, making it a comprehensive
  testbed for both human demonstration capture and high-speed agents.
</p>
<h4>Keyboard Controls (6 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>Space</b> / <b>Ctrl</b></td><td>ATTACK (fire weapon)</td></tr>
  <tr><td><b>A</b></td><td>MOVE_LEFT (strafe left)</td></tr>
  <tr><td><b>D</b></td><td>MOVE_RIGHT (strafe right)</td></tr>
  <tr><td><b>W</b> / <b>↑</b></td><td>MOVE_FORWARD</td></tr>
  <tr><td><b>Q</b> / <b>←</b></td><td>TURN_LEFT</td></tr>
  <tr><td><b>E</b> / <b>→</b></td><td>TURN_RIGHT</td></tr>
</table>
<p><i>Note: No backward movement (S) in this scenario.</i></p>
""" + VIZDOOM_CONTROLS_HTML

__all__ = [
    "VIZDOOM_DEFEND_THE_CENTER_HTML",
    "VIZDOOM_DEFEND_THE_LINE_HTML",
    "VIZDOOM_DEADLY_CORRIDOR_HTML",
]
