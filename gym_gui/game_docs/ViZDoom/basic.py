"""Documentation snippets for introductory ViZDoom scenarios."""

from .controls import VIZDOOM_CONTROLS_HTML

VIZDOOM_BASIC_HTML = """
<h3>ViZDoom – Basic</h3>
<p>
  Learn the fundamentals of ViZDoom by eliminating a single stationary enemy in
  a rectangular room. Observations come directly from the screen buffer and the
  action space is limited to strafing left/right and firing the pistol. This is
  ideal for smoke-testing adapters, human control, or imitation-learning demos.
</p>
<h4>Keyboard Controls (3 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>Space</b> / <b>Ctrl</b></td><td>ATTACK (fire weapon)</td></tr>
  <tr><td><b>A</b> / <b>←</b></td><td>MOVE_LEFT (strafe left)</td></tr>
  <tr><td><b>D</b> / <b>→</b></td><td>MOVE_RIGHT (strafe right)</td></tr>
</table>
<p><i>Note: No forward/backward movement (W/S) in this scenario.</i></p>
""" + VIZDOOM_CONTROLS_HTML

VIZDOOM_PREDICT_POSITION_HTML = """
<h3>ViZDoom – Predict Position</h3>
<p>
  A timing-focused target practice scenario where the agent (or human) must fire
  at enemies that briefly appear from hiding. Only rotation and the attack
  button are enabled, making it a compact benchmark for reaction-based policies.
</p>
<h4>Keyboard Controls (3 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>Space</b> / <b>Ctrl</b></td><td>ATTACK (fire weapon)</td></tr>
  <tr><td><b>A</b> / <b>←</b></td><td>TURN_LEFT</td></tr>
  <tr><td><b>D</b> / <b>→</b></td><td>TURN_RIGHT</td></tr>
</table>
<p><i>Note: No movement keys (W/S) in this scenario - rotation only.</i></p>
""" + VIZDOOM_CONTROLS_HTML

VIZDOOM_TAKE_COVER_HTML = """
<h3>ViZDoom – Take Cover</h3>
<p>
  Dodging rockets is the name of the game. Only lateral movement is available,
  forcing players to rely on spatial awareness and fast observation-to-action
  loops. Great for testing human control overlays and curriculum learning.
</p>
<h4>Keyboard Controls (2 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>A</b> / <b>←</b></td><td>MOVE_LEFT</td></tr>
  <tr><td><b>D</b> / <b>→</b></td><td>MOVE_RIGHT</td></tr>
</table>
<p><i>Note: No attack, forward/backward, or turning in this scenario.</i></p>
""" + VIZDOOM_CONTROLS_HTML

__all__ = [
    "VIZDOOM_BASIC_HTML",
    "VIZDOOM_PREDICT_POSITION_HTML",
    "VIZDOOM_TAKE_COVER_HTML",
]
