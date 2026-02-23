"""Documentation snippets for multiplayer-style ViZDoom scenarios."""

from .controls import VIZDOOM_CONTROLS_HTML

VIZDOOM_DEATHMATCH_HTML = """
<h3>ViZDoom – Deathmatch</h3>
<p>
  Full Doom mechanics unlocked: move freely, switch weapons, open doors, and
  battle multiple bots. Ideal for stress-testing human/agent hybrid modes and
  showcasing telemetry overlays (health, armor, kill count) within the GUI.
</p>
<h4>Keyboard Controls (8 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>Space</b> / <b>Ctrl</b></td><td>ATTACK (fire weapon)</td></tr>
  <tr><td><b>E</b> / <b>Enter</b></td><td>USE (open doors, interact)</td></tr>
  <tr><td><b>W</b> / <b>↑</b></td><td>MOVE_FORWARD</td></tr>
  <tr><td><b>S</b> / <b>↓</b></td><td>MOVE_BACKWARD</td></tr>
  <tr><td><b>A</b></td><td>MOVE_LEFT (strafe left)</td></tr>
  <tr><td><b>D</b></td><td>MOVE_RIGHT (strafe right)</td></tr>
  <tr><td><b>Q</b> / <b>←</b></td><td>TURN_LEFT</td></tr>
  <tr><td><b>→</b></td><td>TURN_RIGHT</td></tr>
</table>
<p><i>This is the only scenario with full WASD movement + backward (S).</i></p>
""" + VIZDOOM_CONTROLS_HTML

__all__ = ["VIZDOOM_DEATHMATCH_HTML"]
