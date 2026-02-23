"""Documentation snippets for navigation-oriented ViZDoom scenarios."""

from .controls import VIZDOOM_CONTROLS_HTML

VIZDOOM_HEALTH_GATHERING_HTML = """
<h3>ViZDoom – Health Gathering</h3>
<p>
  Roam a maze-like arena collecting medkits to stay alive while your health
  constantly decays. Only movement and rotation are available, highlighting
  navigation policies, curriculum learning, and human steering overlays.
</p>
<h4>Keyboard Controls (3 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>W</b> / <b>↑</b></td><td>MOVE_FORWARD</td></tr>
  <tr><td><b>A</b> / <b>←</b></td><td>TURN_LEFT</td></tr>
  <tr><td><b>D</b> / <b>→</b></td><td>TURN_RIGHT</td></tr>
</table>
<p><i>Note: No attack or backward movement in this scenario.</i></p>
""" + VIZDOOM_CONTROLS_HTML

VIZDOOM_HEALTH_GATHERING_SUPREME_HTML = """
<h3>ViZDoom – Health Gathering Supreme</h3>
<p>
  A tougher variant with more hazards and sparser health packs. This scenario is
  well-suited for evaluating reward-shaping controls in the GUI.
</p>
<h4>Keyboard Controls (3 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>W</b> / <b>↑</b></td><td>MOVE_FORWARD</td></tr>
  <tr><td><b>A</b> / <b>←</b></td><td>TURN_LEFT</td></tr>
  <tr><td><b>D</b> / <b>→</b></td><td>TURN_RIGHT</td></tr>
</table>
<p><i>Note: No attack or backward movement in this scenario.</i></p>
""" + VIZDOOM_CONTROLS_HTML

VIZDOOM_MY_WAY_HOME_HTML = """
<h3>ViZDoom – My Way Home</h3>
<p>
  Focus on sparse-reward navigation: find a goal item hidden in a multi-room
  maze. Great for testing observation overlays and manual exploration in Human
  Control Mode.
</p>
<h4>Keyboard Controls (3 actions)</h4>
<table border="1" cellpadding="5" cellspacing="0">
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><b>W</b> / <b>↑</b></td><td>MOVE_FORWARD</td></tr>
  <tr><td><b>A</b> / <b>←</b></td><td>TURN_LEFT</td></tr>
  <tr><td><b>D</b> / <b>→</b></td><td>TURN_RIGHT</td></tr>
</table>
<p><i>Note: No attack or backward movement in this scenario.</i></p>
""" + VIZDOOM_CONTROLS_HTML

__all__ = [
    "VIZDOOM_HEALTH_GATHERING_HTML",
    "VIZDOOM_HEALTH_GATHERING_SUPREME_HTML",
    "VIZDOOM_MY_WAY_HOME_HTML",
]
