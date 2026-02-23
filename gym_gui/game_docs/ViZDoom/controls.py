"""Shared human control documentation for all ViZDoom scenarios."""

VIZDOOM_CONTROLS_HTML = """
<hr>
<h4>Mouse Control (FPS-style, 360°)</h4>
<ul>
  <li><b>Click</b> on Video tab to capture mouse (cursor hides)</li>
  <li><b>Move mouse</b> left/right to turn camera horizontally</li>
  <li><b>Move mouse</b> up/down to look up/down (vertical aim)</li>
  <li><b>Press ESC</b> to release mouse and return cursor to normal</li>
</ul>
<p><i>Mouse capture is also released automatically when the window loses focus.</i></p>
<p><b>Tip:</b> Click the Video tab to enable FPS-style mouse look, use keyboard for
movement/attacks, and press ESC when you need to interact with other GUI elements.</p>
"""

__all__ = ["VIZDOOM_CONTROLS_HTML"]
