"""Documentation for Jumanji Snake environment.

Snake is the classic arcade game where the player controls a growing
snake that must eat food while avoiding walls and its own body.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations


def get_snake_html(env_id: str = "Snake-v1") -> str:
    """Generate Snake HTML documentation."""
    return f"""
<h2>Jumanji {env_id}</h2>

<p>
Snake is a classic arcade game where the player controls a snake that
grows longer each time it eats food. The challenge is to avoid hitting
the walls or the snake's own body.
</p>

<h4>Game Rules</h4>
<ul>
    <li>Snake moves continuously in current direction</li>
    <li>Eating food makes the snake grow longer</li>
    <li>Game ends if snake hits wall or itself</li>
    <li>Goal: Grow as long as possible</li>
</ul>

<h4>Observation Space</h4>
<p><code>Dict</code> with:</p>
<ul>
    <li><strong>grid</strong>: 2D grid showing snake body and food</li>
    <li><strong>head_position</strong>: Location of snake's head</li>
    <li><strong>direction</strong>: Current movement direction</li>
    <li><strong>length</strong>: Current snake length</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(4)</code> - Change direction:</p>
<ul>
    <li><strong>0</strong>: Up</li>
    <li><strong>1</strong>: Right</li>
    <li><strong>2</strong>: Down</li>
    <li><strong>3</strong>: Left</li>
</ul>
<p><em>Note: Cannot reverse direction (180-degree turn)</em></p>

<h4>Rewards</h4>
<ul>
    <li><strong>+1</strong>: Eating food</li>
    <li><strong>-1</strong>: Hitting wall or self (game over)</li>
    <li><strong>0</strong>: Normal movement</li>
</ul>

<h4>Episode End</h4>
<ul>
    <li><strong>Termination</strong>: Snake collides with wall or itself</li>
    <li><strong>Win</strong>: Fill entire grid (rare)</li>
</ul>

<h4>Strategies</h4>
<ul>
    <li>Plan ahead to avoid trapping yourself</li>
    <li>Follow the walls in a consistent pattern</li>
    <li>Leave escape routes open</li>
    <li>Use Hamiltonian cycle for perfect play</li>
</ul>

<h4>Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W / Up Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Turn Up (action 0)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D / Right Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Turn Right (action 1)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">S / Down Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Turn Down (action 2)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A / Left Arrow</td><td style="border: 1px solid #ddd; padding: 8px;">Turn Left (action 3)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/google-deepmind/jumanji" target="_blank">Jumanji GitHub Repository</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Snake_(video_game_genre)" target="_blank">Snake (Wikipedia)</a></li>
</ul>
"""


SNAKE_HTML = get_snake_html()

__all__ = ["get_snake_html", "SNAKE_HTML"]
