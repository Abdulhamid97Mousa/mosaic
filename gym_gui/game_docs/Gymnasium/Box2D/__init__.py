"""Documentation for Gymnasium Box2D environments."""
from __future__ import annotations

LUNAR_LANDER_HTML = """
<h2>LunarLander-v3</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px;">
Part of the <strong>Gymnasium Box2D</strong> environment family. 
<a href="https://gymnasium.farama.org/environments/box2d/">Box2D Overview</a>
</p>

<h3>Description</h3>
<p>Control a lunar lander and guide it to a safe landing pad with minimal fuel usage.</p>

<h3>Observation Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>state</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(8,)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">8-dimensional continuous vector: x/y position, x/y velocity, angle, angular velocity, left leg contact, right leg contact</td>
  </tr>
</table>

<h3>Action Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #e3f2fd;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Key</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Action</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">ID</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>Space</kbd> or <kbd>S</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Do nothing</td>
    <td style="border: 1px solid #ddd; padding: 8px;">0</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Idle / cut thrust</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>A</kbd> or <kbd>←</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Fire left engine</td>
    <td style="border: 1px solid #ddd; padding: 8px;">1</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Apply left thruster</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>W</kbd> or <kbd>↑</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Fire main engine</td>
    <td style="border: 1px solid #ddd; padding: 8px;">2</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Apply main thruster</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>D</kbd> or <kbd>→</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Fire right engine</td>
    <td style="border: 1px solid #ddd; padding: 8px;">3</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Apply right thruster</td>
  </tr>
</table>

<h3>Rewards & Episode End</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Condition</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Successful landing</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>+100 to +140</code> (scaled by closeness to target pad)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Crashing or flying away</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>-100</code> or more</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Main engine per frame</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>-0.3</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Side thruster per frame</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>-0.03</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Episode terminates</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Lander comes to rest, crashes, or 1000 steps reached</td>
  </tr>
</table>

<h3>References</h3>
<ul>
<li><strong>Official Documentation:</strong> <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">LunarLander (Gymnasium)</a></li>
<li><strong>GitHub:</strong> <a href="https://github.com/Farama-Foundation/Gymnasium">Gymnasium Repository</a></li>
</ul>
"""


CAR_RACING_HTML = """
<h2>CarRacing-v3</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px;">
Part of the <strong>Gymnasium Box2D</strong> environment family. 
<a href="https://gymnasium.farama.org/environments/box2d/">Box2D Overview</a>
</p>

<h3>Description</h3>
<p>Drive a car around a procedurally generated track using continuous steering, gas, and brake controls.</p>

<h3>Observation Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>image</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(0, 255, (96, 96, 3), uint8)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">96×96 RGB image rendered from overhead camera</td>
  </tr>
</table>

<h3>Action Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #e3f2fd;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Key</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Action</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value Range</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>A</kbd> or <kbd>←</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Steering (left)</td>
    <td style="border: 1px solid #ddd; padding: 8px;">[-1, 0]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Steer left with gentle throttle</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>D</kbd> or <kbd>→</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Steering (right)</td>
    <td style="border: 1px solid #ddd; padding: 8px;">[0, 1]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Steer right with gentle throttle</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>W</kbd> or <kbd>↑</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Gas</td>
    <td style="border: 1px solid #ddd; padding: 8px;">[0, 1]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Accelerate forward</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>S</kbd> or <kbd>↓</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Brake</td>
    <td style="border: 1px solid #ddd; padding: 8px;">[0, 1]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Apply brakes</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>Space</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Coast</td>
    <td style="border: 1px solid #ddd; padding: 8px;">[0, 0, 0]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">No throttle, no steering</td>
  </tr>
</table>

<p><strong>Note:</strong> The environment uses continuous control: <code>[steering, gas, brake]</code> where steering ∈ [-1, 1], gas ∈ [0, 1], brake ∈ [0, 1]</p>

<h3>Rewards & Episode End</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Condition</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Visit new track tile</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>+1000 / N</code> (N = total tiles)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Engine load per frame</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>-0.1</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Episode terminates</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Car drives off track too long or 1000 steps reached</td>
  </tr>
</table>

<h3>References</h3>
<ul>
<li><strong>Official Documentation:</strong> <a href="https://gymnasium.farama.org/environments/box2d/car_racing/">CarRacing (Gymnasium)</a></li>
<li><strong>GitHub:</strong> <a href="https://github.com/Farama-Foundation/Gymnasium">Gymnasium Repository</a></li>
</ul>
"""


BIPEDAL_WALKER_HTML = """
<h2>BipedalWalker-v3</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px;">
Part of the <strong>Gymnasium Box2D</strong> environment family. 
<a href="https://gymnasium.farama.org/environments/box2d/">Box2D Overview</a>
</p>

<h3>Description</h3>
<p>Teach a 2D bipedal robot to walk across rough terrain. Rewards forward progress and penalizes motor torque.</p>

<h3>Observation Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>state</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(24,)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">24-dimensional continuous vector: hull angle/velocity, joint angles, joint speeds, leg ground contact, and 10 lidar rangefinder measurements</td>
  </tr>
</table>

<h3>Action Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #e3f2fd;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Continuous</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(4,)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">4D continuous vector: torque for each joint [hip_1, knee_1, hip_2, knee_2] ∈ [-1, 1]</td>
  </tr>
</table>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; border-left: 4px solid #ff9800;">
<strong>Human Control Note:</strong> This environment requires continuous 4D torque actions (not discrete keyboard commands). 
Manual keyboard control is not currently supported in the Qt shell. Use an automated agent or policy for this environment.
</p>

<h3>Rewards & Episode End</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Condition</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Successful traversal to goal</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>+300</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Forward progress per step</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>~+1.0</code> (proportional to distance moved)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Robot falls (hull touches ground)</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>-100</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Motor torque penalty</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>-0.00035 × Σ(torque²)</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Episode terminates</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Robot hull touches ground, reaches goal, or 1600 steps reached</td>
  </tr>
</table>

<h3>References</h3>
<ul>
<li><strong>Official Documentation:</strong> <a href="https://gymnasium.farama.org/environments/box2d/bipedal_walker/">BipedalWalker (Gymnasium)</a></li>
<li><strong>GitHub:</strong> <a href="https://github.com/Farama-Foundation/Gymnasium">Gymnasium Repository</a></li>
</ul>
"""

__all__ = [
    "LUNAR_LANDER_HTML",
    "CAR_RACING_HTML",
    "BIPEDAL_WALKER_HTML",
]
