"""Documentation for HoverAviary environment.

HoverAviary is a single-agent RL environment where a Crazyflie 2.x quadcopter
must reach a target altitude and stabilize (hover) in place.

Paper: "Learning to Fly - a Gym Environment with PyBullet Physics for
       Reinforcement Learning of Multi-agent Quadcopter Control"
       (Panerati et al., 2021)
Repository: https://github.com/utiasDSL/gym-pybullet-drones
"""

from __future__ import annotations


def get_hover_aviary_html(env_id: str = "hover-aviary-v0") -> str:
    """Generate HoverAviary HTML documentation.

    Args:
        env_id: Environment identifier (default: "hover-aviary-v0")

    Returns:
        HTML string containing environment documentation.
    """
    return f"""
<h2>{env_id}</h2>

<p>
HoverAviary is a single-agent reinforcement learning environment for quadcopter control.
A simulated <strong>Crazyflie 2.x</strong> nanoquadcopter must learn to reach a target
altitude (z=1.0m) and stabilize in a hovering position using PyBullet physics.
</p>

<h4>Drone Model: Crazyflie 2.x</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Mass</td><td style="border: 1px solid #ddd; padding: 8px;">27g</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Arm Length</td><td style="border: 1px solid #ddd; padding: 8px;">39.7mm</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Thrust-to-Weight</td><td style="border: 1px solid #ddd; padding: 8px;">2.25</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Speed</td><td style="border: 1px solid #ddd; padding: 8px;">30 km/h</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Configuration</td><td style="border: 1px solid #ddd; padding: 8px;">X-configuration (CF2X)</td></tr>
</table>

<h4>Observation Space</h4>
<p><code>Box</code> - Kinematic state vector (12 features by default):</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Index</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Feature</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Range</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0-2</td><td style="border: 1px solid #ddd; padding: 8px;">Position (x, y, z)</td><td style="border: 1px solid #ddd; padding: 8px;">meters</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3-5</td><td style="border: 1px solid #ddd; padding: 8px;">Roll, Pitch, Yaw</td><td style="border: 1px solid #ddd; padding: 8px;">radians</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6-8</td><td style="border: 1px solid #ddd; padding: 8px;">Linear Velocity (vx, vy, vz)</td><td style="border: 1px solid #ddd; padding: 8px;">m/s</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">9-11</td><td style="border: 1px solid #ddd; padding: 8px;">Angular Velocity (wx, wy, wz)</td><td style="border: 1px solid #ddd; padding: 8px;">rad/s</td></tr>
</table>
<p>Optional: RGB vision observations (64x48x4) when <code>obs=ObservationType.RGB</code></p>

<h4>Action Space</h4>
<p>Multiple action types available:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Shape</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">RPM</td><td style="border: 1px solid #ddd; padding: 8px;">(4,)</td><td style="border: 1px solid #ddd; padding: 8px;">Normalized motor speeds [-1, 1]</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">ONE_D_RPM</td><td style="border: 1px solid #ddd; padding: 8px;">(1,)</td><td style="border: 1px solid #ddd; padding: 8px;">Single RPM for all 4 motors</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">PID</td><td style="border: 1px solid #ddd; padding: 8px;">(3,)</td><td style="border: 1px solid #ddd; padding: 8px;">Target position (uses internal PID)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">VEL</td><td style="border: 1px solid #ddd; padding: 8px;">(4,)</td><td style="border: 1px solid #ddd; padding: 8px;">Velocity command (vx, vy, vz, yaw_rate)</td></tr>
</table>

<h4>Reward Function</h4>
<p>Simple distance-based reward encouraging the drone to reach target altitude z=1.0m:</p>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
reward = max(0, 2 - ||target_pos - drone_pos||^4)
       = max(0, 2 - ||[0, 0, 1] - [x, y, z]||^4)
</pre>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: Target reached within 0.0001m tolerance</li>
    <li><strong>Truncation</strong>: Episode timeout (8 seconds at 30Hz control = 240 steps)</li>
    <li><strong>Truncation</strong>: Drone exceeds position bounds or excessive tilt</li>
</ul>

<h4>Physics Simulation</h4>
<p>Built on PyBullet with configurable aerodynamic effects:</p>
<ul>
    <li><strong>PYB</strong>: Base PyBullet physics</li>
    <li><strong>PYB_GND</strong>: + Ground effect (increased thrust near surface)</li>
    <li><strong>PYB_DRAG</strong>: + Air drag proportional to velocity</li>
    <li><strong>PYB_DW</strong>: + Downwash between drones</li>
    <li><strong>PYB_GND_DRAG_DW</strong>: All effects combined</li>
</ul>

<h4>Simulation Parameters</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Parameter</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Default</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Physics frequency</td><td style="border: 1px solid #ddd; padding: 8px;">240 Hz</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Control frequency</td><td style="border: 1px solid #ddd; padding: 8px;">30 Hz</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode length</td><td style="border: 1px solid #ddd; padding: 8px;">8 seconds</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Gravity</td><td style="border: 1px solid #ddd; padding: 8px;">9.8 m/s^2</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Hover RPM</td><td style="border: 1px solid #ddd; padding: 8px;">~14,500 RPM</td></tr>
</table>

<h4>Usage Example</h4>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;">
import gymnasium as gym
from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

env = HoverAviary(
    gui=True,
    obs=ObservationType.KIN,
    act=ActionType.RPM
)

obs, info = env.reset()
for _ in range(240):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
</pre>

<h4>Benchmark Results (PPO, 30k timesteps)</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Algorithm</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Mean Reward</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">SAC</td><td style="border: 1px solid #ddd; padding: 8px;">Best convergence</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">PPO</td><td style="border: 1px solid #ddd; padding: 8px;">Good convergence</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A2C</td><td style="border: 1px solid #ddd; padding: 8px;">Slower convergence</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/utiasDSL/gym-pybullet-drones" target="_blank">GitHub Repository</a></li>
    <li><a href="https://utiasdsl.github.io/gym-pybullet-drones" target="_blank">Documentation</a></li>
    <li>Paper: Panerati, J., et al. (2021). Learning to Fly - a Gym Environment with PyBullet Physics for RL of Multi-agent Quadcopter Control.</li>
</ul>
"""


# Backward compatibility constant
HOVER_AVIARY_HTML = get_hover_aviary_html("hover-aviary-v0")

__all__ = [
    "get_hover_aviary_html",
    "HOVER_AVIARY_HTML",
]
