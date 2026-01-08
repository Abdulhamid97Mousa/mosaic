"""Documentation for MultiHoverAviary environment.

MultiHoverAviary is a multi-agent RL environment where multiple Crazyflie 2.x
quadcopters must learn to hover at different target altitudes.

Paper: "Learning to Fly - a Gym Environment with PyBullet Physics for
       Reinforcement Learning of Multi-agent Quadcopter Control"
       (Panerati et al., 2021)
Repository: https://github.com/utiasDSL/gym-pybullet-drones
"""

from __future__ import annotations


def get_multihover_aviary_html(env_id: str = "multihover-aviary-v0") -> str:
    """Generate MultiHoverAviary HTML documentation.

    Args:
        env_id: Environment identifier (default: "multihover-aviary-v0")

    Returns:
        HTML string containing environment documentation.
    """
    return f"""
<h2>{env_id}</h2>

<p>
MultiHoverAviary is a <strong>multi-agent reinforcement learning</strong> environment
for coordinated quadcopter control. Multiple simulated <strong>Crazyflie 2.x</strong>
nanoquadcopters must learn to hover at different target altitudes while accounting
for aerodynamic interactions (downwash effects) between drones.
</p>

<h4>Multi-Agent Features</h4>
<ul>
    <li><strong>Scalable</strong>: Configure 2 to N drones</li>
    <li><strong>Downwash modeling</strong>: Realistic aerodynamic interference between drones</li>
    <li><strong>Centralized/Decentralized</strong>: Supports both CTDE and independent learning</li>
    <li><strong>RLlib compatible</strong>: Native multi-agent RL interface</li>
</ul>

<h4>Observation Space (per agent)</h4>
<p><code>Box</code> - Kinematic state vector (12 features):</p>
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
<p>Optional: Adjacency matrix for neighboring drones within radius R</p>

<h4>Action Space (per agent)</h4>
<p>Same action types as HoverAviary, applied per drone:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Shape</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">RPM</td><td style="border: 1px solid #ddd; padding: 8px;">(num_drones, 4)</td><td style="border: 1px solid #ddd; padding: 8px;">Motor speeds per drone</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">ONE_D_RPM</td><td style="border: 1px solid #ddd; padding: 8px;">(num_drones, 1)</td><td style="border: 1px solid #ddd; padding: 8px;">Single RPM per drone</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">PID</td><td style="border: 1px solid #ddd; padding: 8px;">(num_drones, 3)</td><td style="border: 1px solid #ddd; padding: 8px;">Target position per drone</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">VEL</td><td style="border: 1px solid #ddd; padding: 8px;">(num_drones, 4)</td><td style="border: 1px solid #ddd; padding: 8px;">Velocity command per drone</td></tr>
</table>

<h4>Reward Function (Leader-Follower Example)</h4>
<p>Each drone has a different altitude target based on its index:</p>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
# Leader (drone 0): target z=0.5m
reward_0 = -||[0, 0, 0.5] - pos_0||^2

# Follower (drone 1): track leader's altitude
reward_1 = -0.5 * (z_1 - z_0)^2

# Total reward
total_reward = sum(reward_i for all drones)
</pre>

<h4>Downwash Effect</h4>
<p>
When drones fly at different altitudes, the upper drone's propeller wash
reduces lift on the lower drone. This is modeled as:
</p>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
W = k_D1 * (r_P / 4*delta_z)^2 * exp(-0.5 * (sqrt(delta_x^2 + delta_y^2) / (k_D2*delta_z + k_D3))^2)
</pre>
<p>where delta_x, delta_y, delta_z are distances between drones.</p>

<h4>Episode End Conditions</h4>
<ul>
    <li><strong>Termination</strong>: All drones reach targets within tolerance</li>
    <li><strong>Truncation</strong>: Episode timeout (8 seconds)</li>
    <li><strong>Truncation</strong>: Any drone exceeds position bounds</li>
</ul>

<h4>Usage Example (2 Drones)</h4>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;">
import numpy as np
from gym_pybullet_drones.envs import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

env = MultiHoverAviary(
    num_drones=2,
    obs=ObservationType.KIN,
    act=ActionType.ONE_D_RPM
)

obs, info = env.reset()
for _ in range(240):
    # Action shape: (2, 1) for 2 drones with 1D RPM action
    action = np.random.uniform(-1, 1, (2, 1))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
</pre>

<h4>RLlib Multi-Agent Training</h4>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;">
# Central critic with 2 action models
# Critic inputs: 25 features (combined observations)
# Actor inputs: 12 features per drone
# Hidden layers: 2 x 256 with tanh activation

from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig()
config.multi_agent(
    policies={{"drone_0", "drone_1"}},
    policy_mapping_fn=lambda agent_id, *args: agent_id,
)
</pre>

<h4>MARL Challenges</h4>
<ul>
    <li><strong>Credit Assignment</strong>: Disentangling individual contributions to team reward</li>
    <li><strong>Non-Stationarity</strong>: Other agents' policies change during training</li>
    <li><strong>Coordination</strong>: Avoiding collisions while reaching targets</li>
    <li><strong>Physical Coupling</strong>: Downwash creates dynamic interdependencies</li>
</ul>

<h4>Supported MARL Algorithms</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Algorithm</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Library</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">MAPPO</td><td style="border: 1px solid #ddd; padding: 8px;">RLlib</td><td style="border: 1px solid #ddd; padding: 8px;">Centralized Critic</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">MADDPG</td><td style="border: 1px solid #ddd; padding: 8px;">RLlib</td><td style="border: 1px solid #ddd; padding: 8px;">Centralized Critic</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">QMIX</td><td style="border: 1px solid #ddd; padding: 8px;">RLlib</td><td style="border: 1px solid #ddd; padding: 8px;">Value Decomposition</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Independent PPO</td><td style="border: 1px solid #ddd; padding: 8px;">RLlib/SB3</td><td style="border: 1px solid #ddd; padding: 8px;">Decentralized</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/utiasDSL/gym-pybullet-drones" target="_blank">GitHub Repository</a></li>
    <li><a href="https://utiasdsl.github.io/gym-pybullet-drones" target="_blank">Documentation</a></li>
    <li>Paper: Panerati, J., et al. (2021). Learning to Fly - a Gym Environment with PyBullet Physics for RL of Multi-agent Quadcopter Control.</li>
</ul>
"""


# Backward compatibility constant
MULTIHOVER_AVIARY_HTML = get_multihover_aviary_html("multihover-aviary-v0")

__all__ = [
    "get_multihover_aviary_html",
    "MULTIHOVER_AVIARY_HTML",
]
