"""Documentation for MiniGrid Lava Gap environments."""
from __future__ import annotations


def get_lavagap_html(env_id: str) -> str:
    """Generate Lava Gap HTML documentation for a specific variant."""
    size = "7×7"
    desc = "standard benchmark"
    
    if "S5" in env_id:
        size = "5×5"
        desc = "compact challenge"
    elif "S6" in env_id:
        size = "6×6"
        desc = "intermediate difficulty"
    else:  # S7
        size = "7×7"
        desc = "standard benchmark"
    
    return f"""
<h2>{env_id}</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> MiniGrid (Gymnasium) -- Single-agent grid navigation.
<a href="https://minigrid.farama.org/environments/minigrid/" target="_blank">Documentation</a>
</p>

<p>The agent must reach the green goal square at the opposite corner of a <strong>{size}</strong> room, passing through a narrow gap in a vertical strip of deadly lava.
Touching the lava terminates the episode with zero reward. This {desc} is useful for studying safety and safe exploration.</p>

<h4>Available Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Grid Size</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MiniGrid-LavaGapS5-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">5×5</td><td style="border: 1px solid #ddd; padding: 8px;">Compact challenge</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MiniGrid-LavaGapS6-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">6×6</td><td style="border: 1px solid #ddd; padding: 8px;">Intermediate difficulty</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MiniGrid-LavaGapS7-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">7×7</td><td style="border: 1px solid #ddd; padding: 8px;">Standard benchmark</td></tr>
</table>

<h4>Mission</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Obstacle Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Mission String</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Lava</td><td style="border: 1px solid #ddd; padding: 8px;">"avoid the lava and get to the green goal square"</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Other</td><td style="border: 1px solid #ddd; padding: 8px;">"find the opening and get to the green goal square"</td></tr>
</table>

<h4>Observation Space</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Component</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>image</code></td><td style="border: 1px solid #ddd; padding: 8px;">Box(0, 255, (7,7,3), uint8)</td><td style="border: 1px solid #ddd; padding: 8px;">7×7 RGB grid, (OBJECT_IDX, COLOR_IDX, STATE)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>direction</code></td><td style="border: 1px solid #ddd; padding: 8px;">Discrete(4)</td><td style="border: 1px solid #ddd; padding: 8px;">0=right, 1=down, 2=left, 3=up</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>mission</code></td><td style="border: 1px solid #ddd; padding: 8px;">MissionSpace</td><td style="border: 1px solid #ddd; padding: 8px;">Task description</td></tr>
</table>

<h4>Action Space (Discrete(7))</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #e8f5e9;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Use</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A or Left</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Turn left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D or Right</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Turn right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W or Up</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Move forward</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Unused</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Unused</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Unused</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q or <em>(no key)</em></td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td><td style="border: 1px solid #ddd; padding: 8px;">No-op</td></tr>
</table>

<h4>Rewards & Episode End</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Success Reward</td><td style="border: 1px solid #ddd; padding: 8px;"><code>1 - 0.9 * (step_count / max_steps)</code> (×10 multiplier)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Lava Contact</td><td style="border: 1px solid #ddd; padding: 8px;">0 (episode terminates)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Failure</td><td style="border: 1px solid #ddd; padding: 8px;">0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">Agent reaches goal or touches lava</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Truncation</td><td style="border: 1px solid #ddd; padding: 8px;">max_episode_steps timeout</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://minigrid.farama.org/environments/minigrid/LavaGapEnv/" target="_blank">Official Documentation</a></li>
    <li><a href="https://github.com/Farama-Foundation/Minigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


# For backward compatibility
MINIGRID_LAVAGAP_HTML = get_lavagap_html("MiniGrid-LavaGapS7-v0")

__all__ = ["MINIGRID_LAVAGAP_HTML", "get_lavagap_html"]
