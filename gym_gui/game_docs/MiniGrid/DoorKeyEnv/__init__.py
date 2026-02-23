"""Documentation for MiniGrid DoorKey environments."""
from __future__ import annotations


def get_doorkey_html(env_id: str) -> str:
    """Generate DoorKey HTML documentation for a specific variant."""
    # Extract grid size from env_id (e.g., "MiniGrid-DoorKey-5x5-v0" -> "5×5")
    size = "8×8"  # default
    if "5x5" in env_id:
        size = "5×5"
        description = "This tiny map is perfect for fast curriculum starts and initial learning."
    elif "6x6" in env_id:
        size = "6×6"
        description = "This intermediate-sized map provides a balance between simplicity and challenge."
    elif "16x16" in env_id:
        size = "16×16"
        description = "This long-horizon environment provides a sparse reward challenge."
    else:  # 8x8
        size = "8×8"
        description = "This standard benchmark is a classic test-bed for curiosity and curriculum learning."
    
    return f"""
<h2>{env_id}</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> MiniGrid (Gymnasium) -- Single-agent grid navigation.
<a href="https://minigrid.farama.org/environments/minigrid/" target="_blank">Documentation</a>
</p>

<p>Collect the yellow key, unlock a door, and reach the goal in a <strong>{size}</strong> room split by a wall. {description}</p>

<h4>Available Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Grid Size</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MiniGrid-DoorKey-5x5-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">5×5</td><td style="border: 1px solid #ddd; padding: 8px;">Tiny map for fast curriculum starts</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MiniGrid-DoorKey-6x6-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">6×6</td><td style="border: 1px solid #ddd; padding: 8px;">Intermediate difficulty</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MiniGrid-DoorKey-8x8-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">8×8</td><td style="border: 1px solid #ddd; padding: 8px;">Standard benchmark</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MiniGrid-DoorKey-16x16-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">16×16</td><td style="border: 1px solid #ddd; padding: 8px;">Long-horizon sparse reward challenge</td></tr>
</table>

<h4>Observation Space</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Component</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>image</code></td><td style="border: 1px solid #ddd; padding: 8px;">Box(0, 255, (7,7,3), uint8)</td><td style="border: 1px solid #ddd; padding: 8px;">7×7 RGB grid observation</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>direction</code></td><td style="border: 1px solid #ddd; padding: 8px;">Discrete(4)</td><td style="border: 1px solid #ddd; padding: 8px;">0=right, 1=down, 2=left, 3=up</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>mission</code></td><td style="border: 1px solid #ddd; padding: 8px;">MissionSpace</td><td style="border: 1px solid #ddd; padding: 8px;">"use the key to open the door and then get to the goal"</td></tr>
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
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pick up key</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Unused</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Open door</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q or <em>(no key)</em></td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td><td style="border: 1px solid #ddd; padding: 8px;">No-op</td></tr>
</table>

<h4>Rewards & Episode End</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Success Reward</td><td style="border: 1px solid #ddd; padding: 8px;"><code>1 - 0.9 × (step_count / max_steps)</code> (×10 multiplier)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Failure Reward</td><td style="border: 1px solid #ddd; padding: 8px;">0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">Agent reaches goal tile</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Truncation</td><td style="border: 1px solid #ddd; padding: 8px;">max_steps timeout (default: 100)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://minigrid.farama.org/environments/minigrid/doorkey/" target="_blank">Official Documentation</a></li>
    <li><a href="https://github.com/Farama-Foundation/Minigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


# For backward compatibility - generic HTML
MINIGRID_DOORKEY_HTML = get_doorkey_html("MiniGrid-DoorKey-8x8-v0")

__all__ = ["MINIGRID_DOORKEY_HTML", "get_doorkey_html"]
