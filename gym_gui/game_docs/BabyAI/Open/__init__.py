"""Documentation for BabyAI Open environment."""
from __future__ import annotations


def get_open_html(env_id: str = "BabyAI-Open-v0") -> str:
    """Generate Open HTML documentation."""
    return f"""
<h2>{env_id}</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> BabyAI (Gymnasium) -- Single-agent instructional grid navigation.
<a href="https://minigrid.farama.org/environments/babyai/" target="_blank">Documentation</a>
</p>

<p>Open a door that may be in another room. Requires navigation, exploration, and multi-step planning.</p>

<h4>Mission</h4>
<p><em>"open a {{color}} door"</em> — colors: red, green, blue, purple, yellow, grey</p>

<h4>Registered Variants</h4>
<ul>
<li><strong>BabyAI-Open-v0</strong></li>
</ul>

<h4>Observation Space</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Component</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>image</code></td><td style="border: 1px solid #ddd; padding: 8px;">Box(0, 255, (7,7,3), uint8)</td><td style="border: 1px solid #ddd; padding: 8px;">7×7 RGB grid observation</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>direction</code></td><td style="border: 1px solid #ddd; padding: 8px;">Discrete(4)</td><td style="border: 1px solid #ddd; padding: 8px;">0=right, 1=down, 2=left, 3=up</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>mission</code></td><td style="border: 1px solid #ddd; padding: 8px;">MissionSpace</td><td style="border: 1px solid #ddd; padding: 8px;">Text instruction for the task</td></tr>
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
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pick up object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Drop object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Toggle/activate (open doors)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Complete action</td></tr>
</table>

<h4>Rewards & Episode End</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Success Reward</td><td style="border: 1px solid #ddd; padding: 8px;"><code>1 - 0.9 × (step_count / max_steps)</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Failure Reward</td><td style="border: 1px solid #ddd; padding: 8px;">0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">Agent opens the target door</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Truncation</td><td style="border: 1px solid #ddd; padding: 8px;">max_steps timeout</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://minigrid.farama.org/environments/babyai/Open/" target="_blank">Official Documentation</a></li>
    <li><a href="https://github.com/Farama-Foundation/Minigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


BABYAI_OPEN_HTML = get_open_html()

__all__ = ["BABYAI_OPEN_HTML", "get_open_html"]
