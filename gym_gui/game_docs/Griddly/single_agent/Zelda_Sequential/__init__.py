"""
Zelda Sequential environment documentation for MOSAIC.
"""

GRIDDLY_ZELDA_SEQUENTIAL_HTML = """
<h2>Zelda Sequential</h2>

<p>Port of GVGAI Zelda. Pick up keys to reach doors in the correct order. Avoid enemies. Sequential key-door progression.</p>

<h3>Environment ID</h3>
<table style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Environment</th>
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">ID</th>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;">Zelda Sequential</td>
        <td style="padding: 8px; border: 1px solid #ddd;"><code>GDY-Zelda-Sequential-v0</code></td>
    </tr>
</table>

<h3>Observation Space</h3>
<table style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Component</th>
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Space</th>
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Description</th>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;">Image</td>
        <td style="padding: 8px; border: 1px solid #ddd;"><code>Box(0, 255, (H×24, W×24, 3), uint8)</code></td>
        <td style="padding: 8px; border: 1px solid #ddd;">RGB pixel array from global sprite observer. Dimensions scale with grid size and 24px tiles.</td>
    </tr>
</table>

<h3>Action Space (Discrete(5))</h3>
<table style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #d4edda;">
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Key</th>
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Action</th>
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">ID</th>
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Use</th>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;">Space</td>
        <td style="padding: 8px; border: 1px solid #ddd;">NOOP</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0</td>
        <td style="padding: 8px; border: 1px solid #ddd;">Do nothing</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;">W / ↑</td>
        <td style="padding: 8px; border: 1px solid #ddd;">UP</td>
        <td style="padding: 8px; border: 1px solid #ddd;">1</td>
        <td style="padding: 8px; border: 1px solid #ddd;">Move up</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;">S / ↓</td>
        <td style="padding: 8px; border: 1px solid #ddd;">DOWN</td>
        <td style="padding: 8px; border: 1px solid #ddd;">2</td>
        <td style="padding: 8px; border: 1px solid #ddd;">Move down</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;">A / ←</td>
        <td style="padding: 8px; border: 1px solid #ddd;">LEFT</td>
        <td style="padding: 8px; border: 1px solid #ddd;">3</td>
        <td style="padding: 8px; border: 1px solid #ddd;">Move left</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;">D / →</td>
        <td style="padding: 8px; border: 1px solid #ddd;">RIGHT</td>
        <td style="padding: 8px; border: 1px solid #ddd;">4</td>
        <td style="padding: 8px; border: 1px solid #ddd;">Move right</td>
    </tr>
</table>

<h3>Levels</h3>
<p>3 levels with grid sizes from 8×8 to 13×10.</p>

<h3>Rewards & Episode End</h3>
<table style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Condition</th>
        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Reward</th>
    </tr>
    <tr><td style='padding: 8px; border: 1px solid #ddd;'>Reach goal</td><td style='padding: 8px; border: 1px solid #ddd;'>+1 (episode ends)</td></tr>
    <tr><td style='padding: 8px; border: 1px solid #ddd;'>Hit by enemy</td><td style='padding: 8px; border: 1px solid #ddd;'>-1 (episode ends)</td></tr>
    <tr><td style='padding: 8px; border: 1px solid #ddd;'>Each step</td><td style='padding: 8px; border: 1px solid #ddd;'>0</td></tr>
</table>

<h3>References</h3>
<ul>
    <li><a href="https://griddly.readthedocs.io/">Griddly Documentation</a></li>
    <li><a href="https://github.com/Bam4d/Griddly">Griddly GitHub Repository</a></li>
</ul>
"""

__all__ = ["GRIDDLY_ZELDA_SEQUENTIAL_HTML"]
