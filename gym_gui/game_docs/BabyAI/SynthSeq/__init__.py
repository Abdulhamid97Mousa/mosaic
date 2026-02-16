"""Documentation for BabyAI SynthSeq environment."""
from __future__ import annotations


def get_synthseq_html(env_id: str = "BabyAI-SynthSeq-v0") -> str:
    """Generate SynthSeq HTML documentation."""
    return f"""
<h2>{env_id}</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px;">
Part of the <strong>BabyAI</strong> environment family. 
<a href="https://minigrid.farama.org/environments/babyai/">BabyAI Overview</a>
</p>

<h3>Description</h3>
<p>Extends SynthLoc with sequential command execution. Combines GoToSeq-style sequencing with spatial language. No implicit unlocking.</p>

<h3>Mission Structure</h3>
<p>Actions connected with 'and', 'then', or 'after you':</p>
<ul>
<li><em>"go to the green key and put the box next to the yellow ball"</em></li>
<li><em>"pick up the red ball then go to the blue door"</em></li>
</ul>

<h3>Competencies</h3>
<p>Maze navigation, unblocking, unlocking, navigation, pickup, placement, door opening, spatial reasoning, sequential execution</p>

<h3>Registered Variants</h3>
<ul>
<li><strong>BabyAI-SynthSeq-v0</strong></li>
</ul>

<h3>Observation Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Component</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Space</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>image</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(0, 255, (7, 7, 3), uint8)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">7×7 grid RGB observation (partially observable view)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>direction</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>Discrete(4)</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Agent facing direction (0=right, 1=down, 2=left, 3=up)</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>mission</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>MissionSpace</code></td>
    <td style="border: 1px solid #ddd; padding: 8px;">Task instruction as natural language string</td>
  </tr>
</table>

<h3>Action Space</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #e8f5e9;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Key</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Action</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">ID</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Use</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>A</kbd> or <kbd>←</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">LEFT</td>
    <td style="border: 1px solid #ddd; padding: 8px;">0</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Turn left</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>D</kbd> or <kbd>→</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td>
    <td style="border: 1px solid #ddd; padding: 8px;">1</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Turn right</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>W</kbd> or <kbd>↑</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td>
    <td style="border: 1px solid #ddd; padding: 8px;">2</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Move forward</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>Space</kbd> or <kbd>G</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td>
    <td style="border: 1px solid #ddd; padding: 8px;">3</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Pick up an object</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>H</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">DROP</td>
    <td style="border: 1px solid #ddd; padding: 8px;">4</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Drop the object being carried</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>E</kbd> or <kbd>Enter</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td>
    <td style="border: 1px solid #ddd; padding: 8px;">5</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Open doors or toggle objects</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;"><kbd>Q</kbd></td>
    <td style="border: 1px solid #ddd; padding: 8px;">DONE</td>
    <td style="border: 1px solid #ddd; padding: 8px;">6</td>
    <td style="border: 1px solid #ddd; padding: 8px;">Complete the task</td>
  </tr>
</table>

<h3>Rewards & Episode End</h3>
<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
  <tr style="background-color: #f0f0f0;">
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Condition</th>
    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Success (all sequential tasks completed)</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>1 - 0.9 × (step_count / max_steps)</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Failure (timeout)</td>
    <td style="border: 1px solid #ddd; padding: 8px;"><code>0</code></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">Episode terminates</td>
    <td style="border: 1px solid #ddd; padding: 8px;">On task completion or max_steps reached</td>
  </tr>
</table>

<h3>References</h3>
<ul>
<li><strong>Official Documentation:</strong> <a href="https://minigrid.farama.org/environments/babyai/SynthSeq/">BabyAI SynthSeq</a></li>
<li><strong>GitHub:</strong> <a href="https://github.com/Farama-Foundation/Minigrid">Minigrid Repository</a></li>
</ul>
"""


BABYAI_SYNTHSEQ_HTML = get_synthseq_html()

__all__ = ["BABYAI_SYNTHSEQ_HTML", "get_synthseq_html"]
