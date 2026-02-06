"""MultiGrid game documentation module.

gym-multigrid is a multi-agent extension of MiniGrid for training cooperative
and competitive multi-agent RL policies. All agents act simultaneously each step.

Supported Packages:
    - Old gym-multigrid (ArnaudFickinger): Soccer, Collect
      Repository: https://github.com/ArnaudFickinger/gym-multigrid
      Location: 3rd_party/gym-multigrid/

    - New multigrid (INI): 13 environments including Empty, RedBlueDoors, LockedHallway, etc.
      Repository: https://github.com/ini/multigrid
      Location: 3rd_party/multigrid-ini/

Legacy Environments:
    - MultiGrid-Soccer-v0: 4 agents (2v2 soccer), zero-sum competitive
    - MultiGrid-Collect-v0: 3 agents collecting balls, cooperative/competitive

INI Environments:
    - Empty (6 variants): Training grounds with configurable sizes and spawn positions
    - RedBlueDoors (2 variants): Sequential door-opening puzzle
    - LockedHallway (3 variants): Multi-room navigation with keys
    - BlockedUnlockPickup: Complex pickup with obstacles
    - Playground: Diverse object interactions
"""

from __future__ import annotations


def get_multigrid_html(env_id: str) -> str:
    """Generate MultiGrid HTML documentation for a specific variant.

    Args:
        env_id: Environment identifier (e.g., "MultiGrid-Soccer-v0")

    Returns:
        HTML string containing environment documentation.
    """
    # Legacy gym-multigrid environments
    if "Soccer" in env_id:
        return _get_soccer_html()
    elif "Collect" in env_id:
        return _get_collect_html()
    # INI multigrid environments
    elif "RedBlueDoors" in env_id:
        return _get_redbluedoors_html(env_id)
    elif "Empty" in env_id:
        return _get_empty_html(env_id)
    elif "LockedHallway" in env_id:
        return _get_lockedhallway_html(env_id)
    elif "BlockedUnlockPickup" in env_id:
        return _get_blockedunlockpickup_html()
    elif "Playground" in env_id:
        return _get_playground_html()
    else:
        return _get_overview_html()


def _get_overview_html() -> str:
    """Return overview HTML for MultiGrid family."""
    return """
<h2>gym-multigrid</h2>

<p>
<strong>gym-multigrid</strong> is a multi-agent extension of MiniGrid designed for training
cooperative and competitive multi-agent reinforcement learning policies.
Unlike single-agent MiniGrid, all agents act <em>simultaneously</em> each step.
</p>

<h4>Key Features</h4>
<ul>
    <li><strong>Multi-Agent</strong>: 2-4 agents controlled by Operators</li>
    <li><strong>Simultaneous Actions</strong>: All agents act at once (PARALLEL paradigm)</li>
    <li><strong>Cooperative/Competitive</strong>: Team-based or individual rewards</li>
    <li><strong>Compatible with RLlib</strong>: Native multi-agent policy training</li>
</ul>

<h4>Available Environments</h4>

<h5>Legacy gym-multigrid (ArnaudFickinger)</h5>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Agents</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Soccer-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">4</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Zero-sum</td>
        <td style="border: 1px solid #ddd; padding: 8px;">2v2 soccer, score goals</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Collect-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Competitive</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Collect balls of your color</td>
    </tr>
</table>

<h5>INI multigrid (Modern)</h5>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Agents</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Empty-5x5-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Empty 5x5 grid training ground</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Empty-Random-5x5-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Empty 5x5 with random spawn</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Empty-6x6-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Empty 6x6 grid training ground</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Empty-Random-6x6-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Empty 6x6 with random spawn</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Empty-8x8-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Empty 8x8 grid (default size)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Empty-16x16-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Large 16x16 empty grid</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-RedBlueDoors-6x6-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Open red then blue door (6x6)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-RedBlueDoors-8x8-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Open red then blue door (8x8)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-LockedHallway-2Rooms-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Navigate through 2 locked rooms</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-LockedHallway-4Rooms-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Navigate through 4 locked rooms</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-LockedHallway-6Rooms-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Navigate through 6 locked rooms</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-BlockedUnlockPickup-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Unlock door and pickup object</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MultiGrid-Playground-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1+</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cooperative</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Diverse objects and interactions</td>
    </tr>
</table>

<h4>Stepping Paradigm: SIMULTANEOUS</h4>
<p>
Unlike turn-based games (PettingZoo AEC), gym-multigrid uses <strong>simultaneous stepping</strong>:
</p>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
# All agents submit actions at once
actions = [agent0_action, agent1_action, agent2_action, agent3_action]
obs_list, rewards_list, done, info = env.step(actions)
</pre>

<h4>⚠️ API Differences (Important for Reproducibility!)</h4>
<p>
The two MultiGrid packages use <strong>different Python APIs</strong> due to the
<a href="https://gymnasium.farama.org/introduction/migration_guide/" target="_blank">Gym → Gymnasium migration</a>.
This is critical for <strong>reproducible research</strong>!
</p>

<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #fff3e0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Feature</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Legacy (Soccer, Collect)</th>
        <th style="border: 1px solid #ddd; padding: 8px;">INI (Empty, BlockedUnlockPickup, etc.)</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Framework</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">OpenAI Gym (2020)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Gymnasium (2023+)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Seeding</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>env.seed(42)</code> then <code>env.reset()</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>env.reset(seed=42)</code></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Reset return</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>obs</code> (single value)</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>(obs, info)</code> tuple</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Step return</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>(obs, reward, done, info)</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>(obs, reward, terminated, truncated, info)</code></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Actions</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">8 (includes STILL=0)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">7 (no STILL, LEFT=0)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Status</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Unmaintained</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Actively maintained</td>
    </tr>
</table>

<p style="background-color: #e8f5e9; padding: 10px; border-radius: 4px; margin-top: 10px;">
<strong>📝 Note:</strong> MOSAIC handles these API differences automatically. Use the shared seed in the
Operators tab for reproducible experiments across both environment types.
</p>

<h4>Multi-Keyboard Support (NEW!)</h4>
<p>
MultiGrid now supports <strong>multi-human gameplay</strong> with multiple USB keyboards!
Up to 4 players can control 4 agents simultaneously using separate keyboards.
</p>

<h5>Setup Requirements</h5>
<ul>
    <li><strong>Control Mode</strong>: Set to <code>HUMAN_ONLY</code></li>
    <li><strong>Input Mode</strong>: Set to <code>State-Based (Real-time)</code></li>
    <li><strong>Keyboard Assignment</strong>: Assign each USB keyboard to an agent</li>
</ul>

<h5>Keyboard Controls (Per Player)</h5>
<p>
<strong>⚠️ Important:</strong> Legacy and INI environments use <em>different action IDs</em> for the same keys!
</p>

<h6>Legacy gym-multigrid (Soccer, Collect) - 8 Actions</h6>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #fff3e0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><em>(no key)</em></td><td style="border: 1px solid #ddd; padding: 8px;">STILL (idle)</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A or ←</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D or →</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W or ↑</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>7</strong></td></tr>
</table>

<h6>INI multigrid (Empty, BlockedUnlockPickup, etc.) - 7 Actions</h6>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #e3f2fd;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A or ←</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D or →</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W or ↑</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q or <em>(no key)</em></td><td style="border: 1px solid #ddd; padding: 8px;">DONE (idle)</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td></tr>
</table>

<p style="background-color: #ffecb3; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>💡 Note:</strong> INI environments have no STILL action. When no key is pressed, agents send <code>DONE (6)</code> as the idle action.
</p>

<p>
See <code>docs/MULTI_KEYBOARD_SETUP.md</code> for detailed setup instructions.
</p>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ArnaudFickinger/gym-multigrid" target="_blank">GitHub Repository</a></li>
    <li>Based on: <a href="https://minigrid.farama.org/" target="_blank">MiniGrid</a></li>
</ul>
"""


def _get_soccer_html() -> str:
    """Return HTML documentation for Soccer environment."""
    return """
<h2>MultiGrid-Soccer-v0</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>🔧 API:</strong> Legacy gym-multigrid (OpenAI Gym) — Uses <code>env.seed(N)</code> then <code>env.reset()</code>.
<a href="https://github.com/ArnaudFickinger/gym-multigrid" target="_blank">GitHub</a>
</p>

<p>
A 4-player (2v2) soccer game where two teams compete to score goals.
Agents must pick up the ball, navigate to the opponent's goal, and drop it to score.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">15 x 10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Number of Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2 per team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1 (Red)</td><td style="border: 1px solid #ddd; padding: 8px;">Agent 0, Agent 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2 (Green)</td><td style="border: 1px solid #ddd; padding: 8px;">Agent 2, Agent 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball</td><td style="border: 1px solid #ddd; padding: 8px;">1 neutral ball</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>10,000</strong> (fixed)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (scoring penalizes opponents)</td></tr>
</table>

<h4>Observation Space</h4>
<p>
Each agent receives a partial observation of the grid:
<code>Box(low=0, high=255, shape=(view_size, view_size, 6), dtype=uint8)</code>
</p>
<p>The 6 channels encode: object type, color, state, carrying object, carrying color, direction.</p>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> - Same actions for all agents:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">STILL</td><td style="border: 1px solid #ddd; padding: 8px;">Do nothing</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Turn left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Turn right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move forward</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;">Pick up ball (or steal from agent)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;">Drop ball (scores if at goal)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;">Toggle object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;">Signal completion</td></tr>
</table>

<h4>Rewards</h4>
<p>Zero-sum rewards when a team scores:</p>
<ul>
    <li><strong>Scoring team</strong>: +1.0 (shared by both teammates)</li>
    <li><strong>Opposing team</strong>: -1.0 (shared by both opponents)</li>
</ul>

<h4>Game Mechanics</h4>
<ul>
    <li><strong>Ball Pickup</strong>: Walk to ball and use PICKUP action</li>
    <li><strong>Ball Stealing</strong>: Can PICKUP from agent carrying ball</li>
    <li><strong>Scoring</strong>: DROP ball at opponent's goal</li>
    <li><strong>Teamwork</strong>: Can DROP ball to pass to teammate, they PICKUP</li>
</ul>

<h4>Multi-Keyboard Gameplay</h4>
<p>
Play <strong>4-player local co-op</strong> using multiple USB keyboards!
Each player controls one agent using their own keyboard simultaneously.
</p>

<h5>Setup</h5>
<ol>
    <li>Set <strong>Control Mode</strong> to <code>HUMAN_ONLY</code></li>
    <li>Set <strong>Input Mode</strong> to <code>State-Based (Real-time)</code></li>
    <li>Connect 4 USB keyboards</li>
    <li>Assign each keyboard to an agent (Agent 0-3) in the Keyboard Assignment widget</li>
    <li>Press keys on each keyboard to control your agent!</li>
</ol>

<h5>Keyboard Mapping (Legacy MultiGrid)</h5>
<p>
For human control, the following keyboard keys map to action IDs:
</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #fff3e0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Soccer Use</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><em>(no key)</em></td><td style="border: 1px solid #ddd; padding: 8px;">STILL</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Idle (stand still)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A or ←</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Rotate counter-clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D or →</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Rotate clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W or ↑</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Move in facing direction</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Grab ball (or steal from opponent!)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Drop ball (scores at goal, or pass to teammate)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Toggle object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>7</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Signal completion</td></tr>
</table>
<p style="background-color: #ffecb3; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>💡 Note:</strong> Legacy MultiGrid uses <code>STILL (0)</code> as the idle action when no key is pressed.
This is different from INI MultiGrid which uses <code>DONE (6)</code>.
</p>

<p>
<strong>Strategy Tip:</strong> Team Red (Agent 0 & 1) vs Team Green (Agent 2 & 3).
Coordinate with your teammate to steal the ball and score goals!
</p>

<h4>Training Configurations</h4>
<p>Example Ray RLlib multi-agent config:</p>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
multi_agent:
  policies:
    team_red: [agent_0, agent_1]   # Share policy
    team_green: [agent_2, agent_3] # Share policy
  policy_mapping_fn: lambda agent_id: "team_red" if agent_id < 2 else "team_green"
</pre>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ArnaudFickinger/gym-multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_collect_html() -> str:
    """Return HTML documentation for Collect environment."""
    return """
<h2>MultiGrid-Collect-v0</h2>

<p>
A 3-player ball collection game where agents compete to pick up balls.
Each agent is assigned a color and can collect balls of any color,
but the reward structure can be competitive or cooperative.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Number of Agents</td><td style="border: 1px solid #ddd; padding: 8px;">3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agent 0</td><td style="border: 1px solid #ddd; padding: 8px;">Red</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agent 1</td><td style="border: 1px solid #ddd; padding: 8px;">Green</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agent 2</td><td style="border: 1px solid #ddd; padding: 8px;">Blue</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">5 neutral balls</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>10,000</strong> (fixed)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (collecting penalizes others)</td></tr>
</table>

<h4>Observation Space</h4>
<p>
Each agent receives a partial observation of the grid:
<code>Box(low=0, high=255, shape=(view_size, view_size, 6), dtype=uint8)</code>
</p>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> - Same as Soccer environment.</p>

<h4>Rewards</h4>
<p>Zero-sum rewards when an agent collects a ball:</p>
<ul>
    <li><strong>Collecting agent</strong>: +1.0</li>
    <li><strong>Other agents</strong>: -1.0 (split among non-collectors)</li>
</ul>

<h4>Game Mechanics</h4>
<ul>
    <li><strong>Ball Collection</strong>: Walk to ball and use PICKUP action</li>
    <li><strong>Balls disappear</strong>: Once collected, balls are removed from grid</li>
    <li><strong>Competition</strong>: Race to collect before opponents</li>
    <li><strong>Episode ends</strong>: All balls collected or max steps reached</li>
</ul>

<h4>Cooperative Variant</h4>
<p>
The environment can be configured for cooperative play where all agents
share rewards. Configure via environment kwargs:
</p>
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
env = CollectGameEnv(zero_sum=False)  # Shared rewards
</pre>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ArnaudFickinger/gym-multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_redbluedoors_html(env_id: str) -> str:
    """Return HTML documentation for RedBlueDoors environment."""
    if "6x6" in env_id:
        size = "6x6"
        max_steps = 720
    else:
        size = "8x8"
        max_steps = 1280
    return f"""
<h2>MultiGrid-RedBlueDoors-{size}-v0</h2>

<p>
A cooperative multi-agent puzzle where agents must open doors in sequence:
first the <strong>red door</strong>, then the <strong>blue door</strong>.
Opening them in the wrong order results in failure.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">{size}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1-4 (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Task</td><td style="border: 1px solid #ddd; padding: 8px;">Open red door, then blue door</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Mission</td><td style="border: 1px solid #ddd; padding: 8px;">"open the red door then the blue door"</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>{max_steps}</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Reward</td><td style="border: 1px solid #ddd; padding: 8px;">1 - 0.9 * (step_count / max_steps)</td></tr>
</table>

<h4>Success/Failure Conditions</h4>
<ul>
    <li><strong>Success</strong>: Any agent opens blue door while red door is already open</li>
    <li><strong>Failure</strong>: Any agent opens blue door while red door is still closed</li>
    <li><strong>Truncation</strong>: Maximum steps reached</li>
</ul>

<h4>Key Mechanics</h4>
<ul>
    <li>Agents must use <code>TOGGLE</code> action to open doors</li>
    <li>Doors can be opened by any agent (coordination required)</li>
    <li>Red door must be opened first (order matters!)</li>
    <li>Cooperative reward: all agents get same reward on success</li>
</ul>

<h4>Training Tips</h4>
<ul>
    <li>Teaches <strong>temporal reasoning</strong>: understanding action sequences</li>
    <li>Requires <strong>coordination</strong>: agents must agree on who opens which door</li>
    <li>Good for testing <strong>curriculum learning</strong>: start with single agent</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://ini.github.io/docs/multigrid/multigrid/multigrid.envs.redbluedoors.html" target="_blank">Official Documentation</a></li>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_empty_html(env_id: str) -> str:
    """Return HTML documentation for Empty environment."""
    if "5x5" in env_id:
        size, grid_size, max_steps = "5x5", "5", 100
    elif "6x6" in env_id:
        size, grid_size, max_steps = "6x6", "6", 144
    elif "16x16" in env_id:
        size, grid_size, max_steps = "16x16", "16", 1024
    else:
        size, grid_size, max_steps = "8x8", "8", 256

    is_random = "Random" in env_id
    spawn_mode = "random positions" if is_random else "fixed corner position"

    return f"""
<h2>MultiGrid-Empty-{size}-v0</h2>

<p>
An empty {size} grid environment with no obstacles or objects.
Agents spawn at {spawn_mode} and must navigate to the green goal square.
This is the simplest MultiGrid environment, ideal for testing basic navigation.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">{grid_size} x {grid_size}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1-4 (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Spawn Position</td><td style="border: 1px solid #ddd; padding: 8px;">{spawn_mode.capitalize()}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Task</td><td style="border: 1px solid #ddd; padding: 8px;">Navigate to green goal square</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Obstacles</td><td style="border: 1px solid #ddd; padding: 8px;">None (empty grid)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>{max_steps}</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Reward</td><td style="border: 1px solid #ddd; padding: 8px;">1 - 0.9 * (step_count / max_steps)</td></tr>
</table>

<h4>Use Cases</h4>
<ul>
    <li><strong>Baseline Testing</strong>: Verify basic navigation works</li>
    <li><strong>Algorithm Debugging</strong>: Simple environment for troubleshooting</li>
    <li><strong>Curriculum Start</strong>: Begin training before adding complexity</li>
    <li><strong>Performance Benchmarking</strong>: Measure optimal navigation</li>
</ul>

<h4>Training Tips</h4>
<ul>
    <li>Should solve quickly (< 1M steps for simple RL algorithms)</li>
    <li>If failing here, check observation/action space configuration</li>
    <li>Random spawn variant tests generalization</li>
    <li>Larger grids ({grid_size}x{grid_size}) test long-horizon planning</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://ini.github.io/docs/multigrid/multigrid/multigrid.envs.empty.html" target="_blank">Official Documentation</a></li>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_lockedhallway_html(env_id: str) -> str:
    """Return HTML documentation for LockedHallway environment."""
    if "2Rooms" in env_id:
        num_rooms = 2
        max_steps = 400
    elif "4Rooms" in env_id:
        num_rooms = 4
        max_steps = 800
    else:
        num_rooms = 6
        max_steps = 1200

    return f"""
<h2>MultiGrid-LockedHallway-{num_rooms}Rooms-v0</h2>

<p>
A multi-room navigation puzzle with {num_rooms} rooms connected by locked doors.
Agents must find keys to unlock doors and navigate through all rooms to reach the goal.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Number of Rooms</td><td style="border: 1px solid #ddd; padding: 8px;">{num_rooms}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1-4 (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Keys</td><td style="border: 1px solid #ddd; padding: 8px;">{num_rooms - 1} (one per locked door)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Task</td><td style="border: 1px solid #ddd; padding: 8px;">Unlock all doors and reach goal</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Complexity</td><td style="border: 1px solid #ddd; padding: 8px;">{"Low" if num_rooms == 2 else "Medium" if num_rooms == 4 else "High"}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>{max_steps}</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Reward</td><td style="border: 1px solid #ddd; padding: 8px;">1 - 0.9 * (step_count / max_steps)</td></tr>
</table>

<h4>Key Mechanics</h4>
<ul>
    <li><strong>Key Collection</strong>: Use <code>PICKUP</code> to collect keys</li>
    <li><strong>Door Unlocking</strong>: Use <code>TOGGLE</code> on locked door while carrying matching key</li>
    <li><strong>Key-Door Matching</strong>: Each key unlocks specific colored door</li>
    <li><strong>Sequential Progress</strong>: Must unlock doors in order to progress</li>
</ul>

<h4>Multi-Agent Cooperation</h4>
<ul>
    <li>Agents can <strong>pass keys</strong> by dropping and picking up</li>
    <li>One agent can collect key, another can unlock door</li>
    <li>Splitting work across rooms can speed up completion</li>
    <li>Requires <strong>implicit coordination</strong> without communication</li>
</ul>

<h4>Difficulty Progression</h4>
<ul>
    <li><strong>2 Rooms</strong>: Simple linear progression, good for initial training</li>
    <li><strong>4 Rooms</strong>: Medium complexity, tests planning and memory</li>
    <li><strong>6 Rooms</strong>: High complexity, requires long-horizon reasoning</li>
</ul>

<h4>Training Tips</h4>
<ul>
    <li>Start with 2 rooms, gradually increase complexity</li>
    <li>May need <strong>recurrent policies</strong> (LSTM/GRU) for 6 rooms</li>
    <li>Consider <strong>intrinsic motivation</strong> (curiosity) to encourage exploration</li>
    <li>Multi-agent versions benefit from <strong>communication channels</strong></li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://ini.github.io/docs/multigrid/multigrid/multigrid.envs.locked_hallway.html" target="_blank">Official Documentation</a></li>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_blockedunlockpickup_html() -> str:
    """Return HTML documentation for BlockedUnlockPickup environment."""
    return """
<h2>MultiGrid-BlockedUnlockPickup-v0</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>🔬 API:</strong> INI multigrid (Gymnasium) — Uses <code>env.reset(seed=N)</code> for reproducibility.
<a href="https://github.com/ini/multigrid" target="_blank">GitHub</a>
</p>

<p>
The objective is to <strong>pick up a box</strong> which is placed in another room,
behind a <strong>locked door</strong>. The door is also <strong>blocked by a ball</strong>
which must be moved before the door can be unlocked.
</p>

<p>
Agents must learn to: move the ball → pick up the key → open the door → pick up the box.
The standard setting is <strong>cooperative</strong>, where all agents receive the reward
when the task is completed.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Layout</td><td style="border: 1px solid #ddd; padding: 8px;">2 rooms (1 row × 2 columns)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Room Size</td><td style="border: 1px solid #ddd; padding: 8px;">6×6 (default)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1+ (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Mission</td><td style="border: 1px solid #ddd; padding: 8px;">"pick up the {color} box"</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Joint Reward</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (all agents share reward)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>576</strong> (formula: 16 × room_size², room_size=6)</td></tr>
</table>

<h4>Objects in Environment</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Object</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Location</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Purpose</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">🔑 Key</td><td style="border: 1px solid #ddd; padding: 8px;">Left room</td><td style="border: 1px solid #ddd; padding: 8px;">Unlocks the door (color matches door)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">⚽ Ball</td><td style="border: 1px solid #ddd; padding: 8px;">Blocking door</td><td style="border: 1px solid #ddd; padding: 8px;">Must be moved before door can be toggled</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">🚪 Locked Door</td><td style="border: 1px solid #ddd; padding: 8px;">Between rooms</td><td style="border: 1px solid #ddd; padding: 8px;">Separates agents from target box</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">📦 Box</td><td style="border: 1px solid #ddd; padding: 8px;">Right room</td><td style="border: 1px solid #ddd; padding: 8px;">Target object to pick up (success!)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">🤖 Agents</td><td style="border: 1px solid #ddd; padding: 8px;">Left room</td><td style="border: 1px solid #ddd; padding: 8px;">Start in the left room</td></tr>
</table>

<h4>Task Sequence</h4>
<ol>
    <li><strong>Move the Ball</strong>: Push or pick up the ball blocking the door</li>
    <li><strong>Pick Up Key</strong>: Use <code>PICKUP</code> action on the key</li>
    <li><strong>Unlock Door</strong>: Face door and use <code>TOGGLE</code> while carrying key</li>
    <li><strong>Enter Right Room</strong>: Move through the now-open door</li>
    <li><strong>Pick Up Box</strong>: Use <code>PICKUP</code> on the target box → Success!</li>
</ol>

<h4>Reward Structure</h4>
<p>
<code>reward = 1 - 0.9 × (step_count / max_steps)</code>
</p>
<ul>
    <li><strong>Success</strong>: Any agent picks up the correct box → reward between 0.1 and 1.0</li>
    <li><strong>Failure</strong>: Timeout (max steps exceeded) → reward = 0</li>
    <li><strong>Joint reward</strong>: All agents receive the same reward on success</li>
</ul>

<h4>Action Space</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Turn left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Turn right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move forward</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;">Pick up an object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;">Drop an object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;">Toggle / activate an object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;">Done completing task</td></tr>
</table>

<h4>Keyboard Mapping (INI MultiGrid)</h4>
<p>
For human control, the following keyboard keys map to action IDs:
</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #e3f2fd;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Use in This Environment</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A or ←</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Turn to face a different direction</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D or →</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Turn to face a different direction</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W or ↑</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Move into adjacent cell</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pick up key, ball, or box</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Drop carried object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Unlock door (while carrying key)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q or <em>(no key)</em></td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Idle (wait/no action)</td></tr>
</table>
<p style="background-color: #ffecb3; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>💡 Tip:</strong> INI MultiGrid has <em>no STILL action</em>. When no key is pressed, <code>DONE (6)</code> is sent as the idle action.
This is different from Legacy MultiGrid which uses <code>STILL (0)</code>.
</p>

<h4>Multi-Agent Strategies</h4>
<ul>
    <li><strong>Division of Labor</strong>: One agent moves ball, another gets key</li>
    <li><strong>Key Relay</strong>: Pass key to agent closer to the door</li>
    <li><strong>Parallel Progress</strong>: Move ball and collect key simultaneously</li>
</ul>

<h4>Training Tips</h4>
<ul>
    <li>Use <strong>curriculum learning</strong>: start with simpler environments (Empty, then LockedHallway)</li>
    <li>Consider <strong>shaped rewards</strong> for intermediate sub-goals</li>
    <li><strong>Hindsight Experience Replay (HER)</strong> helps with sparse rewards</li>
    <li>May need <strong>recurrent policies</strong> (LSTM/GRU) for memory of sub-task progress</li>
</ul>

<h4>Termination</h4>
<ul>
    <li><strong>Success</strong>: Any agent picks up the correct box</li>
    <li><strong>Timeout</strong>: Maximum steps reached</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://ini.github.io/docs/multigrid/multigrid/multigrid.envs.blockedunlockpickup.html" target="_blank">Official Documentation</a></li>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


def _get_playground_html() -> str:
    """Return HTML documentation for Playground environment."""
    return """
<h2>MultiGrid-Playground-v0</h2>

<p>
A diverse environment filled with various objects and interactive elements.
Acts as a sandbox for testing agent behaviors with multiple object types,
colors, and interaction mechanics all in one place.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1-4 (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Objects</td><td style="border: 1px solid #ddd; padding: 8px;">Keys, doors, boxes, balls, and more</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Purpose</td><td style="border: 1px solid #ddd; padding: 8px;">Exploratory sandbox, testing interactions</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Task</td><td style="border: 1px solid #ddd; padding: 8px;">Flexible - depends on goal configuration</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>100</strong></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Complexity</td><td style="border: 1px solid #ddd; padding: 8px;">Variable (many possible interactions)</td></tr>
</table>

<h4>Available Objects</h4>
<ul>
    <li><strong>Doors</strong>: Can be opened (if unlocked) or require keys</li>
    <li><strong>Keys</strong>: Match colored doors for unlocking</li>
    <li><strong>Balls</strong>: Can be picked up, carried, and dropped</li>
    <li><strong>Boxes</strong>: Can contain other objects, can be opened/closed</li>
    <li><strong>Walls</strong>: Static obstacles blocking movement</li>
    <li><strong>Goals</strong>: Target locations for success</li>
</ul>

<h4>Use Cases</h4>
<ul>
    <li><strong>Behavior Testing</strong>: See how agents interact with diverse objects</li>
    <li><strong>Transfer Learning</strong>: Pre-train on varied interactions</li>
    <li><strong>Multi-Task Learning</strong>: Handle multiple objective types</li>
    <li><strong>Emergent Behaviors</strong>: Observe unexpected agent strategies</li>
</ul>

<h4>Interaction Mechanics</h4>
<ul>
    <li><code>PICKUP</code>: Pick up keys, balls, or other portable objects</li>
    <li><code>DROP</code>: Place carried object on ground</li>
    <li><code>TOGGLE</code>: Open/close doors, interact with boxes</li>
    <li><code>FORWARD</code>: Move in facing direction (blocked by walls/closed doors)</li>
</ul>

<h4>Training Tips</h4>
<ul>
    <li><strong>Exploration Bonus</strong>: Reward discovering new object types</li>
    <li><strong>Multi-Goal RL</strong>: Handle multiple possible objectives</li>
    <li><strong>Goal Conditioning</strong>: Specify what to accomplish via instruction</li>
    <li>Good for testing <strong>generalization</strong> across task types</li>
</ul>

<h4>References</h4>
<ul>
    <li><a href="https://ini.github.io/docs/multigrid/multigrid/multigrid.envs.playground.html" target="_blank">Official Documentation</a></li>
    <li><a href="https://github.com/ini/multigrid" target="_blank">GitHub Repository</a></li>
</ul>
"""


# Pre-generated HTML constants for convenience
MULTIGRID_OVERVIEW_HTML = _get_overview_html()
MULTIGRID_SOCCER_HTML = _get_soccer_html()
MULTIGRID_COLLECT_HTML = _get_collect_html()

__all__ = [
    "get_multigrid_html",
    "MULTIGRID_OVERVIEW_HTML",
    "MULTIGRID_SOCCER_HTML",
    "MULTIGRID_COLLECT_HTML",
]
