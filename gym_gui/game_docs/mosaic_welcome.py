"""MOSAIC Project Welcome Message.

This module provides the welcome HTML displayed in the Game Info panel
when the GUI launches, before any environment is selected.

MOSAIC: Multi-Agent Orchestration System with Adaptive Intelligent Control for Heterogeneous Agent Workloads
"""

MOSAIC_WELCOME_HTML = """
<table width="100%" cellpadding="15" cellspacing="0" style="background-color: #5b4b8a; border-radius: 8px;">
<tr><td>
    <h1 style="color: white; margin: 0; letter-spacing: 3px; font-size: 28px;">MOSAIC</h1>
    <p style="color: #e0d8f0; margin: 5px 0 0 0; font-size: 12px;">
        Multi-Agent Orchestration System with Adaptive Intelligent Control for Heterogeneous Agent Workloads
    </p>
</td></tr>
</table>

<p style="color: #444; font-size: 13px; line-height: 1.6; margin-top: 15px;">
A unified platform that orchestrates diverse agents, paradigms, and workers
to create cohesive intelligent systems — like tiles in a mosaic forming a complete picture.
</p>

<h3 style="color: #333; margin-top: 15px; margin-bottom: 10px;">Supported Frameworks</h3>
<table width="100%" cellpadding="8" cellspacing="4">
<tr>
    <td style="background-color: #f5f0ff; border-left: 4px solid #5b4b8a; width: 33%;">
        <b style="color: #5b4b8a;">Gymnasium</b><br/>
        <span style="color: #666; font-size: 11px;">Standard RL environment API</span><br/>
        <a href="https://gymnasium.farama.org/" style="color: #1565c0; font-size: 10px;">Docs</a> |
        <a href="https://github.com/Farama-Foundation/Gymnasium" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
    <td style="background-color: #e8f5e9; border-left: 4px solid #2e7d32; width: 33%;">
        <b style="color: #2e7d32;">PettingZoo</b><br/>
        <span style="color: #666; font-size: 11px;">Multi-agent environments (AEC & Parallel)</span><br/>
        <a href="https://pettingzoo.farama.org/" style="color: #1565c0; font-size: 10px;">Docs</a> |
        <a href="https://github.com/Farama-Foundation/PettingZoo" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
    <td style="background-color: #e0f7fa; border-left: 4px solid #00bcd4; width: 33%;">
        <b style="color: #00838f;">Ray RLlib</b><br/>
        <span style="color: #666; font-size: 11px;">Scalable RL library</span><br/>
        <a href="https://docs.ray.io/en/latest/rllib/" style="color: #1565c0; font-size: 10px;">Docs</a> |
        <a href="https://github.com/ray-project/ray" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
</tr>
<tr>
    <td style="background-color: #e0f2f1; border-left: 4px solid #00838f; width: 33%;">
        <b style="color: #00695c;">MuJoCo</b><br/>
        <span style="color: #666; font-size: 11px;">Physics simulation for robotics</span><br/>
        <a href="https://mujoco.org/" style="color: #1565c0; font-size: 10px;">Docs</a> |
        <a href="https://github.com/google-deepmind/mujoco" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
    <td style="background-color: #f3e5f5; border-left: 4px solid #7b1fa2; width: 33%;">
        <b style="color: #7b1fa2;">Godot Engine</b><br/>
        <span style="color: #666; font-size: 11px;">Free & open source 2D/3D game engine</span><br/>
        <a href="https://docs.godotengine.org/en/stable/" style="color: #1565c0; font-size: 10px;">Docs</a> |
        <a href="https://github.com/godotengine/godot" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
    <td style="background-color: #efebe9; border-left: 4px solid #37474f; width: 33%;">
        <b style="color: #37474f;">CleanRL</b><br/>
        <span style="color: #666; font-size: 11px;">Single-file RL implementations</span><br/>
        <a href="https://docs.cleanrl.dev/" style="color: #1565c0; font-size: 10px;">Docs</a> |
        <a href="https://github.com/vwxyzjn/cleanrl" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
</tr>
<tr>
    <td style="background-color: #e3f2fd; border-left: 4px solid #1976d2; width: 33%;">
        <b style="color: #1976d2;">XuanCe</b><br/>
        <span style="color: #666; font-size: 11px;">Comprehensive DRL library (PyTorch/TF/MS)</span><br/>
        <a href="https://xuance.readthedocs.io/" style="color: #1565c0; font-size: 10px;">Docs</a> |
        <a href="https://github.com/agi-brain/xuance" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
    <td style="background-color: #fce4ec; border-left: 4px solid #e91e63; width: 33%;">
        <b style="color: #c2185b;">Jumanji</b><br/>
        <span style="color: #666; font-size: 11px;">JAX-based logic puzzles (2048, Sudoku, etc.)</span><br/>
        <a href="https://github.com/instadeepai/jumanji" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://instadeepai.github.io/jumanji/" style="color: #1565c0; font-size: 10px;">Docs</a>
    </td>
    <td style="background-color: #f5f5f5; width: 33%;"></td>
</tr>
</table>

<h3 style="color: #333; margin-top: 15px; margin-bottom: 10px;">Research Environments</h3>
<table width="100%" cellpadding="8" cellspacing="4">
<tr>
    <td style="background-color: #fff5f5; border-left: 4px solid #8b0000; width: 33%;">
        <b style="color: #8b0000;">NetHack (NLE)</b><br/>
        <span style="color: #666; font-size: 11px;">Procedurally generated roguelike for hard exploration</span><br/>
        <a href="https://github.com/facebookresearch/nle" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
    <td style="background-color: #fff8f0; border-left: 4px solid #ff6600; width: 33%;">
        <b style="color: #ff6600;">MiniHack</b><br/>
        <span style="color: #666; font-size: 11px;">Customizable sandbox built on NLE</span><br/>
        <a href="https://github.com/facebookresearch/minihack" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://minihack.readthedocs.io/" style="color: #1565c0; font-size: 10px;">Docs</a>
    </td>
    <td style="background-color: #f0fff0; border-left: 4px solid #228b22; width: 33%;">
        <b style="color: #228b22;">Crafter</b><br/>
        <span style="color: #666; font-size: 11px;">Open world survival for agent capabilities</span><br/>
        <a href="https://github.com/danijar/crafter" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://danijar.com/project/crafter/" style="color: #1565c0; font-size: 10px;">Project</a>
    </td>
</tr>
<tr>
    <td style="background-color: #fff3e0; border-left: 4px solid #ef6c00; width: 33%;">
        <b style="color: #ef6c00;">MiniGrid</b><br/>
        <span style="color: #666; font-size: 11px;">Minimalistic gridworld environments</span><br/>
        <a href="https://minigrid.farama.org/" style="color: #1565c0; font-size: 10px;">Docs</a> |
        <a href="https://github.com/Farama-Foundation/Minigrid" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
    <td style="background-color: #ffebee; border-left: 4px solid #c62828; width: 33%;">
        <b style="color: #c62828;">ViZDoom</b><br/>
        <span style="color: #666; font-size: 11px;">Doom-based visual RL platform</span><br/>
        <a href="https://vizdoom.farama.org/" style="color: #1565c0; font-size: 10px;">Docs</a> |
        <a href="https://github.com/Farama-Foundation/ViZDoom" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
    <td style="background-color: #eceff1; border-left: 4px solid #455a64; width: 33%;">
        <b style="color: #455a64;">ALE / Atari</b><br/>
        <span style="color: #666; font-size: 11px;">Arcade Learning Environment</span><br/>
        <a href="https://github.com/Farama-Foundation/Arcade-Learning-Environment" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
</tr>
<tr>
    <td style="background-color: #f3e5f5; border-left: 4px solid #9c27b0; width: 33%;">
        <b style="color: #9c27b0;">BabaIsAI</b><br/>
        <span style="color: #666; font-size: 11px;">Rule manipulation puzzles for LLM reasoning</span><br/>
        <a href="https://github.com/nacloos/baba-is-ai" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://arxiv.org/abs/2407.13729" style="color: #1565c0; font-size: 10px;">Paper</a>
    </td>
    <td style="background-color: #e8eaf6; border-left: 4px solid #3f51b5; width: 33%;">
        <b style="color: #3f51b5;">OpenSpiel</b><br/>
        <span style="color: #666; font-size: 11px;">Collection of games for research (DeepMind)</span><br/>
        <a href="https://github.com/google-deepmind/open_spiel" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://openspiel.readthedocs.io/en/latest/" style="color: #1565c0; font-size: 10px;">Docs</a>
    </td>
    <td style="background-color: #e0f7fa; border-left: 4px solid #0097a7; width: 33%;">
        <b style="color: #0097a7;">Procgen</b><br/>
        <span style="color: #666; font-size: 11px;">16 procedurally generated games for generalization</span><br/>
        <a href="https://github.com/openai/procgen" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://arxiv.org/abs/1912.01588" style="color: #1565c0; font-size: 10px;">Paper</a>
    </td>
</tr>
<tr>
    <td style="background-color: #e1f5fe; border-left: 4px solid #0288d1; width: 33%;">
        <b style="color: #0288d1;">BabyAI</b><br/>
        <span style="color: #666; font-size: 11px;">Language-grounded instruction following</span><br/>
        <a href="https://github.com/mila-iqia/babyai" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://arxiv.org/abs/1810.08272" style="color: #1565c0; font-size: 10px;">Paper</a>
    </td>
    <td style="background-color: #fbe9e7; border-left: 4px solid #d84315; width: 33%;">
        <b style="color: #d84315;">TextWorld</b><br/>
        <span style="color: #666; font-size: 11px;">Text-based games framework (Microsoft)</span><br/>
        <a href="https://github.com/microsoft/TextWorld" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://arxiv.org/abs/1806.11532" style="color: #1565c0; font-size: 10px;">Paper</a>
    </td>
    <td style="background-color: #e8f5e9; border-left: 4px solid #43a047; width: 33%;">
        <b style="color: #43a047;">MultiGrid</b><br/>
        <span style="color: #666; font-size: 11px;">Multi-agent MiniGrid extensions (Soccer, Collect)</span><br/>
        <a href="https://github.com/ArnaudFickinger/gym-multigrid" style="color: #1565c0; font-size: 10px;">GitHub</a>
    </td>
</tr>
<tr>
    <td style="background-color: #fff3e0; border-left: 4px solid #ef6c00; width: 33%;">
        <b style="color: #ef6c00;">Melting Pot</b><br/>
        <span style="color: #666; font-size: 11px;">Multi-agent social scenarios (50+ substrates, up to 16 agents, DeepMind)</span><br/>
        <a href="https://github.com/google-deepmind/meltingpot" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://shimmy.farama.org/environments/meltingpot/" style="color: #1565c0; font-size: 10px;">Docs</a>
    </td>
    <td style="background-color: #fff8e1; border-left: 4px solid #f57c00; width: 33%;">
        <b style="color: #f57c00;">Overcooked-AI</b><br/>
        <span style="color: #666; font-size: 11px;">2-agent cooperative cooking for human-AI coordination (UC Berkeley CHAI)</span><br/>
        <a href="https://github.com/HumanCompatibleAI/overcooked_ai" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://arxiv.org/abs/1910.05789" style="color: #1565c0; font-size: 10px;">Paper</a> |
        <a href="https://bair.berkeley.edu/blog/2019/10/21/coordination/" style="color: #1565c0; font-size: 10px;">Blog</a>
    </td>
    <td style="background-color: #e0f2f1; border-left: 4px solid #00695c; width: 33%;">
        <b style="color: #00695c;">PyBullet Drones</b><br/>
        <span style="color: #666; font-size: 11px;">Multi-agent quadcopter control with realistic physics (University of Toronto)</span><br/>
        <a href="https://github.com/utiasDSL/gym-pybullet-drones" style="color: #1565c0; font-size: 10px;">GitHub</a> |
        <a href="https://arxiv.org/abs/2103.02142" style="color: #1565c0; font-size: 10px;">Paper</a>
    </td>
</tr>
</table>

<h3 style="color: #333; margin-top: 15px; margin-bottom: 10px;">Core Features</h3>
<table width="100%" cellpadding="8" cellspacing="4">
<tr>
    <td style="background-color: #f5f5f5; border-left: 4px solid #5b4b8a;">
        <b style="color: #333;">Multi-Paradigm Support</b><br/>
        <span style="color: #666; font-size: 12px;">Single-agent, multi-agent (AEC/Parallel), and hybrid environments</span>
    </td>
</tr>
<tr>
    <td style="background-color: #f5f5f5; border-left: 4px solid #2e7d32;">
        <b style="color: #333;">Agent Integration</b><br/>
        <span style="color: #666; font-size: 12px;">Human, RL (CleanRL, Ray RLlib, XuanCe), and LLM agents</span>
    </td>
</tr>
<tr>
    <td style="background-color: #f5f5f5; border-left: 4px solid #1565c0;">
        <b style="color: #333;">Policy Mapping</b><br/>
        <span style="color: #666; font-size: 12px;">Assign different policies to different agents with flexible configuration</span>
    </td>
</tr>
</table>

<table width="100%" cellpadding="10" cellspacing="0" style="background-color: #e8f5e9; margin-top: 12px;">
<tr><td>
    <b style="color: #2e7d32; font-size: 13px;">Getting Started</b>
    <ol style="margin: 8px 0 0 0; padding-left: 20px; color: #444; font-size: 12px;">
        <li><b>Select Environment</b> — Choose from the dropdown in the sidebar</li>
        <li><b>Configure Settings</b> — Adjust parameters for your environment</li>
        <li><b>Load Environment</b> — Click "Load" to initialize</li>
        <li><b>Start Training/Playing</b> — Begin your experiment</li>
    </ol>
</td></tr>
</table>

<p style="color: #888; font-size: 11px; margin-top: 12px; text-align: center;">
    <i>Select an environment from the sidebar to see detailed documentation.</i>
</p>
"""

MULTI_KEYBOARD_HTML = """
<h4>Multi-Keyboard Setup</h4>
<p>Multiple humans can play together using separate USB keyboards (Linux X11 recommended).</p>
<ol>
  <li><strong>Connect Keyboards</strong>: Plug in USB keyboards</li>
  <li><strong>Assign in Human Control tab</strong>: Use the Keyboard Assignment widget to map each keyboard to an agent</li>
  <li><strong>Start Game</strong>: Each player uses their assigned keyboard</li>
</ol>

<table border="1" style="border-collapse: collapse; margin: 10px 0; font-size: 0.9em;">
  <tr>
    <th style="padding: 4px;">Platform</th>
    <th style="padding: 4px;">Support</th>
  </tr>
  <tr><td style="padding: 4px;">Linux X11</td><td style="padding: 4px; color: green;">✓ Full</td></tr>
  <tr><td style="padding: 4px;">Windows/macOS/Wayland</td><td style="padding: 4px; color: orange;">⚠ Limited</td></tr>
</table>
"""

__all__ = ["MOSAIC_WELCOME_HTML", "MULTI_KEYBOARD_HTML"]
