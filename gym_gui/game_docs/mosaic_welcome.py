"""MOSAIC Project Welcome Message.

This module provides the welcome HTML displayed in the Game Info panel
when the GUI launches, before any environment is selected.

MOSAIC: Multi-paradigm Orchestration System for Agent Integration & Composition
"""

MOSAIC_WELCOME_HTML = """
<table width="100%" cellpadding="15" cellspacing="0" style="background-color: #5b4b8a; border-radius: 8px;">
<tr><td>
    <h1 style="color: white; margin: 0; letter-spacing: 3px; font-size: 28px;">MOSAIC</h1>
    <p style="color: #e0d8f0; margin: 5px 0 0 0; font-size: 12px;">
        Multi-paradigm Orchestration System for Agent Integration &amp; Composition
    </p>
</td></tr>
</table>

<p style="color: #444; font-size: 13px; line-height: 1.6; margin-top: 15px;">
A unified platform that orchestrates diverse agents, paradigms, and workers
to create cohesive intelligent systems — like tiles in a mosaic forming a complete picture.
</p>

<h3 style="color: #333; margin-top: 20px; margin-bottom: 10px;">Supported Paradigms</h3>
<table cellpadding="4" cellspacing="3">
<tr>
    <td style="background-color: #5b4b8a; color: white; padding: 4px 10px; border-radius: 10px; font-size: 11px;">Gymnasium</td>
    <td style="background-color: #2e7d32; color: white; padding: 4px 10px; border-radius: 10px; font-size: 11px;">PettingZoo AEC</td>
    <td style="background-color: #1565c0; color: white; padding: 4px 10px; border-radius: 10px; font-size: 11px;">PettingZoo Parallel</td>
    <td style="background-color: #ef6c00; color: white; padding: 4px 10px; border-radius: 10px; font-size: 11px;">MiniGrid</td>
</tr>
<tr>
    <td style="background-color: #c62828; color: white; padding: 4px 10px; border-radius: 10px; font-size: 11px;">ViZDoom</td>
    <td style="background-color: #00838f; color: white; padding: 4px 10px; border-radius: 10px; font-size: 11px;">MuJoCo MPC</td>
    <td style="background-color: #7b1fa2; color: white; padding: 4px 10px; border-radius: 10px; font-size: 11px;">Godot UE</td>
    <td style="background-color: #455a64; color: white; padding: 4px 10px; border-radius: 10px; font-size: 11px;">ALE Atari</td>
</tr>
</table>

<h3 style="color: #333; margin-top: 20px; margin-bottom: 10px;">Core Features</h3>

<table width="100%" cellpadding="10" cellspacing="5">
<tr>
    <td style="background-color: #f5f5f5; border-left: 4px solid #5b4b8a;">
        <b style="color: #333;">Multi-Paradigm Support</b><br/>
        <span style="color: #666; font-size: 12px;">Seamlessly switch between single-agent, multi-agent (AEC/Parallel), and hybrid environments</span>
    </td>
</tr>
<tr>
    <td style="background-color: #f5f5f5; border-left: 4px solid #2e7d32;">
        <b style="color: #333;">Agent Integration</b><br/>
        <span style="color: #666; font-size: 12px;">Human, RL (CleanRL, Ray), BDI (SPADE, Jason), and future LLM agents in the same framework</span>
    </td>
</tr>
<tr>
    <td style="background-color: #f5f5f5; border-left: 4px solid #1565c0;">
        <b style="color: #333;">Policy Mapping</b><br/>
        <span style="color: #666; font-size: 12px;">Assign different policies to different agents with flexible configuration</span>
    </td>
</tr>
<tr>
    <td style="background-color: #f5f5f5; border-left: 4px solid #ef6c00;">
        <b style="color: #333;">3D Engine Support</b><br/>
        <span style="color: #666; font-size: 12px;">MuJoCo MPC for robotics, Godot for game environments, AirSim planned</span>
    </td>
</tr>
</table>

<table width="100%" cellpadding="12" cellspacing="0" style="background-color: #e8f5e9; margin-top: 15px;">
<tr><td>
    <b style="color: #2e7d32; font-size: 14px;">Getting Started</b>
    <ol style="margin: 10px 0 0 0; padding-left: 20px; color: #444; font-size: 13px;">
        <li><b>Select Environment</b> — Choose from the dropdown in the sidebar</li>
        <li><b>Configure Settings</b> — Adjust parameters for your environment</li>
        <li><b>Load Environment</b> — Click "Load" to initialize</li>
        <li><b>Start Training/Playing</b> — Begin your experiment</li>
    </ol>
</td></tr>
</table>

<p style="color: #888; font-size: 11px; margin-top: 15px; text-align: center;">
    <i>Select an environment from the sidebar to see detailed documentation.</i>
</p>
"""

__all__ = ["MOSAIC_WELCOME_HTML"]
