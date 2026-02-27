MOSAIC
======

.. raw:: html

   <a href="https://github.com/Abdulhamid97Mousa/MOSAIC">
        <img alt="GitHub" src="https://img.shields.io/github/stars/Abdulhamid97Mousa/MOSAIC?style=social">
   </a>
   <a href="https://github.com/Abdulhamid97Mousa/mosaic/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
   </a>
   <a href="https://www.python.org/downloads/">
        <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg">
   </a>
   <a href="https://pytorch.org/get-started/locally/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-red">
   </a>
   <a href="https://www.gymlibrary.dev/">
        <img alt="Gymnasium" src="https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue">
   </a>
   <a href="https://pettingzoo.farama.org/">
        <img alt="PettingZoo" src="https://img.shields.io/badge/PettingZoo-%3E%3D1.24.0-blue">
   </a>

.. raw:: html

   <br><br>

**A Unified Platform for Cross-Paradigm Comparison and Evaluation of Homogeneous and Heterogeneous Multi-Agent RL, LLM, VLM, and Human Decision-Makers**

MOSAIC is a visual-first platform that enables researchers to configure, run, and
compare experiments across RL, LLM, VLM, and human decision-makers in the same
multi-agent environment.  Different paradigms like tiles in a mosaic come
together to form a complete picture of agent performance.


.. figure:: _static/figures/A_Full_Architecture.png
   :alt: MOSAIC Platform Overview
   :align: center
   :width: 100%
   :target: documents/architecture/workers/architecture.html

   The architecture shows the
   :doc:`Evaluation Phase <documents/architecture/operators/index>` (operators containing workers),
   :doc:`Training Phase <documents/architecture/workers/architecture>` (TrainerClient, TrainerService, Workers),
   Daemon Process (gRPC Server, RunRegistry, Dispatcher, Broadcasters),
   and :doc:`Worker Processes <documents/architecture/workers/integrated_workers/index>`
   (:doc:`CleanRL <documents/architecture/workers/integrated_workers/CleanRL_Worker/index>`,
   :doc:`XuanCe <documents/architecture/workers/integrated_workers/XuanCe_Worker/index>`,
   :doc:`Ray RLlib <documents/architecture/workers/integrated_workers/RLlib_Worker/index>`,
   :doc:`BALROG <documents/architecture/workers/integrated_workers/BALROG_Worker/index>`).

.. raw:: html

   <br>


MOSAIC provides two evaluation modes designed for reproducibility:

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/ea9ebc18-2216-4fb2-913c-5d354ebea56e" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>Manual Mode</strong> Side-by-side lock-step evaluation with shared seeds.
     See <a href="documents/architecture/operators/index.html">Operators &amp; Evaluation Modes</a>
     and <a href="documents/rendering_tabs/slow_lane.html">Slow Lane (Render View)</a>.
   </p>

- **Manual Mode:** side-by-side comparison where multiple operators step through
  the same environment with shared seeds, letting researchers visually inspect
  decision-making differences between paradigms in real time.

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/a9b3f6f4-661c-492f-b43f-34d7125a6d2e" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>Script Mode:</strong> Automated batch evaluation with deterministic seed sequences.
     See <a href="documents/architecture/operators/architecture.html">IPC Architecture</a>
     and <a href="documents/runtime_logging/index.html">Runtime Logging</a>.
   </p>

- **Script Mode:** automated, long-running evaluation driven by Python scripts
  that define operator configurations, worker assignments, seed sequences, and
  episode counts.  Scripts execute deterministically with no manual intervention,
  producing reproducible telemetry logs (JSONL) for every step and episode.

All evaluation runs share **identical conditions**: same environment seeds, same
observations, and unified telemetry.  Script Mode additionally supports
**procedural seeds** (different seed per episode to test generalization) and
**fixed seeds** (same seed every episode to isolate agent behaviour), with
configurable step pacing for visual inspection or headless batch execution.

| **GitHub**: `https://github.com/Abdulhamid97Mousa/MOSAIC <https://github.com/Abdulhamid97Mousa/MOSAIC>`_

Why MOSAIC?
-----------

Today's AI landscape offers powerful but **fragmented** tools: RL frameworks
(`CleanRL <https://github.com/vwxyzjn/cleanrl>`_,
`RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_,
`XuanCe <https://github.com/agi-brain/xuance>`_),
language models (GPT, Claude), and robotics simulators (MuJoCo).
Each excels in isolation, but **no platform bridges them together**
under a unified, visual-first interface.

**MOSAIC provides:**

- **Visual-First Design**: Configure experiments through an intuitive PyQt6 interface, **Almost no code required**.
- **Heterogeneous Agent Mixing**: Deploy Human(Agent),  RL, and LLM agents in the same environment
- **Resource Management & Quotas**: GPU allocation, queue limits, credit-based backpressure, health monitoring.
- **Per-Agent Policy Binding**: Route each agent to different workers via ``PolicyMappingService``.
- **Worker Lifecycle Orchestration**: Subprocess management with heartbeat monitoring and graceful termination.

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/ded17cdc-f23c-404f-a9f6-074fbe74816c" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>Human vs Human:</strong> Two human players competing via dedicated USB keyboards.
     See <a href="documents/human_control/index.html">Human Control</a>
     and <a href="documents/human_control/multi_keyboard_evdev.html">Multi-Keyboard Support (Evdev)</a>.
   </p>

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/2625a8f8-476c-4171-86cc-a9970cbf1665" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>Random Agents:</strong> Baseline agents across 26 environment families.
     See <a href="documents/architecture/workers/integrated_workers/MOSAIC_Random_Worker/index.html">MOSAIC Random Worker</a>
     and <a href="documents/environments/index.html">Supported Environments</a>.
   </p>

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/f2d79901-a93d-465b-9058-1b9cdabf311a" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>Heterogeneous Multi-Agent Ad-Hoc Teamwork in Adversarial Settings:</strong> Different decision-making paradigms (RL, LLM, Random) competing head-to-head in the same multi-agent environment.
     See <a href="documents/architecture/operators/heterogeneous_decision_maker/index.html">Heterogeneous Decision-Maker</a>.
   </p>

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/2ae1665b-3a57-44be-98a3-4e7223b37628" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>Homogeneous Teams: Random vs LLM:</strong> Two homogeneous teams (all-Random vs all-LLM) competing in the same multi-agent environment.
     See <a href="documents/architecture/operators/homogenous_decision_makers/index.html">Homogeneous Decision-Makers</a>.
   </p>

Agent-Level Interface and Cross-Paradigm Evaluation
-----------------------------------------------------

**Agent-Level Interface.** Existing infrastructure lacks the ability to deploy
agents from different decision-making paradigms within the same environment.
The root cause is an **interface mismatch**: RL agents expect tensor
observations and produce integer actions, while LLM agents expect text prompts
and produce text responses.  MOSAIC addresses this through an *operator
abstraction* that forms an agent-level interface by mapping workers to agents:
each operator, regardless of whether it is backed by an RL policy, an LLM, or
a human, conforms to a minimal unified interface
(``select_action(obs) → action``).  The environment never needs to know what
kind of decision-maker it is communicating with.  This is the agent-side
counterpart to what `Gymnasium <https://gymnasium.farama.org/>`_ did for
environments: Gymnasium standardized the environment interface
(``reset()`` / ``step()``), so any algorithm can interact with any environment;
MOSAIC's Operator Protocol standardizes the agent interface, so any
decision-maker can be plugged into any compatible environment without modifying
either side.

**Cross-Paradigm Evaluation.** Cross-paradigm evaluation is the ability to
deploy decision-makers from *different paradigms* (RL, LLM/VLM, Human,
scripted baselines) within the same multi-agent environment under identical
conditions, and to produce directly comparable results.  Both evaluation modes described above
(:doc:`Manual Mode <documents/architecture/operators/lifecycle>` and
:doc:`Script Mode <documents/architecture/operators/architecture>`) guarantee
that all decision-makers face the same environment states, observations, and
shared seeds, making this the first infrastructure to enable fair, reproducible
cross-paradigm evaluation.

See :doc:`Operator Concept <documents/architecture/operators/concept>` for the
full Agent-Level Interface specification,
:doc:`Heterogeneous Decision-Maker <documents/architecture/operators/heterogeneous_decision_maker/index>`
for the research gap and design rationale, and
:doc:`IPC Architecture <documents/architecture/operators/architecture>` for
Manual Mode and Script Mode implementation details.

Comparison with Existing Frameworks
------------------------------------

Existing frameworks are paradigm-siloed. No prior framework allowed fair,
reproducible, head-to-head comparison between RL agents and LLM agents in the
same multi-agent environment.

*Agent Paradigms*: which decision-maker types are supported.
*Framework*: algorithms can be integrated without source-code modifications.
*Platform GUI*: real-time visualization during execution.
*Cross-Paradigm*: infrastructure for comparing different agent types (e.g., RL
vs. LLM) on identical environment instances with shared random seeds for
reproducible head-to-head evaluation.
Legend: ✓ Supported, ✗ Not supported, ◉ Partial.

.. raw:: html

   <style>
     .cmp-table { width:100%; border-collapse:collapse; margin:1.5em 0; font-size:0.95em; }
     .cmp-table th, .cmp-table td { padding:6px 10px; text-align:center; }
     .cmp-table th { background:#f5f5f5; }
     .cmp-table td:first-child { text-align:left; }
     .cmp-table tbody tr { border-bottom:1px solid #eee; }
     .cmp-table .section-row td { font-style:italic; background:#f8f8f8; padding:8px 10px; }
     .cmp-table .mosaic-row { border-top:2.5px solid #333; background:#eef6ff; font-weight:bold; }
     .cmp-yes { color:#1a7f37; font-size:1.2em; } /* green checkmark */
     .cmp-no  { color:#cf222e; font-size:1.2em; } /* red cross */
     .cmp-part { color:#0969da; font-size:1.1em; } /* blue partial */
   </style>
   <table class="cmp-table">
     <thead>
       <tr style="border-bottom:2px solid #333;">
         <th rowspan="2" style="text-align:left;">System</th>
         <th colspan="3" style="border-bottom:1px solid #aaa;">Agent Paradigms</th>
         <th colspan="2" style="border-bottom:1px solid #aaa;">Infrastructure</th>
         <th style="border-bottom:1px solid #aaa;">Evaluation</th>
       </tr>
       <tr style="border-bottom:2px solid #333;">
         <th>RL</th><th>LLM/VLM</th><th>Human</th>
         <th>Framework</th><th>Platform GUI</th><th>Cross-Paradigm</th>
       </tr>
     </thead>
     <tbody>
       <tr class="section-row"><td colspan="7"><strong>RL Frameworks</strong></td></tr>
       <tr>
         <td>RLlib <a href="#ref1">[1]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>CleanRL <a href="#ref2">[2]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>Tianshou <a href="#ref3">[3]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>Acme <a href="#ref4">[4]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>XuanCe <a href="#ref5">[5]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>OpenRL <a href="#ref6">[6]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>Stable-Baselines3 <a href="#ref7">[7]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>Coach <a href="#ref8">[8]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>BenchMARL <a href="#ref15">[15]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr class="section-row"><td colspan="7"><strong>LLM/VLM Benchmarks</strong></td></tr>
       <tr>
         <td>BALROG <a href="#ref9">[9]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>TextArena <a href="#ref10">[10]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-yes">&#10003;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>GameBench <a href="#ref11">[11]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>lmgame-Bench <a href="#ref12">[12]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>LLM Chess <a href="#ref13">[13]</a></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>LLM-Game-Bench <a href="#ref14">[14]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-part">&#9673;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>AgentBench <a href="#ref16">[16]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>MultiAgentBench <a href="#ref17">[17]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>GAMEBoT <a href="#ref18">[18]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>Collab-Overcooked <a href="#ref19">[19]</a></td>
         <td><span class="cmp-part">&#9673;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>BotzoneBench <a href="#ref20">[20]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr>
         <td>AgentGym <a href="#ref21">[21]</a></td>
         <td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-no">&#10007;</span></td><td><span class="cmp-no">&#10007;</span></td>
       </tr>
       <tr class="mosaic-row">
         <td>MOSAIC (Ours)</td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-yes">&#10003;</span></td>
         <td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-yes">&#10003;</span></td><td><span class="cmp-yes">&#10003;</span></td>
       </tr>
     </tbody>
   </table>
   <p style="font-size:0.85em; color:#555; margin-top:4px;">
     <span class="cmp-yes">&#10003;</span> Supported &nbsp;&nbsp;
     <span class="cmp-no">&#10007;</span> Not supported &nbsp;&nbsp;
     <span class="cmp-part">&#9673;</span> Partial
   </p>


Experimental Configurations
---------------------------

Heterogeneous decision-making enables a systematic ablation matrix for
cross-paradigm research. The following configurations illustrate the design
using 2v2 soccer in :doc:`MOSAIC MultiGrid <documents/environments/mosaic_multigrid>`.
Notation follows the :ref:`paper's appendix <paper-notation>`:
:math:`\pi^{RL}` denotes a solo‑trained RL policy (frozen at evaluation),
:math:`\lambda^{LLM}` an LLM agent, :math:`\rho` a uniform random policy, and
:math:`\nu` a no‑op (null action) policy.

Adversarial Cross‑Paradigm Matchups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Testing how paradigms perform against each other. Each configuration pits two
homogeneous teams against one another.

.. list-table::
   :widths: 20 25 25 30
   :header-rows: 1

   * - Configuration
     - Team A
     - Team B
     - Purpose
   * - **A1** (RL vs RL)
     - :math:`\pi^{RL}_1 + \pi^{RL}_2`
     - :math:`\pi^{RL}_3 + \pi^{RL}_4`
     - Homogeneous RL baseline (ceiling)
   * - **A2** (LLM vs LLM)
     - :math:`\lambda^{LLM}_1 + \lambda^{LLM}_2`
     - :math:`\lambda^{LLM}_3 + \lambda^{LLM}_4`
     - Homogeneous LLM baseline
   * - **A3** (RL vs LLM)
     - :math:`\pi^{RL}_1 + \pi^{RL}_2`
     - :math:`\lambda^{LLM}_1 + \lambda^{LLM}_2`
     - Central cross‑paradigm comparison
   * - **A4** (RL vs Random)
     - :math:`\pi^{RL}_1 + \pi^{RL}_2`
     - :math:`\rho_1 + \rho_2`
     - Sanity check (trained vs random)

Cooperative Heterogeneous Teams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Testing how paradigms work together **within** a team. All RL policies are
trained solo (1v1) and frozen before deployment; LLM agents are zero‑shot.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Configuration
     - Green Team
     - Blue Team
   * - **C1** (Heterogeneous vs Crippled)
     - :math:`\pi^{RL} + \lambda^{LLM}`
     - :math:`\pi^{RL} + \rho`
   * - **C2** (Heterogeneous vs Solo)
     - :math:`\pi^{RL} + \lambda^{LLM}`
     - :math:`\pi^{RL} + \nu`
   * - **C3** (Solo‑pair vs Solo‑pair)
     - :math:`\pi^{RL}_i + \pi^{RL}_j`
     - :math:`\pi^{RL}_k + \pi^{RL}_l`
   * - **C4** (Heterogeneous vs Co‑trained)
     - :math:`\pi^{RL} + \lambda^{LLM}`
     - :math:`\pi^{RL}_{2v2} + \pi^{RL}_{2v2}`


.. admonition:: 1v1‑to‑2v2 Transfer Design – Why Solo Training?
   :class: important


   RL agents are trained as **solo experts in 1v1** (single‑agent environment),
   then deployed as teammates in 2v2 **without any fine‑tuning**. This design
   eliminates the *co‑training confound*: if agents were trained together in
   2v2 via MAPPO self‑play, their policies would encode implicit partner models
   calibrated against another MAPPO agent. Swapping one teammate with an LLM
   would then conflate two effects – the paradigm difference **and** the partner
   mismatch. With 1v1‑trained agents, the RL policy carries **zero partner
   expectations** because it never had a partner, cleanly isolating the paradigm
   variable.

   This is distinct from **zero‑shot coordination (ZSC)** in the ad‑hoc teamwork
   literature. ZSC studies RL agents cooperating with unknown *RL* partners,
   agents that share the same observation and action representations
   (:math:`\mathcal{O} = \mathbb{R}^d`, :math:`\mathcal{A}` discrete). Here we
   study an LLM as an ad‑hoc partner for a frozen RL policy – the partner is not
   only unknown but operates through a fundamentally different paradigm
   (text‑based reasoning vs. learned tensor‑to‑action mapping). The fair
   comparison baseline also changes: in ZSC the reference is a co‑trained
   RL+RL team, while here the appropriate baseline is **C3**: two independently
   trained 1v1 solo experts paired in 2v2, since neither agent was trained with
   any partner.

   For full mathematical details and further configurations, see
   :ref:`Appendix A <paper-appendix>` of the companion paper.

Supported Environment Families
------------------------------

MOSAIC supports **26 environment families** spanning single-agent, multi-agent,
and cooperative/competitive paradigms.  See the full
:doc:`Environment Families <documents/environments/index>` reference for
installation instructions, environment lists, and academic citations.


.. list-table::
   :widths: 28 42 30
   :header-rows: 1

   * - Family
     - Description
     - Example Environments
   * - **Gymnasium**
     - Standard single-agent RL (Toy Text, Classic Control, Box2D, MuJoCo)
     - .. image:: images/envs/gymnasium/cartpole.gif
          :width: 200px
   * - **Atari / ALE**
     - 128 classic Atari 2600 games
     - .. image:: images/envs/atari/atari.gif
          :width: 200px
   * - **MiniGrid**
     - Procedural grid-world navigation
     - .. image:: images/envs/minigrid/minigrid.gif
          :width: 200px
   * - **BabyAI**
     - Language-grounded instruction following
     - .. image:: images/envs/babyai/GoTo.gif
          :width: 200px
   * - **ViZDoom**
     - Doom-based first-person visual RL
     - .. image:: images/envs/vizdoom/vizdoom.gif
          :width: 200px
   * - **MiniHack / NetHack**
     - Roguelike dungeon crawling (NLE)
     - .. image:: images/envs/minihack/minihack.gif
          :width: 200px
   * - **Crafter**
     - Open-world survival benchmark
     - .. image:: images/envs/crafter/crafter.gif
          :width: 200px
   * - **Procgen**
     - 16 procedurally generated environments
     - .. image:: images/envs/procgen/coinrun.gif
          :width: 200px
   * - **BabaIsAI**
     - Rule-manipulation puzzles
     - .. image:: images/envs/babaisai/babaisai.png
          :width: 200px
   * - **Jumanji**
     - JAX-accelerated logic/routing/packing (25 envs)
     - .. image:: images/envs/jumanji/jumanji.gif
          :width: 200px
   * - **PyBullet Drones**
     - Quadcopter physics simulation
     - .. image:: images/envs/pybullet_drones/pybullet_drones.gif
          :width: 200px
   * - **PettingZoo Classic**
     - Turn-based board games (AEC)
     - .. image:: images/envs/pettingzoo/pettingzoo.gif
          :width: 200px
   * - **MOSAIC MultiGrid**
     - Competitive team sports (view_size=3)
     - .. image:: images/envs/mosaic_multigrid/mosaic_multigrid.gif
          :width: 200px
   * - **INI MultiGrid**
     - Cooperative exploration (view_size=7)
     - .. image:: images/envs/multigrid_ini/multigrid_ini.gif
          :width: 200px
   * - **Melting Pot**
     - Social multi-agent scenarios (up to 16 agents)
     - .. image:: images/envs/meltingpot/meltingpot.gif
          :width: 200px
   * - **Overcooked**
     - Cooperative cooking (2 agents)
     - .. image:: images/envs/overcooked/overcooked_layouts.gif
          :width: 200px
   * - **SMAC**
     - StarCraft Multi-Agent Challenge (hand-designed maps)
     - .. image:: images/envs/smac/smac.gif
          :width: 200px
   * - **SMACv2**
     - StarCraft Multi-Agent Challenge v2 (procedural units)
     - .. image:: images/envs/smacv2/smacv2.png
          :width: 200px
   * - **RWARE**
     - Cooperative warehouse delivery
     - .. image:: images/envs/rware/rware.gif
          :width: 200px
   * - **MuJoCo**
     - Continuous-control robotics tasks
     - .. image:: images/envs/mujoco/ant.gif
          :width: 200px

Supported Workers (8)
---------------------

* :doc:`CleanRL <documents/architecture/workers/integrated_workers/CleanRL_Worker/index>`: Single-file RL implementations (PPO, DQN, SAC, TD3, DDPG, C51)
* :doc:`XuanCe <documents/architecture/workers/integrated_workers/XuanCe_Worker/index>`: Modular RL framework with flexible algorithm composition and custom environments.
  Multi-agent algorithms (MAPPO, QMIX, MADDPG, VDN, COMA)
* :doc:`Ray RLlib <documents/architecture/workers/integrated_workers/RLlib_Worker/index>`: RL with distributed training and large-batch optimization (PPO, IMPALA, APPO)
* :doc:`BALROG <documents/architecture/workers/integrated_workers/BALROG_Worker/index>`: LLM/VLM agentic evaluation (GPT-4o, Claude 3, Gemini · NetHack, BabyAI, Crafter)
* :doc:`MOSAIC LLM <documents/architecture/workers/integrated_workers/MOSAIC_LLM_Worker/index>`: Multi-agent LLM with coordination strategies and Theory of Mind (MultiGrid, BabyAI, MeltingPot, PettingZoo)
* :doc:`Chess LLM <documents/architecture/workers/integrated_workers/Chess_LLM_Worker/index>`: LLM chess play with multi-turn dialog (PettingZoo Chess)
* :doc:`MOSAIC Human Worker <documents/architecture/workers/integrated_workers/MOSAIC_Human_Worker/index>`: Human-in-the-loop play via keyboard for any Gymnasium-compatible environment (MiniGrid, Crafter, Chess, NetHack)
* :doc:`MOSAIC Random Worker <documents/architecture/workers/integrated_workers/MOSAIC_Random_Worker/index>`: Baseline agents with random, no-op, and cycling action behaviours across all 26 environment families

Citing MOSAIC
-------------

If you use MOSAIC in your research, please cite the following paper:

.. code-block:: bibtex

   @article{mousa2026mosaic,
     title   = {{MOSAIC}: A Unified Platform for Cross-Paradigm Comparison
                and Evaluation of Homogeneous and Heterogeneous Multi-Agent
                {RL}, {LLM}, {VLM}, and Human Decision-Makers},
     author  = {Mousa, Abdulhamid M. and Daoui, Zahra and Khajiev, Rakhmonberdi
                and Azzabi, Jalaledin M. and Mousa, Abdulkarim M. and Liu, Ming},
     year    = {2026},
     url     = {https://github.com/Abdulhamid97Mousa/MOSAIC},
     note    = {Available at \url{https://github.com/Abdulhamid97Mousa/MOSAIC}}
   }

References
----------

.. raw:: html

   <p style="font-size:0.9em; color:#555;">
     <span id="ref1">[1]</span> E. Liang et al., "RLlib: Abstractions for Distributed Reinforcement Learning," <em>ICML</em>, 2018.<br>
     <span id="ref2">[2]</span> S. Huang et al., "CleanRL: High-quality Single-file Implementations of Deep RL Algorithms," <em>JMLR</em>, 2022.<br>
     <span id="ref3">[3]</span> J. Weng et al., "Tianshou: A Highly Modularized Deep RL Library," <em>JMLR</em>, 2022.<br>
     <span id="ref4">[4]</span> M. Hoffman et al., "Acme: A Research Framework for Distributed RL," <em>arXiv:2006.00979</em>, 2020.<br>
     <span id="ref5">[5]</span> W. Liu et al., "XuanCe: A Comprehensive and Unified Deep RL Library," <em>arXiv:2312.16248</em>, 2023.<br>
     <span id="ref6">[6]</span> S. Huang et al., "OpenRL: A Unified Reinforcement Learning Framework," <em>arXiv:2312.16189</em>, 2023.<br>
     <span id="ref7">[7]</span> A. Raffin et al., "Stable-Baselines3: Reliable RL Implementations," <em>JMLR</em>, 2021.<br>
     <span id="ref8">[8]</span> I. Caspi et al., "Reinforcement Learning Coach," 2017.<br>
     <span id="ref9">[9]</span> D. Paglieri et al., "BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games," <em>arXiv:2411.13543</em>, 2024.<br>
     <span id="ref10">[10]</span> G. De Magistris et al., "TextArena," 2025.<br>
     <span id="ref11">[11]</span> D. Costarelli et al., "GameBench: Evaluating Strategic Reasoning Abilities of LLM Agents," <em>arXiv:2406.06613</em>, 2024.<br>
     <span id="ref12">[12]</span> Y. Huang et al., "lmgame-Bench: Evaluating LLMs on Game-Theoretic Decision-Making," 2025.<br>
     <span id="ref13">[13]</span> M. Saplin, "LLM Chess," 2025.<br>
     <span id="ref14">[14]</span> J. Guo et al., "LLM-Game-Bench: Evaluating LLM Reasoning through Game-Playing," 2024.<br>
     <span id="ref15">[15]</span> M. Bettini et al., "BenchMARL: Benchmarking Multi-Agent Reinforcement Learning," <em>JMLR</em>, 2024. arXiv:2312.01472.<br>
     <span id="ref16">[16]</span> X. Liu et al., "AgentBench: Evaluating LLMs as Agents," <em>ICLR</em>, 2024. arXiv:2308.03688.<br>
     <span id="ref17">[17]</span> K. Zhu et al., "MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents," <em>ACL</em>, 2025. arXiv:2503.01935.<br>
     <span id="ref18">[18]</span> Y. Lin et al., "GAMEBoT: Transparent Assessment of LLM Reasoning in Games," <em>ACL</em>, 2025. arXiv:2412.13602.<br>
     <span id="ref19">[19]</span> H. Sun et al., "Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents," <em>EMNLP</em>, 2025. arXiv:2502.20073.<br>
     <span id="ref20">[20]</span> "BotzoneBench: Scalable LLM Evaluation via Graded AI Anchors," arXiv:2602.13214, 2026.<br>
     <span id="ref21">[21]</span> Z. Xi et al., "AgentGym: Evolving Large Language Model-based Agents across Diverse Environments," <em>ACL</em>, 2025. arXiv:2406.04151.
   </p>

.. raw:: html

   <br><hr>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   documents/tutorials/installation/index
   documents/tutorials/quickstart

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Environments

   documents/environments/index

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Architecture

   documents/architecture/overview
   documents/architecture/paradigms
   documents/architecture/policy_mapping
   documents/architecture/workers/index
   documents/architecture/operators/index

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Rendering

   documents/rendering_tabs/index

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Runtime Logs

   documents/runtime_logging/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Human Control

   documents/human_control/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   documents/api/core
   documents/api/services
   documents/api/adapters

.. toctree::
   :hidden:
   :caption: Development

   GitHub <https://github.com/Abdulhamid97Mousa/MOSAIC>
   documents/contributing
   documents/changelog
