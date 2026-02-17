Human Control
=============

Human Control is the interactive play layer of MOSAIC -- keyboard input,
mouse interaction, and the render view that displays environment state in
real time.  Together these subsystems let a human player step into any
supported environment and interact with it through the GUI.

Signal Flow
-----------

.. mermaid::

   graph LR
       KB[Keyboard] --> HIC[HumanInputController]
       HIC --> SC[SessionController]
       SC --> ADAPT[Adapter]
       ADAPT --> ENV[Environment]
       ENV --> |obs| ADAPT
       ADAPT --> |payload| REG[RendererRegistry]
       REG --> STRAT[RendererStrategy]
       STRAT --> RV[Render View]

       style KB fill:#4a90d9,stroke:#2e5a87,color:#fff
       style RV fill:#4a90d9,stroke:#2e5a87,color:#fff
       style HIC fill:#50c878,stroke:#2e8b57,color:#fff
       style SC fill:#50c878,stroke:#2e8b57,color:#fff
       style REG fill:#ff7f50,stroke:#cc5500,color:#fff
       style STRAT fill:#ff7f50,stroke:#cc5500,color:#fff
       style ADAPT fill:#9370db,stroke:#6a0dad,color:#fff
       style ENV fill:#9370db,stroke:#6a0dad,color:#fff

| Blue = GUI widgets | Green = Controllers | Orange = Rendering | Purple = Environment / Adapter |

Subsystems
----------

**Keyboard Input** --
Two input modes (shortcut-based for turn-based games, state-based for
real-time arcade games), per-environment key mappings covering all 26
environment families, and multi-keyboard support for multi-agent play via
Linux ``evdev``.  See :doc:`keyboard_input` for the full reference.

**Render View** --
A strategy-pattern rendering pipeline that converts environment
observations into visual output.  Three built-in strategies cover grid
tile maps, RGB pixel arrays, and interactive board games.  See
:doc:`render_view` for the full reference.

Control Modes
-------------

The ``ControlMode`` enum determines who provides actions for each agent.
Human input is active in every mode except ``AGENT_ONLY``.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Control Mode
     - Description
   * - ``HUMAN_ONLY``
     - All actions come from the human keyboard.  No AI agent is involved.
   * - ``AGENT_ONLY``
     - All actions come from an AI agent.  The keyboard is disabled.
   * - ``HYBRID_TURN_BASED``
     - Human and agent alternate turns within the same environment.
   * - ``HYBRID_HUMAN_AGENT``
     - Human controls some agents while AI controls others simultaneously.
   * - ``MULTI_AGENT_COOP``
     - Multiple human players cooperate, each with a dedicated USB keyboard.
   * - ``MULTI_AGENT_COMPETITIVE``
     - Multiple human players compete against each other, each with a
       dedicated USB keyboard.

.. toctree::
   :maxdepth: 2

   keyboard_input
   render_view
   multi_keyboard_evdev
   single_agent/index
