Human Control
=============

MOSAIC provides a comprehensive human control system that lets users play
environments directly through the GUI.  Every supported environment has
keyboard (and sometimes mouse) bindings that map physical keys to
discrete or continuous actions.

The system operates in two fundamentally different configurations:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Mode
     - Description
   * - **Single-Agent**
     - One human controls one agent.  Standard keyboard input via Qt.
   * - **Multi-Agent**
     - Multiple humans each control a separate agent using dedicated USB
       keyboards routed through Linux ``evdev``.

.. mermaid::

   graph TD
       subgraph "Human Control System"
           KC[Keyboard Events] --> HIC[HumanInputController]
           HIC -->|Turn-based| SB[Shortcut-Based Mode]
           HIC -->|Real-time| STB[State-Based Mode]
           SB --> SA[SessionController.perform_human_action]
           STB --> RES[KeyCombinationResolver]
           RES --> SA
       end

       subgraph "Multi-Agent Extension"
           USB1[USB Keyboard 1] --> EVDEV[evdev Monitor]
           USB2[USB Keyboard 2] --> EVDEV
           EVDEV --> ROUTE[Agent Router]
           ROUTE --> A1[Agent 0 Keys]
           ROUTE --> A2[Agent 1 Keys]
       end

       style KC fill:#4a90d9,stroke:#2e5a87,color:#fff
       style HIC fill:#50c878,stroke:#2e8b57,color:#fff
       style EVDEV fill:#ff7f50,stroke:#cc5500,color:#fff
       style RES fill:#9370db,stroke:#6a0dad,color:#fff

How It Works
------------

When a human presses a key in the MOSAIC GUI:

1. **Qt captures the key event** (``keyPressEvent`` / ``keyReleaseEvent``)
2. **HumanInputController** routes it through the configured input mode
3. A **KeyCombinationResolver** (or QShortcut) maps the key(s) to an action index
4. The action is forwarded to the environment via ``SessionController``

The input mode is user-configurable in the **Game Configuration** panel:

- **Shortcut-Based (Immediate)**: Each key press instantly fires one action.
  Best for turn-based games (Chess, FrozenLake, MiniGrid).
- **State-Based (Real-time)**: Tracks all currently pressed keys and computes
  combined actions (e.g., Up+Right → diagonal).  Best for arcade games
  (Procgen, Atari, ViZDoom).

.. tip::

   For multi-agent environments, MOSAIC automatically forces **state-based**
   mode because shortcut-based mode conflicts with per-device keyboard routing.

.. toctree::
   :maxdepth: 2

   single_agent/index
   multi_agent/index
