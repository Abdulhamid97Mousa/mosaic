Single-Agent Human Control
==========================

In single-agent mode, one human player controls the environment using the
system keyboard.  MOSAIC provides two input processing modes and
per-environment keyboard mappings for every supported environment family.

Input Modes
-----------

.. _input-mode-shortcut:

Shortcut-Based (Immediate)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Uses Qt's ``QShortcut`` mechanism.  Each key press immediately triggers a
single action.  There is no key-state tracking — pressing two keys at once
fires whichever arrives first.

**Best for:** Turn-based and grid-world environments (FrozenLake, MiniGrid,
Chess, MiniHack, Jumanji puzzles).

.. _input-mode-state:

State-Based (Real-time)
^^^^^^^^^^^^^^^^^^^^^^^

Tracks all currently pressed keys in a ``set``.  On each game tick
(~16–33 ms), a **KeyCombinationResolver** examines the pressed-key set
and produces a combined action.

**Best for:** Arcade and real-time environments (Procgen, Atari, ViZDoom,
Box2D, MeltingPot).

.. mermaid::

   sequenceDiagram
       participant User as Player
       participant Qt as Qt Event Loop
       participant HIC as HumanInputController
       participant Res as KeyCombinationResolver
       participant Env as Environment

       Note over User,Env: State-Based Mode (Real-time)
       User->>Qt: Press Up arrow
       Qt->>HIC: keyPressEvent(Up)
       HIC->>HIC: _pressed_keys.add(Up)
       User->>Qt: Press Right arrow (while Up held)
       Qt->>HIC: keyPressEvent(Right)
       HIC->>HIC: _pressed_keys.add(Right)
       HIC->>Res: resolve({Up, Right})
       Res-->>HIC: action=8 (up_right)
       HIC->>Env: perform_human_action(8)

Continuous Action Mapping
-------------------------

Some environments (Box2D family) use continuous action spaces.  MOSAIC
provides a ``ContinuousActionMapper`` that converts discrete keyboard
actions into continuous vectors:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Environment
     - Vector Size
     - Mapping
   * - **Lunar Lander**
     - 2
     - ``0 → (0,0)`` idle, ``1 → (0,-1)`` left engine, ``2 → (1,0)`` main engine, ``3 → (0,1)`` right engine
   * - **Car Racing**
     - 3
     - ``0 → (0,0,0)`` coast, ``1 → (1,0.3,0)`` right, ``2 → (-1,0.3,0)`` left, ``3 → (0,1,0)`` gas, ``4 → (0,0,0.8)`` brake
   * - **Bipedal Walker**
     - 4
     - ``0 → (0,0,0,0)`` neutral, ``1 → (0.8,0.6,-0.8,-0.6)`` forward, ``2 → (-0.8,-0.6,0.8,0.6)`` back

.. toctree::
   :maxdepth: 2

   keyboard_mappings
