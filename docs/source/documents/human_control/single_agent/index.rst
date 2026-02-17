Single-Agent Human Control
==========================

In single-agent mode, one human player controls the environment using
the system keyboard.  This page traces the full journey of a keypress
from the physical keyboard to the updated render view.

End-to-End Flow
---------------

.. mermaid::

   sequenceDiagram
       participant User
       participant HIC as HumanInputController
       participant SC as SessionController
       participant Adapter
       participant Env as Gymnasium Environment
       participant RR as RendererRegistry
       participant RV as Render View

       User->>HIC: keyPress (e.g., Arrow Right)
       HIC->>SC: perform_human_action(action=1)
       SC->>Adapter: _apply_action(1)
       Adapter->>Env: env.step(1)
       Env-->>Adapter: obs, reward, terminated, info
       Adapter-->>SC: step result
       SC->>SC: step_processed signal
       SC-->>RR: render payload
       RR->>RV: strategy.render(payload)
       RV-->>User: Updated visual display

Control Modes
-------------

Three ``ControlMode`` values permit human input in single-agent
environments:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Mode
     - Behaviour
   * - ``HUMAN_ONLY``
     - All actions come from the keyboard.  The environment waits for
       human input before advancing.
   * - ``HYBRID_TURN_BASED``
     - Human and AI agent alternate turns.  The GUI indicates whose turn
       it is and blocks keyboard input during the agent's turn.
   * - ``HYBRID_HUMAN_AGENT``
     - Human controls some agents while AI controls others.  Both act
       simultaneously each step.

Example: Playing FrozenLake
---------------------------

FrozenLake is a turn-based grid-world environment with 4 discrete
actions.  It uses **shortcut-based** input mode because every action is
a single keypress and there is no need for simultaneous key combinations.

**Key mappings (shortcut-based)**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Key
     - Action Index
     - Effect
   * - Arrow Left / A
     - 0
     - Move left
   * - Arrow Down / S
     - 1
     - Move down
   * - Arrow Right / D
     - 2
     - Move right
   * - Arrow Up / W
     - 3
     - Move up

**Renderer**: ``GridRendererStrategy`` renders the FrozenLake map as a
tile grid using ``FrozenLakeAssets``.  The elf sprite moves from tile to
tile as the agent position updates.

**Walkthrough**

1. The player presses **Right** (or **D**).
2. The ``QShortcut`` fires and calls
   ``SessionController.perform_human_action(2)``.
3. ``SessionController`` checks that the control mode allows human input,
   then calls ``_apply_action(2)``.
4. The adapter calls ``env.step(2)``, which returns a new observation
   (the elf moved one tile right), reward, and termination flag.
5. ``SessionController`` emits the ``step_processed`` signal.
6. The ``RendererRegistry`` passes the new observation to
   ``GridRendererStrategy.render()``, which updates the elf sprite
   position on the tile map.
7. The player sees the elf in its new position and presses the next key.

Example: Playing Atari
----------------------

Atari games are real-time environments with up to 18 discrete actions
that include directional movement, fire, and all direction+fire
combinations.  They use **state-based** input mode to support
simultaneous key presses (e.g., Up+Space = UPFIRE).

**Key handling (state-based)**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Keys Held
     - Action Index
     - Effect
   * - (none)
     - 0
     - NOOP
   * - Space
     - 1
     - FIRE
   * - Arrow Up
     - 2
     - UP
   * - Arrow Right
     - 3
     - RIGHT
   * - Arrow Left
     - 4
     - LEFT
   * - Arrow Down
     - 5
     - DOWN
   * - Arrow Up + Arrow Right
     - 6
     - UPRIGHT
   * - Arrow Up + Arrow Left
     - 7
     - UPLEFT
   * - Arrow Down + Arrow Right
     - 8
     - DOWNRIGHT
   * - Arrow Down + Arrow Left
     - 9
     - DOWNLEFT
   * - Space + Arrow Up
     - 10
     - UPFIRE
   * - Space + Arrow Right
     - 11
     - RIGHTFIRE
   * - Space + Arrow Left
     - 12
     - LEFTFIRE
   * - Space + Arrow Down
     - 13
     - DOWNFIRE

**Resolver**: ``AleKeyCombinationResolver`` inspects the set of
currently pressed keys and returns the correct composite action index.

**Renderer**: ``RgbRendererStrategy`` displays each game frame as a
scaled RGB image.

**Walkthrough**

1. The player holds **Up** and presses **Space** simultaneously.
2. ``HumanInputController.eventFilter`` tracks both keys in
   ``_pressed_keys``.
3. ``AleKeyCombinationResolver.resolve({Up, Space})`` returns action
   index 10 (``UPFIRE``).
4. ``perform_human_action(10)`` is called on ``SessionController``.
5. The adapter calls ``env.step(10)``; the Atari emulator advances one
   frame and returns an RGB observation.
6. ``RgbRendererStrategy.render()`` converts the NumPy array to a
   ``QPixmap`` and paints it in the render tab.
7. The loop repeats at the environment's native frame rate.
