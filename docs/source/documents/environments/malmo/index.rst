Microsoft Malmo / MalmoEnv
==========================

`Microsoft Malmo <https://github.com/microsoft/malmo>`_ is a platform for AI research and
experimentation **built on top of Minecraft** (Java Edition 1.11.2).  MOSAIC integrates with
Malmo through the **MalmoEnv** Python interface, which connects directly to a running
Minecraft JVM process over TCP — no native compilation or Minecraft account required.

.. note::

   Unlike most MOSAIC environments, Malmo requires a **running Minecraft server** before any
   experiments can start.  Follow the :doc:`installation` guide to set it up once;
   subsequent launches take only a few seconds.

:Install: see :doc:`installation`
:Paradigm: Single-agent (Minecraft missions)
:Stepping: ``SINGLE_AGENT``
:Render mode: RGB frame from Minecraft (first-person view)
:Action space: Discrete (8 movement / action commands)
:Keyboard: W/S (forward/back), A/D (strafe), Q/E (turn left/right), Space (jump), F (attack)

.. toctree::
   :maxdepth: 2

   installation
   environments


Overview
--------

MOSAIC uses the **MalmoEnv** protocol, which communicates with the Malmo Mod running inside
Minecraft over a lightweight TCP socket on port **9000**.  This means:

* The Python agent process and the Minecraft JVM are **separate** — they communicate over localhost.
* Minecraft must be **started first** before launching any MOSAIC experiment.
* The mission is described by an **XML file** (shipped with MalmoEnv); each call to
  ``env.init()`` sends the XML to Minecraft, which resets the world accordingly.

Architecture
~~~~~~~~~~~~

.. code-block:: text

   MOSAIC GUI
   ├── MalmoEnvAdapter (gym_gui/core/adapters/mosaic_malmo.py)
   │     ├── malmoenv.Env  ←→  TCP socket (port 9000)
   │     │                          │
   │     │                    Minecraft JVM
   │     │                    (Malmo Mod 0.37.0 running inside ForgeGradle)
   │     └── Render: RGB frame → MosaicMalmoRendererStrategy
   └── Human input: W/A/S/D + Space + F → discrete action index

Keyboard Controls
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Action index
     - Description
   * - W / ↑
     - 0
     - Move forward
   * - S / ↓
     - 1
     - Move backward
   * - A / ←
     - 2
     - Strafe left
   * - D / →
     - 3
     - Strafe right
   * - Q
     - 4
     - Turn left
   * - E
     - 5
     - Turn right
   * - Space
     - 6
     - Jump
   * - F
     - 7
     - Attack / break block

Available Missions
------------------

All missions are bundled with the MalmoEnv package and located in
``3rd_party/environments/malmo/MalmoEnv/missions/``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - MOSAIC Game ID
     - Mission / Description
   * - ``MalmoEnv-MobChase-v0``
     - Chase and catch a mob in an open arena
   * - ``MalmoEnv-MazeRunner-v0``
     - Navigate a procedurally generated maze to the goal
   * - ``MalmoEnv-Vertical-v0``
     - Climb a vertical tower of blocks
   * - ``MalmoEnv-CliffWalking-v0``
     - Walk along a cliff edge without falling (dense reward)
   * - ``MalmoEnv-CatchTheMob-v0``
     - Catch a moving mob in an enclosed space
   * - ``MalmoEnv-FindTheGoal-v0``
     - Find a goal block hidden in a large flat world
   * - ``MalmoEnv-Attic-v0``
     - Navigate an indoor attic environment
   * - ``MalmoEnv-DefaultFlatWorld-v0``
     - Flat creative-mode world (open exploration)
   * - ``MalmoEnv-DefaultWorld-v0``
     - Default Minecraft world generation (survival-style)
   * - ``MalmoEnv-Eating-v0``
     - Collect food items for a positive reward
   * - ``MalmoEnv-Obstacles-v0``
     - Navigate over and around obstacles
   * - ``MalmoEnv-TrickyArena-v0``
     - Survive in a tricky arena with pits and hazards
   * - ``MalmoEnv-TreasureHunt-v0``
     - Find treasure chests scattered across the world

Citation
--------

.. code-block:: bibtex

   @article{johnson2016malmo,
     author  = {Matthew Johnson and Katja Hofmann and Tim Hutton and David Bignell},
     title   = {The Malmo Platform for Artificial Intelligence Experimentation},
     journal = {Proceedings of the 25th International Joint Conference on Artificial Intelligence},
     year    = {2016},
     pages   = {4246--4247},
   }
