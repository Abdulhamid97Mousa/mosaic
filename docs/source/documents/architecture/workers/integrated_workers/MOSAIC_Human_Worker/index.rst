MOSAIC Human Worker
===================

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../../../_static/videos/human_vs_human.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br/><br/>

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../../../_static/videos/random_worker.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br/><br/>

The MOSAIC Human Worker enables **human-in-the-loop** which is essentially to control the agent via keyboard for any Gymnasium-compatible environment. It bridges human decision-making with
MOSAIC's multi-agent evaluation framework, allowing researchers to play
alongside or against RL, LLM, and random agents.

**Important:** The human owns the agent for the entire episode. Once an
operator is configured with a human worker, that agent slot is controlled
exclusively by the human. There is no automatic switching to an AI
policy mid-execution. The human remains in control from reset to episode
end, ensuring clean and comparable evaluation data.

The worker operates in two modes: **interactive mode** where the worker
owns the environment and the GUI sends human-chosen actions via action
buttons, and **board-game mode** where the GUI owns the environment
(PettingZoo games) and the worker handles move selection with legal-move
validation.

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Human-in-the-loop (single-agent and multi-agent)
   * - **Task Type**
     - Human vs AI, human + AI cooperative, human baseline evaluation
   * - **Modes**
     - ``interactive`` (env-owning), ``board-game`` (action-selector)
   * - **Environments**
     - MiniGrid, BabyAI, MosaicMultiGrid, Crafter, MiniHack/NetHack,
       PettingZoo (Chess, Go, Connect Four), Gymnasium Classic Control
   * - **Execution**
     - Subprocess (interactive step-by-step via GUI)
   * - **GPU required**
     - No
   * - **Source**
     - ``3rd_party/mosaic/human_worker/human_worker/``
   * - **Entry point**
     - ``human-worker`` (CLI)

Overview
--------

The MOSAIC Human Worker turns any MOSAIC environment into a playable game.
The GUI displays rendered frames, action buttons with environment-specific
labels, and episode statistics. The human clicks an action, the worker
steps the environment, and the next frame appears.

This enables several research workflows:

- **Human baseline:** Establish human-level performance benchmarks for
  comparison against RL and LLM agents.
- **Human-AI teams (Cooperation):** Deploy a human teammate alongside RL or LLM agents
  in cooperative multi-agent environments (Soccer 2v2, Overcooked).
- **Human-AI Adversarial (Competition):** Deploy human players against trained RL policies or LLM
  agents in competitive environments.
- **Environment debugging:** Manually explore environments to understand
  dynamics, test reward functions, and verify rendering.

Key features:

- **Environment-aware action labels:** "Turn Left", "Forward", "Pickup" for
  MiniGrid; "Push Left", "Push Right" for CartPole; "Noop"..."Make Iron Sword"
  for Crafter (17 actions)
- **Legal move validation:** for board games (Chess, Go), invalid moves are
  rejected with feedback
- **Custom initial states:** MiniGrid environments support JSON-based grid
  state injection for reproducible scenarios
- **Crafter support:** Custom gymnasium wrapper with configurable render
  resolution (64x64 to 512x512)
- **RGB frame rendering:** Real-time visualization in the GUI
- **Episode telemetry:** Step count, reward, success/failure, duration
- **Dual runtime modes:** Interactive (worker owns env) and board-game
  (GUI owns env)

Architecture
------------

The worker follows the standard MOSAIC :doc:`shim pattern <../../concept>` with
two runtime classes:

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           RENDER["Render View<br/>(RGB frames)"]
           BUTTONS["Action Buttons<br/>(env-specific labels)"]
           DAEMON["Operator Launcher"]
       end

       subgraph "Human Worker Subprocess"
           CLI["cli.py<br/>(human-worker)"]
           CFG["config.py<br/>(HumanWorkerConfig)"]
           IRT["HumanInteractiveRuntime<br/>(env-owning)"]
           LRT["HumanWorkerRuntime<br/>(board-game)"]
       end

       subgraph "Environment"
           ENV["Gymnasium / PettingZoo<br/>(MiniGrid, Crafter, Chess...)"]
       end

       DAEMON -->|"spawn"| CLI
       CLI --> CFG
       CFG --> IRT
       CFG --> LRT
       IRT -->|"reset / step"| ENV
       IRT -->|"RGB frames"| RENDER
       BUTTONS -->|"action click"| IRT
       LRT -->|"waiting_for_human"| BUTTONS

       style RENDER fill:#4a90d9,stroke:#2e5a87,color:#fff
       style BUTTONS fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style IRT fill:#ff7f50,stroke:#cc5500,color:#fff
       style LRT fill:#ff7f50,stroke:#cc5500,color:#fff
       style ENV fill:#e8e8e8,stroke:#999

Runtime Modes
-------------

**Interactive mode** (worker owns the environment):

Used for grid-world and continuous environments (MiniGrid, BabyAI, Crafter,
Classic Control). The worker creates the gymnasium environment, renders
frames, and accepts human-chosen actions.

.. code-block:: bash

   human-worker --mode interactive --run-id game_001 \
       --env-name minigrid --task MiniGrid-DoorKey-8x8-v0 --seed 42

Protocol:

.. code-block:: json

   {"cmd": "reset", "seed": 42, "env_name": "minigrid", "task": "MiniGrid-DoorKey-8x8-v0"}
   {"cmd": "step", "action": 2}
   {"cmd": "stop"}

**Board-game mode** (GUI owns the environment):

Used for PettingZoo turn-based games (Chess, Go, Connect Four). The GUI
owns the environment and sends observations with legal moves. The worker
displays options and waits for human selection.

.. code-block:: bash

   human-worker --mode board-game --run-id chess_001 \
       --player-name "Alice"

Protocol:

.. code-block:: json

   {"cmd": "init_agent", "game_name": "chess_v6", "player_id": "player_0"}
   {"cmd": "select_action", "observation": "...", "info": {"legal_moves": ["e2e4", "d2d4"]}}
   {"cmd": "human_input", "move": "e2e4", "player_id": "player_0"}
   {"cmd": "stop"}

Action Labels
-------------

The worker provides environment-specific action labels for the GUI's action
buttons:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Environment
     - Actions
     - Labels
   * - MiniGrid / BabyAI
     - 7
     - Turn Left, Turn Right, Forward, Pickup, Drop, Toggle, Done
   * - MosaicMultiGrid
     - 8
     - Still, Turn Left, Turn Right, Forward, Pickup, Drop, Toggle, Done
   * - Crafter
     - 17
     - Noop, Move Left/Right/Up/Down, Do, Sleep, Place Stone/Table/Furnace/Plant,
       Make Wood/Stone/Iron Pickaxe, Make Wood/Stone/Iron Sword
   * - NetHack / NLE
     - 24
     - North, East, South, West, NE, SE, SW, NW, Wait, Kick, Open, Search, ...
   * - FrozenLake
     - 4
     - Left, Down, Right, Up
   * - Taxi
     - 6
     - South, North, East, West, Pickup, Dropoff
   * - CartPole
     - 2
     - Push Left, Push Right
   * - LunarLander
     - 4
     - Noop, Fire Left, Fire Main, Fire Right

For unknown environments, generic labels (``Action 0``, ``Action 1``, ...) are
generated automatically.

Configuration
-------------

**CLI arguments:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--mode``
     - ``interactive``
     - ``interactive`` (env-owning) or ``board-game`` (action-selector)
   * - ``--run-id``
     - ``""``
     - Unique run identifier (assigned by GUI)
   * - ``--player-name``
     - ``"Human"``
     - Display name for the human player
   * - ``--env-name``
     - ``""``
     - Environment family (minigrid, babyai, crafter, etc.)
   * - ``--task``
     - ``""``
     - Gymnasium environment ID
   * - ``--seed``
     - ``42``
     - Random seed for environment
   * - ``--game-resolution``
     - ``512x512``
     - Render resolution for Crafter (e.g., ``64x64``, ``512x512``)
   * - ``--timeout``
     - ``0.0``
     - Timeout for human input in seconds (0 = no timeout)
   * - ``--show-legal-moves``
     - ``true``
     - Highlight legal moves in board-game mode
   * - ``--confirm-moves``
     - ``false``
     - Require move confirmation before submitting

**HumanWorkerConfig dataclass:**

.. code-block:: python

   @dataclass
   class HumanWorkerConfig:
       run_id: str = ""
       player_name: str = "Human"
       env_name: str = ""
       task: str = ""
       render_mode: str = "rgb_array"
       seed: int = 42
       game_resolution: Tuple[int, int] = (512, 512)
       timeout_seconds: float = 0.0
       show_legal_moves: bool = True
       confirm_moves: bool = False
       telemetry_dir: str = "var/telemetry"

Supported Environments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Environment Family
     - Mode
     - Notes
   * - MiniGrid
     - Interactive
     - All variants; supports custom initial state injection
   * - BabyAI
     - Interactive
     - Language-grounded instruction following
   * - MosaicMultiGrid
     - Interactive
     - Soccer, Collect, Basketball (multi-agent via hybrid operators)
   * - Crafter
     - Interactive
     - Custom gymnasium wrapper, configurable render size
   * - Gymnasium Classic
     - Interactive
     - CartPole, MountainCar, Acrobot, FrozenLake, Taxi, etc.
   * - PettingZoo
     - Board-game
     - Chess, Connect Four, Go, Tic-Tac-Toe (legal move validation)
   * - NetHack / MiniHack
     - Interactive
     - Roguelike dungeon crawling

Test Coverage
-------------

The Human Worker has **35 tests** across 8 test classes:

.. list-table::
   :header-rows: 1
   :widths: 40 10 50

   * - Test Class
     - Tests
     - Coverage
   * - TestHumanWorkerConfig
     - 5
     - Config defaults, custom values, serialization (to_dict/from_dict)
   * - TestHumanWorkerRuntime
     - 4
     - Init state, agent init, human input request, move validation
   * - TestHumanWorkerRuntimeInteractive
     - 3
     - init_agent command, select_action, human_input processing
   * - TestWorkerMetadata
     - 3
     - Metadata values, capabilities (worker_type, paradigms, GPU)
   * - TestHumanWorkerEdgeCases
     - 3
     - Empty legal moves, multiple init calls, empty move string
   * - TestActionLabels
     - 5
     - MiniGrid, FrozenLake, Taxi, unknown env, label truncation
   * - TestHumanWorkerConfigNew
     - 3
     - Environment config fields, serialization with env fields
   * - TestHumanInteractiveRuntime(WithEnv)
     - 9
     - Import, initial state, emit, reset with MiniGrid, step,
       invalid action, step without reset

.. toctree::
   :maxdepth: 1

   installation
