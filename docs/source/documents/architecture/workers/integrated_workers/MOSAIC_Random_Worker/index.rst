MOSAIC Random Worker
====================

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../../../_static/videos/random_worker.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br><br>

The MOSAIC Random Worker is MOSAIC's **lightweight baseline agent** for
multi-agent and single-agent environments. It selects actions without any
learned policy, providing a performance floor for comparison against RL,
LLM, and human decision-makers.

The worker supports three action-selection behaviors; **random**
(uniform sampling), **noop** (always action 0), and **cycling**
(round-robin through the action space), making it useful for sanity
checks, environment debugging, and establishing baselines in
:doc:`heterogeneous </documents/architecture/operators/heterogeneous_decision_maker/index>`
experiments (e.g., RL + LLM teammates vs Random opponents).

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Baseline agent (single-agent and multi-agent)
   * - **Task Type**
     - Random baseline, noop baseline, cycling baseline
   * - **Behaviors**
     - ``random`` (uniform), ``noop`` (action 0), ``cycling`` (round-robin)
   * - **Environments**
     - All Gymnasium-compatible: MiniGrid, BabyAI, MosaicMultiGrid, Atari,
       FrozenLake, Taxi, Blackjack, MeltingPot, PettingZoo, and more
   * - **Execution**
     - Subprocess (autonomous or interactive step-by-step)
   * - **GPU required**
     - No
   * - **Source**
     - ``3rd_party/mosaic/random_worker/random_worker/``
   * - **Entry point**
     - ``random-worker`` (CLI)

Overview
--------

The MOSAIC Random Worker provides a zero-intelligence baseline for any
Gymnasium-compatible environment. It requires no training, no API keys,
and no GPU, just point it at an environment and it will select actions
according to one of three simple strategies.

This enables several research workflows:

- **Baseline comparison:** Measure how much better a trained RL policy or
  LLM agent performs compared to random play.
- **Environment debugging:** Verify that environments, rendering, and
  telemetry pipelines work end-to-end before deploying expensive workers.
- **Heterogeneous experiments:** Fill opponent or teammate slots with random agents
  in mixed-paradigm evaluations (e.g., RL team vs Random team).

Key features:

- **3 action behaviors:** random (uniform sampling), noop (always still), cycling (round-robin)
- **Automatic action space resolution:** creates a temporary env to detect ``Discrete(N)``
- **Multi-agent support:** handles Dict action spaces (MosaicMultiGrid Soccer, Basketball, etc.)
- **Dual runtime modes:** autonomous (batch episodes) or interactive (GUI step-by-step)
- **Action-selector mode:** for PettingZoo games where the GUI owns the environment
- **Deterministic seeding:** reproducible action sequences via ``--seed``
- **RGB rendering:** captures frames for GUI visualization
- **117 tests** across 15 test classes covering all behaviors and environments

Architecture
------------

The worker follows the standard MOSAIC :doc:`shim pattern <../../concept>` with
two runtime modes:

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Operator Config<br/>(Baseline worker)"]
           DAEMON["Operator Launcher"]
       end

       subgraph "Random Worker Subprocess"
           CLI["cli.py<br/>(random-worker)"]
           CFG["config.py<br/>(RandomWorkerConfig)"]
           RT["runtime.py<br/>(RandomWorkerRuntime)"]
       end

       subgraph "Environment"
           ENV["Gymnasium<br/>(any compatible env)"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT -->|"reset / step"| ENV

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style ENV fill:#e8e8e8,stroke:#999

Action Behaviors
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Behavior
     - Strategy
     - Description
   * - **random**
     - Uniform sampling
     - Samples uniformly from ``Discrete(N)``. Default behavior.
       Provides a true random baseline for comparison.
   * - **noop**
     - Always action 0
     - Always selects action 0 (``still`` / do nothing).
       Tests environment behavior with a passive agent.
   * - **cycling**
     - Round-robin
     - Cycles through actions ``0, 1, 2, ..., N-1, 0, 1, ...``
       Deterministically exercises every action in order.

Supported Environments
----------------------

The Random Worker automatically resolves the action space by creating a
temporary environment instance. It supports **any Gymnasium-compatible
environment**, including:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Environment
     - Action Space
     - Notes
   * - MiniGrid (all variants)
     - Discrete(7)
     - Empty, DoorKey, KeyCorridor, etc.
   * - BabyAI (all variants)
     - Discrete(7)
     - GoTo, Pickup, Open tasks
   * - MosaicMultiGrid Soccer 1v1
     - Discrete(8) x 2
     - Multi-agent Dict action space
   * - MosaicMultiGrid Soccer 2v2
     - Discrete(8) x 4
     - 4-agent team play
   * - MosaicMultiGrid Basketball 3v3
     - Discrete(8) x 6
     - 6-agent team play
   * - MosaicMultiGrid Collect
     - Discrete(8) x 2--4
     - Ball collection variants
   * - Gymnasium Toy Text
     - Discrete(N)
     - FrozenLake, Taxi, Blackjack, CliffWalking
   * - Atari / ALE
     - Discrete(N)
     - All 128 ALE games
   * - PettingZoo
     - varies
     - Chess, Connect Four, Go (action-selector mode)

For multi-agent environments with ``Dict`` action spaces, the worker
automatically unwraps to individual ``Discrete`` spaces and builds
per-agent action dictionaries at each step.

Runtime Modes
-------------

**Autonomous mode** (batch episodes, for Script Experiments):

.. code-block:: bash

   random-worker --run-id test123 \
       --task MiniGrid-Empty-8x8-v0 \
       --behavior random --seed 42

**Interactive mode** (GUI step-by-step, action-selector protocol):

.. code-block:: bash

   random-worker --run-id test123 --interactive \
       --task MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0

Interactive mode reads JSON commands from stdin and emits telemetry to stdout:

.. code-block:: json

   {"cmd": "init_agent", "game_name": "chess_v6", "player_id": "player_0"}
   {"cmd": "select_action", "observation": [...], "player_id": "player_0"}
   {"cmd": "stop"}

Autonomous mode uses the env-owning protocol:

.. code-block:: json

   {"cmd": "reset", "seed": 42}
   {"cmd": "step"}
   {"cmd": "stop"}

Configuration
-------------

**CLI arguments:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--run-id``
     - (required)
     - Unique run identifier (assigned by GUI)
   * - ``--task``
     - ``""``
     - Gymnasium environment ID (required for autonomous mode)
   * - ``--env-name``
     - ``""``
     - Environment family name
   * - ``--seed``
     - ``None``
     - Random seed for reproducible action sequences
   * - ``--behavior``
     - ``random``
     - Action selection strategy: ``random``, ``noop``, or ``cycling``
   * - ``--interactive``
     - ``false``
     - Run in interactive (action-selector) mode

**RandomWorkerConfig dataclass:**

.. code-block:: python

   @dataclass
   class RandomWorkerConfig:
       run_id: str = ""
       env_name: str = ""
       task: str = ""
       seed: Optional[int] = None
       behavior: str = "random"  # random | noop | cycling

Test Coverage
-------------

The Random Worker has **117 tests** across 15 test classes:

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Test Class
     - Tests
     - Coverage
   * - TestRandomWorkerConfig
     - 2
     - Config dataclass defaults and construction
   * - TestCLI
     - 3
     - Argument parsing and entry point
   * - TestRuntimeProtocol
     - 6
     - JSON protocol with fallback Discrete(7)
   * - TestFullLoop
     - 6
     - Stdin/stdout loop, malformed JSON, empty lines
   * - TestActionSpaceResolution
     - 12
     - MiniGrid, BabyAI, MosaicMultiGrid, FrozenLake, Taxi, Blackjack
   * - TestBehaviorsWithRealEnvs
     - 12
     - random/noop/cycling across Discrete(2,4,6,7,8)
   * - TestMultiAgent
     - 4
     - 4-agent Soccer, 6-agent Basketball, reinit
   * - TestReproducibility
     - 3
     - Seed determinism
   * - TestFullLoopRealEnvs
     - 3
     - Full protocol with Soccer 2v2, FrozenLake, BabyAI
   * - TestSubprocessIntegration
     - 7
     - Real subprocess launches
   * - TestEdgeCases
     - 7
     - Large obs, missing fields, rapid fire
   * - TestAutonomousMode
     - 16
     - Env-owning reset/step/stop protocol
   * - TestMosaicMultigridAutonomous
     - 18
     - All 11 MosaicMultiGrid envs, render, episode end
   * - TestMosaicMultigridInteractive
     - 14
     - Action space resolution, 4-agent Soccer, 6-agent Basketball
   * - TestMosaicMultigridSubprocess
     - 3
     - Real subprocess: Soccer 2v2, Soccer 1v1, Basketball 3v3

.. toctree::
   :maxdepth: 1

   installation
