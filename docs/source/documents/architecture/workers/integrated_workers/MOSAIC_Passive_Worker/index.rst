MOSAIC Passive Worker
=====================

The MOSAIC Passive Worker is MOSAIC's **passive baseline agent** for
multi-agent and single-agent environments.  It always selects the
environment's "do nothing" action, providing a passive performance floor
for comparison against RL, LLM, and human decision-makers.

**NOOP / STILL resolution:** the worker automatically determines the
correct passive action for each environment:

1. **NOOP** — action 0, used when the environment maps index 0 to a
   passive meaning (``"still"``, ``"noop"``, ``"idle"``, ``"wait"``).
   This covers MosaicMultiGrid (action 0 = "still"), Crafter (action 0 =
   "noop"), and most Gymnasium environments.
2. **STILL** — if action 0 is *not* passive (e.g. MiniGrid where 0 is
   "Turn Left"), the worker scans the environment's action-meaning list
   for the first entry matching a passive keyword and uses that index
   instead.

Unlike the :doc:`Random Worker </documents/architecture/workers/integrated_workers/MOSAIC_Random_Worker/index>`
which samples uniformly from the action space, the Passive Worker
deterministically takes no action, making it ideal for:

- Measuring environment **natural dynamics** (what happens when agents
  are idle)
- **Fault-isolation testing** — verify the environment, rendering, and
  telemetry pipelines work end-to-end without any policy-driven actions
- Establishing a true **do-nothing baseline** in heterogeneous experiments

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Passive baseline agent (single-agent and multi-agent)
   * - **Task Type**
     - NOOP / STILL baseline (environment-aware passive action)
   * - **Behavior**
     - ``noop`` — action 0 when it maps to a passive meaning;
       ``still`` — scans action meanings for the first passive keyword
   * - **Environments**
     - All Gymnasium-compatible: MiniGrid, BabyAI, MosaicMultiGrid, Atari,
       FrozenLake, Taxi, Blackjack, MeltingPot, PettingZoo, and more
   * - **Execution**
     - Subprocess (autonomous or interactive step-by-step)
   * - **GPU required**
     - No
   * - **Source**
     - ``3rd_party/mosaic/passive_worker/passive_worker/``
   * - **Entry point**
     - ``passive-worker`` (CLI)

Overview
--------

The Passive Worker provides a deterministic do-nothing baseline for any
Gymnasium-compatible environment.  It requires no training, no API keys,
and no GPU.  Point it at an environment and every agent will select the
resolved passive action at every timestep.

The worker resolves the correct passive action automatically:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Environment
     - Action
     - How resolved
   * - MosaicMultiGrid (Soccer, Collect)
     - ``0`` (still)
     - Action 0 = ``"still"`` → NOOP
   * - Crafter
     - ``0`` (noop)
     - Action 0 = ``"noop"`` → NOOP
   * - Procgen
     - ``4`` (noop)
     - Action 0 = ``"down_left"`` → scans meanings → ``"noop"`` at index 4
   * - MiniGrid / BabyAI
     - ``6`` (done)
     - Action 0 = ``"Turn Left"`` → scans meanings → ``"done"`` at index 6
   * - FrozenLake, Taxi, etc.
     - ``0``
     - No action meanings → defaults to NOOP (action 0)

**Passive keywords** recognised: ``still``, ``noop``, ``idle``, ``wait``,
``done``.

This enables several research workflows:

- **Natural dynamics baseline:** Measure what happens in an environment
  when all agents are completely idle — useful for environments where
  passivity is meaningful (e.g., MultiGrid Soccer: the ball doesn't move).
- **Fault isolation:** If something breaks with the passive worker, the
  bug is in the environment, rendering, or telemetry — not the policy.
- **Heterogeneous experiments:** Fill opponent or teammate slots with
  passive agents in mixed-paradigm evaluations (e.g., RL team vs Passive
  opponents).

Key features:

- **Environment-aware passive action:** resolves NOOP or STILL from
  action meanings automatically
- **Automatic action space resolution:** creates a temporary env to detect
  ``Discrete(N)``
- **Multi-agent support:** handles Dict action spaces (MosaicMultiGrid
  Soccer, Basketball, etc.)
- **Dual runtime modes:** autonomous (batch episodes) or interactive
  (GUI step-by-step)
- **Action-selector mode:** for PettingZoo games where the GUI owns the
  environment
- **RGB rendering:** captures frames for GUI visualization

Architecture
------------

The worker follows the standard MOSAIC :doc:`shim pattern <../../concept>`
with two runtime modes:

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Operator Config<br/>(Passive baseline)"]
           DAEMON["Operator Launcher"]
       end

       subgraph "Passive Worker Subprocess"
           CLI["cli.py<br/>(passive-worker)"]
           CFG["config.py<br/>(PassiveWorkerConfig)"]
           RT["runtime.py<br/>(PassiveWorkerRuntime)"]
       end

       subgraph "Environment"
           ENV["Gymnasium<br/>(any compatible env)"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT -->|"reset / step (action=0)"| ENV

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#9b59b6,stroke:#6c3483,color:#fff
       style CFG fill:#9b59b6,stroke:#6c3483,color:#fff
       style RT fill:#9b59b6,stroke:#6c3483,color:#fff
       style ENV fill:#e8e8e8,stroke:#999

Comparison with Random Worker
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Passive Worker
     - Random Worker
   * - **Action strategy**
     - NOOP (action 0) or STILL (env-aware lookup)
     - Uniform random sampling
   * - **Determinism**
     - Fully deterministic
     - Stochastic (seed-dependent)
   * - **Use case**
     - Natural dynamics, fault isolation
     - Random baseline comparison
   * - **Multi-agent**
     - All agents idle (same passive action)
     - All agents sample independently

Runtime Modes
-------------

**Autonomous mode** (batch episodes, for Script Experiments):

.. code-block:: bash

   passive-worker --run-id test123 \
       --task MiniGrid-Empty-8x8-v0 --seed 42

**Interactive mode** (GUI step-by-step, action-selector protocol):

.. code-block:: bash

   passive-worker --run-id test123 --interactive

Interactive mode reads JSON commands from stdin and emits telemetry to
stdout:

.. code-block:: json

   {"cmd": "init_agent", "game_name": "chess_v6", "player_id": "player_0"}
   {"cmd": "select_action", "observation": ["..."], "player_id": "player_0"}
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
     - Random seed for environment resets
   * - ``--interactive``
     - ``false``
     - Run in interactive (action-selector) mode

**PassiveWorkerConfig dataclass:**

.. code-block:: python

   @dataclass
   class PassiveWorkerConfig:
       run_id: str = ""
       env_name: str = ""
       task: str = ""
       seed: Optional[int] = None

Installation
------------

.. code-block:: bash

   cd 3rd_party/mosaic/passive_worker
   pip install -e .

.. toctree::
   :maxdepth: 1

   installation
