Integrated Workers
==================

MOSAIC ships with four production-ready workers that wrap major RL
frameworks and LLM evaluation suites.  Each worker follows the
:doc:`shim pattern <../concept>`: upstream libraries are **never
modified**; a thin integration layer translates between MOSAIC and
the library.

.. list-table::
   :header-rows: 1
   :widths: 18 18 22 22 20

   * - Worker
     - Paradigm
     - Algorithms / Models
     - Environments
     - Execution Model
   * - :doc:`CleanRL <CleanRL_Worker/index>`
     - Single-Agent
     - PPO, DQN, SAC, TD3, DDPG, C51
     - Gymnasium, Atari, MiniGrid, BabyAI, Procgen
     - Subprocess
   * - :doc:`XuanCe <XuanCe_Worker/index>`
     - Multi-Agent
     - MAPPO, QMIX, MADDPG, VDN, COMA + 40 more
     - PettingZoo, SMAC, MultiGrid, MPE
     - In-process
   * - :doc:`Ray RLlib <RLlib_Worker/index>`
     - Both
     - PPO, IMPALA, APPO, DQN, A2C
     - PettingZoo (SISL, Classic, Butterfly, MPE)
     - Ray cluster
   * - :doc:`BALROG <BALROG_Worker/index>`
     - LLM/VLM Evaluation
     - GPT-4o, Claude 3, Gemini, vLLM (local)
     - NetHack, MiniHack, BabyAI, Crafter, TextWorld
     - Subprocess (parallel)

Each worker provides:

- **CLI entry point** for subprocess launching by the Trainer Daemon
- **Configuration dataclass** implementing the ``WorkerConfig`` protocol
- **Runtime orchestrator** managing the training lifecycle
- **FastLane telemetry** for real-time frame streaming to the GUI
- **GUI form widgets** for visual experiment configuration
- **Automatic discovery** via Python entry points

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Training Form<br/>(per-worker UI)"]
           DAEMON["Trainer Daemon"]
       end

       subgraph "Worker Subprocess"
           CLI["cli.py"]
           CFG["config.py"]
           RT["runtime.py"]
           FL["fastlane.py"]
           SITE["sitecustomize.py"]
       end

       subgraph "Upstream Library"
           LIB["CleanRL / XuanCe / RLlib<br/>(unmodified)"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT --> FL
       RT --> LIB
       SITE -.->|"import-time patches"| LIB

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style FL fill:#ff7f50,stroke:#cc5500,color:#fff
       style SITE fill:#ff7f50,stroke:#cc5500,color:#fff
       style LIB fill:#e8e8e8,stroke:#999

GUI Integration
---------------

Each worker has dedicated GUI form widgets for experiment configuration:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Worker
     - Form Widgets
     - Purpose
   * - **CleanRL**
     - ``cleanrl_train_form.py``
       ``cleanrl_script_form.py``
       ``cleanrl_resume_form.py``
       ``cleanrl_policy_form.py``
     - Standard training, custom scripts,
       checkpoint resume, policy evaluation
   * - **XuanCe**
     - ``xuance_train_form.py``
       ``xuance_script_form.py``
     - Standard training (with backend selection),
       custom scripts
   * - **Ray RLlib**
     - (Configured via Advanced Config)
     - Distributed training setup

.. toctree::
   :maxdepth: 1

   CleanRL_Worker/index
   XuanCe_Worker/index
   RLlib_Worker/index
   BALROG_Worker/index
