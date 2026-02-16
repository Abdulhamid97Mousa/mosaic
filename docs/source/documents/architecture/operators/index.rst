Operators
=========

An **Operator** is an agent-level abstraction that selects actions from
observations.  While :doc:`Workers <../workers/index>` handle process-level
concerns (training, telemetry, GPU isolation), Operators are the
**decision-makers** -- the entities that answer the question
*"given this observation, what action should I take?"*

.. mermaid::

   graph TB
       subgraph "MOSAIC Core"
           GUI["Qt6 GUI<br/>(Main Process)"]
           LAUNCHER["OperatorLauncher<br/>(Subprocess Manager)"]
       end

       subgraph "Operator Sub-Processes"
           O1["Human Operator<br/>Keyboard Input"]
           O2["LLM Operator<br/>BALROG / Chess"]
           O3["RL Operator<br/>CleanRL PPO"]
           O4["Baseline Operator<br/>Random / Scripted"]
       end

       GUI --> LAUNCHER
       LAUNCHER -- "stdin/stdout JSON" --> O1
       LAUNCHER -- "stdin/stdout JSON" --> O2
       LAUNCHER -- "stdin/stdout JSON" --> O3
       LAUNCHER -- "stdin/stdout JSON" --> O4

       style GUI fill:#4a90d9,stroke:#2e5a87,color:#fff
       style LAUNCHER fill:#50c878,stroke:#2e8b57,color:#fff
       style O1 fill:#9370db,stroke:#6a0dad,color:#fff
       style O2 fill:#9370db,stroke:#6a0dad,color:#fff
       style O3 fill:#9370db,stroke:#6a0dad,color:#fff
       style O4 fill:#9370db,stroke:#6a0dad,color:#fff

Key Principles
--------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Protocol-Based**
     - Operators implement Python ``Protocol`` classes -- no base class
       inheritance required.  Any object with ``select_action(obs)`` is
       a valid operator.
   * - **Category System**
     - Every operator belongs to a category: ``human``, ``llm``, ``rl``,
       ``bdi``, or ``baseline``.  The GUI adapts its configuration UI
       based on category.
   * - **Interactive Mode**
     - Operators run as subprocesses with ``--interactive`` flag, enabling
       step-by-step JSON commands over stdin/stdout.  This keeps the GUI
       responsive while operators compute.
   * - **Multi-Operator Comparison**
     - Multiple operators can run side-by-side on the same environment
       with shared seeds for scientific comparison (e.g., LLM vs RL on
       the same MiniGrid layout).
   * - **Decoupled Execution**
     - Manual mode (click-to-step) and Script mode (automated experiments)
       are fully independent code paths with separate state machines.

Available Operators
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 15 32 35

   * - Operator
     - Category
     - Backend
     - Use Case
   * - **Human**
     - human
     - Keyboard input via GUI
     - Manual play and debugging
   * - **BALROG LLM**
     - llm
     - barlog_worker (vLLM, OpenRouter)
     - LLM agent benchmarking on MiniGrid/BabyAI
   * - **MOSAIC LLM**
     - llm
     - operators_worker (vLLM, OpenAI API)
     - Multi-agent LLM evaluation
   * - **Chess LLM**
     - llm
     - chess_worker (llm_chess prompting)
     - LLM chess play with multi-turn dialog
   * - **CleanRL**
     - rl
     - cleanrl_worker (PPO, DQN)
     - Trained RL policy evaluation
   * - **Random Baseline**
     - baseline
     - operators_worker (random action)
     - Baseline comparison for experiments

.. tip::

   Operators are distinct from Workers.  A Worker trains policies and
   manages library lifecycles.  An Operator *uses* a trained policy
   (or LLM, or human input) to select actions step-by-step.  See
   :doc:`concept` for the full distinction.


.. toctree::
   :maxdepth: 2

   concept
   architecture
   lifecycle
   development
   examples
