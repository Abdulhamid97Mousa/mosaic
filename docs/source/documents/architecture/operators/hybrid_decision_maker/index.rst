Hybrid Decision-Maker
=====================

A **hybrid** setup is one where agents in the same experiment use
**different decision-making paradigms** -- for example, an RL-trained
policy and an LLM playing side-by-side as teammates, or an RL agent
competing against an LLM agent.

This is MOSAIC's key innovation and what distinguishes it from every
other RL or LLM framework.

The Research Gap
----------------

Existing frameworks are paradigm-siloed:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Framework
     - RL
     - LLM
     - Cross-Paradigm
   * - RLlib, CleanRL, XuanCe
     - Yes
     - No
     - No
   * - BALROG, AgentBench
     - No
     - Yes
     - No
   * - TextArena
     - No
     - Yes (vs Human)
     - No
   * - **MOSAIC**
     - **Yes**
     - **Yes**
     - **Yes**

No prior framework allowed **fair, reproducible, head-to-head
comparison** between RL agents and LLM agents in the same multi-agent
environment.  The root cause is an interface mismatch -- RL agents
expect tensor observations and produce integer actions, while LLM
agents expect text prompts and produce text responses.

The Gymnasium Analogy
---------------------

Gymnasium (Towers et al., 2024) standardized the **environment**
interface: every environment implements ``reset()`` and ``step()``, so
any algorithm can interact with any environment without modification.

No equivalent standardization existed for the **agent** side.  MOSAIC's
Operator Protocol fills this gap:

.. mermaid::

   graph LR
       subgraph "Gymnasium (Environments)"
           E1["CartPole"]
           E2["MiniGrid"]
           E3["Chess"]
       end

       EPROTO["reset() / step()<br/>Unified Env Interface"]

       subgraph "MOSAIC (Agents)"
           A1["RL Policy"]
           A2["LLM Agent"]
           A3["Human Player"]
       end

       APROTO["select_action(obs)<br/>Unified Agent Interface"]

       E1 --> EPROTO
       E2 --> EPROTO
       E3 --> EPROTO
       A1 --> APROTO
       A2 --> APROTO
       A3 --> APROTO

       style EPROTO fill:#4a90d9,stroke:#2e5a87,color:#fff
       style APROTO fill:#50c878,stroke:#2e8b57,color:#fff
       style E1 fill:#ddd,stroke:#999,color:#333
       style E2 fill:#ddd,stroke:#999,color:#333
       style E3 fill:#ddd,stroke:#999,color:#333
       style A1 fill:#ddd,stroke:#999,color:#333
       style A2 fill:#ddd,stroke:#999,color:#333
       style A3 fill:#ddd,stroke:#999,color:#333

Just as Gymnasium made environments interchangeable, the Operator
Protocol makes **agents interchangeable** -- any decision-maker can be
plugged into any compatible environment without modifying either side.

How Hybrid Teams Work
---------------------

The ``WorkerAssignment`` system maps each agent slot in a multi-agent
environment to a specific worker subprocess.  A single
``OperatorConfig`` can freely mix RL, LLM, human, and baseline workers
across agent slots:

.. code-block:: python

   # Hybrid team: RL + LLM in 2v2 soccer
   config = OperatorConfig.multi_agent(
       operator_id="hybrid_team",
       display_name="RL + LLM Hybrid vs RL + Random",
       env_name="multigrid",
       task="MultiGrid-Soccer-2v2-v0",
       player_workers={
           # Green team: hybrid (RL + LLM)
           "green_0": WorkerAssignment(
               worker_id="cleanrl_worker",
               worker_type="rl",
               settings={"algorithm": "ppo", "checkpoint": "mappo_1v1.pt"},
           ),
           "green_1": WorkerAssignment(
               worker_id="mosaic_llm_worker",
               worker_type="llm",
               settings={
                   "client_name": "openrouter",
                   "model_id": "gpt-4o",
                   "temperature": 0,
                   "coordination_level": 2,
                   "observation_mode": "visible_teammates",
               },
           ),
           # Blue team: crippled baseline (RL + Random)
           "blue_0": WorkerAssignment(
               worker_id="cleanrl_worker",
               worker_type="rl",
               settings={"algorithm": "ppo", "checkpoint": "mappo_1v1.pt"},
           ),
           "blue_1": WorkerAssignment(
               worker_id="operators_worker",
               worker_type="baseline",
               settings={},
           ),
       },
   )

This creates four agent slots, each backed by a different subprocess:

.. mermaid::

   graph TB
       ENV["MultiGrid Soccer 2v2<br/>(PettingZoo AEC)"]

       subgraph "Green Team (Hybrid)"
           G0["green_0: RL<br/>cleanrl_worker<br/>MAPPO checkpoint"]
           G1["green_1: LLM<br/>mosaic_llm_worker<br/>GPT-4o"]
       end

       subgraph "Blue Team (Crippled)"
           B0["blue_0: RL<br/>cleanrl_worker<br/>MAPPO checkpoint"]
           B1["blue_1: Baseline<br/>operators_worker<br/>Random actions"]
       end

       ENV -- "obs" --> G0
       ENV -- "obs" --> G1
       ENV -- "obs" --> B0
       ENV -- "obs" --> B1
       G0 -- "action" --> ENV
       G1 -- "action" --> ENV
       B0 -- "action" --> ENV
       B1 -- "action" --> ENV

       style ENV fill:#4a90d9,stroke:#2e5a87,color:#fff
       style G0 fill:#50c878,stroke:#2e8b57,color:#fff
       style G1 fill:#50c878,stroke:#2e8b57,color:#fff
       style B0 fill:#ff7f50,stroke:#cc5500,color:#fff
       style B1 fill:#ff7f50,stroke:#cc5500,color:#fff

All four agents receive observations from the same environment, with
the same seed, on the same timestep -- yet each uses a completely
different decision-making mechanism.  The environment only sees
``select_action(obs) -> action``, regardless of what runs inside.

Multi-Worker Pattern
--------------------

The hybrid setup uses the **multi-worker pattern**: one Operator wraps
N Worker subprocesses via the ``OperatorController`` protocol:

.. code-block:: python

   class OperatorController(Protocol):
       """Multi-agent extension of the Operator Protocol."""

       def select_action(
           self, agent_id: str, observation: Any, info: Any = None,
       ) -> Any:
           """AEC mode: one agent acts at a time."""
           ...

       def select_actions(
           self, observations: Dict[str, Any],
       ) -> Dict[str, Any]:
           """Parallel mode: all agents act simultaneously."""
           ...

Each worker runs as a **separate OS process**, communicating via
JSONL-over-stdout.  This process isolation means:

- A crashed worker never takes down the GUI or other workers
- Each worker can use different Python dependencies, GPU allocations,
  or even different Python versions
- Integration effort is minimal and non-invasive to upstream libraries

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Worker
     - LOC Added
     - Modifications to Original Library
   * - CleanRL DQN (~300 LOC)
     - ~50 LOC (harness)
     - Zero
   * - BALROG Agent (~500 LOC)
     - ~80 LOC (runtime.py)
     - Zero
   * - XuanCe MAPPO (~2000 LOC)
     - ~120 LOC (wrapper)
     - Zero

Experimental Configurations
---------------------------

Hybrid decision-making enables a systematic ablation matrix for
cross-paradigm research.  Here are examples using 2v2 soccer:

Adversarial Cross-Paradigm
^^^^^^^^^^^^^^^^^^^^^^^^^^

Testing how paradigms perform **against** each other:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Configuration
     - Team A
     - Team B
     - Purpose
   * - RL vs RL
     - MAPPO + MAPPO
     - MAPPO + MAPPO
     - Homogeneous RL baseline
   * - LLM vs LLM
     - GPT-4o + GPT-4o
     - GPT-4o + GPT-4o
     - Homogeneous LLM baseline
   * - RL vs LLM
     - MAPPO + MAPPO
     - GPT-4o + GPT-4o
     - Cross-paradigm matchup
   * - RL vs Random
     - MAPPO + MAPPO
     - Random + Random
     - Sanity check

Cooperative Hybrid Teams
^^^^^^^^^^^^^^^^^^^^^^^^

Testing how paradigms work **together** as teammates:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Configuration
     - Green Team
     - Blue Team
     - Research Question
   * - Hybrid vs Crippled
     - RL + LLM
     - RL + Random
     - Does LLM reasoning beat random as a teammate?
   * - Hybrid vs Solo
     - RL + LLM
     - RL + NoOp
     - Does the LLM actively help or just "not hurt"?
   * - Solo-pair vs Solo-pair
     - RL + RL
     - RL + RL
     - Uncoordinated RL pair baseline
   * - Hybrid vs Co-trained
     - RL + LLM
     - RL(2v2) + RL(2v2)
     - Can zero-shot LLM match million-step co-training?

.. important::

   The **1v1-to-2v2 transfer design** is critical: RL agents are
   trained as solo experts (1v1), then deployed as teammates alongside
   an LLM in 2v2.  This eliminates the co-training confound -- the RL
   agent has no partner expectations because it never had a partner.

Deterministic Cross-Paradigm Evaluation
---------------------------------------

Shared seed schedules are distributed to all operators via
``OperatorService.seed()``.  Full trajectories are logged under unified
telemetry.  This produces **directly comparable results** across
paradigms -- the first time this has been possible.

.. mermaid::

   graph LR
       SEEDS["Seed Schedule<br/>[42, 43, 44, ..., 141]"]

       SEEDS --> RL["RL Operator<br/>same 100 seeds"]
       SEEDS --> LLM["LLM Operator<br/>same 100 seeds"]
       SEEDS --> HYB["Hybrid Team<br/>same 100 seeds"]
       SEEDS --> BASE["Baseline<br/>same 100 seeds"]

       RL --> TEL["Unified Telemetry<br/>JSONL logs"]
       LLM --> TEL
       HYB --> TEL
       BASE --> TEL

       style SEEDS fill:#4a90d9,stroke:#2e5a87,color:#fff
       style RL fill:#9370db,stroke:#6a0dad,color:#fff
       style LLM fill:#9370db,stroke:#6a0dad,color:#fff
       style HYB fill:#50c878,stroke:#2e8b57,color:#fff
       style BASE fill:#ddd,stroke:#999,color:#333
       style TEL fill:#ff7f50,stroke:#cc5500,color:#fff

The ``MultiOperatorService`` manages N operators in parallel, each
with its own environment instance but sharing the same seed:

.. code-block:: python

   class MultiOperatorService:
       def add_operator(self, config: OperatorConfig) -> None: ...
       def remove_operator(self, operator_id: str) -> None: ...
       def get_active_operators(self) -> list[OperatorConfig]: ...
       def start_all(self) -> None: ...
       def stop_all(self) -> None: ...

Research Questions
------------------

The hybrid decision-maker architecture enables research questions that
were previously impossible to investigate:

.. list-table::
   :header-rows: 1
   :widths: 10 60 30

   * - #
     - Question
     - Priority
   * - Q1
     - Does an LLM teammate help a solo-trained RL agent in 2v2?
     - High
   * - Q2
     - Does the LLM actively help or just "not hurt"? (vs NoOp)
     - High
   * - Q3
     - Is an LLM teammate better than a second solo RL agent?
     - High
   * - Q4
     - Can zero-shot LLM teaming compete with million-step co-training?
     - High
   * - Q5
     - What role does the LLM play? (ball possession, assists, defense)
     - High
   * - Q6
     - Is hybrid advantage robust across seeds?
     - High
   * - Q7
     - Does hybrid advantage generalize across environments?
     - Medium

These questions span four possible outcomes:

- **Hypothesis A (RL Carries)**: the LLM adds nothing -- hybrid
  performance matches RL + dummy
- **Hypothesis B (LLM Carries)**: the RL agent adds nothing -- hybrid
  performance matches dummy + LLM
- **Hypothesis C (True Synergy)**: hybrid outperforms both homogeneous
  teams -- the most publishable result
- **Hypothesis D (Interference)**: hybrid underperforms both
  homogeneous teams -- still publishable as a negative result

GUI Integration
---------------

The hybrid decision-maker is configured through the
``OperatorsTab`` in the GUI, which provides two execution modes:

**Manual Mode** -- step-by-step execution where the user clicks
"Step All" or "Step Player" to advance each timestep.  Useful for
debugging and observing agent behavior:

.. mermaid::

   graph LR
       USER["User"]
       OT["OperatorsTab<br/>Manual Mode"]
       OCW["OperatorConfigWidget<br/>(up to 8 operators)"]
       ORC["OperatorRenderContainer<br/>(per-operator viewport)"]

       USER -- "configure" --> OCW
       USER -- "Step All" --> OT
       OT -- "step signal" --> ORC

       style USER fill:#eee,stroke:#999,color:#333
       style OT fill:#4a90d9,stroke:#2e5a87,color:#fff
       style OCW fill:#4a90d9,stroke:#2e5a87,color:#fff
       style ORC fill:#4a90d9,stroke:#2e5a87,color:#fff

**Script Mode** -- automated batch experiments that run N episodes
across M seed values without user interaction.  Managed by
``OperatorScriptExecutionManager``:

.. mermaid::

   graph LR
       SCRIPT["ScriptExperimentWidget<br/>define seed range + episodes"]
       MGR["ScriptExecutionManager<br/>state machine"]
       OPS["MultiOperatorService<br/>N operators in parallel"]
       TEL["Unified Telemetry<br/>JSONL logs per operator"]

       SCRIPT --> MGR
       MGR --> OPS
       OPS --> TEL

       style SCRIPT fill:#4a90d9,stroke:#2e5a87,color:#fff
       style MGR fill:#50c878,stroke:#2e8b57,color:#fff
       style OPS fill:#ff7f50,stroke:#cc5500,color:#fff
       style TEL fill:#ff7f50,stroke:#cc5500,color:#fff

Each operator gets its own ``OperatorRenderContainer`` with a
color-coded type badge (LLM=blue, RL=purple, Human=orange) and
live observation rendering, making it easy to visually compare how
different paradigms behave on the same environment state.
