What Is a Worker?
=================

Workers solve a fundamental problem in multi-framework RL research:
**how do you run CleanRL, Ray RLlib, XuanCe, and LLM agents in the
same platform without them conflicting?**

The Problem
-----------

RL libraries are designed to run standalone.  They each have their own:

- Dependency trees (often conflicting versions of NumPy, PyTorch, etc.)
- Logging systems (TensorBoard, Weights & Biases, custom loggers)
- Configuration formats (YAML, JSON, CLI args, Python dicts)
- GPU management strategies
- Error handling and signal traps

Running them inside a single process leads to import conflicts,
segfaults, and unpredictable behaviour.

.. mermaid::

   graph LR
       subgraph "❌ Single-Process (Fragile)"
           APP["Monolithic App"]
           APP --> CRL["CleanRL"]
           APP --> RAY["Ray RLlib"]
           APP --> XUA["XuanCe"]
           CRL -.->|"torch 2.0"| CONFLICT["⚠️ Version Conflict"]
           RAY -.->|"torch 2.1"| CONFLICT
       end

       style CONFLICT fill:#ff4444,stroke:#cc0000,color:#fff
       style APP fill:#ddd,stroke:#999

The Solution: Process Isolation
-------------------------------

MOSAIC runs every RL library in its **own OS process**.  The only
communication channel is a simple, well-defined IPC protocol.

.. mermaid::

   graph TB
       subgraph "✅ Multi-Process (Robust)"
           DAEMON["Trainer Daemon"]
           DAEMON -->|"spawn"| P1["Process 1<br/>CleanRL + torch 2.0"]
           DAEMON -->|"spawn"| P2["Process 2<br/>Ray RLlib + torch 2.1"]
           DAEMON -->|"spawn"| P3["Process 3<br/>XuanCe + torch 2.0"]
           P1 -->|"JSONL stdout"| DAEMON
           P2 -->|"JSONL stdout"| DAEMON
           P3 -->|"JSONL stdout"| DAEMON
       end

       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style P1 fill:#ff7f50,stroke:#cc5500,color:#fff
       style P2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style P3 fill:#ff7f50,stroke:#cc5500,color:#fff

This gives us:

- **Fault containment**: a crashed worker cannot freeze the GUI or
  kill other workers
- **Dependency freedom**: each worker can pin its own library versions
- **GPU isolation**: ``CUDA_VISIBLE_DEVICES`` is set per-process
- **Clean shutdown**: ``SIGTERM`` to the process group kills the
  entire worker tree

Worker vs Operator
------------------

MOSAIC distinguishes two concepts that are easy to confuse:

.. list-table::
   :header-rows: 1
   :widths: 15 42 43

   * - Concept
     - Definition
     - Examples
   * - **Worker**
     - A *process-level* execution unit that handles **training**, API
       calls, or bash script for custom workflows.  Workers live in ``3rd_party/``
       and communicate via JSONL over stdout.  The Trainer Daemon
       manages the worker process lifecycle, passing configuration
       via JSON files and environment variables.
     - ``cleanrl_worker``, ``xuance_worker``, ``ray_worker``,
       ``balrog_worker``, ``mosaic_llm_worker``
   * - **Operator**
     - An *agent-level* interface used strictly for **evaluation**.
       The operator assigns workers to agents, loads trained policies
       (or connects to LLM endpoints), and exposes
       ``select_action(obs) -> action`` to the GUI.  Operators support
       two modes: **Manual** (side-by-side comparison) and **Script**
       (automated long-running evaluation).
     - RL Operator, LLM Operator, Human Operator

Workers handle training **separately** from operators.  An RL worker
trains a policy and saves a checkpoint; an LLM worker connects to a
vLLM instance.  The :doc:`operator </documents/architecture/operators/index>`
then takes the result (trained policy or LLM endpoint) and assigns it
to an agent for evaluation via ``select_action()``.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph LR
       subgraph Training["Training Phase (Worker)"]
           WORKER["CleanRL Worker"]
           WORKER -->|"produces"| CKPT["Checkpoint"]
       end

       subgraph Eval["Evaluation Phase (Operator)"]
           OP["RL Operator"]
           SA["select_action(obs)"]
           OP --> SA
       end

       CKPT -->|"loaded by"| OP

       style Training fill:#ff7f50,stroke:#cc5500,color:#fff
       style Eval fill:#4a90d9,stroke:#2e5a87,color:#fff
       style CKPT fill:#f0e68c,stroke:#bdb76b

The Shim Pattern
----------------

Workers never modify the upstream library.  Instead, a thin **shim
layer** sits between MOSAIC and the library, translating everything:

.. mermaid::

   graph TB
       subgraph Core["MOSAIC Core"]
           D["Trainer Daemon"] ~~~ G["Qt6 GUI"]
       end

       subgraph Shim["MOSAIC Shim Layer"]
           S1["config.py<br/>Translate JSON → CLI"] ~~~ S2["runtime.py<br/>Manage lifecycle"] ~~~ S3["telemetry.py<br/>Emit JSONL"] ~~~ S4["analytics.py<br/>Generate manifests"] ~~~ S5["fastlane.py<br/>Shared-memory rendering"]
       end

       subgraph Upstream["Upstream Library -- Unmodified"]
           U1["ppo.py"] ~~~ U2["dqn.py"] ~~~ U3["sac.py"]
       end

       Core -->|"gRPC + JSONL"| Shim
       Shim --> Upstream

       style Core fill:#4a90d9,stroke:#2e5a87,color:#fff
       style Shim fill:#ff7f50,stroke:#cc5500,color:#fff
       style Upstream fill:#e8e8e8,stroke:#999

**Benefits:**

- Upstream libraries can be updated independently
- The shim is the only code that needs testing against MOSAIC
- Adding a new worker means writing a new shim, not forking a library
- Protocol changes only affect the shim layer

Directory Layout
----------------

Every worker follows the same structure:

.. code-block:: text

   3rd_party/my_worker/
   ├── pyproject.toml              # Package metadata + entry point
   ├── README.md
   ├── my_worker/                  # MOSAIC shim layer
   │   ├── __init__.py             # get_worker_metadata()
   │   ├── config.py               # WorkerConfig implementation
   │   ├── runtime.py              # Training orchestration
   │   ├── analytics.py            # Analytics manifest generation
   │   ├── telemetry.py            # JSONL lifecycle events
   │   ├── fastlane.py             # Shared-memory rendering
   │   └── cli.py                  # CLI entry point
   ├── upstream_lib/               # Vendored or pip-installed
   └── tests/
       └── test_standardization.py # Protocol compliance tests
