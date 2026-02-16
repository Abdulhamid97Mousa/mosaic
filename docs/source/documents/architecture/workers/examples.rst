Worker Examples
===============

MOSAIC ships with three production workers.  Each demonstrates a
different integration strategy: vendored source, pip dependency,
and local package.

CleanRL Worker
--------------

**Strategy:** Git submodule (vendored source)

The CleanRL worker wraps `CleanRL <https://github.com/vwxyzjn/cleanrl>`_,
a collection of single-file RL implementations, as a MOSAIC-compatible
subprocess.

.. code-block:: bash

   pip install -e ".[cleanrl]"

**Supported Algorithms:** PPO, DQN, SAC, TD3, DDPG, C51

.. mermaid::

   graph TB
       subgraph "cleanrl_worker/ (Shim)"
           CLI["cli.py<br/>Entry point"]
           CFG["config.py<br/>MOSAIC JSON → CLI args"]
           RT["runtime.py<br/>Lifecycle management"]
           TEL["telemetry.py<br/>JSONL emission"]
           FL["fastlane.py<br/>Shared-memory rendering"]
           AN["analytics.py<br/>TB/W&B manifests"]
       end

       subgraph "cleanrl/ (Upstream)"
           PPO["ppo.py"]
           DQN["dqn.py"]
           SAC["sac.py"]
       end

       CLI --> CFG --> RT
       RT --> TEL
       RT --> FL
       RT --> AN
       RT -->|"subprocess"| PPO
       RT -->|"subprocess"| DQN
       RT -->|"subprocess"| SAC

       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style TEL fill:#ff7f50,stroke:#cc5500,color:#fff
       style FL fill:#ff7f50,stroke:#cc5500,color:#fff
       style AN fill:#ff7f50,stroke:#cc5500,color:#fff
       style PPO fill:#e8e8e8,stroke:#999
       style DQN fill:#e8e8e8,stroke:#999
       style SAC fill:#e8e8e8,stroke:#999

**Directory layout:**

.. code-block:: text

   3rd_party/cleanrl_worker/
   ├── cleanrl/                  ← Git submodule (upstream)
   │   └── cleanrl/
   │       ├── ppo.py
   │       ├── dqn.py
   │       └── sac.py
   ├── cleanrl_worker/           ← MOSAIC shim
   │   ├── __init__.py
   │   ├── cli.py
   │   ├── config.py
   │   ├── runtime.py
   │   ├── telemetry.py
   │   ├── analytics.py
   │   ├── fastlane.py
   │   └── algorithms/
   │       └── ppo_with_save.py
   ├── pyproject.toml
   └── tests/

**How it works:**

1. The Daemon spawns ``python -m cleanrl_worker.cli --config worker.json``
2. ``cli.py`` parses the MOSAIC config and creates a ``CleanRLWorkerRuntime``
3. ``runtime.py`` translates config fields to CleanRL CLI arguments
4. CleanRL runs as a subprocess; ``sitecustomize.py`` injects telemetry hooks
5. Training metrics are emitted as JSONL to stdout
6. The Telemetry Proxy reads stdout and streams to the Daemon

**Configuration example:**

.. code-block:: python

   config = {
       "algorithm": "ppo",
       "learning_rate": 3e-4,
       "total_timesteps": 1_000_000,
       "num_envs": 4,
       "capture_video": True,
   }


XuanCe Worker
-------------

**Strategy:** Local package (``3rd_party/xuance_worker/``)

The XuanCe worker integrates `XuanCe <https://github.com/agi-brain/xuance>`_,
a comprehensive multi-agent RL library, for cooperative and competitive
training.

.. code-block:: bash

   pip install -e ".[xuance]"

**Supported Algorithms:** MAPPO, QMIX, MADDPG, VDN, COMA, IPPO, IQL

.. mermaid::

   graph TB
       subgraph "xuance_worker/ (Shim)"
           CLI2["cli.py"]
           CFG2["config.py"]
           RT2["runtime.py"]
           TEL2["telemetry.py"]
       end

       subgraph "xuance/ (Upstream)"
           MAPPO["MAPPO"]
           QMIX["QMIX"]
           VDN["VDN"]
       end

       CLI2 --> CFG2 --> RT2 --> TEL2
       RT2 -->|"in-process"| MAPPO
       RT2 -->|"in-process"| QMIX
       RT2 -->|"in-process"| VDN

       style CLI2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style TEL2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style MAPPO fill:#e8e8e8,stroke:#999
       style QMIX fill:#e8e8e8,stroke:#999
       style VDN fill:#e8e8e8,stroke:#999

**Key difference from CleanRL:** XuanCe algorithms are called
**in-process** (imported as Python modules) rather than spawned as
subprocesses.  This allows tighter integration with MOSAIC's multi-agent
environment adapters.

**Directory layout:**

.. code-block:: text

   3rd_party/xuance_worker/
   ├── xuance/                   ← Upstream XuanCe source
   │   └── xuance/
   │       ├── learner/
   │       ├── agent/
   │       └── environment/
   ├── xuance_worker/            ← MOSAIC shim
   │   ├── __init__.py
   │   ├── cli.py
   │   ├── config.py
   │   ├── runtime.py
   │   └── telemetry.py
   └── pyproject.toml

**Configuration example:**

.. code-block:: python

   config = {
       "algorithm": "mappo",
       "learning_rate": 5e-4,
       "batch_size": 256,
       "backend": "torch",
   }

RLlib Worker
------------

**Strategy:** Pip dependency (``ray[rllib]``)

The RLlib worker uses `Ray RLlib <https://docs.ray.io/en/latest/rllib/>`_
for distributed, GPU-accelerated training at scale.

.. code-block:: bash

   pip install -e ".[ray-rllib]"

**Supported Algorithms:** PPO, IMPALA, APPO, SAC, DQN

.. mermaid::

   graph TB
       subgraph "ray_worker/ (Shim)"
           CLI3["cli.py"]
           CFG3["config.py"]
           RT3["runtime.py"]
           CB["callbacks.py<br/>RLlib callback hooks"]
       end

       subgraph "Ray Cluster"
           HEAD["Ray Head"]
           ACT1["Actor 1"]
           ACT2["Actor 2"]
           ACT3["Actor N"]
           HEAD --> ACT1
           HEAD --> ACT2
           HEAD --> ACT3
       end

       CLI3 --> CFG3 --> RT3
       RT3 --> CB
       RT3 -->|"ray.init()"| HEAD

       style CLI3 fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG3 fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT3 fill:#ff7f50,stroke:#cc5500,color:#fff
       style CB fill:#ff7f50,stroke:#cc5500,color:#fff
       style HEAD fill:#e8e8e8,stroke:#999
       style ACT1 fill:#e8e8e8,stroke:#999
       style ACT2 fill:#e8e8e8,stroke:#999
       style ACT3 fill:#e8e8e8,stroke:#999

**Key difference:** RLlib manages its own cluster of Ray actors.
The shim translates MOSAIC configs to RLlib ``AlgorithmConfig``
objects and hooks into RLlib's callback system to emit MOSAIC
telemetry.

**Directory layout:**

.. code-block:: text

   3rd_party/ray_worker/
   ├── ray_worker/               ← MOSAIC shim
   │   ├── __init__.py
   │   ├── cli.py
   │   ├── config.py
   │   ├── runtime.py
   │   ├── telemetry.py
   │   ├── callbacks.py
   │   └── fastlane.py
   └── pyproject.toml

**Configuration example:**

.. code-block:: python

   config = {
       "algorithm": "PPO",
       "num_workers": 8,
       "num_envs_per_worker": 4,
       "framework": "torch",
   }

Comparison
----------

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26

   * - Aspect
     - CleanRL
     - XuanCe
     - RLlib
   * - **Upstream Source**
     - Git submodule
     - Local package
     - pip (``ray[rllib]``)
   * - **Execution Model**
     - Subprocess
     - In-process
     - Ray cluster
   * - **Paradigm**
     - Single-agent
     - Multi-agent
     - Both
   * - **GPU Usage**
     - 0–1 GPU
     - 0–1 GPU
     - Multi-GPU
   * - **Best For**
     - Quick experiments,
       reproducible baselines
     - Multi-agent RL research
       (SMAC, PettingZoo)
     - Large-scale distributed
       training
   * - **Complexity**
     - Low
     - Medium
     - High
