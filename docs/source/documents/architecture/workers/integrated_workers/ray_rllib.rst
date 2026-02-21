Ray RLlib Worker
================

The Ray RLlib worker is MOSAIC's **distributed multi-agent RL** integration.
It wraps `Ray RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_ —
Ray's scalable reinforcement learning library — behind the standard
:doc:`shim pattern <../concept>`, providing distributed training across
multiple CPUs or GPUs, PettingZoo multi-agent environment support, and
flexible policy configurations including self-play and independent learning.

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Single-agent, Multi-agent (parameter sharing, independent, self-play,
       `CTDE <https://arxiv.org/abs/1706.02275>`_)
   * - **Algorithms**
     - PPO, DQN, A2C,
       `IMPALA <https://arxiv.org/abs/1802.01561>`_,
       `APPO <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#appo>`_
   * - **Environments**
     - `PettingZoo <https://pettingzoo.farama.org/>`_ (SISL, Classic, Butterfly, MPE)
   * - **Execution**
     - Ray cluster (distributed across workers, optionally multi-GPU)
   * - **GPU required**
     - No (optional CUDA acceleration)
   * - **Source**
     - ``3rd_party/ray_worker/ray_worker/``

Architecture
------------

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Training Form<br/>(Advanced Config)"]
           DAEMON["Trainer Daemon"]
       end

       subgraph "Ray Head Process"
           CLI["cli.py<br/>entry point"]
           CFG["config.py<br/>RayWorkerConfig"]
           RT["runtime.py<br/>RayWorkerRuntime"]
           FL["fastlane.py<br/>FastLane telemetry"]
           SITE["sitecustomize.py"]
           AP["algo_params.py<br/>schema-based hyperparams"]
           PA["policy_actor.py<br/>inference actors"]
       end

       subgraph "Ray Workers"
           W0["Rollout Worker 0"]
           W1["Rollout Worker 1"]
           WN["Rollout Worker N"]
       end

       subgraph "Upstream RLlib"
           ALGO["PPO / DQN / IMPALA / APPO<br/>(unmodified RLlib algorithms)"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT --> AP
       RT --> ALGO
       RT --> PA
       ALGO --> W0
       ALGO --> W1
       ALGO --> WN
       FL -.->|"shared-memory frames"| DAEMON

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style FL fill:#ff7f50,stroke:#cc5500,color:#fff
       style SITE fill:#ff7f50,stroke:#cc5500,color:#fff
       style AP fill:#ff7f50,stroke:#cc5500,color:#fff
       style PA fill:#ff7f50,stroke:#cc5500,color:#fff
       style W0 fill:#c8e6c9,stroke:#388e3c
       style W1 fill:#c8e6c9,stroke:#388e3c
       style WN fill:#c8e6c9,stroke:#388e3c
       style ALGO fill:#e8e8e8,stroke:#999

**Lifecycle of a training run:**

1. The GUI form builds a config JSON and hands it to the Trainer Daemon.
2. The daemon spawns ``python -m ray_worker.cli --config <path>``.
3. ``cli.py`` loads the config into ``RayWorkerConfig`` and delegates
   to ``RayWorkerRuntime``.
4. ``RayWorkerRuntime.run()`` initialises Ray (``ray.init()``), builds
   the RLlib ``AlgorithmConfig``, and calls ``algorithm.train()``.
5. Ray distributes rollout workers across available CPUs/GPUs.
6. On completion the runtime saves a checkpoint and writes an
   analytics manifest to ``var/trainer/runs/``.

Supported Algorithms
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Algorithm
     - Type
     - Notes
   * - **PPO**
     - On-policy policy gradient
     - Default choice; works for both discrete and continuous action spaces
   * - **DQN**
     - Off-policy Q-learning
     - Discrete action spaces only
   * - **A2C**
     - On-policy actor-critic
     - Synchronous variant of A3C
   * - `IMPALA <https://arxiv.org/abs/1802.01561>`_
     - Distributed policy gradient
     - High-throughput asynchronous training
   * - `APPO <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#appo>`_
     - Asynchronous PPO
     - IMPALA with PPO-style clipping

Policy Configurations
---------------------

The worker supports four multi-agent policy configurations, controlled by
``PolicyConfiguration`` in ``config.py``:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Configuration
     - ``PolicyConfiguration`` value
     - Description
   * - **Parameter Sharing**
     - ``PARAMETER_SHARING``
     - All agents share one policy.  Sample-efficient for cooperative,
       homogeneous teams.
   * - **Independent**
     - ``INDEPENDENT``
     - Each agent has its own policy.  No coordination signal.
       Equivalent to running N independent PPO agents.
   * - **Self-Play**
     - ``SELF_PLAY``
     - Agent plays against frozen copies of itself.  Produces competitive
       policies without a fixed opponent.  Supports population-based training.
   * - **Shared Value Function**
     - ``SHARED_VALUE_FUNCTION``
     - `CTDE <https://arxiv.org/abs/1706.02275>`_: separate actors per agent,
       shared centralised critic. Equivalent to
       `MAPPO <https://arxiv.org/abs/2103.01955>`_ but within the RLlib framework.

Supported Environments
----------------------

The Ray worker integrates with `PettingZoo <https://pettingzoo.farama.org/>`_ environment families:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Family
     - Example Environments
     - Notes
   * - **SISL**
     - ``waterworld_v4``, ``multiwalker_v9``, ``pursuit_v4``
     - Cooperative continuous control; multiple agents share a reward
   * - **Classic**
     - ``chess_v6``, ``go_v5``, ``connect_four_v3``, ``tictactoe_v3``
     - Turn-based board games; AEC API
   * - **Butterfly**
     - ``knights_archers_zombies_v10``, ``cooperative_pong_v5``, ``pistonball_v6``
     - Real-time cooperative/competitive games
   * - **MPE**
     - ``simple_spread_v3``, ``simple_adversary_v3``, ``simple_tag_v3``
     - Multi-agent particle environments; cooperative and adversarial

Configuration
-------------

The ``RayWorkerConfig`` dataclass (``config.py``) composes several
sub-configs:

.. code-block:: python

   @dataclass
   class RayWorkerConfig:
       run_id: str
       environment: EnvironmentConfig    # env family + env_id + wrappers
       policy_configuration: PolicyConfiguration  # sharing / independent / self-play
       training: TrainingConfig          # algorithm, timesteps, hyperparams
       resources: ResourceConfig         # num_workers, num_gpus, num_cpus
       checkpoint: CheckpointConfig      # save frequency, checkpoint dir

   @dataclass
   class EnvironmentConfig:
       family: str       # "sisl", "classic", "butterfly", "mpe"
       env_id: str       # e.g. "waterworld_v4"
       api_type: PettingZooAPIType  # AEC or PARALLEL

   @dataclass
   class ResourceConfig:
       num_workers: int = 2     # Rollout workers (default: 2)
       num_gpus: float = 0.0    # GPU fraction for the head process
       num_cpus: int = 1        # CPUs per worker

Algorithm hyperparameters are schema-driven via ``algo_params.py``.
Each algorithm exposes a versioned JSON schema; the GUI reads the schema
to generate form fields dynamically.

Policy Actor and Evaluation
----------------------------

The worker ships a dedicated inference layer (``policy_actor.py``) for
loading trained RLlib checkpoints and running policy evaluation without
starting a full Ray cluster:

.. code-block:: python

   from ray_worker import RayPolicyConfig, create_ray_actor, run_evaluation

   actor = create_ray_actor(RayPolicyConfig(
       checkpoint_path="var/trainer/runs/my_run/checkpoint_000100",
       algorithm="PPO",
       env_id="waterworld_v4",
   ))

   results = run_evaluation(EvaluationConfig(
       actor=actor,
       num_episodes=20,
   ))

``RayPolicyController`` wraps multiple actors for multi-agent evaluation,
mapping each agent ID to its corresponding policy checkpoint.

FastLane Telemetry
------------------

FastLane streams render frames to the MOSAIC GUI via shared memory.
The Ray worker's ``fastlane.py`` hooks into RLlib's callback system to
emit frames on each rollout step without modifying the upstream algorithm.

GUI Integration
---------------

The Ray RLlib worker is configured via the **Advanced Config** panel in
the MOSAIC training dashboard.  Unlike CleanRL and XuanCe, it does not
currently have a dedicated form widget; all parameters are passed as a
raw JSON config.

Worker Discovery
----------------

The worker registers itself via the ``mosaic.workers`` entry point in
``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."mosaic.workers"]
   ray = "ray_worker:get_worker_metadata"

``get_worker_metadata()`` returns a ``WorkerCapabilities`` descriptor
advertising support for self-play, population-based training, pause/resume,
and up to 100 agents across the ``sisl``, ``classic``, ``butterfly``, and
``mpe`` environment families.

File Layout
-----------

.. code-block:: text

   3rd_party/ray_worker/ray_worker/
   +-- __init__.py            # Exports, get_worker_metadata()
   +-- cli.py                 # CLI entry point (--config, --dry-run)
   +-- config.py              # RayWorkerConfig and sub-config dataclasses
   +-- runtime.py             # RayWorkerRuntime, EnvironmentFactory
   +-- fastlane.py            # FastLane telemetry hooks
   +-- sitecustomize.py       # Import-time patches
   +-- algo_params.py         # Schema-based algorithm hyperparameter registry
   +-- policy_actor.py        # RayPolicyActor, RayPolicyController (inference)
   +-- policy_evaluator.py    # PolicyEvaluator, run_evaluation()
   +-- evaluation_results.py  # EpisodeMetrics, EvaluationResults dataclasses
   +-- analytics.py           # Run manifest
