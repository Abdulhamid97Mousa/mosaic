XuanCe Worker
=============

The XuanCe worker is MOSAIC's **multi-agent and comprehensive RL** integration.
It wraps the `XuanCe <https://github.com/agi-brain/xuance>`_ library — a
unified deep RL library with 46+ algorithms across single-agent, multi-agent,
and offline RL — behind the standard :doc:`shim pattern <../concept>`, adding
subprocess isolation, FastLane telemetry, curriculum learning, and GUI
configuration.

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Single-agent, Multi-agent (parameter sharing, independent)
   * - **Algorithms**
     - 46+ including PPO, DQN, SAC, MAPPO, QMIX, MADDPG, VDN, COMA
   * - **Backends**
     - PyTorch (primary), TensorFlow, MindSpore
   * - **Environments**
     - `Gymnasium <https://gymnasium.farama.org/>`_, Atari, MuJoCo,
       `PettingZoo <https://pettingzoo.farama.org/>`_,
       `SMAC <https://github.com/oxwhirl/smac>`_, MPE, MultiGrid, MultiGrid Soccer
   * - **Execution**
     - In-process (single OS process, vectorized environments)
   * - **GPU required**
     - No (optional CUDA acceleration)
   * - **Source**
     - ``3rd_party/xuance_worker/xuance_worker/``

.. note::

   **Known Import Issue**: The ``mpi4py`` package triggers ``MPI_Init()`` at
   import time, which blocks indefinitely outside an MPI launch environment.
   The worker sets ``MPI4PY_RC_INITIALIZE=0`` to suppress this. If you see
   the worker hanging on startup, verify this environment variable is set.
   Multi-agent runs that require MPI (e.g. SMAC on HPC clusters) must be
   launched via ``mpirun``.

Architecture
------------

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Training Form<br/>(XuanCe widgets)"]
           DAEMON["Trainer Daemon"]
       end

       subgraph "Worker Process"
           CLI["cli.py<br/>entry point"]
           CFG["config.py<br/>XuanCeWorkerConfig"]
           RT["runtime.py<br/>XuanCeWorkerRuntime"]
           FL["fastlane.py<br/>FastLane telemetry"]
           SITE["sitecustomize.py<br/>import-time patches"]
           AR["algorithm_registry.py<br/>Backend / Paradigm index"]
           SHIMS["xuance_shims.py<br/>path + dir redirects"]
       end

       subgraph "Upstream XuanCe"
           RUNNER["RunnerDRL / RunnerMARL<br/>RunnerPettingzoo<br/>(unmodified)"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT --> AR
       RT --> RUNNER
       SITE -.->|"import-time patches"| RUNNER
       SHIMS -.->|"redirect logs/checkpoints to var/"| RUNNER
       FL -.->|"shared-memory frames"| DAEMON

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style FL fill:#ff7f50,stroke:#cc5500,color:#fff
       style SITE fill:#ff7f50,stroke:#cc5500,color:#fff
       style AR fill:#ff7f50,stroke:#cc5500,color:#fff
       style SHIMS fill:#ff7f50,stroke:#cc5500,color:#fff
       style RUNNER fill:#e8e8e8,stroke:#999

**Lifecycle of a training run:**

1. The GUI form builds a config JSON and hands it to the Trainer Daemon.
2. The daemon spawns ``python -m xuance_worker.cli --config <path>``.
3. ``cli.py`` loads the config into ``XuanCeWorkerConfig`` and delegates
   to ``XuanCeWorkerRuntime``.
4. ``xuance_shims.py`` redirects XuanCe's hardcoded output paths
   (logs, checkpoints, TensorBoard) into MOSAIC's ``var/`` directory.
5. ``runtime.py`` calls ``xuance.get_runner()`` with the resolved
   algorithm, environment family, and parser args.
6. XuanCe's runner (``RunnerDRL`` for single-agent, ``RunnerMARL`` for
   multi-agent) executes the training loop.
7. FastLane telemetry streams render frames to the GUI via shared memory.

Supported Algorithms
--------------------

Algorithms are indexed in ``algorithm_registry.py`` by ``Backend`` and
``Paradigm``. The table below shows the primary families:

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Family
     - Algorithms
     - Paradigm
     - Notes
   * - **Policy Gradient**
     - PPO, A2C, A3C, PG, PDPG
     - Single-agent
     - Stable on-policy training
   * - **Q-Learning**
     - DQN, DDQN, Dueling DQN, NoisyDQN, PerDQN, C51, QRDQN
     - Single-agent
     - Discrete action spaces
   * - **Actor-Critic (continuous)**
     - SAC, TD3, DDPG, MASAC
     - Single-agent
     - Continuous control
   * - **Model-Based**
     - DreamerV3
     - Single-agent
     - World-model planning
   * - **Cooperative MARL**
     - `MAPPO <https://arxiv.org/abs/2103.01955>`_,
       `QMIX <https://arxiv.org/abs/1803.11605>`_,
       VDN, `COMA <https://arxiv.org/abs/1705.08926>`_, MADDPG, IDDPG
     - Multi-agent
     - `CTDE <https://arxiv.org/abs/1706.02275>`_ paradigm
   * - **Competitive MARL**
     - MAPPO (self-play), MADDPG
     - Multi-agent
     - Adversarial training

Runners
-------

XuanCe selects the training runner based on the environment family:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Runner
     - Environment Family
     - Use Case
   * - ``RunnerDRL``
     - ``classic_control``, ``atari``, ``mujoco``, ``minigrid``
     - Standard single-agent Gymnasium environments
   * - ``RunnerMARL``
     - ``multigrid``, ``smac``
     - Cooperative multi-agent (CTDE algorithms)
   * - ``RunnerPettingzoo``
     - ``mpe``, ``pettingzoo``
     - `PettingZoo <https://pettingzoo.farama.org/>`_ AEC and parallel API environments
   * - ``RunnerStarCraft2``
     - ``smac``
     - StarCraft Multi-Agent Challenge (requires SC2 installation)
   * - ``RunnerFootball``
     - ``football``
     - Google Research Football

Configuration
-------------

The ``XuanCeWorkerConfig`` dataclass (``config.py``) is the single source
of truth for all run parameters:

.. code-block:: python

   @dataclass
   class XuanCeWorkerConfig:
       run_id: str           # ULID-format unique run identifier
       method: str           # Algorithm name ("ppo", "mappo", "qmix", ...)
       env: str              # Environment family ("classic_control", "multigrid", ...)
       env_id: str           # Specific env ID ("CartPole-v1", "soccer_1vs1", ...)
       dl_toolbox: str       # Backend: "torch" (default), "tensorflow", "mindspore"
       running_steps: int    # Total training timesteps (default: 1_000_000)
       seed: int | None      # Random seed (None = random)
       device: str           # "cpu" or "cuda:0"
       parallels: int        # Number of parallel environments (default: 8)
       test_mode: bool       # True = evaluation mode (load checkpoint, no training)
       config_path: str | None  # Custom YAML config (None = XuanCe defaults)
       extras: dict          # Algorithm-specific overrides

Key ``extras`` fields:

- ``training_mode`` — ``"cooperative"`` or ``"competitive"`` (for MARL)
- ``curriculum_schedule`` — list of ``{"env_id": ..., "steps": ...}`` dicts
- ``tensorboard_dir`` — relative path for TensorBoard logs
- ``checkpoint_dir`` — relative path for model checkpoints
- ``num_envs`` — alias for ``parallels`` (used by some MARL algorithms)

Curriculum Training
-------------------

The XuanCe worker supports **single-process curriculum training** via
``multi_agent_curriculum_training.py``.  Unlike the two-process approach,
it hot-swaps environments in memory, preserving the Adam optimizer
momentum and learning-rate schedule across phases.

.. code-block:: json

   {
     "curriculum_schedule": [
       {"env_id": "collect_1vs1", "steps": 1000000},
       {"env_id": "soccer_1vs1",  "steps": 4000000}
     ]
   }

Both environments must share the same observation and action spaces so
the network architecture requires no modification between phases.

Multi-Agent Configuration
--------------------------

For MARL algorithms (MAPPO, QMIX, etc.), two key choices affect the
checkpoint format and deployment:

**Parameter Sharing** (``use_parameter_sharing=True``) — see `MARL Book ch. 5 <https://www.marl-book.com/>`_:

All agents share one policy network.  The network input is
``obs_dim + n_agents`` (a one-hot agent identity is appended).
This is more sample-efficient for symmetric games but creates a
dimension dependency on ``n_agents`` at inference time.

**Independent Networks** (``use_parameter_sharing=False``):

Each agent has its own separate policy.  Network input is ``obs_dim``
only.  Checkpoints are fully self-contained and can be loaded for
any agent slot without configuration.

.. warning::

   If you train with parameter sharing on a 1v1 environment
   (``n_agents=2``) and then deploy in a 2v2 environment
   (``n_agents=4``), the actor's first linear layer will have an
   input dimension mismatch (``obs+2`` vs ``obs+4``).  Either train
   with ``use_parameter_sharing=False``, or bypass
   ``agent.action()`` and construct the one-hot manually at inference.

FastLane Telemetry
------------------

FastLane streams render frames from the training process to the MOSAIC
GUI via shared memory.  Environment variables controlling behaviour:

- ``GYM_GUI_FASTLANE_ONLY`` — ``1`` to stream, ``0`` to disable
- ``GYM_GUI_FASTLANE_SLOT`` — which parallel env index to probe
- ``GYM_GUI_FASTLANE_VIDEO_MODE`` — ``"single"`` or ``"grid"``
- ``GYM_GUI_FASTLANE_GRID_LIMIT`` — max envs to tile in grid mode

GUI Integration
---------------

The XuanCe worker provides two form widgets in ``gym_gui/ui/widgets/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Form
     - Purpose
   * - ``xuance_train_form.py``
     - Primary training dialog.  Algorithm and environment family selection,
       deep learning backend toggle (PyTorch / TensorFlow / MindSpore),
       hyperparameter configuration, FastLane and TensorBoard settings.
   * - ``xuance_script_form.py``
     - Custom shell script launcher for multi-phase curriculum runs.
       Reads ``*.sh`` scripts with inline ``@description``, ``@phases``,
       and ``@total_timesteps`` metadata.

Worker Discovery
----------------

The worker registers itself via the ``mosaic.workers`` entry point in
``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."mosaic.workers"]
   xuance = "xuance_worker:get_worker_metadata"

``get_worker_metadata()`` returns a ``WorkerCapabilities`` descriptor
advertising support for up to 100 agents, discrete and continuous action
spaces, and the ``multigrid``, ``smac``, ``mpe``, and ``pettingzoo``
environment families.

File Layout
-----------

.. code-block:: text

   3rd_party/xuance_worker/xuance_worker/
   +-- __init__.py                       # Lazy exports, get_worker_metadata()
   +-- cli.py                            # CLI entry point (--config, --dry-run)
   +-- config.py                         # XuanCeWorkerConfig dataclass
   +-- runtime.py                        # XuanCeWorkerRuntime, InteractiveRuntime
   +-- fastlane.py                       # FastLane telemetry integration
   +-- sitecustomize.py                  # Import-time patches (mpi4py, paths)
   +-- xuance_shims.py                   # Redirect XuanCe output to var/
   +-- algorithm_registry.py            # Backend / Paradigm / AlgorithmInfo index
   +-- analytics.py                      # Run summary manifest
   +-- multi_agent_curriculum_training.py  # In-memory env-swap curriculum
   +-- single_agent_curriculum_training.py # Single-agent curriculum variant
   +-- _compat.py                        # Backwards-compatibility helpers
   +-- _patches.py                       # XuanCe monkey-patches
   +-- wrappers/
   |   +-- curriculum.py                # Curriculum env wrappers
   +-- environments/
   |   +-- mosaic_multigrid.py          # MultiGrid env registration
   +-- scripts/                         # Pre-built training shell scripts
   +-- configs/                         # Default XuanCe YAML config overrides
