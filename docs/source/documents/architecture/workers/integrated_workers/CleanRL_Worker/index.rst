CleanRL Worker
==============

The CleanRL worker is MOSAIC's **single-agent RL** integration.  It wraps
the `CleanRL <https://github.com/vwxyzjn/cleanrl>`_ library. A
collection of single-file, research-friendly algorithm implementations
behind the standard :doc:`shim pattern <../../concept>`, adding
subprocess isolation, FastLane telemetry, curriculum learning, and GUI
configuration.

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Single-agent (sequential)
   * - **Algorithms**
     - 40+ (PPO, DQN, SAC, TD3, DDPG, C51, Rainbow, and variants)
   * - **Environments**
     - Gymnasium, Atari, MiniGrid, BabyAI, Procgen, MuJoCo, DM Control
   * - **Execution**
     - Subprocess (one OS process per training run)
   * - **GPU required**
     - No (optional CUDA acceleration)
   * - **Source**
     - ``3rd_party/cleanrl_worker/cleanrl_worker/``

Architecture
------------

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Training Form<br/>(CleanRL widgets)"]
           DAEMON["Trainer Daemon"]
       end

       subgraph "Worker Subprocess"
           CLI["cli.py<br/>entry point"]
           CFG["config.py<br/>CleanRLWorkerConfig"]
           RT["runtime.py<br/>CleanRLWorkerRuntime"]
           FL["fastlane.py<br/>FastLaneTelemetryWrapper"]
           SITE["sitecustomize.py<br/>import-time gym.make patch"]
           LAUNCH["launcher.py<br/>algorithm dispatch"]
       end

       subgraph "Upstream CleanRL"
           ALGO["ppo.py / dqn.py / sac.py / ...<br/>(unmodified single-file scripts)"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT --> LAUNCH --> ALGO
       SITE -.->|"patches gym.make()"| ALGO
       FL -.->|"shared-memory frames"| DAEMON

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style FL fill:#ff7f50,stroke:#cc5500,color:#fff
       style SITE fill:#ff7f50,stroke:#cc5500,color:#fff
       style LAUNCH fill:#ff7f50,stroke:#cc5500,color:#fff
       style ALGO fill:#e8e8e8,stroke:#999

**Lifecycle of a training run:**

1. The GUI form builds a config JSON and hands it to the Trainer Daemon.
2. The daemon spawns ``python -m cleanrl_worker.cli --config <path>``.
3. ``cli.py`` loads the config, detects the training mode, and
   delegates to the appropriate runtime.
4. ``CleanRLWorkerRuntime.run()`` resolves the algorithm module from the
   registry, prepares the run directory, sets FastLane / W&B / TensorBoard
   environment variables, and launches the algorithm as a subprocess via
   ``cleanrl_worker.launcher``.
5. ``sitecustomize.py`` patches ``gym.make()`` at import time so every
   environment is automatically wrapped with ``FastLaneTelemetryWrapper``.
6. The runtime polls the subprocess, emits heartbeats every 30 seconds,
   and on completion writes an ``analytics.json`` manifest.

Supported Algorithms
--------------------

The ``DEFAULT_ALGO_REGISTRY`` in ``runtime.py`` maps algorithm names to
importable modules.  The first entry (``ppo``) points to the
MOSAIC-patched version; all others delegate to upstream CleanRL.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Family
     - Algorithms
     - Notes
   * - **PPO**
     - ``ppo`` (MOSAIC-patched),
       ``ppo_continuous_action``,
       ``ppo_atari``,
       ``ppo_atari_multigpu``,
       ``ppo_atari_lstm``,
       ``ppo_atari_envpool``,
       ``ppo_atari_envpool_xla_jax``,
       ``ppo_atari_envpool_xla_jax_scan``,
       ``ppo_procgen``,
       ``ppo_rnd_envpool``,
       ``ppo_pettingzoo_ma_atari``
     - Primary algorithm family; ``ppo`` is the default
   * - **Policy Optimization Variants**
     - ``ppg_procgen``,
       ``pqn``,
       ``pqn_atari_envpool``,
       ``pqn_atari_envpool_lstm``,
       ``rpo_continuous_action``
     - Phasic Policy Gradient, Periodic Q-Network, Reward-Policy Optimization
   * - **Q-Learning**
     - ``dqn``,
       ``dqn_atari``,
       ``dqn_jax``,
       ``dqn_atari_jax``,
       ``rainbow_atari``,
       ``qdagger_dqn_atari_impalacnn``,
       ``qdagger_dqn_atari_jax_impalacnn``
     - Deep Q-Network and extensions
   * - **Distributional RL**
     - ``c51``,
       ``c51_jax``,
       ``c51_atari``,
       ``c51_atari_jax``
     - Categorical DQN (C51)
   * - **Continuous Control**
     - ``ddpg_continuous_action``,
       ``ddpg_continuous_action_jax``,
       ``td3_continuous_action``,
       ``td3_continuous_action_jax``,
       ``sac_continuous_action``,
       ``sac_atari``
     - DDPG, TD3, and SAC for continuous action spaces

Agent Architectures
-------------------

The worker ships with two built-in neural network architectures
used by the MOSAIC-patched PPO and curriculum training modes.

MinigridCNN
^^^^^^^^^^^^

Defined in ``agents/minigrid.py``.  Designed for 7x7x3 partially
observable grid-world images (MiniGrid / BabyAI environments).

.. code-block:: text

   Input: (B, 7, 7, 3) uint8
     -> permute to (B, 3, 7, 7), normalize to [0, 1]
     -> Conv2d(3, 32, 3, padding=1) + ReLU
     -> Conv2d(32, 64, 3, padding=1) + ReLU
     -> Conv2d(64, 64, 3, padding=1) + ReLU
     -> Flatten
     -> Linear(3136, 128)

``MinigridAgent`` pairs this backbone with separate actor and critic
heads (each ``Linear(128, 128) -> ReLU -> Linear(128, out)``), using
orthogonal weight initialization.

MLPAgent
^^^^^^^^

Defined in ``agents/mlp.py``.  Used for flat observation spaces
(CartPole, MountainCar, LunarLander, etc.).

.. code-block:: text

   Input: (B, obs_dim)
     -> Linear(obs_dim, 64) + Tanh
     -> Linear(64, 64) + Tanh

Separate actor and critic heads branch from the shared trunk.
Hidden size is 64 with Tanh activations and orthogonal initialization.

Training Modes
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Mode
     - Trigger
     - Description
   * - **Standard training**
     - Default (no special mode flag)
     - Spawns the algorithm as a subprocess via ``cleanrl_worker.launcher``.
       Full FastLane telemetry, TensorBoard, and W&B support.
   * - **Curriculum training**
     - ``extras.curriculum_schedule`` present
     - Runs ``run_curriculum_training()`` with Syllabus-RL environment
       switching.  Environments are swapped according to the schedule
       without restarting the training loop.
   * - **Resume training**
     - ``extras.mode == "resume_training"``
     - Loads a ``.cleanrl_model`` checkpoint from ``extras.checkpoint_path``
       and continues training for additional timesteps.
   * - **Policy evaluation**
     - ``extras.mode == "policy_eval"``
     - In-process batched evaluation using ``run_batched_evaluation()``.
       Supports configurable episodes, gamma, capture video, and repeat.
   * - **Interactive**
     - ``--interactive`` CLI flag
     - Stdin/stdout JSON-lines IPC protocol.  The GUI sends ``reset``,
       ``step``, and ``stop`` commands; the worker responds with
       observations, rewards, and render frames.
   * - **Dry run**
     - ``--dry-run`` CLI flag
     - Resolves the algorithm module and validates the config, then
       exits without launching training.  Useful for pre-flight checks.

Curriculum Training
^^^^^^^^^^^^^^^^^^^

Curriculum training uses `Syllabus-RL <https://github.com/RyanNavillus/Syllabus>`_
to progressively advance through a sequence of environments.  The
schedule is a list of stages, each specifying an ``env_id`` and an
optional stopping condition:

.. code-block:: json

   {
     "curriculum_schedule": [
       {"env_id": "BabyAI-GoToRedBallNoDists-v0", "steps": 200000},
       {"env_id": "BabyAI-GoToRedBall-v0", "steps": 200000},
       {"env_id": "BabyAI-GoToObj-v0", "steps": 200000},
       {"env_id": "BabyAI-GoToLocal-v0"}
     ]
   }

Stopping conditions per stage: ``steps>=N``, ``episodes>=N``,
``episode_return>=X``.  Multiple conditions can be combined with ``|``
(OR logic).  If no condition is specified, the default is
``steps>=100000``.

The ``BabyAITaskWrapper`` (a ``ReinitTaskWrapper`` subclass) handles
environment switching at runtime.  The training loop (PPO) requires
no modification, curriculum learning operates entirely at the
environment level.

Built-in preset schedules are available in ``wrappers/curriculum.py``:

- ``BABYAI_GOTO_CURRICULUM``: four-stage GoTo progression
- ``BABYAI_DOORKEY_CURRICULUM``: four-stage DoorKey progression (5x5 to 16x16)

FastLane Telemetry
------------------

FastLane provides **real-time frame streaming** from the training
subprocess to the MOSAIC GUI via shared memory.

**How it works:**

1. ``sitecustomize.py`` patches ``gym.make()`` at import time.
2. Every environment created by the training script is automatically
   wrapped with ``FastLaneTelemetryWrapper``.
3. On each ``step()``, the wrapper calls ``env.render()`` to grab an
   RGB frame and publishes it through ``FastLaneWriter``.
4. The GUI reads frames from shared memory and displays them in the
   training dashboard.

**Video modes:**

- ``single``: only the probe environment (selected by ``fastlane_slot``)
  emits frames.
- ``grid``: multiple environments contribute frames; slot 0 coordinates
  tiling via ``_GridCoordinator``.  The ``fastlane_grid_limit`` parameter
  controls how many environments participate.
- ``off``: no frame emission.

**Metrics published alongside each frame:**

- ``last_reward``: reward from the most recent step
- ``rolling_return``: exponentially smoothed episode return
- ``step_rate_hz``: current training throughput

**Tuning parameters (environment variables):**

- ``CLEANRL_FASTLANE_INTERVAL_MS``: minimum milliseconds between frames
  (throttling)
- ``CLEANRL_FASTLANE_MAX_DIM``: maximum pixel dimension before
  downscaling

GUI Integration
---------------

The CleanRL worker provides four dedicated form widgets for experiment
configuration, all located in ``gym_gui/ui/widgets/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Form
     - Purpose
   * - ``cleanrl_train_form.py``
     - Primary training dialog.  Algorithm and environment selection,
       hyperparameter tuning (dynamically generated from schema files),
       FastLane settings, TensorBoard/W&B tracking, GPU toggle.
   * - ``cleanrl_script_form.py``
     - Custom shell script launcher for multi-phase training.  Reads
       ``*.sh`` scripts and parses inline metadata (``@description``,
       ``@phases``, ``@total_timesteps``).
   * - ``cleanrl_resume_form.py``
     - Resume training from a ``.cleanrl_model`` checkpoint.  Auto-discovers
       checkpoints under ``var/trainer/runs/``.
   * - ``cleanrl_policy_form.py``
     - Policy evaluation dialog.  Loads a trained checkpoint, configures
       evaluation episodes, gamma, and optional video capture.

Worker Discovery
----------------

The worker registers itself via the ``mosaic.workers`` entry point
group in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."mosaic.workers"]
   cleanrl = "cleanrl_worker:get_worker_metadata"

``get_worker_metadata()`` returns a ``WorkerCapabilities`` descriptor:

.. code-block:: python

   WorkerCapabilities(
       worker_type="cleanrl",
       supported_paradigms=("sequential",),
       env_families=(
           "gymnasium", "atari", "procgen",
           "mujoco", "dm_control", "minigrid", "babyai",
       ),
       action_spaces=("discrete", "continuous"),
       observation_spaces=("vector", "image"),
       max_agents=1,
       supports_checkpointing=True,
       requires_gpu=False,
       estimated_memory_mb=512,
   )

Configuration
-------------

The ``CleanRLWorkerConfig`` dataclass (``config.py``) is the single
source of truth for all run parameters:

.. code-block:: python

   @dataclass(frozen=True)
   class CleanRLWorkerConfig:
       run_id: str                    # ULID-format unique run identifier
       algo: str                      # Algorithm name (e.g. "ppo", "dqn")
       env_id: str                    # Gymnasium environment ID
       total_timesteps: int           # Training budget
       seed: Optional[int] = None
       extras: dict[str, Any] = ...   # All additional config
       worker_id: Optional[str] = None
       raw: dict[str, Any] = ...      # Full raw payload (for debugging)

The config loader accepts two JSON formats:

- **Nested (GUI):** the config lives at ``metadata.worker.config`` inside
  the full job descriptor.
- **Flat (standalone):** the JSON maps directly to ``CleanRLWorkerConfig``
  fields.

Key ``extras`` fields:

- ``mode``: ``"train"`` (default), ``"policy_eval"``, ``"resume_training"``, ``"interactive"``
- ``cuda`` / ``use_cuda``: enable GPU acceleration
- ``tensorboard_dir``: relative path for TensorBoard logs
- ``track_wandb``: enable Weights & Biases logging
- ``algo_params``: dict of algorithm-specific hyperparameters passed
  as CLI flags to the upstream script
- ``curriculum_schedule``: list of stage dicts (triggers curriculum
  mode)
- ``fastlane_video_mode``: ``"single"``, ``"grid"``, or ``"off"``
- ``policy_path``: path to trained model (for eval/resume)


.. toctree::
   :maxdepth: 1

   installation
   common_errors
