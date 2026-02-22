Application Constants
=====================

MOSAIC centralises all tuning knobs, default values, and magic numbers into
a single ``gym_gui/constants/`` package.  Every subsystem — from the
:doc:`/documents/rendering_tabs/slow_lane` queue sizes to the
:doc:`/documents/rendering_tabs/strategies` FPS caps — imports its defaults
from here rather than scattering literals across the codebase.

Architecture
------------

The package follows a **domain-based submodule** pattern.  Each submodule
is a focused Python file exporting frozen dataclasses and/or module-level
constants.  The top-level ``__init__.py`` re-exports everything
(**160 symbols** in ``__all__``) so consumers can write:

.. code-block:: python

   from gym_gui.constants import (
       OPERATOR_DEFAULTS,
       UI_DEFAULTS,
       TELEMETRY_HUB_BUFFER_SIZE,
       INITIAL_CREDITS,
       format_episode_id,
   )

Constant Domains
----------------

.. list-table::
   :widths: 26 10 64
   :header-rows: 1

   * - Module
     - Lines
     - Key Exports
   * - ``constants_core.py``
     - 203
     - ``EpisodeCounterConfig``, ``format_episode_id()``,
       ``parse_episode_id()``, ``MAX_COUNTER_VALUE`` (999 999)
   * - ``constants_ui.py``
     - 220
     - ``UIDefaults``, ``RenderDefaults``, ``SliderDefaults``,
       ``BufferDefaults``, ``LayoutDefaults``
   * - ``constants_telemetry.py``
     - 122
     - ``STEP_BUFFER_SIZE``, ``RENDER_QUEUE_SIZE``,
       ``DB_SINK_BATCH_SIZE``, ``TELEMETRY_KEY_*`` prefixes
   * - ``constants_telemetry_bus.py``
     - 90
     - ``BusDefaults``, ``CreditDefaults``, ``RunBusQueueDefaults``
   * - ``constants_telemetry_db.py``
     - 47
     - ``TelemetryDBSinkDefaults``, ``DatabaseDefaults`` (WAL / journal)
   * - ``constants_trainer.py``
     - 97
     - ``TrainerDefaults`` (gRPC port, daemon lifecycle),
       ``TrainerRetryDefaults`` (exponential backoff)
   * - ``constants_worker.py``
     - 162
     - ``WorkerDefaults`` (heartbeat interval, resource limits),
       ``WORKER_ENTRY_POINT_GROUP`` (plugin discovery)
   * - ``constants_operator.py``
     - 204
     - ``OperatorDefaults``, ``BarlogDefaults``,
       ``BALROG_SUPPORTED_ENVS``, operator category labels
   * - ``constants_tensorboard.py``
     - 54
     - ``TensorboardDefaults``, ``build_tensorboard_log_dir()``
   * - ``constants_wandb.py``
     - 41
     - ``WandbDefaults``, ``build_wandb_run_url()``
   * - ``constants_replay.py``
     - 126
     - ``ReplayDefaults``, ``REPLAY_COMPRESSION``, ``FRAME_REF_*``
       (HDF5 storage config)
   * - ``constants_vector.py``
     - 22
     - ``VECTOR_ENV_BATCH_SIZE_KEY``, ``SUPPORTED_AUTORESET_MODES``
   * - ``optional_deps.py``
     - 379
     - ``require_torch()``, ``is_pettingzoo_available()``,
       and 12+ optional-dependency gates
   * - ``mosaic_welcome.py``
     - —
     - Welcome widget strings
   * - ``loader.py``
     - —
     - Dynamic constant discovery

Rendering Defaults
------------------

``constants_ui.py`` provides the defaults used by the
:doc:`/documents/rendering_tabs/index` subsystem:

.. code-block:: python

   @dataclass(frozen=True)
   class RenderDefaults:
       min_delay_ms: int = 10
       max_delay_ms: int = 500
       tick_interval_ms: int = 50
       default_delay_ms: int = 100     # ~10 FPS for slow lane
       queue_size: int = 32            # RenderingSpeedRegulator max
       bootstrap_timeout_ms: int = 500

   @dataclass(frozen=True)
   class UIDefaults:
       render: RenderDefaults = RenderDefaults()
       sliders: SliderDefaults = SliderDefaults()
       buffers: BufferDefaults = BufferDefaults()
       layout: LayoutDefaults = LayoutDefaults()

These values directly govern the
:doc:`/documents/rendering_tabs/slow_lane` ``RenderingSpeedRegulator``
drain interval and queue capacity.

Telemetry Constants
-------------------

``constants_telemetry.py`` defines queue sizes, buffer bounds, and batch
parameters for the :doc:`/documents/rendering_tabs/slow_lane` pipeline:

.. list-table::
   :widths: 45 12 43
   :header-rows: 1

   * - Constant
     - Value
     - Used by
   * - ``STEP_BUFFER_SIZE``
     - 64
     - ``LiveTelemetryTab`` step deque
   * - ``EPISODE_BUFFER_SIZE``
     - 32
     - ``LiveTelemetryTab`` episode deque
   * - ``RENDER_QUEUE_SIZE``
     - 32
     - ``RenderingSpeedRegulator`` max queue
   * - ``RUNBUS_DEFAULT_QUEUE_SIZE``
     - 2 048
     - ``RunBus`` main queue
   * - ``RUNBUS_UI_PATH_QUEUE_SIZE``
     - 512
     - UI subscriber path
   * - ``RUNBUS_DB_PATH_QUEUE_SIZE``
     - 1 024
     - DB subscriber path
   * - ``LIVE_STEP_QUEUE_SIZE``
     - 64
     - ``LiveTelemetryController``
   * - ``LIVE_EPISODE_QUEUE_SIZE``
     - 64
     - ``LiveTelemetryController``
   * - ``LIVE_CONTROL_QUEUE_SIZE``
     - 32
     - ``LiveTelemetryController``
   * - ``TELEMETRY_HUB_MAX_QUEUE``
     - 4 096
     - ``TelemetryAsyncHub``
   * - ``TELEMETRY_HUB_BUFFER_SIZE``
     - 100 000
     - Hub in-memory buffer
   * - ``DB_SINK_BATCH_SIZE``
     - 256
     - ``TelemetryDBSink`` batch writes
   * - ``DB_SINK_CHECKPOINT_INTERVAL``
     - 4 096
     - WAL checkpoint trigger
   * - ``DB_SINK_WRITER_QUEUE_SIZE``
     - 16 384
     - Writer thread queue
   * - ``INITIAL_CREDITS``
     - 200
     - ``CreditManager`` initial grant
   * - ``MIN_CREDITS_THRESHOLD``
     - 10
     - Starvation warning
   * - ``RENDER_BOOTSTRAP_TIMEOUT_MS``
     - 500
     - Auto-start regulator

Bus & Credit Defaults
---------------------

``constants_telemetry_bus.py`` wraps related defaults into nested dataclasses:

.. code-block:: python

   @dataclass(frozen=True)
   class CreditDefaults:
       initial_credits: int = 200
       starvation_threshold: int = 10

   @dataclass(frozen=True)
   class BusDefaults:
       run_bus: RunBusQueueDefaults = ...
       run_events: RunEventDefaults = ...
       telemetry_streams: TelemetryStreamDefaults = ...
       hub: TelemetryHubDefaults = ...
       logging: TelemetryLoggingDefaults = ...
       credit: CreditDefaults = ...

These are consumed by the :doc:`/documents/rendering_tabs/slow_lane`
``CreditManager`` and ``RunBus``.

Operator Constants
------------------

``constants_operator.py`` defines categories used by the
:doc:`/documents/architecture/operators/index` system:

.. code-block:: python

   OPERATOR_CATEGORY_HUMAN   = "human"
   OPERATOR_CATEGORY_LLM     = "llm"
   OPERATOR_CATEGORY_RL      = "rl"
   OPERATOR_CATEGORY_HYBRID  = "hybrid"

It also provides BALROG LLM worker defaults:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Constant
     - Value
   * - ``BALROG_SUPPORTED_ENVS``
     - ``("babyai", "minigrid", "minihack", "crafter", "nle",
       "textworld", "toytext")``
   * - ``BALROG_SUPPORTED_CLIENTS``
     - ``("openai", "anthropic", "google", "vllm")``
   * - ``BALROG_AGENT_TYPES``
     - ``("naive", "cot", "robust_naive", "robust_cot",
       "few_shot", "dummy")``
   * - ``BALROG_DEFAULT_MODEL``
     - ``"gpt-4o-mini"``
   * - ``BALROG_DEFAULT_TEMPERATURE``
     - ``0.7``
   * - ``BALROG_DEFAULT_NUM_EPISODES``
     - ``5``

Optional Dependency Gates
-------------------------

``optional_deps.py`` provides safe import wrappers that raise clear errors
when an optional library is missing.  Seven dependency groups are guarded:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Dependency
     - Check
     - Require
   * - PyTorch
     - ``is_torch_available()``
     - ``require_torch()``
   * - PettingZoo
     - ``is_pettingzoo_available()``
     - ``require_pettingzoo()``
   * - CleanRL
     - ``is_cleanrl_available()``
     - ``require_cleanrl()``
   * - ViZDoom
     - ``is_vizdoom_available()``
     - ``require_vizdoom()``
   * - Stockfish
     - ``is_stockfish_available()``
     - ``require_stockfish()``
   * - MuJoCo MPC
     - ``is_mjpc_available()``
     - ``get_mjpc_launcher()``
   * - Godot
     - ``is_godot_available()``
     - ``get_godot_launcher()``

Each guard uses lazy-loaded flags with caching and raises
``OptionalDependencyError`` with an installation hint.

Type Stubs
----------

``__init__.pyi`` provides full type annotations for IDE auto-completion,
since the runtime ``__init__.py`` uses dynamic imports.

Directory Layout
----------------

.. code-block:: text

   gym_gui/
     constants/
       __init__.py                # Re-exports 160 symbols
       __init__.pyi               # Type stubs for IDE support
       constants_core.py          # Episode IDs, counter bounds
       constants_ui.py            # Window / render / slider defaults
       constants_telemetry.py     # Queue sizes, buffer configs
       constants_telemetry_bus.py # RunBus queue & credit defaults
       constants_telemetry_db.py  # SQLite / WAL tuning
       constants_trainer.py       # gRPC & daemon lifecycle
       constants_worker.py        # Worker discovery & heartbeat
       constants_operator.py      # Operator categories, BALROG config
       constants_tensorboard.py   # TensorBoard log paths
       constants_wandb.py         # W&B URL builder
       constants_replay.py        # HDF5 replay storage
       constants_vector.py        # Vector env metadata
       constants_game.py          # Legacy game config (dynamic import)
       optional_deps.py           # Safe optional-dependency gates
       loader.py                  # Dynamic constant discovery
       mosaic_welcome.py          # Welcome widget strings
       README.md                  # Internal architecture guide

See Also
--------

- :doc:`log_constants` — ``LogConstant`` codes reference specific constant
  values (e.g., ``LOG_TELEMETRY_QUEUE_OVERFLOW`` uses ``RUNBUS_DEFAULT_QUEUE_SIZE``).
- :doc:`validation` — Pydantic models validate payloads whose schema
  constraints come from these constants.
- :doc:`/documents/rendering_tabs/slow_lane` — the telemetry pipeline whose
  queue sizes, batch sizes, and credit thresholds are all defined here.
