Render Tabs & Views
====================

All rendering widgets — :doc:`fastlane` live views, :doc:`slow_lane`
telemetry streams, and multi-operator evaluation — are hosted inside
**RenderTabs**, the central ``QTabWidget`` at the heart of the MOSAIC UI.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       RT["RenderTabs<br/>(QTabWidget)"]

       subgraph Static["Built-in Tabs (created at startup)"]
           GRID["Grid Tab<br/>(GridRendererStrategy)"]
           RAW["Raw Tab<br/>(text output)"]
           VIDEO["Video Tab<br/>(RgbRendererStrategy)"]
           REPLAY["Human Replay Tab"]
           MULTI["Multi-Operator Tab<br/>(MultiOperatorRenderView)"]
       end

       subgraph Dynamic["Dynamic Tabs (created per-run)"]
           FLT["FastLaneTab<br/>(Qt Quick · ~60 Hz)"]
           LTT["LiveTelemetryTab<br/>(slow lane · ~10 FPS)"]
       end

       RT --> GRID & RAW & VIDEO & REPLAY & MULTI
       RT --> FLT & LTT

       style Static fill:#e8f5e9,stroke:#2e8b57,color:#333
       style Dynamic fill:#e3f2fd,stroke:#1565c0,color:#333

RenderTabs
----------

``RenderTabs`` (``gym_gui/ui/widgets/render_tabs.py``) is a ``QTabWidget``
that owns all rendering surfaces.

.. code-block:: python

   RenderTabs(
       parent: QWidget | None = None,
       *,
       telemetry_service: TelemetryService | None = None,
       run_manager: TrainingRunManager | None = None,
   )

**Built-in tabs** are created at startup:

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - Tab
     - Purpose
   * - **Grid**
     - Hosts :doc:`strategies` ``GridRendererStrategy`` and
       ``BoardGameRendererStrategy`` with a welcome overlay when idle.
   * - **Raw**
     - Plain-text environment output (ASCII renders, debug info).
   * - **Video**
     - Hosts ``RgbRendererStrategy`` for pixel-based environments.
   * - **Human Replay**
     - Episode review for previously recorded human sessions.
   * - **Multi-Operator**
     - Side-by-side agent comparison via ``MultiOperatorRenderView``.

**Dynamic tabs** are created on demand by worker presenters:

- ``add_dynamic_tab(run_id, name, widget)`` — adds or focuses an agent-specific
  tab (e.g., ``CleanRL-Live-agent_0``).
- Tabs are named by the presenter using the pattern
  ``{WorkerName}-Live-{agent_id}``.

**Signals forwarded** from :doc:`strategies` board-game renderers:

- ``chess_move_made(str, str)``
- ``connect_four_column_clicked(int)``
- ``go_intersection_clicked(int, int)`` / ``go_pass_requested()``
- ``sudoku_cell_selected(int, int)`` / ``sudoku_digit_entered(int, int, int)``
  / ``sudoku_cell_cleared(int, int)``
- ``checkers_cell_clicked(int, int)``
- ``human_action_submitted(str, int)`` — operator_id, action_index
- ``board_game_move_made(str, str, str)`` — operator_id, from_sq, to_sq
- ``chess_move_button_clicked(str, str)`` — operator_id, uci_move

LiveTelemetryTab
----------------

``LiveTelemetryTab`` (``gym_gui/ui/widgets/live_telemetry_tab.py``) displays
a live :doc:`slow_lane` telemetry stream for a specific
``(run_id, agent_id)`` pair.

.. code-block:: python

   LiveTelemetryTab(
       run_id: str,
       agent_id: str,
       *,
       game_id: GameId | None = None,
       buffer_size: int = 100,            # DEFAULT_TELEMETRY_BUFFER_SIZE
       episode_buffer_size: int = 10,     # DEFAULT_EPISODE_BUFFER_SIZE
       render_throttle_interval: int = 1, # UI_RENDERING_THROTTLE_MIN
       render_delay_ms: int = 100,        # DEFAULT_RENDER_DELAY_MS
       live_render_enabled: bool = True,
       renderer_registry: RendererRegistry | None = None,
       parent: QWidget | None = None,
   )

**Data flow:**

1. ``LiveTelemetryController`` (see :doc:`slow_lane`) emits step/episode events.
2. Events are queued in bounded ``Deque`` buffers (oldest dropped on overflow):
   ``_step_buffer`` (maxlen=100), ``_episode_buffer`` (maxlen=10).
3. ``RenderingSpeedRegulator`` drains the render queue at ~10 FPS
   (``render_delay_ms=100``).
4. The tab delegates to a :doc:`strategies` ``RendererStrategy`` (resolved via
   ``RendererRegistry``) for the visual frame, and updates a stats table
   independently.
5. ``_render_throttle_interval`` controls which steps trigger a render
   (every *N*-th step).

FastLaneTab
-----------

``FastLaneTab`` (``gym_gui/ui/widgets/fastlane_tab.py``) renders frames from
the :doc:`fastlane` shared-memory ring buffer using **Qt Quick / QML**.

.. code-block:: python

   FastLaneTab(
       run_id: str,
       agent_id: str,
       *,
       mode_label: str | None = None,   # default "Fast lane"
       run_mode: str | None = None,     # "train" | "policy_eval"
       parent: QWidget | None = None,
   )

**Architecture:**

- Owns a ``FastLaneConsumer`` that polls shared memory every 16 ms (~60 Hz).
- ``FastLaneConsumer`` emits ``frame_ready(FastLaneFrameEvent)`` with a
  ``QImage`` + HUD text.
- The tab hosts a ``QQuickWidget`` loading ``FastLaneView.qml`` for
  GPU-accelerated rendering.
- A status label shows connection state (``connected``, ``reconnecting``,
  ``fastlane-unavailable``).

**Modes:**

- ``"train"`` (default) — live training display with reward / step-rate HUD.
- ``"policy_eval"`` — adds an evaluation summary overlay that reloads
  ``eval_summary.json`` every 1 s (batch count, episodes, avg/min/max return).

OperatorRenderContainer
-----------------------

``OperatorRenderContainer``
(``gym_gui/ui/widgets/operator_render_container.py``) wraps a single
:doc:`/documents/architecture/operators/index` rendering area during
multi-operator evaluation.

.. code-block:: python

   OperatorRenderContainer(
       config: OperatorConfig,
       renderer_registry: RendererRegistry | None = None,
       parent: QWidget | None = None,
   )

**Layout:**

- **Header**: operator name + type badge (``llm`` / ``vlm`` / ``rl`` /
  ``human``) with status-coloured indicator (pending / loaded / running /
  stopped / error).
- **Render area**: hosts a :doc:`strategies` ``RendererStrategy``
  (Grid, RGB, or BoardGame).
- **Stats bar**: compact step / episode / reward counters.
- **Human interaction**: detected via
  ``operator_type == "human" or any(worker.type == "human")``.

**Signals:**

- ``status_changed(str, str)`` — operator_id, new_status
- ``human_action_submitted(str, int)`` — operator_id, action_index
- ``board_game_move_made(str, str, str)`` — operator_id, from_sq, to_sq
- ``chess_move_button_clicked(str, str)`` — operator_id, uci_move

MultiOperatorRenderView
-----------------------

``MultiOperatorRenderView``
(``gym_gui/ui/widgets/multi_operator_render_view.py``) arranges *N*
``OperatorRenderContainer`` widgets in a dynamic grid layout:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Operators
     - Auto-layout
     - Manual override
   * - 1
     - Full width (1 col)
     - "Auto", "1 Column"
   * - 2
     - 1 × 2
     - "2 Columns"
   * - 3–4
     - 2 × 2
     - "3 Columns"
   * - 5–6
     - 2 × 3
     - (auto or override)
   * - 7–9
     - 3 × 3
     - (auto or override)
   * - 10+
     - 4 columns
     - (auto or override)

This is the primary view for side-by-side
:doc:`/documents/architecture/paradigms` comparison during evaluation — e.g.,
watching an RL :doc:`/documents/architecture/operators/index` and an LLM
operator solve the same environment simultaneously.

**Signals forwarded:**

- ``operator_status_changed(str, str)``
- ``human_action_submitted(str, int)``
- ``board_game_move_made(str, str, str)``
- ``chess_move_button_clicked(str, str)``

Worker Presenters
-----------------

Worker presenters (``gym_gui/ui/presenters/workers/``) are responsible for
creating the correct tab type when a training run starts:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Presenter
     - Tab created
   * - ``CleanRlWorkerPresenter``
     - ``FastLaneTab`` (live training) + ``LiveTelemetryTab`` (telemetry)
   * - ``XuanCeWorkerPresenter``
     - ``FastLaneTab`` with XuanCe-specific naming
   * - ``RayWorkerPresenter``
     - ``RayMultiWorkerFastLaneTab`` (multi-agent tiled view)
   * - ``ChessWorkerPresenter``
     - ``BoardGameFastLaneTab`` with move metadata parsing
   * - ``HumanWorkerPresenter``
     - ``LiveTelemetryTab`` with human input integration

The ``WorkerPresenterRegistry`` (``registry.py``) maps worker IDs to presenter
classes so ``MainWindow`` can instantiate the correct presenter automatically.

See Also
--------

- :doc:`fastlane` — ``FastLaneTab`` and ``FastLaneConsumer`` details.
- :doc:`slow_lane` — ``LiveTelemetryController`` and ``RenderingSpeedRegulator``
  that feed ``LiveTelemetryTab``.
- :doc:`strategies` — the ``RendererStrategy`` protocol and ``RendererRegistry``
  used by both ``LiveTelemetryTab`` and ``OperatorRenderContainer``.
- :doc:`/documents/architecture/operators/index` — the operator system that
  populates ``MultiOperatorRenderView``.
- :doc:`/documents/runtime_logging/log_constants` — ``LOG1001``–``LOG1015``
  for operator UI and render container events.
