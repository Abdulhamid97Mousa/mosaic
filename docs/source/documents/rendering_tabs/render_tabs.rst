Render Tabs
===========

All rendering widgets: :doc:`fastlane` live views, :doc:`slow_lane`
telemetry streams, and multi-operator evaluation: are hosted inside
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

- ``add_dynamic_tab(run_id, name, widget)``: adds or focuses an agent-specific
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
- ``human_action_submitted(str, int)``: operator_id, action_index
- ``board_game_move_made(str, str, str)``: operator_id, from_sq, to_sq
- ``chess_move_button_clicked(str, str)``: operator_id, uci_move

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

- :doc:`views`: the individual view widgets (``FastLaneTab``,
  ``LiveTelemetryTab``, ``OperatorRenderContainer``,
  ``MultiOperatorRenderView``) that are placed inside ``RenderTabs``.
- :doc:`fastlane`: shared-memory ring buffer and ``FastLaneConsumer``.
- :doc:`slow_lane`: ``LiveTelemetryController`` and ``RenderingSpeedRegulator``.
- :doc:`strategies`: ``RendererStrategy`` protocol and ``RendererRegistry``.
- :doc:`/documents/architecture/operators/index`: the operator system that
  populates ``MultiOperatorRenderView``.
- :doc:`/documents/runtime_logging/log_constants`: ``LOG1001``--``LOG1015``
  for operator UI and render container events.
