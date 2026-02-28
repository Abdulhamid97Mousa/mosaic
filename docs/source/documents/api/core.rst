Core API
========

Core enums and data structures used throughout MOSAIC.

Enums
-----

SteppingParadigm
^^^^^^^^^^^^^^^^

.. autoclass:: gym_gui.core.enums.SteppingParadigm
   :members:
   :undoc-members:

WorkerCapabilities
^^^^^^^^^^^^^^^^^^

.. autoclass:: gym_gui.core.adapters.base.WorkerCapabilities
   :members:
   :undoc-members:
   :no-index:

UI Widgets
----------

.. note::

   Qt widget classes cannot be auto-documented by Sphinx in a headless
   environment.  The API signatures below are maintained manually.

PlayerAssignmentPanel
^^^^^^^^^^^^^^^^^^^^^

.. py:class:: gym_gui.ui.widgets.operator_config_widget.PlayerAssignmentPanel(env_family, env_id, num_agents, agent_ids=None, agent_labels=None, parent=None)

   Panel showing all agent/player assignments for a multi-agent environment.
   Contains one :class:`PlayerAssignmentRow` per agent slot.

   .. py:attribute:: assignments_changed
      :type: pyqtSignal

      Emitted when any player assignment row changes (type, worker, or settings).

   .. py:method:: get_worker_assignments() -> Dict[str, WorkerAssignment]

      Return a dict mapping each ``player_id`` to its :class:`~gym_gui.services.operator.WorkerAssignment`.

   .. py:method:: has_llm_agent() -> bool

      Return ``True`` if at least one agent row has Type set to **LLM**.

   .. py:method:: set_vllm_servers(servers)

      Propagate the current list of vLLM servers to every row.

PlayerAssignmentRow
^^^^^^^^^^^^^^^^^^^

.. py:class:: gym_gui.ui.widgets.operator_config_widget.PlayerAssignmentRow(player_id, player_label, parent=None)

   Single row inside a :class:`PlayerAssignmentPanel`.  Exposes a
   **Type** dropdown (LLM / RL / Human / Random), a **Worker** dropdown,
   and type-specific settings (LLM provider/model, RL policy path, etc.).

   .. py:attribute:: assignment_changed
      :type: pyqtSignal

      Emitted when the user changes any field in this row.

   .. py:method:: get_assignment() -> WorkerAssignment

      Build and return a :class:`~gym_gui.services.operator.WorkerAssignment`
      from the current UI state.  The ``Random`` type is mapped to
      ``worker_type="baseline"`` with ``behavior="random"``.

OperatorConfigWidget
^^^^^^^^^^^^^^^^^^^^

.. py:class:: gym_gui.ui.widgets.operator_config_widget.OperatorConfigWidget(operator_id, parent=None)

   Per-operator configuration widget.  For multi-agent environments it
   creates a :class:`PlayerAssignmentPanel` and an environment-specific
   settings section (observation mode, coordination strategy for
   MultiGrid / MeltingPot).

   .. py:method:: get_config() -> OperatorConfig

      Build an :class:`~gym_gui.services.operator.OperatorConfig` from
      the current UI state (single-agent or multi-agent depending on the
      selected environment).
