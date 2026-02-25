Validation
==========

MOSAIC validates data at three layers: **Pydantic models** for telemetry
payloads, a **ValidationService** for runtime checking with
:doc:`log_constants` structured logging, and **UI validators** for form
fields in the PyQt6 interface.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph LR
       subgraph Pydantic["Pydantic Models"]
           TSE["TelemetryEventBase"]
           SE["StepEvent"]
           EE["EpisodeEvent"]
           RSE["RunStartedEvent"]
           RCE["RunCompletedEvent"]
           TC["TrainingConfig"]
           TCU["TrainerControlUpdate"]
           SC["SubprocessCommand"]
       end

       subgraph Service["ValidationService"]
           VS["validate_telemetry_event()"]
           VTC["validate_training_config()"]
           VCU["validate_trainer_control_update()"]
           VSD["validate_step_data()"]
       end

       subgraph UI["UI Validators & Widgets"]
           IR["IntRangeValidator"]
           FR["FloatRangeValidator"]
           NS["NonEmptyStringValidator"]
           FP["FilePathValidator"]
           VLE["ValidatedLineEdit"]
           VSB["ValidatedSpinBox"]
       end

       Pydantic --> Service
       Service -->|"log_constant()"| LOG[":doc:`log_constants`"]
       UI --> FORM["PyQt6 Forms"]

       style Pydantic fill:#e8f5e9,stroke:#2e8b57,color:#333
       style Service fill:#fff3e0,stroke:#e65100,color:#333
       style UI fill:#e3f2fd,stroke:#1565c0,color:#333

Pydantic Models
---------------

Defined in ``gym_gui/validations/validations_pydantic.py``, these models
enforce schema and value constraints on telemetry data flowing through the
:doc:`/documents/rendering_tabs/slow_lane`:

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Model
     - Validated fields
   * - ``TelemetryEventBase``
     - ``type: str`` (non-empty), ``ts: datetime`` (UTC timestamp),
       extra fields allowed.
   * - ``RunStartedEvent``
     - ``run_id: str`` (non-empty), ``config: Dict[str, Any]``.
   * - ``RunCompletedEvent``
     - ``run_id``, ``status`` ∈ {``completed``, ``failed``, ``cancelled``},
       optional ``error: str``.
   * - ``StepEvent``
     - ``run_id``, ``episode`` (≥ 0), ``step`` (≥ 0), ``action``,
       ``reward``, ``state``, ``next_state``, ``terminated``, ``truncated``,
       ``q_before``, ``q_after``, ``epsilon`` ∈ [0, 1],
       ``observation``, ``next_observation``.
   * - ``EpisodeEvent``
     - ``run_id``, ``episode``, ``reward``, ``steps``, ``success``,
       ``metadata``.
   * - ``ArtifactEvent``
     - ``run_id``, ``kind`` ∈ {``policy``, ``video``, ``checkpoint``,
       ``log``, ``tensorboard``}, ``path``.
   * - ``TrainingConfig``
     - ``run_id``, ``game_id``, ``seed``, ``max_episodes``,
       ``max_steps_per_episode``, ``policy_strategy``, ``policy_path``,
       ``agent_id``, ``capture_video``, ``headless``, ``extra``.
   * - ``TrainerControlUpdate``
     - ``run_id``, ``reason``, ``params: Dict`` with bounded
       ``epsilon`` / ``per_alpha`` / ``per_beta`` / ``tau`` /
       ``lr_multiplier``.
   * - ``SubprocessCommand``
     - ``args: list[str]`` (non-empty, all strings).

All models use ``@field_validator`` decorators for constraint checking
(non-negative indices, bounded epsilon, enum membership).

ValidationService
-----------------

``ValidationService`` (``gym_gui/validations/validations_telemetry.py``)
wraps Pydantic validation with :doc:`log_constants` structured logging:

.. code-block:: python

   from gym_gui.validations import ValidationService

   validator = ValidationService(strict_mode=False)
   event = validator.validate_telemetry_event(raw_data)
   # On success → logs LOG_SERVICE_VALIDATION_DEBUG
   # On failure → logs LOG_SERVICE_VALIDATION_WARNING (or raises if strict)

**Methods:**

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - Method
     - Returns
   * - ``validate_telemetry_event(data)``
     - ``TelemetryEventBase | None``
   * - ``validate_training_config(data)``
     - ``TrainingConfig | None``
   * - ``validate_trainer_control_update(data)``
     - ``TrainerControlUpdate | None``
   * - ``validate_step_data(*, episode, step, …)``
     - ``bool``
   * - ``get_validation_errors()``
     - ``list[str]``
   * - ``get_validation_warnings()``
     - ``list[str]``
   * - ``get_step_schema(key)``
     - ``Dict | None``
   * - ``clear_errors()`` / ``clear_warnings()``
     - ``None``

**Modes:**

- ``strict_mode=True``: raises ``ValidationError`` on any failure.
  Suitable for CI/test environments.
- ``strict_mode=False`` (default): logs a warning via ``log_constant()``
  and returns ``None``.  Suitable for production where partial data should
  not crash the pipeline.

The service collects all errors in ``_validation_errors`` for later
inspection via ``get_validation_errors()``.

UI Validators
-------------

``gym_gui/validations/validations_ui.py`` provides Qt-friendly validators for
form fields.  Each validator returns a ``ValidationResult(is_valid, message)``:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Validator
     - Rule
   * - ``IntRangeValidator``
     - Integer within ``[min, max]``.
   * - ``FloatRangeValidator``
     - Float with inclusive bounds.
   * - ``NonEmptyStringValidator``
     - Non-empty, optional min/max length and regex pattern.
   * - ``FilePathValidator``
     - File or directory exists, readable / writable.

Validated Widgets
^^^^^^^^^^^^^^^^^

Custom PyQt6 widgets that integrate validators with real-time feedback:

.. code-block:: python

   class ValidatedLineEdit(QtWidgets.QLineEdit):
       validation_changed = pyqtSignal(bool)

       def is_valid(self) -> bool: ...
       def get_error_label(self) -> QtWidgets.QLabel: ...

   class ValidatedSpinBox(QtWidgets.QSpinBox):
       validation_changed = pyqtSignal(bool)

       def is_valid(self) -> bool: ...
       def get_error_label(self) -> QtWidgets.QLabel: ...

Both emit ``validation_changed(bool)`` on every keystroke/value change
and display inline error labels.

``FormValidationState`` aggregates results across multiple fields, and
``ValidationErrorDialog`` displays collected errors in a PyQt6 dialog.

Helper: ``create_validated_input_row(label_text, validator, parent, *, placeholder, initial_value)``
creates a label + ``ValidatedLineEdit`` + error label row.

Worker Form Validators
----------------------

Specialised validators for training configuration forms:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Module
     - What it validates
   * - ``validation_cleanrl_worker_form.py``
     - :doc:`/documents/architecture/workers/integrated_workers/CleanRL_Worker/index`
       dry-run via subprocess execution.
   * - ``validation_xuance_worker_form.py``
     - :doc:`/documents/architecture/workers/integrated_workers/XuanCe_Worker/index`
       worker form fields (backend, algorithm, env).
   * - ``validation_marllib_worker_form.py``
     - MARLlib worker form fields.

Agent Training Form
^^^^^^^^^^^^^^^^^^^

``validations_agent_train_form.py`` validates common training fields via
``validate_agent_train_form(inputs: AgentTrainFormInputs) → list[str]``:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Field
     - Constraint
   * - ``episodes``
     - 1 – 1 000 000
   * - ``max_steps_per_episode``
     - 1 – 100 000
   * - ``learning_rate``
     - (0, 1]
   * - ``discount``
     - [0, 1)
   * - ``epsilon_decay``
     - (0, 1]
   * - ``seed``
     - 0 – 999 999
   * - ``agent_id``
     - ≤ 256 chars
   * - ``worker_id``
     - Alphanumeric + underscore + hyphen

Directory Layout
----------------

.. code-block:: text

   gym_gui/
     validations/
       __init__.py
       validations_pydantic.py              # 8 Pydantic models
       validations_telemetry.py             # ValidationService (strict/lenient)
       validations_ui.py                    # Qt validators, ValidatedLineEdit/SpinBox
       validations_agent_train_form.py      # Agent training form helpers
       validation_cleanrl_worker_form.py    # CleanRL-specific validation
       validation_xuance_worker_form.py     # XuanCe-specific validation
       validation_marllib_worker_form.py    # MARLlib-specific validation

See Also
--------

- :doc:`log_constants`: ``LOG_SERVICE_VALIDATION_*`` codes emitted by
  ``ValidationService``.
- :doc:`constants`: numeric bounds used by ``AgentTrainFormInputs``
  (e.g., ``MAX_COUNTER_VALUE`` from ``constants_core.py``).
- :doc:`/documents/rendering_tabs/slow_lane`: the telemetry pipeline that
  passes through ``ValidationService`` before reaching the UI.
