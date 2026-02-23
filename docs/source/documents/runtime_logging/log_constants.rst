Log Constants
=============

Every loggable event in MOSAIC is represented by a **LogConstant**: an
immutable record carrying a numeric code, severity level, owning component,
and optional tags.  This eliminates magic strings and enables code-based
log search across the entire platform.

LogConstant Dataclass
---------------------

Defined in ``gym_gui/logging_config/log_constants.py`` (~5 400 lines,
500+ constants):

.. code-block:: python

   @dataclass(frozen=True, slots=True)
   class LogConstant:
       code: str              # e.g. "LOG401"
       level: int | str       # logging.INFO, "ERROR", etc.
       message: str           # Default message template
       component: str         # "Controller", "Service", "Adapter", …
       subcomponent: str      # "Session", "Telemetry", "Validation", …
       tags: tuple[str, ...] = ()

Example constants:

.. code-block:: python

   LOG_SESSION_ADAPTER_LOAD_ERROR = LogConstant(
       "LOG401", "ERROR", "Failed to load adapter",
       "Controller", "Session",
   )
   LOG_SESSION_STEP_ERROR = LogConstant(
       "LOG402", "ERROR", "Step execution failed",
       "Controller", "Session",
   )
   LOG_KEYBOARD_DETECTED = LogConstant(
       "LOG406", "INFO", "Keyboard device detected",
       "Controller", "Input",
   )

Code Ranges
-----------

.. list-table::
   :widths: 18 25 57
   :header-rows: 1

   * - Range
     - Component
     - Examples
   * - ``LOG401``–``LOG429``
     - Controller
     - Session lifecycle (``LOG401``–``LOG404``), human input
       (``LOG405``–``LOG407``), live telemetry (``LOG420+``)
   * - ``LOG500``–``LOG599``
     - Service
     - Adapter loading, telemetry forwarding,
       :doc:`validation` events (``LOG_SERVICE_VALIDATION_*``)
   * - ``LOG600``–``LOG699``
     - Core / Adapter
     - Environment stepping, :doc:`/documents/architecture/paradigms`
       switching, adapter creation
   * - ``LOG685``–``LOG689``
     - Operator
     - :doc:`/documents/architecture/operators/index` registration,
       launch, stop
   * - ``LOG700+``
     - Worker
     - :doc:`/documents/architecture/workers/integrated_workers/CleanRL_Worker/index`,
       BALROG, XuanCe, LLM worker events
   * - ``LOG1001``–``LOG1015``
     - Operator UI
     - :doc:`/documents/rendering_tabs/render_tabs` config widget,
       render container, script execution

Lookup Helpers
--------------

.. code-block:: python

   from gym_gui.logging_config.log_constants import (
       get_constant_by_code,
       list_known_components,
       get_component_snapshot,
       validate_log_constants,
   )

   # Find a constant by its numeric code
   c = get_constant_by_code("LOG401")

   # List all registered components
   components = list_known_components()  # ["Adapter", "Controller", ...]

   # Component → subcomponent mapping
   snapshot = get_component_snapshot()
   # {"Controller": {"Session", "Input", ...}, "Service": {...}, ...}

   # Runtime validation: ensures no duplicate codes
   errors = validate_log_constants()  # [] if clean

Directory Layout
----------------

.. code-block:: text

   gym_gui/
     logging_config/
       __init__.py          # Exports configure_logging
       log_constants.py     # 500+ LogConstant definitions (~5,400 lines)
       helpers.py           # log_constant(), LogConstantMixin, _RESERVED_LOG_KEYS
       logger.py            # configure_logging(), filters, formatters, registry

See Also
--------

- :doc:`structured_logging`: ``LogConstantMixin``, filters, formatters, and
  the full logging pipeline that consumes these constants.
- :doc:`constants`: numeric defaults (queue sizes, buffer sizes, credit
  thresholds) that govern :doc:`/documents/rendering_tabs/slow_lane`
  backpressure.
- :doc:`validation`: ``ValidationService`` uses ``log_constant()`` to emit
  structured validation events (``LOG_SERVICE_VALIDATION_*``).
