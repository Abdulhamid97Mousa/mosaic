Log Constants & Structured Logging
====================================

Every loggable event in MOSAIC is represented by a **LogConstant** — an
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

LogConstantMixin
----------------

Services and controllers inherit ``LogConstantMixin``
(``gym_gui/logging_config/helpers.py``) to gain structured logging without
direct ``logger.info()`` / ``logger.error()`` calls:

.. code-block:: python

   from gym_gui.logging_config.helpers import LogConstantMixin
   from gym_gui.logging_config.log_constants import LOG_SESSION_ADAPTER_LOAD_ERROR

   class SessionController(LogConstantMixin):
       def load_adapter(self):
           try:
               adapter = self._create_adapter(env_name)
           except Exception as exc:
               self.log_constant(
                   LOG_SESSION_ADAPTER_LOAD_ERROR,
                   exc_info=exc,
                   extra={"run_id": self.run_id, "env": env_name},
               )

The underlying ``log_constant()`` function:

.. code-block:: python

   def log_constant(
       logger: logging.Logger,
       constant: LogConstant,
       *,
       message: str | None = None,      # override constant.message
       extra: Mapping[str, Any] | None = None,
       exc_info: BaseException | tuple | None = None,
   ) -> None: ...

It:

1. Resolves the log level from the constant.
2. Injects ``log_code``, ``component``, ``subcomponent``, and ``tags`` into the
   record's ``extra`` dict.
3. Filters **reserved record keys** to prevent conflicts with Python's logging
   internals:

.. code-block:: python

   _RESERVED_LOG_KEYS = {
       "name", "msg", "args", "created", "filename", "funcName",
       "levelname", "levelno", "lineno", "module", "msecs", "pathname",
       "process", "processName", "relativeCreated", "stack_info",
       "exc_info", "exc_text", "thread", "threadName", "taskName",
       "message", "asctime",
   }

Logging Configuration
---------------------

``configure_logging()`` (``gym_gui/logging_config/logger.py``) sets up the
full pipeline via ``logging.config.dictConfig``:

.. code-block:: python

   def configure_logging(
       level: int = logging.INFO,
       *,
       stream: bool = True,
       log_to_file: bool = True,
       force: bool = False,
   ) -> None: ...

**Format string** (all records, all handlers):

.. code-block:: text

   %(asctime)s | %(levelname)-7s | %(name)s | [comp=%(component)s sub=%(subcomponent)s run=%(run_id)s agent=%(agent_id)s code=%(log_code)s tags=%(tags)s] | %(message)s

**Filters:**

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Filter
     - Purpose
   * - ``CorrelationIdFilter``
     - Ensures every record has ``run_id``, ``agent_id``, ``log_code``, and
       ``tags`` attributes — even if the emitter didn't provide them.
       Enables per-experiment, per-agent log tracing.
   * - ``ComponentFilter``
     - Resolves the logger name to a ``(component, subcomponent)`` pair via
       the ``_ComponentRegistry`` prefix table.  Also tracks the set of
       observed severity levels per component at runtime.
   * - ``GrpcBlockingIOFilter``
     - Suppresses noisy asyncio ``BlockingIOError`` warnings emitted by
       the gRPC layer.

**Output handlers:**

.. list-table::
   :widths: 22 20 58
   :header-rows: 1

   * - Handler
     - File
     - Notes
   * - ``StreamHandler``
     - stderr
     - All levels.
   * - ``RotatingFileHandler``
     - ``gym_gui.log``
     - 10 MB max, 5 backups — main application log.
   * - ``RotatingFileHandler``
     - ``operators.log``
     - 10 MB max, 5 backups — dedicated
       :doc:`/documents/architecture/operators/index` and UI log.

Component Registry
------------------

The ``_ComponentRegistry`` (``gym_gui/logging_config/logger.py``) maps logger
name prefixes to human-readable component labels:

.. list-table::
   :widths: 40 25
   :header-rows: 1

   * - Logger prefix
     - Component
   * - ``gym_gui.operators``
     - Operator
   * - ``gym_gui.ui``
     - UI
   * - ``gym_gui.controllers``
     - Controller
   * - ``gym_gui.core.adapters``
     - Adapter
   * - ``gym_gui.core``
     - Core
   * - ``gym_gui.services``
     - Service
   * - ``gym_gui.telemetry``
     - Telemetry
   * - ``gym_gui.logging``
     - Logging

``get_component_snapshot()`` returns a dictionary of every component and the
set of severity levels observed at runtime — useful for monitoring dashboards
and :doc:`/documents/rendering_tabs/render_tabs` status indicators.

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

   # Runtime validation — ensures no duplicate codes
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

- :doc:`constants` — :doc:`/documents/runtime_logging/constants` provides
  the numeric defaults (queue sizes, buffer sizes, credit thresholds) that
  govern :doc:`/documents/rendering_tabs/slow_lane` backpressure.
- :doc:`validation` — ``ValidationService`` uses ``log_constant()`` to
  emit structured validation events (``LOG_SERVICE_VALIDATION_*``).
