Structured Logging
==================

MOSAIC's structured logging pipeline converts every ``LogConstant`` into a
rich, filterable log record and routes it to the correct output sink.

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

The underlying ``log_constant()`` function signature:

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
       ``tags`` attributes: even if the emitter did not provide them.
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
     - 10 MB max, 5 backups: main application log.
   * - ``RotatingFileHandler``
     - ``operators.log``
     - 10 MB max, 5 backups: dedicated
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
set of severity levels observed at runtime: useful for monitoring dashboards
and :doc:`/documents/rendering_tabs/render_tabs` status indicators.

See Also
--------

- :doc:`log_constants`: the ``LogConstant`` dataclass, 500+ code definitions,
  and lookup helpers.
- :doc:`constants`: numeric defaults (queue sizes, buffer sizes, credit
  thresholds) used by the rendering and backpressure subsystems.
