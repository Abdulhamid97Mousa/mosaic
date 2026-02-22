Runtime Logs
============

MOSAIC uses a **structured logging** system built on Python's ``logging``
module.  Every log event carries a numeric code, component tag, and optional
correlation IDs so that logs from multi-agent, multi-worker experiments can
be filtered, traced, and aggregated.  This section covers:

- :doc:`log_constants`: the ``LogConstant`` dataclass, 500+ code definitions,
  and lookup helpers.
- :doc:`structured_logging`: ``LogConstantMixin``, filters, formatters, and
  the full logging pipeline.
- :doc:`constants`: the ``gym_gui/constants/`` package that centralises every
  tuning knob, default value, and magic number (160 exports).
- :doc:`validation`: Pydantic models for telemetry payloads,
  ``ValidationService`` for runtime checking, and Qt UI validators for form
  fields.
- :doc:`observability/index`: TensorBoard and Weights and Biases integration.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       subgraph Source["Application Code"]
           SVC["Services / Controllers / UI"]
       end

       subgraph Logging["Logging Pipeline"]
           LC["LogConstant<br/>(code · level · component · tags)"]
           MIXIN["LogConstantMixin<br/>self.log_constant(...)"]
           CIF["CorrelationIdFilter<br/>(run_id · agent_id)"]
           CF["ComponentFilter<br/>(component · subcomponent)"]
           GBIO["GrpcBlockingIOFilter"]
           FMT["CustomFormatter<br/>[comp=X sub=Y run=R agent=A code=LOG401]"]
       end

       subgraph Sinks["Output Handlers"]
           CON["StreamHandler<br/>(stderr)"]
           MAIN["RotatingFileHandler<br/>gym_gui.log · 10 MB × 5"]
           OPS["RotatingFileHandler<br/>operators.log · 10 MB × 5"]
       end

       SVC --> MIXIN --> LC
       LC --> CIF --> CF --> GBIO --> FMT
       FMT --> CON & MAIN & OPS

       style Logging fill:#fff3e0,stroke:#e65100,color:#333
       style Sinks fill:#e3f2fd,stroke:#1565c0,color:#333

The logging pipeline feeds into three sinks.  Every record passes through
``CorrelationIdFilter`` (injects ``run_id``/``agent_id``), ``ComponentFilter``
(resolves the logger name to a component label), and ``GrpcBlockingIOFilter``
(suppresses noisy gRPC warnings) before reaching the ``CustomFormatter``.

The :doc:`constants` package provides all numeric defaults used by the
:doc:`/documents/rendering_tabs/index` subsystem (queue sizes, FPS caps,
credit thresholds) and the :doc:`validation` layer enforces schema correctness
on telemetry payloads before they enter the
:doc:`/documents/rendering_tabs/slow_lane`.

.. toctree::
   :maxdepth: 2

   log_constants
   structured_logging
   constants
   validation
   observability/index
