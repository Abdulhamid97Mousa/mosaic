Backend: Worker Package
=======================

This guide walks through creating the backend package for a new
MOSAIC worker under ``3rd_party/``.  By the end you will have a
fully functional worker that the Daemon can spawn and that emits
standardized JSONL telemetry.

Prerequisites
-------------

- Python 3.10+
- MOSAIC installed (``pip install -e .``)
- Your RL library or agent framework installed

Overview
--------

Every worker consists of five components:

.. mermaid::

   graph LR
       A["1. Config<br/>config.py"] --> B["2. Runtime<br/>runtime.py"]
       B --> C["3. Telemetry<br/>telemetry.py"]
       B --> D["4. Analytics<br/>analytics.py"]
       E["5. Entry Point<br/>pyproject.toml"] -.->|"discovery"| A

       style A fill:#4a90d9,stroke:#2e5a87,color:#fff
       style B fill:#ff7f50,stroke:#cc5500,color:#fff
       style C fill:#50c878,stroke:#2e8b57,color:#fff
       style D fill:#50c878,stroke:#2e8b57,color:#fff
       style E fill:#9370db,stroke:#6a0dad,color:#fff

Step 1: Create the Package
--------------------------

.. code-block:: bash

   mkdir -p 3rd_party/my_worker/my_worker
   touch 3rd_party/my_worker/my_worker/__init__.py

Step 2: Configuration (``config.py``)
--------------------------------------

Implement the ``WorkerConfig`` protocol — a dataclass with ``run_id``,
``seed``, ``to_dict()``, and ``from_dict()``:

.. code-block:: python

   from __future__ import annotations
   from dataclasses import dataclass
   from typing import Any, Dict

   @dataclass
   class MyWorkerConfig:
       # Protocol-required fields
       run_id: str
       seed: int | None = None

       # Worker-specific fields
       env_id: str = "CartPole-v1"
       algorithm: str = "dqn"
       total_steps: int = 100_000
       learning_rate: float = 1e-4

       def __post_init__(self) -> None:
           if not self.run_id:
               raise ValueError("run_id is required")

       def to_dict(self) -> Dict[str, Any]:
           return {
               "run_id": self.run_id,
               "seed": self.seed,
               "env_id": self.env_id,
               "algorithm": self.algorithm,
               "total_steps": self.total_steps,
               "learning_rate": self.learning_rate,
           }

       @classmethod
       def from_dict(cls, data: Dict[str, Any]) -> "MyWorkerConfig":
           fields = cls.__dataclass_fields__
           return cls(**{k: v for k, v in data.items() if k in fields})

Step 3: Runtime (``runtime.py``)
---------------------------------

Manage the worker lifecycle — emit ``run_started``, run the training
loop, and emit ``run_completed`` or ``run_failed``:

.. code-block:: python

   import json
   import sys
   import time
   from typing import Any, Dict

   class MyWorkerRuntime:
       def __init__(self, config: MyWorkerConfig):
           self.config = config

       def run(self) -> Dict[str, Any]:
           self._emit_lifecycle("run_started", {
               "env_id": self.config.env_id,
               "algorithm": self.config.algorithm,
           })

           try:
               result = self._train()
               summary = {"status": "completed", **result}
               self._emit_lifecycle("run_completed", summary)
               return summary
           except Exception as e:
               self._emit_lifecycle("run_failed", {"error": str(e)})
               raise

       def _train(self) -> Dict[str, Any]:
           """Your training logic goes here."""
           for step in range(self.config.total_steps):
               # ... train one step ...

               # Emit step telemetry every N steps
               if step % 100 == 0:
                   self._emit_step(step, reward=1.0)

               # Emit heartbeat every 60s
               if step % 10_000 == 0:
                   self._emit_lifecycle("heartbeat", {"step": step})

           return {"total_steps": self.config.total_steps}

       def _emit_step(self, step: int, reward: float):
           event = {
               "event_type": "step",
               "run_id": self.config.run_id,
               "step_index": step,
               "reward": reward,
           }
           print(json.dumps(event), file=sys.stdout, flush=True)

       def _emit_lifecycle(self, event_name: str, payload: dict):
           event = {
               "event": event_name,
               "run_id": self.config.run_id,
               "timestamp": time.time(),
               "payload": payload,
           }
           print(json.dumps(event), file=sys.stdout, flush=True)

.. important::

   Always use ``flush=True`` when printing telemetry.  The Telemetry
   Proxy reads ``stdout`` line-by-line — buffered output causes
   delayed or missing telemetry.

Step 4: Worker Metadata (``__init__.py``)
------------------------------------------

Register the worker's identity and capabilities for automatic
discovery:

.. code-block:: python

   def get_worker_metadata() -> tuple:
       from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

       metadata = WorkerMetadata(
           name="My Worker",
           version="1.0.0",
           description="My RL worker for MOSAIC",
           author="Your Name",
           homepage="https://github.com/...",
           upstream_library="mylib",
           upstream_version="1.0.0",
           license="MIT",
       )

       capabilities = WorkerCapabilities(
           worker_type="myworker",
           supported_paradigms=("sequential",),
           env_families=("gymnasium",),
           action_spaces=("discrete", "continuous"),
           observation_spaces=("vector", "image"),
           max_agents=1,
           supports_checkpointing=True,
           requires_gpu=False,
       )

       return metadata, capabilities

Step 5: Entry Point (``pyproject.toml``)
-----------------------------------------

Register the worker so MOSAIC discovers it automatically:

.. code-block:: toml

   [project]
   name = "my-worker"
   version = "1.0.0"
   requires-python = ">=3.10"

   [project.entry-points."mosaic.workers"]
   myworker = "my_worker:get_worker_metadata"

   [build-system]
   requires = ["setuptools>=64"]
   build-backend = "setuptools.backends._legacy:_Backend"

Then install in development mode:

.. code-block:: bash

   cd 3rd_party/my_worker
   pip install -e .

Step 6: CLI Entry Point (``cli.py``)
--------------------------------------

Create the command-line interface that the Daemon invokes:

.. code-block:: python

   import argparse
   import json
   from my_worker.config import MyWorkerConfig
   from my_worker.runtime import MyWorkerRuntime

   def main(argv=None):
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", required=True)
       parser.add_argument("--grpc", action="store_true")
       parser.add_argument("--grpc-target", default="127.0.0.1:50055")
       args = parser.parse_args(argv)

       # Load config from JSON file
       with open(args.config) as f:
           config = MyWorkerConfig.from_dict(json.load(f))

       # Run the worker
       runtime = MyWorkerRuntime(config)
       runtime.run()

   if __name__ == "__main__":
       main()

Step 7: Add Requirements
-------------------------

Create ``requirements/myworker.txt`` in the MOSAIC root:

.. code-block:: text

   # My Worker dependencies
   mylib>=1.0.0
   -e 3rd_party/my_worker

And add to MOSAIC's ``pyproject.toml`` optional dependencies:

.. code-block:: toml

   [project.optional-dependencies]
   myworker = ["-r requirements/myworker.txt"]

Testing
-------

Verify your worker passes the MOSAIC standardization tests:

.. code-block:: python

   # tests/test_my_worker_standardization.py

   def test_config_protocol():
       """Config implements WorkerConfig protocol."""
       from my_worker.config import MyWorkerConfig
       config = MyWorkerConfig(run_id="test-001", seed=42)
       assert config.run_id == "test-001"
       d = config.to_dict()
       restored = MyWorkerConfig.from_dict(d)
       assert restored.run_id == config.run_id

   def test_metadata():
       """Worker provides valid metadata."""
       from my_worker import get_worker_metadata
       metadata, capabilities = get_worker_metadata()
       assert metadata.name
       assert capabilities.worker_type

   def test_lifecycle_events(capsys):
       """Worker emits required lifecycle events."""
       from my_worker.config import MyWorkerConfig
       from my_worker.runtime import MyWorkerRuntime
       import json

       config = MyWorkerConfig(run_id="test-001", total_steps=10)
       runtime = MyWorkerRuntime(config)
       runtime.run()

       output = capsys.readouterr().out
       lines = [json.loads(l) for l in output.strip().split("\\n")]
       events = [l.get("event") for l in lines if "event" in l]
       assert "run_started" in events
       assert "run_completed" in events

