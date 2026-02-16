Developing an Operator
======================

This guide walks through creating a new operator type for MOSAIC.
Unlike Workers (which are standalone packages in ``3rd_party/``),
Operators are lightweight -- they need an ``InteractiveRuntime`` inside
an existing worker subprocess plus a few GUI integration points.

Prerequisites
-------------

- Python 3.10+
- MOSAIC installed (``pip install -e .``)
- An existing worker package (or create a new one following the
  :doc:`Worker development guide <../workers/development>`)

Overview
--------

Adding a new operator requires changes in three areas:

.. mermaid::

   graph LR
       A["1. InteractiveRuntime<br/>(Worker subprocess)"] --> B["2. OperatorLauncher<br/>(Command dispatch)"]
       B --> C["3. GUI Integration<br/>(Config widget)"]

       style A fill:#ff7f50,stroke:#cc5500,color:#fff
       style B fill:#50c878,stroke:#2e8b57,color:#fff
       style C fill:#4a90d9,stroke:#2e5a87,color:#fff

Step 1: Create the InteractiveRuntime
--------------------------------------

The ``InteractiveRuntime`` is the core of your operator.  It reads
JSON commands from stdin, interacts with the environment and your
decision-making logic, and writes JSON responses to stdout.

.. code-block:: python

   # 3rd_party/my_worker/my_worker/runtime.py

   import json
   import sys
   from typing import Any, Optional

   class InteractiveRuntime:
       """Interactive runtime for GUI step-by-step control.

       Protocol:
           Input (stdin):
               {"cmd": "reset", "seed": 42}
               {"cmd": "step"}
               {"cmd": "stop"}

           Output (stdout):
               {"type": "ready", "run_id": "...", "env_id": "...", "seed": 42}
               {"type": "step", "step_index": 0, "action": 2, "reward": 0.5, ...}
               {"type": "episode_end", "total_reward": 0.95, "episode_length": 15}
               {"type": "error", "message": "..."}
       """

       def __init__(self, config):
           self._config = config
           self._env = None
           self._obs = None
           self._step_idx = 0
           self._episode_reward = 0.0

       def _emit(self, data: dict) -> None:
           """Emit a JSON response to stdout."""
           print(json.dumps(data), flush=True)

       def _handle_reset(self, seed: Optional[int] = None) -> None:
           """Reset the environment and prepare for stepping."""
           import gymnasium as gym

           if self._env is None:
               self._env = gym.make(
                   self._config.env_id,
                   render_mode="rgb_array",
               )

           self._obs, info = self._env.reset(seed=seed)
           self._step_idx = 0
           self._episode_reward = 0.0

           self._emit({
               "type": "ready",
               "run_id": self._config.run_id,
               "env_id": self._config.env_id,
               "seed": seed,
           })

       def _select_action(self, observation: Any) -> int:
           """Your decision-making logic goes here."""
           # Example: random action
           return self._env.action_space.sample()

       def _handle_step(self) -> None:
           """Execute one environment step."""
           action = self._select_action(self._obs)
           obs_new, reward, terminated, truncated, info = self._env.step(action)

           self._episode_reward += reward
           self._step_idx += 1

           # Build step response
           step_data = {
               "type": "step",
               "step_index": self._step_idx,
               "action": int(action),
               "reward": float(reward),
               "terminated": terminated,
               "truncated": truncated,
               "episode_reward": self._episode_reward,
           }

           # Add render payload for GUI display
           try:
               rgb_frame = self._env.render()
               if rgb_frame is not None:
                   import numpy as np
                   step_data["render_payload"] = {
                       "mode": "rgb",
                       "rgb": rgb_frame.tolist(),
                       "width": rgb_frame.shape[1],
                       "height": rgb_frame.shape[0],
                   }
           except Exception:
               pass

           self._emit(step_data)

           # Check for episode end
           if terminated or truncated:
               self._emit({
                   "type": "episode_end",
                   "total_reward": self._episode_reward,
                   "episode_length": self._step_idx,
                   "terminated": terminated,
                   "truncated": truncated,
               })
               # Auto-reset for next episode
               self._obs, _ = self._env.reset()
               self._step_idx = 0
               self._episode_reward = 0.0
           else:
               self._obs = obs_new

       def run(self) -> None:
           """Main loop -- read commands from stdin, dispatch."""
           for line in sys.stdin:
               line = line.strip()
               if not line:
                   continue

               try:
                   cmd = json.loads(line)
               except json.JSONDecodeError as e:
                   self._emit({"type": "error", "message": f"Invalid JSON: {e}"})
                   continue

               cmd_type = cmd.get("cmd")
               if cmd_type == "reset":
                   self._handle_reset(cmd.get("seed"))
               elif cmd_type == "step":
                   self._handle_step()
               elif cmd_type == "stop":
                   self._emit({"type": "stopped"})
                   break
               else:
                   self._emit({
                       "type": "error",
                       "message": f"Unknown command: {cmd_type}",
                   })

.. important::

   Always use ``flush=True`` when printing responses.  The GUI reads
   stdout line-by-line -- buffered output causes delayed or missing
   responses.

Step 2: Add the ``--interactive`` CLI Flag
-------------------------------------------

Add the interactive mode flag to your worker's CLI entry point:

.. code-block:: python

   # 3rd_party/my_worker/my_worker/cli.py

   import argparse

   def main(argv=None):
       parser = argparse.ArgumentParser()
       parser.add_argument("--run-id", required=True)
       parser.add_argument("--env-id", default="CartPole-v1")
       parser.add_argument("--render-mode", default=None)
       parser.add_argument(
           "--interactive",
           action="store_true",
           help="Run in interactive mode for GUI step-by-step control.",
       )
       args = parser.parse_args(argv)

       config = MyWorkerConfig(
           run_id=args.run_id,
           env_id=args.env_id,
           render_mode=args.render_mode,
       )

       if args.interactive:
           from my_worker.runtime import InteractiveRuntime
           runtime = InteractiveRuntime(config)
           runtime.run()
       else:
           # Normal training/evaluation mode
           from my_worker.runtime import MyWorkerRuntime
           runtime = MyWorkerRuntime(config)
           runtime.run()

   if __name__ == "__main__":
       main()

Step 3: Register with OperatorLauncher
---------------------------------------

Add your operator type to the command dispatch in
``gym_gui/services/operator_launcher.py``:

.. code-block:: python

   def _build_my_operator_command(
       self,
       config: OperatorConfig,
       run_id: str,
       *,
       interactive: bool = False,
   ) -> list[str]:
       settings = config.settings or {}

       cmd = [
           self._python_executable,
           "-m", "my_worker.cli",
           "--run-id", run_id,
           "--env-id", config.task,
           "--render-mode", "rgb_array",
       ]

       if interactive:
           cmd.append("--interactive")

       # Add operator-specific settings
       if settings.get("my_param"):
           cmd.extend(["--my-param", str(settings["my_param"])])

       return cmd

Then add the dispatch case in ``launch_operator()``:

.. code-block:: python

   elif config.operator_type == "my_type":
       cmd = self._build_my_operator_command(config, run_id, interactive=interactive)

Step 4: Add to Worker Catalog
------------------------------

Register your operator's worker in the GUI catalog so it appears in
the operator type dropdown:

.. code-block:: python

   # gym_gui/ui/worker_catalog/catalog.py

   WorkerDefinition(
       worker_id="my_worker",
       display_name="My Custom Operator",
       description="Description of what your operator does.",
       supports_training=False,
       supports_policy_load=True,
       requires_live_telemetry=True,
       supports_multi_agent=False,
   )

Step 5: Add Constants (Optional)
---------------------------------

For a polished integration, add constants for your operator:

.. code-block:: python

   # gym_gui/constants/constants_operator.py

   OPERATOR_CATEGORY_MY_TYPE = "my_type"
   DEFAULT_OPERATOR_ID_MY_TYPE = "my_type_default"
   OPERATOR_DISPLAY_NAME_MY_TYPE = "My Custom Operator"
   OPERATOR_DESCRIPTION_MY_TYPE = (
       "Description shown in the operator dropdown."
   )

Testing
-------

Test your operator's interactive mode end-to-end:

.. code-block:: python

   # tests/test_my_operator_interactive.py
   import json
   import subprocess

   def test_interactive_protocol():
       """Test the stdin/stdout JSON protocol."""
       proc = subprocess.Popen(
           ["python", "-m", "my_worker.cli",
            "--run-id", "test-001",
            "--env-id", "CartPole-v1",
            "--interactive"],
           stdin=subprocess.PIPE,
           stdout=subprocess.PIPE,
           text=True,
       )

       # Send reset
       proc.stdin.write('{"cmd": "reset", "seed": 42}\n')
       proc.stdin.flush()
       response = json.loads(proc.stdout.readline())
       assert response["type"] == "ready"
       assert response["seed"] == 42

       # Send step
       proc.stdin.write('{"cmd": "step"}\n')
       proc.stdin.flush()
       response = json.loads(proc.stdout.readline())
       assert response["type"] == "step"
       assert "action" in response

       # Send stop
       proc.stdin.write('{"cmd": "stop"}\n')
       proc.stdin.flush()
       response = json.loads(proc.stdout.readline())
       assert response["type"] == "stopped"

       proc.wait(timeout=5)

For Qt integration testing, use **pytest-qt**:

.. code-block:: python

   def test_step_triggers_next_step(qtbot, execution_manager):
       """Test that step response triggers paced next step."""
       with qtbot.waitSignal(
           execution_manager.step_operator, timeout=1000
       ) as blocker:
           execution_manager.on_step_received("op1")

       assert blocker.signal_triggered
       assert blocker.args == ["op1"]

Checklist
---------

Use this checklist to verify your operator is complete:

- ``InteractiveRuntime`` reads stdin commands and writes stdout responses
- ``--interactive`` CLI flag dispatches to ``InteractiveRuntime``
- Responses include ``render_payload`` for GUI display
- ``flush=True`` on all stdout writes
- ``OperatorLauncher`` has a ``_build_*_command()`` method for your type
- Worker appears in ``worker_catalog.py``
- Subprocess protocol test passes (reset/step/stop cycle)
- Episodes auto-reset after termination
