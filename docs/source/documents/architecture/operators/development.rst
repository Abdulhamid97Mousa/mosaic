Developing an Operator
======================

This guide walks through creating a new operator type for MOSAIC.
Unlike Workers (which are standalone packages in ``3rd_party/``),
Operators are lightweight, they need an ``InteractiveRuntime`` inside
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

   %%{init: {"flowchart": {"curve": "linear"}} }%%
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

-----

Adding an Environment Family
=============================

An **environment family** groups related environments under a single
dropdown entry in the Operators tab. Adding a new family requires
changes in two files:

- ``gym_gui/ui/widgets/operator_config_widget.py`` -- family registry
  and UI logic
- ``gym_gui/ui/main_window.py`` -- environment creation and preview
  rendering

This section walks through each step. The ``mosaic_multigrid`` /
``ini_multigrid`` split is used as a running example.

Step 1: Add to ``ENV_FAMILIES``
-------------------------------

Open ``gym_gui/ui/widgets/operator_config_widget.py`` and add a new
key to the ``ENV_FAMILIES`` dict:

.. code-block:: python

   ENV_FAMILIES: Dict[str, Tuple[str, ...]] = {
       # ... existing families ...
       "my_family": (
           "MyEnv-TaskA-v0",
           "MyEnv-TaskB-v0",
           "MyEnv-TaskC-v0",
       ),
   }

Each string is a gym-registered environment ID that appears in the
**Task** dropdown when the family is selected.

If your environments are registered dynamically (like ``babyai`` or
``meltingpot``), use an empty tuple and populate it at runtime:

.. code-block:: python

   "my_family": (),  # Loaded dynamically

.. tip::

   Use the correct gym ID strings. If a package registers environments
   via ``gym.register(id="MyEnv-TaskA-v0", ...)``, the ``ENV_FAMILIES``
   tuple must use ``"MyEnv-TaskA-v0"`` exactly.

Step 2: Update ``_auto_detect_agent_count``
-------------------------------------------

If your family is multi-agent, add detection logic so the GUI
knows how many agents to display:

.. code-block:: python

   def _auto_detect_agent_count(env_family: str, env_id: str) -> int:
       # ... existing checks ...

       elif env_family == "my_family":
           # Instantiate and query agent count
           import gym
           env = gym.make(env_id)
           num_agents = getattr(env, 'num_agents', 1)
           env.close()
           return num_agents

For single-agent families, this function can return ``0`` (the
default) and no multi-agent panel will be shown.

Step 3: Update ``_get_execution_mode``
--------------------------------------

Declare whether your family uses turn-based (AEC) or simultaneous
(parallel) stepping:

.. code-block:: python

   def _get_execution_mode(env_family: str) -> str:
       if env_family in ("pettingzoo", "pettingzoo_classic", "open_spiel"):
           return "aec"
       elif env_family in ("mosaic_multigrid", "ini_multigrid",
                           "meltingpot", "overcooked", "my_family"):
           return "parallel"
       return "aec"

Step 4: Update multi-agent guard tuples
----------------------------------------

Several functions use tuples to identify multi-agent or
simultaneous-only families. Add your family name to each:

1. ``_is_multiagent_env_selected()`` -- determines whether to show
   the player assignment panel:

   .. code-block:: python

      return env_family in (
          "pettingzoo", "pettingzoo_classic", "open_spiel",
          "mosaic_multigrid", "ini_multigrid",
          "meltingpot", "overcooked",
          "my_family",  # <-- add here
      )

2. ``simultaneous_only_envs`` in ``_update_multiagent_panel()`` --
   disables the AEC option for families that only support parallel
   mode:

   .. code-block:: python

      simultaneous_only_envs = (
          "overcooked", "mosaic_multigrid", "ini_multigrid",
          "meltingpot",
          "my_family",  # <-- if simultaneous-only
      )

3. Agent ID generation block -- generates ``agent_0``, ``agent_1``,
   etc. for simultaneous families:

   .. code-block:: python

      elif env_family in ("mosaic_multigrid", "ini_multigrid",
                          "meltingpot", "overcooked", "my_family"):
          agent_ids = [f"agent_{i}" for i in range(num_agents)]

4. ``_is_parallel_multiagent()`` in ``main_window.py`` -- identifies
   parallel environments for the shared-environment execution path:

   .. code-block:: python

      if first_config.env_name in (
          "mosaic_multigrid", "ini_multigrid",
          "meltingpot", "overcooked",
          "my_family",  # <-- add here
      ):

Step 5: Add environment creation
---------------------------------

In ``main_window.py``, add a branch to
``_create_parallel_multiagent_env()`` so the runtime can instantiate
your environments:

.. code-block:: python

   elif env_name == "my_family":
       import gymnasium
       import my_package.envs  # triggers gymnasium.register() calls
       env = gymnasium.make(task)
       env.render_mode = 'rgb_array'
       return env

.. important::

   Use ``gymnasium.make(task)`` with the registered environment ID.
   Do **not** hardcode specific class imports -- this ensures all
   variants (current and future) are handled correctly. See
   :doc:`/documents/tutorials/installation/common_errors/operators/index`
   for the preview-hang bug caused by hardcoded class imports.

   Both ``mosaic_multigrid`` and ``ini_multigrid`` use the modern
   Gymnasium API (``import gymnasium``), not the deprecated OpenAI
   Gym (``import gym``).

Step 6: Add preview rendering
------------------------------

In the ``_on_initialize_operator()`` method of ``main_window.py``,
add an ``elif`` branch so "Load Environment" shows a preview frame:

.. code-block:: python

   elif env_name == "my_family":
       try:
           import gymnasium
           import my_package.envs  # noqa: F401

           env = gymnasium.make(task)
           env.render_mode = 'rgb_array'
           env.reset()
           rgb_frame = env.render()
           env.close()
       except ImportError as import_err:
           self._status_bar.showMessage(
               f"my_package not installed - cannot preview: {import_err}",
               5000
           )
           return
       except Exception as e:
           self._status_bar.showMessage(
               f"Cannot preview my_family {task}: {e}",
               5000
           )
           return

Step 7: Add settings panel (optional)
--------------------------------------

If your family needs family-specific UI controls (like the multigrid
observation mode / coordination strategy panel), create a container
widget in the ``_create_multiagent_controls()`` method and
show/hide it based on the selected family:

.. code-block:: python

   if env_family in ("mosaic_multigrid", "ini_multigrid"):
       self._multigrid_settings_container.show()
   elif env_family == "my_family":
       self._my_family_settings_container.show()
   else:
       self._multigrid_settings_container.hide()
       self._my_family_settings_container.hide()

Environment Family Checklist
----------------------------

Use this checklist when adding a new environment family:

- [ ] ``ENV_FAMILIES`` dict has a key with all gym-registered IDs
- [ ] ``_auto_detect_agent_count()`` handles the family (if multi-agent)
- [ ] ``_get_execution_mode()`` returns ``"aec"`` or ``"parallel"``
- [ ] ``_is_multiagent_env_selected()`` includes the family
- [ ] ``simultaneous_only_envs`` tuple includes it (if parallel-only)
- [ ] Agent ID generation block includes it
- [ ] ``_is_parallel_multiagent()`` in ``main_window.py`` includes it
- [ ] ``_create_parallel_multiagent_env()`` has a creation branch
- [ ] ``_on_initialize_operator()`` has a preview rendering branch
- [ ] All gym IDs use ``gym.make()`` -- no hardcoded class imports
- [ ] Preview error messages name the family clearly
- [ ] ``gym_gui/tests/`` has passing tests for the new family

Example: ``mosaic_multigrid`` / ``ini_multigrid`` Split
-------------------------------------------------------

The original ``"multigrid"`` family mixed two distinct environment
sets. It was split into two independent families:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Property
     - ``mosaic_multigrid``
     - ``ini_multigrid``
   * - Source
     - `PyPI: mosaic-multigrid <https://pypi.org/project/mosaic-multigrid/>`_
     - `GitHub: ini/multigrid <https://github.com/ini/multigrid>`_
   * - Type
     - Competitive team sports
     - Cooperative exploration
   * - view_size
     - 3
     - 7
   * - Env count
     - 13 (Soccer, Collect, Basketball)
     - 13 (Empty, RedBlueDoors, LockedHallway, etc.)
   * - ID prefix
     - ``MosaicMultiGrid-*``
     - ``MultiGrid-*``
   * - API
     - Gymnasium (``import gymnasium``)
     - Gymnasium (``import gymnasium``)
   * - Install
     - ``pip install ".[mosaic_multigrid]"``
     - ``pip install ".[multigrid_ini]"``
   * - Role assignment
     - Yes (Soccer forward/defender)
     - No

**Key lesson:** Always use ``gymnasium.make(task)`` for environment
creation rather than importing specific classes. The mosaic_multigrid package
has 13 environments across 4 variant tiers (Original, IndAgObs,
TeamObs, Basketball). Hardcoding class imports like
``SoccerGame4HEnv10x15N2`` will silently break when the user selects a
different variant (e.g., ``MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0``
uses ``SoccerGame4HIndAgObsEnv16x11N2``).
