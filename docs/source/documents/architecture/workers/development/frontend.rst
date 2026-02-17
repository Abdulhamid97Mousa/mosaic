Frontend: Qt6 UI Integration
============================

This guide covers the GUI side of adding a new worker.  Once the
:doc:`backend package <backend>` is in place, the steps below connect
it to the Qt6 interface so users can launch training runs from the
GUI.

Overview
--------

The frontend uses three parallel registries that all share a single
``worker_id`` string:

.. mermaid::

   graph LR
       subgraph "Three Registries"
           FAC["Form Factory<br/>forms/factory.py"]
           CAT["Worker Catalog<br/>worker_catalog/catalog.py"]
           PRES["Presenter Registry<br/>presenters/workers/registry.py"]
       end

       subgraph "Bridge"
           HANDLER["TrainingFormHandler"]
       end

       subgraph "Backend"
           CLIENT["TrainerClient (gRPC)"]
       end

       FAC -->|"create_train_form()"| HANDLER
       HANDLER -->|"submit_config()"| CLIENT
       CAT -.->|"capability flags"| FAC
       PRES -.->|"create_tabs()"| HANDLER

       style FAC fill:#4a90d9,stroke:#2e5a87,color:#fff
       style CAT fill:#4a90d9,stroke:#2e5a87,color:#fff
       style PRES fill:#4a90d9,stroke:#2e5a87,color:#fff
       style HANDLER fill:#50c878,stroke:#2e8b57,color:#fff
       style CLIENT fill:#ff7f50,stroke:#cc5500,color:#fff

.. important::

   The ``worker_id`` string (e.g. ``"cleanrl_worker"``) must be
   identical across all three registries and must match the backend
   package name.

Step 1: Worker Catalog Entry
----------------------------

Add your worker to the catalog so the GUI knows what it supports.
Edit ``gym_gui/ui/worker_catalog/catalog.py`` and add a
``WorkerDefinition`` to the tuple returned by ``get_worker_catalog()``:

.. code-block:: python

   WorkerDefinition(
       worker_id="my_worker",
       display_name="My Worker",
       description="Custom RL training worker",
       supports_training=True,
       supports_policy_load=False,
       requires_live_telemetry=True,
       provides_fast_analytics=False,
       supports_multi_agent=False,
   )

The ``WorkerDefinition`` fields control which buttons and menu items
the GUI enables for this worker:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Effect
   * - ``supports_training``
     - Shows the "Train" button in the control panel
   * - ``supports_policy_load``
     - Shows the "Load Policy" button
   * - ``requires_live_telemetry``
     - Enables the live metrics tab during training
   * - ``provides_fast_analytics``
     - Enables the FastLane real-time visualization tab
   * - ``supports_multi_agent``
     - Exposes multi-agent configuration options

Step 2: Training Form Dialog
-----------------------------

Create a ``QDialog`` subclass that collects user configuration and
returns it as a dictionary.  The only contract the form must fulfill
is implementing ``get_config() -> dict``.

Create ``gym_gui/ui/widgets/my_worker_train_form.py``:

.. code-block:: python

   from PySide6 import QtWidgets
   from typing import Any, Dict

   class MyWorkerTrainForm(QtWidgets.QDialog):
       def __init__(self, parent=None, *, default_game=None, **kwargs):
           super().__init__(parent)
           self.setWindowTitle("My Worker -- Training Configuration")
           self._build_ui()

       def _build_ui(self):
           layout = QtWidgets.QFormLayout(self)

           self._env_combo = QtWidgets.QComboBox()
           self._env_combo.addItems(["CartPole-v1", "LunarLander-v3"])
           layout.addRow("Environment:", self._env_combo)

           self._steps_spin = QtWidgets.QSpinBox()
           self._steps_spin.setRange(1_000, 10_000_000)
           self._steps_spin.setValue(100_000)
           layout.addRow("Total Steps:", self._steps_spin)

           buttons = QtWidgets.QDialogButtonBox(
               QtWidgets.QDialogButtonBox.Ok
               | QtWidgets.QDialogButtonBox.Cancel
           )
           buttons.accepted.connect(self.accept)
           buttons.rejected.connect(self.reject)
           layout.addRow(buttons)

       def get_config(self) -> Dict[str, Any]:
           """Return the trainer payload.

           Called by TrainingFormHandler after the dialog is accepted.
           The dict must contain at minimum: ``run_name``,
           ``entry_point``, ``arguments``, and ``metadata``.
           """
           import sys
           import uuid

           run_id = f"my_worker_{uuid.uuid4().hex[:8]}"
           return {
               "run_name": run_id,
               "entry_point": sys.executable,
               "arguments": ["-m", "my_worker.cli"],
               "metadata": {
                   "ui": {
                       "worker_id": "my_worker",
                       "env_id": self._env_combo.currentText(),
                   },
                   "worker": {
                       "module": "my_worker.cli",
                       "use_grpc": True,
                       "config": {
                           "run_id": run_id,
                           "env_id": self._env_combo.currentText(),
                           "total_steps": self._steps_spin.value(),
                       },
                   },
               },
           }

Step 3: Self-Registration
-------------------------

Register the form with the ``WorkerFormFactory`` at module load time.
Add this block at the **bottom** of your form file:

.. code-block:: python

   # -- Self-registration (bottom of my_worker_train_form.py) --
   from gym_gui.ui.forms.factory import get_worker_form_factory

   _factory = get_worker_form_factory()
   if not _factory.has_train_form("my_worker"):
       _factory.register_train_form(
           "my_worker",
           lambda parent=None, **kw: MyWorkerTrainForm(parent=parent, **kw),
       )

The ``has_train_form()`` guard makes the registration idempotent --
safe if the module is imported more than once.

The ``WorkerFormFactory`` supports five form buckets.  Register
additional forms if your worker supports them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - When to Use
   * - ``register_train_form()``
     - Start a new training run
   * - ``register_policy_form()``
     - Load and evaluate a trained policy
   * - ``register_resume_form()``
     - Resume a previously interrupted run
   * - ``register_evaluation_form()``
     - Run evaluation-only mode
   * - ``register_script_form()``
     - Execute a custom script

Step 4: Worker Presenter
------------------------

The presenter handles two responsibilities:

1. **Build train request** -- convert form data into a gRPC-ready dict
2. **Create analytics tabs** -- provide live QWidget tabs during
   training (e.g. FastLane visualization, reward plots)

Create ``gym_gui/ui/presenters/workers/my_worker_presenter.py``:

.. code-block:: python

   from typing import Any, List, Optional

   class MyWorkerPresenter:
       @property
       def id(self) -> str:
           return "my_worker"

       def build_train_request(
           self, policy_path: Any, current_game: Optional[Any]
       ) -> dict:
           """Build gRPC request dict from form data."""
           raise NotImplementedError(
               "My Worker uses form-based submission, "
               "not presenter-based."
           )

       def create_tabs(
           self,
           run_id: str,
           agent_id: str,
           first_payload: dict,
           parent: Any,
       ) -> List[Any]:
           """Return worker-specific QWidget tabs for live telemetry."""
           return []  # No custom tabs yet

Then register it in
``gym_gui/ui/presenters/workers/__init__.py``:

.. code-block:: python

   from .my_worker_presenter import MyWorkerPresenter
   _registry.register("my_worker", MyWorkerPresenter())

Signal Flow
-----------

No changes are needed to ``TrainingFormHandler`` or ``MainWindow``
wiring.  The existing signal chain routes everything automatically
by ``worker_id``:

.. mermaid::

   sequenceDiagram
       participant CP as ControlPanel
       participant TFH as TrainingFormHandler
       participant FAC as WorkerFormFactory
       participant DLG as MyWorkerTrainForm
       participant TC as TrainerClient (gRPC)
       participant WK as Worker Process

       CP->>TFH: train_agent_requested("my_worker")
       TFH->>FAC: create_train_form("my_worker")
       FAC-->>TFH: dialog instance
       TFH->>DLG: dialog.exec()
       Note over DLG: User fills form,<br/>clicks OK
       DLG-->>TFH: Accepted
       TFH->>DLG: dialog.get_config()
       DLG-->>TFH: config dict
       TFH->>TC: submit_run(config)
       TC->>WK: gRPC spawn

1. The user selects a worker and clicks **Train** in the
   ``ControlPanel``.
2. ``ControlPanel`` emits ``train_agent_requested(worker_id)``.
3. ``TrainingFormHandler.on_train_agent_requested()`` looks up the
   form in ``WorkerFormFactory``, opens the dialog, and waits for
   the user.
4. On accept, it calls ``dialog.get_config()`` and forwards the
   result to ``TrainerClient.submit_run()`` via gRPC.
5. The Daemon spawns the worker subprocess.

