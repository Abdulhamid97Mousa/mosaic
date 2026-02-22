Observability
=============

MOSAIC supports two external experiment-tracking backends for logging metrics,
hyperparameters, and artefacts beyond the built-in
:doc:`structured logging pipeline </documents/runtime_logging/structured_logging>`:
**TensorBoard** and **Weights & Biases (W&B)**.

Both integrations are optional.  Training runs work without them; enabling
either backend adds richer metric visualisation and, in the case of W&B,
cloud-hosted experiment comparison.

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Backend
     - Mode
     - Best for
   * - **TensorBoard**
     - Local
     - Quick scalar plots, lightweight, no account required
   * - **Weights & Biases**
     - Cloud / local
     - Experiment comparison, artefact versioning, team sharing

-----

TensorBoard
-----------

`TensorBoard <https://www.tensorflow.org/tensorboard>`_ is a browser-based
dashboard for visualising training scalars (reward, loss, entropy), histograms,
and media.  MOSAIC workers write TensorBoard summaries via
``torch.utils.tensorboard.SummaryWriter``.

Installation
~~~~~~~~~~~~

TensorBoard is included in MOSAIC's base dependencies.  Verify it is available:

.. code-block:: bash

   source .venv/bin/activate
   python -c "import tensorboard; print(tensorboard.__version__)"

If missing:

.. code-block:: bash

   pip install tensorboard

Enabling per Worker
~~~~~~~~~~~~~~~~~~~

Each worker writes summaries to a run-specific subdirectory under
``var/trainer/runs/<run_id>/tensorboard/``.  The path is controlled by the
worker config:

**CleanRL:**

.. code-block:: bash

   --track --tensorboard-log var/trainer/runs

**XuanCe** (via YAML or GUI extras):

.. code-block:: yaml

   logger: tensorboard
   log_dir: var/trainer/runs

**Ray RLlib** (via ``extras`` in the run config):

.. code-block:: json

   {
     "tensorboard_dir": "var/trainer/runs"
   }

Launching the Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   source .venv/bin/activate
   tensorboard --logdir var/trainer/runs --port 6006

Then open ``http://localhost:6006`` in a browser.  All runs under
``var/trainer/runs/`` appear as separate experiment entries.

For WSL users, forward the port from WSL to Windows:

.. code-block:: bash

   tensorboard --logdir var/trainer/runs --host 0.0.0.0 --port 6006

Then open ``http://localhost:6006`` from the Windows browser.

Key Metrics Logged
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Tag
     - Description
   * - ``charts/episodic_return``
     - Mean episode return per rollout
   * - ``charts/episodic_length``
     - Mean episode length
   * - ``losses/policy_loss``
     - Policy gradient loss (PPO clip loss)
   * - ``losses/value_loss``
     - Value function MSE loss
   * - ``losses/entropy``
     - Policy entropy (exploration signal)
   * - ``charts/learning_rate``
     - Current learning rate (with schedule)
   * - ``charts/SPS``
     - Environment steps per second

-----

Weights and Biases
------------------

`Weights and Biases <https://wandb.ai>`_ (W&B) is a cloud experiment-tracking
platform.  It stores runs, metrics, system stats, model artefacts, and allows
side-by-side comparison of experiments across machines and team members.

Installation
~~~~~~~~~~~~

.. code-block:: bash

   source .venv/bin/activate
   pip install wandb

Authenticate once per machine:

.. code-block:: bash

   wandb login

This stores a credentials token in ``~/.netrc``.  For offline / air-gapped
environments see the Offline Mode section below.

Enabling per Worker
~~~~~~~~~~~~~~~~~~~

**CleanRL:**

.. code-block:: bash

   --track --wandb-project mosaic --wandb-entity <your-team>

Or set environment variables before launching:

.. code-block:: bash

   export WANDB_PROJECT=mosaic
   export WANDB_ENTITY=<your-team>

**XuanCe** (via YAML or GUI extras):

.. code-block:: yaml

   logger: wandb
   wandb_project: mosaic
   wandb_entity: <your-team>

**Ray RLlib** (via ``extras``):

.. code-block:: json

   {
     "wandb_project": "mosaic",
     "wandb_entity": "<your-team>"
   }

Run Naming
~~~~~~~~~~

MOSAIC passes the run ULID as the W&B ``run_name`` so that W&B runs map
one-to-one to MOSAIC run IDs:

.. code-block:: python

   wandb.init(
       project=config.wandb_project,
       name=config.run_id,          # MOSAIC ULID
       config=config.to_dict(),
   )

This makes it straightforward to cross-reference a W&B dashboard entry with
the corresponding checkpoint in ``var/trainer/runs/<run_id>/``.

Offline Mode
~~~~~~~~~~~~

For machines without internet access, W&B can log locally and sync later:

.. code-block:: bash

   export WANDB_MODE=offline

   # After the run completes, sync to the cloud:
   wandb sync var/wandb/offline-run-*

Disabling W&B
~~~~~~~~~~~~~

To suppress all W&B output without removing the flag from the config:

.. code-block:: bash

   export WANDB_DISABLED=true

Or pass ``--no-track`` / set ``logger: tensorboard`` in the worker YAML.

Proxy Configuration
~~~~~~~~~~~~~~~~~~~

If your network requires a proxy, set the W&B proxy environment variables
before launching a training run:

.. code-block:: bash

   export WANDB_VPN_HTTPS_PROXY=https://<proxy-host>:<port>
   export WANDB_VPN_HTTP_PROXY=http://<proxy-host>:<port>

See Also
--------

- :doc:`/documents/runtime_logging/structured_logging`: MOSAIC's internal
  structured log pipeline (``LogConstant``, filters, rotating file handlers).
- :doc:`/documents/rendering_tabs/fastlane`: real-time frame streaming from
  worker processes to the GUI, separate from metric logging.
- :doc:`/documents/runtime_logging/constants`: numeric defaults that govern
  queue sizes and backpressure thresholds in the rendering subsystem.
