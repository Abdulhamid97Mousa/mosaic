Common Errors
=============

This page lists frequently encountered errors when working with the
CleanRL worker, along with their causes and fixes.

.. contents:: On this page
   :local:
   :depth: 1

ModuleNotFoundError: No module named 'tyro'
--------------------------------------------

.. code-block:: text

   ModuleNotFoundError: No module named 'tyro'

**Cause:** The ``cleanrl`` optional dependencies were not installed.
``tyro`` is the CLI argument parser used by upstream CleanRL scripts and
is included in the ``cleanrl`` extras group.

**Fix:**

.. code-block:: bash

   pip install -e ".[cleanrl]"

This installs ``tyro``, ``torch``, ``tensorboard``, ``wandb``,
``tenacity``, and ``moviepy`` in one step.

ModuleNotFoundError: No module named 'cleanrl'
-----------------------------------------------

.. code-block:: text

   ModuleNotFoundError: No module named 'cleanrl'

**Cause:** The upstream CleanRL package is not installed.  MOSAIC's
``cleanrl`` extras group installs the *worker shim* dependencies but
not the CleanRL library itself (it is expected to be available in the
environment).

**Fix:** Install CleanRL:

.. code-block:: bash

   pip install cleanrl

If you are using algorithms that require CleanRL's Atari or EnvPool
extras, install those as well:

.. code-block:: bash

   pip install "cleanrl[atari]"
   pip install "cleanrl[envpool]"

TensorBoard -- "No module named 'pkg_resources'"
--------------------------------------------------

.. code-block:: text

   ModuleNotFoundError: No module named 'pkg_resources'

**Cause:** ``setuptools`` version 78+ removed the bundled ``pkg_resources``
package. TensorBoard imports ``pkg_resources`` at startup, so it fails
when ``setuptools>=78`` is installed.

**Fix:**

.. code-block:: bash

   pip install "setuptools<78"

This constraint is included in ``requirements/base.txt`` and
``requirements/cleanrl_worker.txt``.

CUDA / GPU Errors
-----------------

**"CUDA out of memory"**

.. code-block:: text

   torch.cuda.OutOfMemoryError: CUDA out of memory.

**Cause:** The training run requires more GPU memory than is available.
This commonly happens with large batch sizes, many parallel environments,
or Atari/image-based observations.

**Fixes:**

- Reduce ``num_envs`` in the algorithm parameters.
- Reduce ``num_steps`` (rollout buffer length).
- Disable CUDA and train on CPU (uncheck the GPU toggle in the GUI,
  or set ``"cuda": false`` in the config extras).
- Close other GPU-consuming processes.

**"CUDA not available"**

.. code-block:: text

   RuntimeError: Attempting to use CUDA, but torch.cuda.is_available() is False

**Cause:** PyTorch was installed without CUDA support, or the CUDA
toolkit / GPU drivers are missing.

**Fixes:**

- Install the CUDA-enabled PyTorch build:
  ``pip install torch --index-url https://download.pytorch.org/whl/cu121``
- Verify with: ``python -c "import torch; print(torch.cuda.is_available())"``
- Alternatively, disable CUDA in the run config and train on CPU.

FastLane Telemetry Issues
-------------------------

**No frames appearing in the GUI**

**Possible causes:**

1. ``fastlane_video_mode`` is set to ``"off"``.  Change it to
   ``"single"`` or ``"grid"`` in the training form.
2. The environment does not support ``render(mode="rgb_array")``.
   FastLane calls ``env.render()`` on every step to capture frames.
3. ``GYM_GUI_FASTLANE_ONLY`` is not set.  The ``sitecustomize.py``
   patch checks this environment variable before wrapping environments.
4. The shared-memory segment is not accessible.  Ensure the GUI and the
   worker subprocess are running under the same user.

**Frames are too slow / choppy**

- Increase ``CLEANRL_FASTLANE_INTERVAL_MS`` to reduce frame rate and
  lower overhead (e.g. set to ``100`` for ~10 FPS).
- Set ``CLEANRL_FASTLANE_MAX_DIM`` to downscale large frames before
  publishing (e.g. ``128``).
- Switch from ``grid`` to ``single`` video mode to reduce the number
  of environments rendering simultaneously.

Curriculum Training Errors
--------------------------

**"No module named 'syllabus'"**

.. code-block:: text

   ModuleNotFoundError: No module named 'syllabus'

**Cause:** Syllabus-RL is not installed.  It is required only for
curriculum training mode and is vendored as a Git submodule.

**Fix:**

.. code-block:: bash

   git submodule update --init 3rd_party/Syllabus
   pip install -e 3rd_party/Syllabus

**"Task space mismatch" or unexpected environment switching**

**Cause:** The ``curriculum_schedule`` contains environment IDs that
are not installed or have incompatible observation/action spaces.
All environments in a curriculum must share the same observation and
action space shapes.

**Fix:** Ensure every ``env_id`` in the schedule is installed and
that all environments produce observations of the same shape.
For MiniGrid/BabyAI curricula, all environments use the standard 7x7x3
observation space by default.

Weights & Biases (W&B) Errors
------------------------------

**"wandb.errors.UsageError: api_key not configured"**

.. code-block:: text

   wandb.errors.UsageError: api_key not configured

**Cause:** W&B tracking is enabled but no API key is available.

**Fixes:**

- Set ``WANDB_API_KEY`` in your ``.env`` file.
- Or enter the API key in the training form's W&B section.
- Or run ``wandb login`` in the terminal before launching training.

**W&B upload failures behind a proxy / VPN**

.. code-block:: text

   requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.wandb.ai', ...)

**Cause:** Network requests to ``api.wandb.ai`` are blocked by a
corporate proxy or VPN.

**Fixes:**

- Enable the VPN proxy option in the training form and configure the
  proxy URL (e.g. ``https://127.0.0.1:7890``).
- Or set the proxy variables in ``.env``:

  .. code-block:: bash

     WANDB_VPN_HTTPS_PROXY=https://127.0.0.1:7890
     WANDB_VPN_HTTP_PROXY=http://127.0.0.1:7890

- Or disable W&B tracking entirely and rely on TensorBoard for metrics.

gRPC Handshake Failures
------------------------

**"grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE"**

.. code-block:: text

   grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
       status = StatusCode.UNAVAILABLE
       details = "failed to connect to all addresses"
   >

**Cause:** The worker subprocess could not reach the Trainer Daemon's
gRPC server (default ``127.0.0.1:50055``).

**Possible causes and fixes:**

- The Trainer Daemon is not running.  Start the MOSAIC GUI, which
  launches the daemon automatically.
- The gRPC port is blocked by a firewall.  Ensure port 50055 is open
  for localhost traffic.
- A different ``grpc_target`` was specified.  Verify the target matches
  the daemon's listening address.
- Set ``GRPC_VERBOSITY=debug`` in ``.env`` for detailed connection
  logs.

**"gRPC handshake timeout"**

**Cause:** The daemon is running but too slow to respond (e.g. under
heavy load with many concurrent runs).

**Fixes:**

- Retry the run -- transient timeouts often resolve on the next
  attempt.
- Reduce the number of concurrent training runs.
- Check system resources (CPU, memory) to ensure the daemon is not
  starved.

Environment Import Errors
--------------------------

**"No module named 'minigrid'" / "No module named 'ale_py'"**

.. code-block:: text

   ModuleNotFoundError: No module named 'minigrid'

**Cause:** The environment-specific extras are not installed.

**Fix:** Install the appropriate extras for your target environment:

.. code-block:: bash

   pip install -e ".[minigrid]"    # MiniGrid / BabyAI
   pip install -e ".[atari]"      # Atari (ALE)
   pip install -e ".[mujoco]"     # MuJoCo
   pip install -e ".[procgen]"    # Procgen
