Actor Concept
=============

What is an Actor?
-----------------

In MOSAIC, the term **Actor** refers to any object that can produce an action
given the current environment state.  Actors operate entirely inside the GUI
process and are invoked synchronously on every environment step.

Actors are intentionally **lightweight**.  They do not own environments, do not
spawn subprocesses, and do not perform gradient updates.  Their single
responsibility is: *given a snapshot of the world, return an action*.

This separates the concerns of **training** (handled by Workers) from
**inference** (handled by Actors):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Concern
     - Handled by
     - Lives in
   * - Policy training
     - Worker subprocess (CleanRL, Ray, XuanCe)
     - Isolated subprocess via gRPC or IPC
   * - Action selection
     - Actor (``CleanRLWorkerActor``, ``HumanKeyboardActor``)
     - GUI main process
   * - Actor coordination
     - ``ActorService``
     - GUI main process

The Two Actor Protocols
-----------------------

MOSAIC defines two complementary protocols in ``gym_gui/services/actor.py``:

``Actor`` - Simple, Single-Agent Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Actor`` protocol is the original, lightweight interface designed for
single-agent environments.  Any class that implements these three methods is
a valid ``Actor``:

.. code-block:: python

   class Actor(Protocol):
       id: str

       def select_action(self, step: StepSnapshot) -> Optional[int]: ...
       def on_step(self, step: StepSnapshot) -> None: ...
       def on_episode_end(self, summary: EpisodeSummary) -> None: ...

``PolicyController`` - Paradigm-Aware, Multi-Agent Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``PolicyController`` protocol extends the actor concept for multi-agent
environments.  It adds:

- **Agent-specific action selection** via ``select_action(agent_id, obs, info)``
- **Batch action selection** via ``select_actions(observations)`` for simultaneous
  (POSG) paradigms where all agents act at the same time
- **Paradigm declaration** via the ``paradigm`` property, which signals whether
  the controller targets AEC (turn-based) or POSG (simultaneous) environments
- **Per-agent lifecycle hooks** via ``on_step_result`` and ``on_episode_end``,
  which carry the ``agent_id`` alongside the usual feedback

Data Containers
---------------

Two frozen dataclasses carry data between the environment loop and actors:

``StepSnapshot``
~~~~~~~~~~~~~~~~

Passed to ``select_action`` and ``on_step`` on every environment step:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``step_index``
     - Zero-based step counter within the current episode
   * - ``observation``
     - Raw observation returned by the environment
   * - ``reward``
     - Reward received on the previous step (``0.0`` on the first step)
   * - ``terminated``
     - ``True`` if the episode ended naturally (goal reached, game over)
   * - ``truncated``
     - ``True`` if the episode was cut short (time limit)
   * - ``seed``
     - Optional seed used to reset this episode
   * - ``info``
     - Auxiliary environment info dict

``EpisodeSummary``
~~~~~~~~~~~~~~~~~~

Delivered via ``on_episode_end`` at the end of every episode:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``episode_index``
     - Zero-based episode counter for the current session
   * - ``total_reward``
     - Sum of all rewards across the episode
   * - ``steps``
     - Number of steps taken in the episode
   * - ``metadata``
     - Arbitrary key-value pairs (worker-specific diagnostics)

When to use Actors vs Workers
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - You want to
     - Use
     - Why
   * - Train a new policy from scratch
     - **Worker** (CleanRL, Ray, XuanCe)
     - Workers own the training loop, checkpointing, and telemetry
   * - Evaluate a trained policy in the GUI
     - **Actor** (loads checkpoint, selects actions)
     - Actors are lightweight and run inside the GUI process
   * - Let a human play an environment
     - ``HumanKeyboardActor``
     - Forwards keyboard input captured by ``HumanInputController``
   * - Track which backend is currently active
     - ``CleanRLWorkerActor`` or ``RayRLlibWorkerActor``
     - Placeholder actors represent active training backends in the UI
