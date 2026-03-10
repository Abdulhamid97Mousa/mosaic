Actor Types
===========

MOSAIC ships four built-in actor implementations.  All four are defined in
``gym_gui/services/actor.py`` and implement the ``Actor`` protocol.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       AP["Actor Protocol<br/>(id, select_action, on_step, on_episode_end)"]

       AP --> HKA["HumanKeyboardActor"]
       AP --> CRA["CleanRLWorkerActor"]
       AP --> XCA["XuanCeWorkerActor"]
       AP --> RRA["RayRLlibWorkerActor"]

       classDef real fill:#eafaf1,stroke:#1e8449
       classDef placeholder fill:#fef9e7,stroke:#d4ac0d
       classDef proto fill:#d6eaf8,stroke:#2874a6

       class AP proto
       class HKA real
       class CRA,XCA,RRA placeholder

Green nodes are **real actors** (meaningful action logic).
Yellow nodes are **placeholder actors** (always return ``None``).


HumanKeyboardActor
------------------

.. list-table::
   :stub-columns: 1
   :widths: 20 80

   * - Actor ID
     - ``"human_keyboard"``
   * - Type
     - Real actor
   * - Action source
     - Keyboard / mouse input captured by ``HumanInputController``
   * - Paradigms
     - Single-agent, AEC turn-based, POSG simultaneous
   * - Seeding
     - No RNG; seed has no effect

``HumanKeyboardActor`` bridges human input and the environment step loop.
``HumanInputController`` buffers the latest key press and sets a pending action;
``select_action`` reads that pending action and clears the buffer.

.. code-block:: python

   actor = HumanKeyboardActor(id="human_keyboard")
   actor_service.register_actor(actor, display_name="Human Player")

Interaction flow:

.. mermaid::

   %%{init: {"sequenceDiagram": {"mirrorActors": false}} }%%
   sequenceDiagram
       participant User as User (keyboard)
       participant HIC as HumanInputController
       participant HKA as HumanKeyboardActor
       participant AS as ActorService

       User->>HIC: key press event
       HIC->>HKA: set pending action (e.g. action=3)
       AS->>HKA: select_action(StepSnapshot)
       HKA-->>AS: 3 (pending action cleared)


CleanRLWorkerActor
------------------

.. list-table::
   :stub-columns: 1
   :widths: 20 80

   * - Actor ID
     - ``"cleanrl_worker"``
   * - Type
     - Placeholder actor
   * - Action source
     - CleanRL Worker subprocess (manages its own ``env.step``)
   * - Paradigms
     - Single-agent (PPO, DQN, SAC, TD3, C51, Rainbow)
   * - Seeding
     - No RNG; seed has no effect on this actor

When CleanRL is active, the Worker subprocess runs its own training loop and
steps its own copy of the environment.  The GUI's evaluation loop is paused.
``CleanRLWorkerActor`` exists solely so the Active Actor widget can display
``"CleanRL Worker"`` while training runs in the background.

``select_action`` always returns ``None``.


XuanCeWorkerActor
-----------------

.. list-table::
   :stub-columns: 1
   :widths: 20 80

   * - Actor ID
     - ``"xuance_worker"``
   * - Type
     - Placeholder actor
   * - Action source
     - XuanCe Worker subprocess
   * - Paradigms
     - Multi-agent (MAPPO, QMIX, MADDPG, VDN, COMA)
   * - Seeding
     - No RNG; seed has no effect on this actor

XuanCe is a multi-agent RL library.  Like CleanRL, it owns its training loop
in a separate subprocess.  The actor is a placeholder for the GUI widget.

``select_action`` always returns ``None``.


RayRLlibWorkerActor
--------------------

.. list-table::
   :stub-columns: 1
   :widths: 20 80

   * - Actor ID
     - ``"ray_worker"``
   * - Type
     - Placeholder actor
   * - Action source
     - Ray RLlib Worker subprocess (distributed training)
   * - Paradigms
     - Single-agent and multi-agent (PPO, IMPALA, APPO)
   * - Seeding
     - No RNG; seed has no effect on this actor

Ray RLlib uses distributed rollout workers; the main training loop runs inside
the Ray cluster coordinated by the Ray Worker subprocess.  This actor is a
placeholder for the GUI widget only.

``select_action`` always returns ``None``.


Implementing a Custom Actor
----------------------------

Any object that satisfies the ``Actor`` protocol can be registered with
``ActorService``.  A minimal implementation:

.. code-block:: python

   from gym_gui.services.actor import Actor, StepSnapshot, EpisodeSummary
   import random

   class RandomActor:
       """Selects a random action from a discrete action space."""

       def __init__(self, actor_id: str, n_actions: int, seed: int = 0) -> None:
           self.id = actor_id
           self._rng = random.Random(seed)
           self._n = n_actions

       def select_action(self, step: StepSnapshot) -> int:
           return self._rng.randrange(self._n)

       def on_step(self, step: StepSnapshot) -> None:
           pass  # nothing to update

       def on_episode_end(self, summary: EpisodeSummary) -> None:
           pass  # nothing to reset

       def seed(self, value: int) -> None:
           self._rng.seed(value)  # called by ActorService.seed()

   # Registration
   actor_service.register_actor(
       RandomActor("random_4actions", n_actions=4, seed=42),
       display_name="Random Agent",
       description="Uniform random action selection",
   )

.. note::

   The ``seed`` method is **optional** in the protocol but strongly recommended.
   ``ActorService.seed()`` silently skips actors that do not implement it, but
   without seeding the actor's RNG will not be reproducible across episodes.
