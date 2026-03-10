Actor Lifecycle
===============

This page traces the complete lifecycle of an Actor from registration through
episode end.

Overview
--------

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TD
       A["1. Register Actor<br/>ActorService.register_actor()"]
       B["2. Activate Actor<br/>set_active_actor(id)"]
       C["3. Seed Broadcast<br/>ActorService.seed(n)"]
       D["4. Episode Begin<br/>(env.reset called externally)"]
       E["5. Step Loop<br/>select_action → env.step → on_step"]
       F{"Episode done?"}
       G["6. Episode End<br/>notify_episode_end(summary)"]
       H{"Session over?"}
       I["7. Actor remains registered<br/>(available for next episode)"]

       A --> B --> C --> D --> E --> F
       F -->|"No"| E
       F -->|"Yes"| G --> H
       H -->|"No: next episode"| C
       H -->|"Yes"| I

       style A fill:#d6eaf8,stroke:#2874a6
       style B fill:#d6eaf8,stroke:#2874a6
       style C fill:#d5f5e3,stroke:#1e8449
       style D fill:#fef9e7,stroke:#d4ac0d
       style E fill:#fef9e7,stroke:#d4ac0d
       style G fill:#f9ebea,stroke:#c0392b
       style I fill:#eaecee,stroke:#717d7e

Phase 1: Registration
-----------------------

Actors are registered with ``ActorService.register_actor()`` at session
startup.  Multiple actors can be registered in a single session:

.. code-block:: python

   actor_service = ActorService()

   actor_service.register_actor(
       HumanKeyboardActor(),
       display_name="Human",
       description="Keyboard-controlled agent",
   )
   actor_service.register_actor(
       CleanRLWorkerActor(),
       display_name="CleanRL PPO",
       backend_label="cleanrl_worker",
       activate=False,  # don't immediately switch to this actor
   )

The first actor registered automatically becomes the active actor unless
``activate=False`` is passed.

Phase 2: Activation
---------------------

Only one actor is active at a time.  The GUI switches the active actor via
``set_active_actor(actor_id)`` in response to user interaction (e.g.
selecting a different backend in the Active Actor widget) or programmatically
when ``PolicyMappingService`` routes a specific agent to a different actor:

.. code-block:: python

   # Switch to human control
   actor_service.set_active_actor("human_keyboard")

   # Switch back to autonomous backend
   actor_service.set_active_actor("cleanrl_worker")

Phase 3: Seed Broadcast
-------------------------

Before every episode, ``ActorService.seed(n)`` broadcasts the episode seed to
**all registered actors** (not just the active one).  This ensures every actor
has a deterministic RNG state, even if it is switched in mid-session:

.. code-block:: python

   actor_service.seed(42)  # all actors receive seed(42)

If an actor does not implement a ``seed`` method the call is silently skipped.
Errors during seeding are caught and logged with ``LOG_SERVICE_ACTOR_SEED_ERROR``
rather than crashing the evaluation loop.

Phase 4: Episode Begin
------------------------

Episode setup (``env.reset``) is handled by ``SessionController``, not by the
actor directly.  Actors are stateless with respect to environment reset; they
simply start receiving ``StepSnapshot`` objects once the environment is ready.

Phase 5: Step Loop
--------------------

On each step the evaluation loop:

1. Builds a ``StepSnapshot`` from the environment response.
2. Calls ``ActorService.select_action(snapshot)`` → dispatched to the active actor.
3. Applies the returned action via ``env.step(action)``.
4. Calls ``ActorService.notify_step(snapshot)`` → dispatched to the active actor.

.. mermaid::

   %%{init: {"sequenceDiagram": {"mirrorActors": false}} }%%
   sequenceDiagram
       participant SC as SessionController
       participant AS as ActorService
       participant A as Active Actor
       participant Env as Environment

       SC->>AS: select_action(StepSnapshot)
       AS->>A: actor.select_action(snapshot)
       A-->>AS: action (int or None)
       AS-->>SC: action
       SC->>Env: env.step(action)
       Env-->>SC: obs, reward, terminated, truncated, info
       SC->>AS: notify_step(StepSnapshot)
       AS->>A: actor.on_step(snapshot)

If ``select_action`` returns ``None`` the evaluation loop treats this as a
no-op (typically NOOP action ``0`` or environment-specific default).

Phase 6: Episode End
----------------------

When ``terminated or truncated`` is ``True``, ``SessionController`` builds an
``EpisodeSummary`` and calls ``ActorService.notify_episode_end(summary)``,
which dispatches to the active actor's ``on_episode_end`` hook:

.. code-block:: python

   summary = EpisodeSummary(
       episode_index=3,
       total_reward=18.5,
       steps=200,
       metadata={"truncated": True},
   )
   actor_service.notify_episode_end(summary)

Phase 7: Session End
----------------------

Actors are not explicitly destroyed at session end.  The ``ActorService``
instance is garbage-collected along with the session.  Actors that hold
external resources (e.g. a loaded PyTorch checkpoint) should implement
``__del__`` or a ``close()`` method if cleanup is required.

Error Handling
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Failure
     - Behaviour
   * - ``select_action`` raises
     - Exception propagates to ``SessionController`` → episode aborted
   * - ``on_step`` raises
     - Exception propagates to ``SessionController`` → episode aborted
   * - ``seed(n)`` raises
     - Exception is caught; logged with ``LOG_SERVICE_ACTOR_SEED_ERROR``; seed broadcast continues to remaining actors
   * - ``set_active_actor`` with unknown id
     - ``KeyError`` raised immediately; this fails fast to surface misconfiguration
