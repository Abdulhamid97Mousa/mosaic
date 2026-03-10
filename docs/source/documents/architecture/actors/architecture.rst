Actor Architecture
==================

This page describes the internal structure of the Actor subsystem and how it
integrates with the rest of MOSAIC at evaluation time.

High-Level Position in MOSAIC
------------------------------

Actors sit at the boundary between the GUI evaluation loop and agent decision
logic.  The diagram below shows the four components involved:

.. mermaid::

   graph LR
       SC["SessionController"] --> AS["ActorService"]
       AS --> A["Active Actor"]
       PM["PolicyMappingService"] --> AS
       A -.-> W["Worker Subprocess"]

       style SC fill:#d6eaf8,stroke:#2874a6
       style AS fill:#d6eaf8,stroke:#2874a6
       style PM fill:#d6eaf8,stroke:#2874a6
       style A fill:#eafaf1,stroke:#1e8449
       style W fill:#fef9e7,stroke:#d4ac0d

The dashed arrow from Actor to Worker Subprocess means the actor is a
**placeholder**: the worker subprocess manages its own loop, and the actor
in the GUI simply reports which backend is active.

ActorService Internals
-----------------------

``ActorService`` maintains three internal maps:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Map
     - Purpose
   * - ``_actors``
     - Maps ``actor_id`` to ``Actor`` instance
   * - ``_descriptors``
     - Maps ``actor_id`` to ``ActorDescriptor`` (display name, policy label, backend label for the UI widget)
   * - ``_active_actor_id``
     - The one actor that receives ``select_action`` calls

.. mermaid::

   graph TD
       REG["register_actor()"] --> DB[("_actors + _descriptors")]
       ACT["set_active_actor(id)"] --> DB
       DB --> SEL["select_action(snapshot)"]
       DB --> STEP["notify_step(snapshot)"]
       DB --> END["notify_episode_end(summary)"]
       SEED["seed(n)"] --> DB

       style DB fill:#eaf4fb,stroke:#2874a6

Key design decisions:

- **Only one actor is active at a time.**  Multiple actors can be registered
  (one per training backend), but only the active one receives ``select_action``
  calls.  Switching actors does not restart the session.
- **Seeding is broadcast to all actors.**  ``ActorService.seed(n)`` iterates
  over every registered actor and calls their optional ``seed`` method.  This
  ensures all actors have deterministic state when a new episode begins.
- **Descriptors are UI-only.**  ``ActorDescriptor`` carries display metadata
  for the Active Actor widget and has no effect on action selection.

Policy Mapping Integration
---------------------------

In multi-agent environments, ``PolicyMappingService`` maps each ``agent_id``
to an ``actor_id``.  Before calling ``select_action``, ``SessionController``
uses this mapping to activate the correct actor for the current agent:

.. mermaid::

   sequenceDiagram
       participant Env as Environment
       participant SC as SessionController
       participant PM as PolicyMappingService
       participant AS as ActorService
       participant A as Active Actor

       Env->>SC: obs, reward, done, agent_id
       SC->>PM: get_actor_id(agent_id)
       PM-->>SC: actor_id
       SC->>AS: set_active_actor(actor_id)
       SC->>AS: select_action(StepSnapshot)
       AS->>A: select_action(snapshot)
       A-->>AS: action
       AS-->>SC: action
       SC->>Env: env.step(action)

Placeholder Actors vs Real Actors
-----------------------------------

MOSAIC has two categories of actor:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Actor
     - Category
     - How action is produced
   * - ``HumanKeyboardActor``
     - Real actor
     - Reads the pending key press set by ``HumanInputController`` and returns it as an ``int``
   * - ``CleanRLWorkerActor``
     - Placeholder
     - Always returns ``None``; the CleanRL subprocess manages its own ``env.step``
   * - ``XuanCeWorkerActor``
     - Placeholder
     - Always returns ``None``; the XuanCe subprocess manages its own training loop
   * - ``RayRLlibWorkerActor``
     - Placeholder
     - Always returns ``None``; the Ray cluster manages its own distributed loop

When a placeholder actor is active, the GUI evaluation loop receives ``None``
from ``select_action`` and treats it as a no-op.  The visual widget still shows
which backend is running, but the GUI does not drive the environment steps.

