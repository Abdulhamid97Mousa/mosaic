PolicyMappingService
====================

The ``PolicyMappingService`` is the core abstraction for assigning
policies to agents in multi-agent environments.

Overview
--------

Unlike single-agent RL where one policy controls everything,
multi-agent environments require mapping each agent to its policy.
MOSAIC's ``PolicyMappingService`` handles this with paradigm awareness.

.. code-block:: python

   from gym_gui.services import PolicyMappingService

   service = PolicyMappingService()

   # Configure agents
   service.set_agents(["player_0", "player_1"])

   # Bind different policies
   service.bind_agent_policy("player_0", "human_keyboard")
   service.bind_agent_policy("player_1", "cleanrl_ppo", config={"model_path": "..."})

AgentPolicyBinding
------------------

Each binding stores the relationship between an agent and its policy:

.. code-block:: python

   @dataclass
   class AgentPolicyBinding:
       agent_id: str
       policy_id: str
       worker_id: Optional[str] = None
       config: Dict[str, Any] = field(default_factory=dict)

Action Selection
----------------

The service supports both sequential and simultaneous modes:

**Sequential (AEC)**

.. code-block:: python

   # Get action for current agent
   action = service.select_action(agent_id, snapshot)

**Simultaneous (POSG)**

.. code-block:: python

   # Get actions for all agents at once
   actions = service.select_actions(observations, snapshots)

Step Notification
-----------------

Notify policies of step results for learning:

.. code-block:: python

   # Per-agent notification
   service.notify_step(agent_id, snapshot)

   # Episode end notification
   service.notify_episode_end(agent_id, summary)

Integration with SessionController
----------------------------------

The ``SessionController`` uses ``PolicyMappingService`` for the game loop:

.. code-block:: python

   def _select_agent_action(self) -> Optional[int]:
       agent_id = self._get_active_agent()

       if self._policy_mapping is not None and agent_id:
           return self._policy_mapping.select_action(agent_id, snapshot)

       # Fallback to legacy ActorService
       return self._actor_service.select_action(snapshot)
