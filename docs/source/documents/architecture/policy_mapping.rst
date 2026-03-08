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

Link Groups for Multi-Agent RL
-------------------------------

When deploying RL policies in multi-agent scenarios, MOSAIC supports flexible
policy-to-agent mappings through **link groups**. This is essential because
MAPPO/IPPO checkpoints store all agents' policies in a single file.

Policy Mapping Modes
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mapping Mode
     - Description
   * - **One-to-one**
     - Each agent has its own independent policy checkpoint (default behavior).
       Agents are configured individually with separate policy paths.
   * - **One-to-many**
     - Multiple agents share a single policy checkpoint via link groups.
       The primary agent's policy path is automatically synced to all linked agents.

Why Link Groups Are Essential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-agent RL algorithms like MAPPO and IPPO store all agents' policies in a
single checkpoint file with agent-specific keys:

.. code-block:: python

   # Example checkpoint structure
   checkpoint = {
       "actor.agent_0.model.0.weight": tensor(...),
       "actor.agent_1.model.0.weight": tensor(...),
       "actor.agent_2.model.0.weight": tensor(...),
       "actor.agent_3.model.0.weight": tensor(...),
   }

Without link groups, users must manually enter the same checkpoint path for
each agent, which is error-prone and tedious. Link groups solve this by:

- **Preventing manual copy-paste errors**: One path update propagates to all linked agents
- **Ensuring consistency**: All agents in a group always use the same checkpoint
- **Enabling complex configurations**: Multiple independent groups with different policies
- **Supporting cross-team linking**: Agents from different teams can share policies

LinkGroup Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Link groups are configured via the ``LinkGroup`` dataclass:

.. code-block:: python

   @dataclass
   class LinkGroup:
       group_id: str              # Unique identifier (ULID)
       primary_agent: str         # Primary agent in the group
       linked_agents: list[str]   # Other agents in the group
       policy_path: str           # Group's policy checkpoint
       algorithm: str             # Group's algorithm (IPPO/MAPPO)
       worker_type: str           # Worker type (rl)
       color: str                 # Visual indicator color

Creating Link Groups
~~~~~~~~~~~~~~~~~~~~

Link groups are created manually in the GUI via the "Link Agents" button:

1. Configure the primary agent with policy path and algorithm
2. Click "Link Agents" button on the primary agent's row
3. Select agents to link in the dialog
4. Click OK to create the group

The primary agent's policy path is automatically synced to all linked agents.

Example Configurations
~~~~~~~~~~~~~~~~~~~~~~

**Single group (all agents trained together):**

.. code-block:: python

   # All 4 agents share the same MAPPO checkpoint
   LinkGroup(
       group_id="operator_0_link_0",
       primary_agent="agent_0",
       linked_agents=["agent_1", "agent_2", "agent_3"],
       policy_path="/path/to/checkpoint/final_train_model.pth",
       algorithm="mappo",
       worker_type="rl",
   )

**Multiple independent groups (two teams):**

.. code-block:: python

   # Offense team (agents 0 and 2)
   LinkGroup(
       group_id="operator_0_link_0",
       primary_agent="agent_0",
       linked_agents=["agent_2"],
       policy_path="/path/to/offense_mappo.pth",
       algorithm="mappo",
       worker_type="rl",
   )

   # Defense team (agents 1 and 3)
   LinkGroup(
       group_id="operator_0_link_1",
       primary_agent="agent_1",
       linked_agents=["agent_3"],
       policy_path="/path/to/defense_mappo.pth",
       algorithm="mappo",
       worker_type="rl",
   )

**Mixed evaluation (trained vs random):**

.. code-block:: python

   # Only agents 0 and 1 are linked
   LinkGroup(
       group_id="operator_0_link_0",
       primary_agent="agent_0",
       linked_agents=["agent_1"],
       policy_path="/path/to/checkpoint/final_train_model.pth",
       algorithm="ippo",
       worker_type="rl",
   )
   # agent_2 and agent_3 use Random workers (no link group)
