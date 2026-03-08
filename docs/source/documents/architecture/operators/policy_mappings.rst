Policy Mappings for Heterogeneous Multi-Agent Systems
======================================================

The Challenge of Heterogeneous Multi-Agent Decision-Making
-----------------------------------------------------------

Heterogeneous multi-agent systems present a unique configuration challenge
that does not exist in homogeneous setups. When mixing different decision-making
paradigms (RL, LLM, Human, Random) in the same environment, each agent must be
configured **independently** while maintaining the ability to share resources
where appropriate.

The Core Problem
~~~~~~~~~~~~~~~~

Multi-agent RL algorithms like MAPPO and IPPO store all agents' policies in a
**single checkpoint file** with agent-specific keys:

.. code-block:: python

   # Example: MAPPO checkpoint structure
   checkpoint = {
       "actor.agent_0.model.0.weight": tensor(...),
       "actor.agent_1.model.0.weight": tensor(...),
       "actor.agent_2.model.0.weight": tensor(...),
       "actor.agent_3.model.0.weight": tensor(...),
   }

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       subgraph CHECKPOINT["MAPPO Checkpoint File (mappo_1v1.pth)"]
           A0_W["actor.agent_0.weights"]
           A1_W["actor.agent_1.weights"]
           A2_W["actor.agent_2.weights"]
           A3_W["actor.agent_3.weights"]
       end

       subgraph PROBLEM["The Configuration Problem"]
           direction TB
           MANUAL["Manual Copy-Paste Required"]
           ERROR["Typo in path → Runtime failure"]
           UPDATE["Update checkpoint → Update 4 entries"]
           NOVIS["No visual indication of sharing"]
       end

       CHECKPOINT -.->|"Same file for all agents"| PROBLEM

       style CHECKPOINT fill:#ff7f50,stroke:#cc5500,color:#fff
       style PROBLEM fill:#ffcccc,stroke:#cc0000,color:#333
       style A0_W fill:#fff,stroke:#999,color:#333
       style A1_W fill:#fff,stroke:#999,color:#333
       style A2_W fill:#fff,stroke:#999,color:#333
       style A3_W fill:#fff,stroke:#999,color:#333
       style MANUAL fill:#ffeeee,stroke:#cc0000,color:#333
       style ERROR fill:#ffeeee,stroke:#cc0000,color:#333
       style UPDATE fill:#ffeeee,stroke:#cc0000,color:#333
       style NOVIS fill:#ffeeee,stroke:#cc0000,color:#333

This creates a configuration dilemma:

**Without flexible policy mappings:**
   Users must manually enter the same checkpoint path for each RL agent,
   which is error-prone, tedious, and breaks when mixing paradigms.

**Example of the problem:**

.. code-block:: python

   # Heterogeneous team: RL + LLM + Random
   player_workers={
       "agent_0": WorkerAssignment(
           worker_id="xuance_worker",
           worker_type="rl",
           settings={"policy_path": "/path/to/checkpoint.pth"},  # Manual entry
       ),
       "agent_1": WorkerAssignment(
           worker_id="mosaic_llm_worker",
           worker_type="llm",
           settings={"model_id": "gpt-4o"},  # Different config
       ),
       "agent_2": WorkerAssignment(
           worker_id="xuance_worker",
           worker_type="rl",
           settings={"policy_path": "/path/to/checkpoint.pth"},  # Manual copy-paste!
       ),
       "agent_3": WorkerAssignment(
           worker_id="random_worker",
           worker_type="random",  # No checkpoint needed
       ),
   }

**Problems with this approach:**

1. **Manual copy-paste errors**: Typos in checkpoint paths cause runtime failures
2. **Update fragility**: Changing the checkpoint requires updating multiple entries
3. **No visual indication**: Cannot tell which agents share policies
4. **Breaks heterogeneity**: Forces all agents to be configured identically or manually

Why One-to-One and One-to-Many Mappings Are Essential
------------------------------------------------------

MOSAIC solves this with **flexible policy mappings** that enable both
independent configuration (one-to-one) and resource sharing (one-to-many).

One-to-One Mapping (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each agent has its own independent policy checkpoint. This is the default
behavior and requires no special configuration.

**Use case:** Agents trained separately or using different algorithms

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       subgraph POLICIES["Policy Checkpoints"]
           direction LR
           PPO_FILE["ppo.pth<br/>(CleanRL PPO)"]
           DQN_FILE["dqn.pth<br/>(CleanRL DQN)"]
       end

       subgraph AGENTS["Agents (Independent)"]
           direction LR
           A0["agent_0<br/>cleanrl_worker<br/>PPO"]
           A1["agent_1<br/>cleanrl_worker<br/>DQN"]
       end

       PPO_FILE -->|"One-to-One<br/>Independent"| A0
       DQN_FILE -->|"One-to-One<br/>Independent"| A1

       style POLICIES fill:#fff3e0,stroke:#f57c00,color:#333
       style AGENTS fill:#f5f5f5,stroke:#999,color:#333
       style PPO_FILE fill:#50c878,stroke:#2e8b57,color:#fff
       style DQN_FILE fill:#4a90d9,stroke:#2e5a87,color:#fff
       style A0 fill:#a5d6a7,stroke:#2e8b57,color:#333
       style A1 fill:#90caf9,stroke:#1976d2,color:#333

.. code-block:: python

   # Each agent has its own policy
   player_workers={
       "agent_0": WorkerAssignment(
           worker_id="cleanrl_worker",
           worker_type="rl",
           settings={"policy_path": "/path/to/ppo.pth"},
       ),
       "agent_1": WorkerAssignment(
           worker_id="cleanrl_worker",
           worker_type="rl",
           settings={"policy_path": "/path/to/dqn.pth"},
       ),
   }

One-to-Many Mapping (via Link Groups)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple agents share a single policy checkpoint. The primary agent's policy
path is automatically synced to all linked agents.

**Use case:** Agents trained together in the same MAPPO/IPPO run

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       CHECKPOINT["mappo_team.pth<br/>(Single Checkpoint)"]

       subgraph LINKGROUP["Link Group (operator_0_link_0)"]
           direction TB
           PRIMARY["agent_0 &nbsp;(Primary) &nbsp;xuance_worker"]

           subgraph LINKED[" "]
               direction LR
               LINKED1["agent_1<br/>(Linked)<br/>xuance_worker"]
               LINKED2["agent_2<br/>(Linked)<br/>xuance_worker"]
           end
       end

       CHECKPOINT -->|"Shared Policy"| PRIMARY
       PRIMARY -.->|"Auto-synced"| LINKED1
       PRIMARY -.->|"Auto-synced"| LINKED2

       style CHECKPOINT fill:#ff7f50,stroke:#cc5500,color:#fff
       style LINKGROUP fill:#f3e5f5,stroke:#9c27b0,color:#333
       style LINKED fill:none,stroke:none
       style PRIMARY fill:#9370db,stroke:#6a0dad,color:#fff
       style LINKED1 fill:#ba68c8,stroke:#8e24aa,color:#fff
       style LINKED2 fill:#ba68c8,stroke:#8e24aa,color:#fff

.. code-block:: python

   # All agents share the same checkpoint via link group
   player_workers={
       "agent_0": WorkerAssignment(
           worker_id="xuance_worker",
           worker_type="rl",
           settings={"policy_path": "/path/to/mappo_team.pth"},
       ),
       "agent_1": WorkerAssignment(
           worker_id="xuance_worker",
           worker_type="rl",
           settings={"policy_path": "/path/to/mappo_team.pth"},
       ),
       "agent_2": WorkerAssignment(
           worker_id="xuance_worker",
           worker_type="rl",
           settings={"policy_path": "/path/to/mappo_team.pth"},
       ),
   },
   link_groups={
       "operator_0_link_0": LinkGroup(
           group_id="operator_0_link_0",
           primary_agent="agent_0",
           linked_agents=["agent_1", "agent_2"],
           policy_path="/path/to/mappo_team.pth",
           algorithm="mappo",
           worker_type="rl",
       ),
   }

**Benefits:**

- **Automatic updates**: Change the primary agent's path, all linked agents update
- **Visual indication**: GUI shows which agents are linked
- **Error prevention**: No manual copy-paste required
- **Consistency guarantee**: All linked agents always use the same checkpoint

Enabling Heterogeneous Multi-Agent Decision-Making
---------------------------------------------------

The combination of one-to-one and one-to-many mappings is what makes
heterogeneous multi-agent systems **practical and configurable**.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       subgraph ENV["Soccer 2v2 Environment"]
           direction TB
           GAME["MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0"]
       end

       subgraph GREEN["Green Team (Heterogeneous)"]
           direction TB
           A0["agent_0<br/>RL (xuance_worker)<br/>MAPPO"]
           A1["agent_1<br/>LLM (mosaic_llm_worker)<br/>GPT-4o"]
       end

       subgraph BLUE["Blue Team (RL + Random)"]
           direction TB
           A2["agent_2<br/>RL (xuance_worker)<br/>MAPPO"]
           A3["agent_3<br/>Random (random_worker)"]
       end

       subgraph POLICIES["Policy Storage"]
           direction TB
           MAPPO["mappo_1v1.pth<br/>(Shared via Link Group)"]
           LLM_API["OpenRouter API<br/>(GPT-4o)"]
           RANDOM["No Policy<br/>(Random Actions)"]
       end

       MAPPO -->|"One-to-Many<br/>Link Group"| A0
       MAPPO -->|"One-to-Many<br/>Link Group"| A2
       LLM_API -->|"Independent<br/>Config"| A1
       RANDOM -->|"No Config<br/>Needed"| A3

       A0 --> GAME
       A1 --> GAME
       A2 --> GAME
       A3 --> GAME

       style ENV fill:#f5f5f5,stroke:#999,color:#333
       style GAME fill:#4a90d9,stroke:#2e5a87,color:#fff
       style GREEN fill:#e8f5e9,stroke:#2e8b57,color:#333
       style BLUE fill:#e3f2fd,stroke:#1976d2,color:#333
       style POLICIES fill:#fff3e0,stroke:#f57c00,color:#333
       style A0 fill:#9370db,stroke:#6a0dad,color:#fff
       style A1 fill:#50c878,stroke:#2e8b57,color:#fff
       style A2 fill:#9370db,stroke:#6a0dad,color:#fff
       style A3 fill:#ff7f50,stroke:#cc5500,color:#fff
       style MAPPO fill:#ba68c8,stroke:#8e24aa,color:#fff
       style LLM_API fill:#4db6ac,stroke:#00897b,color:#fff
       style RANDOM fill:#ffb74d,stroke:#f57c00,color:#fff

**Key insight:** Individual agent configuration + flexible policy sharing = heterogeneous teams

Complex Heterogeneous Example: 3vs3 Soccer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario:** 3vs3 soccer with mixed training paradigms

- **Team Green (3 agents):** 2 MAPPO agents (xuance_worker) + 1 PPO agent (cleanrl_worker trained on solo env)
- **Team Blue (3 agents):** 2 MAPPO agents (xuance_worker) + 1 Random agent

This demonstrates the full power of heterogeneous multi-agent configuration:

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       subgraph ENV["Soccer 3v3 Environment"]
           GAME["MosaicMultiGrid-Soccer-3vs3-IndAgObs-v0"]
       end

       subgraph GREEN["Green Team (Heterogeneous)"]
           direction TB
           G0["agent_0<br/>xuance_worker<br/>MAPPO (2v2)"]
           G1["agent_1<br/>xuance_worker<br/>MAPPO (2v2)"]
           G2["agent_2<br/>cleanrl_worker<br/>PPO (Solo)"]
       end

       subgraph BLUE["Blue Team (RL + Random)"]
           direction TB
           B0["agent_3<br/>xuance_worker<br/>MAPPO (2v2)"]
           B1["agent_4<br/>xuance_worker<br/>MAPPO (2v2)"]
           B2["agent_5<br/>random_worker"]
       end

       subgraph POLICIES["Policy Storage"]
           direction TB
           MAPPO["mappo_2v2.pth<br/>(4 agents share)"]
           PPO["ppo_solo.pth<br/>(1 agent)"]
           RANDOM["No Policy"]
       end

       MAPPO -->|"Link Group 0"| G0
       MAPPO -->|"Link Group 0"| G1
       MAPPO -->|"Link Group 1"| B0
       MAPPO -->|"Link Group 1"| B1
       PPO -->|"Independent"| G2
       RANDOM -->|"No Config"| B2

       G0 --> GAME
       G1 --> GAME
       G2 --> GAME
       B0 --> GAME
       B1 --> GAME
       B2 --> GAME

       style ENV fill:#f5f5f5,stroke:#999,color:#333
       style GAME fill:#4a90d9,stroke:#2e5a87,color:#fff
       style GREEN fill:#e8f5e9,stroke:#2e8b57,color:#333
       style BLUE fill:#e3f2fd,stroke:#1976d2,color:#333
       style POLICIES fill:#fff3e0,stroke:#f57c00,color:#333
       style G0 fill:#9370db,stroke:#6a0dad,color:#fff
       style G1 fill:#9370db,stroke:#6a0dad,color:#fff
       style G2 fill:#50c878,stroke:#2e8b57,color:#fff
       style B0 fill:#ba68c8,stroke:#8e24aa,color:#fff
       style B1 fill:#ba68c8,stroke:#8e24aa,color:#fff
       style B2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style MAPPO fill:#ba68c8,stroke:#8e24aa,color:#fff
       style PPO fill:#4db6ac,stroke:#00897b,color:#fff
       style RANDOM fill:#ffb74d,stroke:#f57c00,color:#fff

**Configuration breakdown:**

1. **4 agents share MAPPO checkpoint** (trained on 2v2):
   - Green team: agent_0, agent_1 (Link Group 0)
   - Blue team: agent_3, agent_4 (Link Group 1)
   - All use the same ``mappo_2v2.pth`` checkpoint

2. **1 agent uses independent PPO** (trained on solo environment):
   - Green team: agent_2 (cleanrl_worker)
   - Uses ``ppo_solo.pth`` trained on ``MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0``

3. **1 agent is random baseline**:
   - Blue team: agent_5 (random_worker)
   - No policy needed

**Why this configuration is interesting:**

- **Transfer learning test:** MAPPO agents trained on 2v2 now play 3v3
- **Cross-paradigm cooperation:** PPO agent (solo-trained) cooperates with MAPPO agents (team-trained)
- **Algorithm comparison:** MAPPO vs PPO vs Random in the same environment
- **Ad-hoc teamwork:** Agents trained separately must coordinate without joint training

The Key Insight
~~~~~~~~~~~~~~~

**Individual agent configuration + flexible policy sharing = heterogeneous teams**

Without flexible policy mappings, you are forced to choose:

- **Option A**: Configure each agent independently → manual copy-paste errors
- **Option B**: Force all agents to use the same worker type → no heterogeneity

With flexible policy mappings, you can:

- Configure each agent slot independently (RL, LLM, Human, Random)
- Share resources where appropriate (link groups for RL agents)
- Mix paradigms freely without configuration overhead

Real-World Example: Heterogeneous Soccer Team
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario:** 2v2 soccer with RL + LLM vs RL + Random

.. code-block:: python

   config = OperatorConfig.multi_agent(
       operator_id="heterogeneous_soccer",
       display_name="RL+LLM vs RL+Random",
       env_name="mosaic_multigrid",
       task="MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0",
       player_workers={
           # Green team: heterogeneous (RL + LLM)
           "agent_0": WorkerAssignment(
               worker_id="xuance_worker",
               worker_type="rl",
               settings={
                   "algorithm": "mappo",
                   "policy_path": "/path/to/mappo_1v1.pth",
               },
           ),
           "agent_1": WorkerAssignment(
               worker_id="mosaic_llm_worker",
               worker_type="llm",
               settings={
                   "model_id": "gpt-4o",
                   "coordination_level": 2,
               },
           ),
           # Blue team: RL + Random
           "agent_2": WorkerAssignment(
               worker_id="xuance_worker",
               worker_type="rl",
               settings={
                   "algorithm": "mappo",
                   "policy_path": "/path/to/mappo_1v1.pth",
               },
           ),
           "agent_3": WorkerAssignment(
               worker_id="random_worker",
               worker_type="random",
           ),
       },
       # Link groups: agents 0 and 2 share the same MAPPO checkpoint
       # IDs are scoped to the operator: {operator_id}_link_{N}
       link_groups={
           "heterogeneous_soccer_link_0": LinkGroup(
               group_id="heterogeneous_soccer_link_0",
               primary_agent="agent_0",
               linked_agents=["agent_2"],
               policy_path="/path/to/mappo_1v1.pth",
               algorithm="mappo",
               worker_type="rl",
           ),
       },
   )

**What this configuration achieves:**

1. **Heterogeneity**: Four different decision-making mechanisms in one environment
2. **Resource sharing**: RL agents (0 and 2) share the same MAPPO checkpoint
3. **Independent configuration**: LLM agent (1) has its own settings
4. **Baseline comparison**: Random agent (3) provides reference performance
5. **Maintainability**: Updating the MAPPO checkpoint updates both RL agents automatically

Why This Matters for Research
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heterogeneous multi-agent systems enable novel research questions:

**Cross-paradigm cooperation:**
   How well do RL and LLM agents cooperate as teammates?

**Ad-hoc teamwork:**
   Can an RL agent trained solo adapt to an LLM partner?

**Paradigm comparison:**
   Which paradigm performs better in multi-agent coordination?

**Ablation studies:**
   What is the marginal contribution of each paradigm to team performance?

**None of these questions can be answered without:**

1. The ability to mix paradigms in the same environment
2. Flexible policy configuration that doesn't force homogeneity
3. Automatic resource sharing to prevent configuration errors

Technical Implementation
------------------------

Link Groups
~~~~~~~~~~~

Link groups are the mechanism that enables one-to-many policy mappings:

.. code-block:: python

   @dataclass
   class LinkGroup:
       group_id: str              # Operator-scoped ID (e.g., "operator_0_link_0")
       primary_agent: str         # Primary agent in the group
       linked_agents: list[str]   # Other agents in the group
       policy_path: str           # Group's policy checkpoint
       algorithm: str             # Group's algorithm (IPPO/MAPPO)
       worker_type: str           # Worker type (rl)
       color: str                 # Visual indicator color

**How it works:**

1. User creates a link group via "Link Agents" button in GUI
2. Primary agent's policy path is stored in the link group
3. When operator is launched, linked agents use the group's policy path
4. Changing the primary agent's path updates all linked agents automatically

Multiple Independent Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~

MOSAIC supports multiple independent link groups, enabling complex
team configurations:

.. code-block:: python

   # Two independent teams with different policies
   # Link group IDs are scoped to operator: {operator_id}_link_{N}
   link_groups={
       # Offense team (agents 0 and 1)
       "operator_0_link_0": LinkGroup(
           group_id="operator_0_link_0",
           primary_agent="agent_0",
           linked_agents=["agent_1"],
           policy_path="/path/to/offense_mappo.pth",
           algorithm="mappo",
           worker_type="rl",
       ),
       # Defense team (agents 2 and 3)
       "operator_0_link_1": LinkGroup(
           group_id="operator_0_link_1",
           primary_agent="agent_2",
           linked_agents=["agent_3"],
           policy_path="/path/to/defense_mappo.pth",
           algorithm="mappo",
           worker_type="rl",
       ),
   }

GUI Integration
~~~~~~~~~~~~~~~

The GUI provides visual feedback for link groups:

- **Primary agents**: Show "Link Agents" button
- **Linked agents**: Show "Unlink Agents" button
- **Policy fields**: Hidden for linked agents (only button visible)
- **Editing**: Primary agents can add/remove linked agents via dialog

See :doc:`../policy_mapping` for complete API documentation.

Summary
-------

**The heterogeneous multi-agent challenge:**
   How to configure agents independently while sharing resources where appropriate

**The solution:**
   Flexible policy mappings (one-to-one and one-to-many) via link groups

**Why it matters:**
   Enables mixing RL, LLM, Human, and Random agents in the same environment
   without configuration overhead or manual errors

**The result:**
   Novel research questions in cross-paradigm cooperation, ad-hoc teamwork,
   and heterogeneous decision-making become practical and reproducible
