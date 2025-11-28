# Unreal-MAP: Unreal-Engine-Based General Platform for Multi-Agent Reinforcement Learning


## Abstract
In this paper, we propose Unreal Multi-Agent
Playground (Unreal-MAP), an MARL general
platform based on the Unreal-Engine (UE).
Unreal-MAP allows users to freely create multiagent tasks using the vast visual and physical resources available in the UE community,
and deploy state-of-the-art (SOTA) MARL algorithms within them. Unreal-MAP is userfriendly in terms of deployment, modification,
and visualization, and all its components are
open-source. We also develop an experimental framework compatible with algorithms ranging from rule-based to learning-based provided
by third-party frameworks. Lastly, we deploy
several SOTA algorithms in example tasks developed via Unreal-MAP, and conduct corresponding experimental analyses. We believe
Unreal-MAP can play an important role in the
MARL field by closely integrating existing algorithms with user-customized tasks, thus advancing the field of MARL. The source code
for our project is available at github.com/
binary-husky/unreal-map and github.
com/binary-husky/hmp2g.


## Introduction

Multi-agent reinforcement learning (MARL) has demonstrated remarkable potential in many practical fields, including swarm robotic control (Kalashnikov et al., 2018; Chen
et al., 2020), autonomous vehicles (Peng et al., 2021b),
and video games (Vinyals et al., 2019). There are many
classical and practical algorithms that have emerged in
the field of MARL, such as QMIX (Rashid et al., 2020b),
QPLEX (Wang et al., 2020), MAPPO (Yu et al., 2022) and HAPPO (Zhong et al., 2024). The development and success
of these algorithms could not be achieved without the support of numerous well-designed simulation environments.
These environments provide ample simulated data for testing and comparing algorithms, thereby fostering a variety
of more advanced algorithms.
Obviously, the progression of MARL algorithms will not
stop at mere comparison of performance; their ultimate aspiration lies in the deployment within practical applications.
However, when considering rapid deployment in real-world
applications, current popular simulation environments cannot be freely and effectively customized according to user
needs. Additionally, crafting new domain-specific environments for particular problems frequently encounters challenges in swiftly integrating with existing MARL algorithm
libraries1
, thus hindering the deployment of state-of-the-art
(SOTA) algorithms to meet practical requirements (Oroo jlooy & Hajinezhad, 2023).


The existence of modern game engines offers a possible
solution to the aforementioned issues. Game engines are
software frameworks originally designed to simplify the
development of video games (Boyd & Barbosa, 2017). Over
years of evolution, modern game engines have established
vast development communities, along with rich rendering
resources and physics engine materials (Wheeler, 2023).
If possible, these potential resources could greatly facilitate the application deployment within the MARL domain.
Moreover, game engines are closely linked with the rapidly
developing generative AI, which has the potential to quickly
transform user needs into real products. Some works have
utilized simulation data from game engines for the training of generative AI (Lu et al., 2024), whereas others have
established a feedback loop, incorporating these trained generative models into game engine plugins to facilitate scene
creation (NVIDIA, 2024). This suggests that combining
game engines with MARL could have a very promising
future.
However, there remains a significant gap between the game
development community associated with game engines and
the MARL community. Although some efforts, such as
URLT (Sapio & Ratini, 2022) and Unity-ML Agents (Juliani, 2018), have built frameworks for freely constructing
RL training scenarios using game engines2
, they still face
issues in training efficiency and scaling complexity (Kaup
et al., 2024). To address this, we propose Unreal-MAP,
an MARL general platform based on the Unreal Engine3
.
Compared to existing solutions, Unreal-MAP possesses the
following main features:
(1) Fully Open-Source and Easily Modifiable, UnrealMAP utilizes a layered design, and all components from
the bottom-level engine to the top-level interfaces are opensourced. Users can easily modify all elements of an MARL
task by focusing only on the higher-level operational layers. (2) Optimized Specifically for MARL, the underlying
engine of Unreal-MAP has been optimized to enhance efficiency in large-scale agent simulations and data transmission. This optimization allows users to develop simulations
with heterogeneous, large-scale, multi-team settings that
showcase distinctive multi-agent features through UnrealMAP. (3) Parallel Multi-Process Execution and Controllable Single-Process Time Flow, Unreal-MAP supports
the parallel execution of multiple simulation processes as
well as the adjustment of the simulation time flow speed in a
single process. Users can accelerate simulations to speed up
training or decelerate simulations for detailed slow-motion
2
In this paper, we refer to the works that allow for the free
construction of RL training scenarios as general platforms (Juliani,
2018).
3https://www.unrealengine.com.
analysis.
To fully utilize the capabilities of Unreal-MAP, we also
develop an MARL experimental framework known as the
Hybrid Multi-Agent Playground (HMAP). This framework
includes implementations of rule-based algorithms, builtin learning-based algorithms, and algorithms from thirdparty frameworks such as PyMARL2 (Hu et al., 2021) and
HARL (Zhong et al., 2024). By leveraging Unreal-MAP
and HMAP, users can rapidly customize environments and
deploy algorithms, validate new research ideas, and apply
them in practical scenarios. The overview of the research
workflow for using Unreal-MAP is depicted in Figure 1 and
more details can be found in Appendix C.4.
The contributions of this work are summarized as follows:
firstly, an MARL general platform based on the Unreal
Engine; secondly, an accompanying modular MARL experimental framework; thirdly, a collection of highly extensible
example scenarios based on Unreal-MAP and related experimental analyses. We believe Unreal-MAP can serve as a
comprehensive tool to advance the development of MARL
and ultimately facilitate their application in real-world scenarios.



## Related Works

2. Related Work
Simulation Environments for MARL. Existing MARL
environments can be broadly divided into domain-specific
environments and environment-suites. The former includes
a series of tasks designed around the same domain, which
usually share common genre and similar benchmark metrics.
Notable examples of these works include MPE (Mordatch &
Abbeel, 2017), SMAC (Samvelyan et al., 2019), GRF (Kurach et al., 2020), MAMuJoCo (Peng et al., 2021a), and GoBigger (Zhang et al., 2022). Due to architectural constraints,
domain-specific environments cannot be freely modified
according to user needs. For instance, a StarCraft-based
SMAC scenario can never be reconfigured into soccer-style
tasks. On the other hand, environment suites consist of
sets of environments packaged together, commonly used
to more conveniently benchmark the performance of algorithms. Some works also redesign built-in environments to
enhance training speed. Typical works in this category include PettingZoo (Terry et al., 2021) and JaxMARL (Rutherford et al., 2024). Nevertheless, these suites are just collections of domain-specific environments and thus lack flexible
modification capabilities. More details of related MARL
environments can be found in Appendix B.
General Platforms for RL. These kinds of works enable
users to create environments with arbitrary visual or physical
complexity and deploy existing RL algorithms. Unity-ML
Agents (Juliani, 2018) is a mature general platform built on
the Unity engine. It allows users to develop new scenes us ing the Unity editor and train them using Python-based RL
algorithm libraries. However, its limited open-source implementation hinders further customization (Wheeler, 2023),
and there are still issues in scaling complexity and simulation fidelity (Kaup et al., 2024). Compared to Unity, Unreal
Engine (UE) offers full open-source access and lower learning curves4
(Boyd, 2017), and is more conducive to the
development of a general platform. URLT (Sapio & Ratini, 2022) is an RL general platform based on UE, but its
algorithm part relies on Blueprint scripts and is difficult
to integrate with existing cutting-edge algorithm libraries.
Moreover, it does not support controllable time flow, resulting in very slow training and making it inapplicable to
large-scale agent simulations. Currently, there is a lack of a
powerful, MARL-targeted general platform.
Generative Physics Engines for RL. Emerging generative
physics engines show potential for translating user needs
into actual products, with the ultimate goal of generating
reliable simulations based on user prompts. Research in this
area is still in its infancy. A typical work in this domain
is Genesis (Authors, 2024), which can generate physical
scenes for RL training through prompts and supports differentiable scenes. However, it still relies on underlying
physical engines and 3D models, which is not fully generative. Emerging AI-driven generative physics engines may
replace current lightweight physical engines (Kaup et al.,
2024), but they still depend on the integration with game engines to achieve enhanced generative workflow. Specifically,
game engines can provide generative AI with enough data
for training (Lu et al., 2024), while generative AI can assist
game engines in quickly developing scene-related assets,
such as 3D models or physical engines. In fact, NVIDIA has
proposed a work for high-quality 3D asset generation (Bala
et al., 2024), and provided a generative AI plugin for use
with UE (NVIDIA, 2024). Combining game engine with
MARL naturally aligns with the development trend of generative AI.


## Background

To accommodate various interaction relationships among
multi-agent and multi-team scenarios (Fu et al., 2024), we
use Partially Observable Markov Game (POMG) (Littman,
1994; Gronauer & Diepold, 2022) to model the MARL
problem. A POMG can be represented by an 8-tuple
⟨N, {S
i}i∈N , {Oi}i∈N , {Ω
i}i∈N , {Ai}i∈N , {T i}i∈N , r, γ⟩.
N is the set of all agents, {S
i}i∈N is the global state space
which can be factored as {S
i}i∈N = ×i∈N S
(i) × S
E,
where S
(i)
is the state space of an agent i, and S
E is
the environmental state space, corresponding to all the
4The UE editor employs the graphical programming language
Blueprints, offering a lower learning curve than its native C++
library, even for those familiar with C++ (Boyd, 2017).
non-agent entities. {Oi}i∈N = ×i∈N O(i)
is the joint
observation space and {Ω
i}i∈N is the set of observation
functions. Similarly, {Ai}i∈N is the joint action space
of all agents. {T i}i∈N is the collection of all agents’
transitions and the environmental transition. Finally, γ is
the discount factor and r : {S
i}i∈N × {Ai}i∈N × N → R
is the agent-level reward function.
We define team as a collection of agents, which all share
the same overall goal in a purely cooperative form. Agents
within the same team aim to find an optimal joint policy
that maximizes the cumulative reward for the whole team.
Denoting the joint policy of a certain team A ⊆ N as π¯A,
the optimal policy π¯
∗
A can be represented as:
π¯
∗
A = arg max
π¯A
Eπ¯A
"X∞
k=0
γ
kX
i∈A
r
i
t+k
| s¯t = ¯s
#
, (1)
where s¯ is the initial global state, γ
k P
i∈A r
i
t+k
is the discounted return of team A, r
i
t+k
is the reward of an agent
i ∈ A at timestep t + k


## 4. Unreal-MAP

###  4.1. Basic Concepts in Unreal-MA
Multi-agent simulation can demonstrate great diversity in
different domains. We introduce several new concepts that
align with human intuition as well as the requirements of
multi-agent simulation.
Agents and Teams: Agents are the basic decision-making
units in the environments. Unreal-MAP introduces a new
concept team to distinguish agents with different goals.
Unreal-MAP supports numbers of teams, where teams may
engage in competition or cooperation. Each team possesses
its own independent goal and is equipped with a separate
learning-based (or rule-based) algorithm.
Entities: Entities are objects in simulation that do not make
decisions but still has important functionality. For instance
a street lamp or a dynamic obstacle. A shared characteristic
of these objects is that they must be removed or reinitialized
when an episode ends or a new episode starts.
Tasks and Scenarios: Tasks corresponds to POMGs defined
in Section 3. The properties of tasks in Unreal-MAP include
the types and numbers of agents, their team affiliations,
as well as each agent’s state space, action space, etc. A
scenario can give rise to a series of tasks, which typically
share similar reward functions, implying that the objectives
to be achieved by the multi-agent systems are the same.
Maps: Maps in Unreal-MAP determine where the task takes
place. A map can be a small room, or a city full of buildings.
It is a great advantage that Unreal-MAP decouples the concept of tasks and maps, as users can conveniently deploy a

### USE THE IMAGE HERE /home/hamid/Desktop/Projects/GUI_BDI_RL/docs/1.0_DAY_27/TASK_2/images/figure_2_Architecture_of_unreal_map.png

Figure 2. Architecture of Unreal-MAP. Unreal-MAP employs a hierarchical, five-layered architecture, all of which are open source.
Users can modify all elements within POMG by configuring parameters through the Python-based interface layer. For more advanced
development requirements, users can conveniently adjust scenario elements using Blueprint through the advanced module layer.


task in new maps (as long as the agent has the appropriate
size and a suitable position initialization function).
Events: We define an event system to simplify the reward
crafting procedure. For instance, an event will be generated
when an agent is destroyed or an episode is ended. When
it is time to compute next-step reward, these events will
provide convenient reference.


### 4.2. Utilizing Unreal-MAP to customize tasks

Unreal-MAP employs a hierarchical five-layer architecture,
where each layer builds upon the previous one. From bottom
to top, the five layers are: native layer, specification layer,
base class layer, advanced module layer, and interface
layer. Users only need to focus on the advanced module
layer and the interface layer. In most cases, modifying the
interface layer is sufficient to alter all elements of tasks.
Figure 2 shows the internal architecture of Unreal-MAP.
Specifically, the native layer includes assets from the Unreal
community and the Unreal Engine, some part of which have
been optimized for MARL compatibility. The specification
layer consists of Unreal-MAP’s underlying systems and
programming specifications, all implemented in C++. The
base class layer includes all basic classes implemented
using Blueprints. These three layers, also known as the
fundamental layers, form the foundation of Unreal-MAP, .
The top two layers of Unreal-MAP are user operation layers.
The advanced module layer, based on Blueprints, allows
for the modification of agents’ physical properties such as
appearances, perceptions and kinematics, thereby enabling
the development of various agents. This layer also facilitates
the development of environmental entities and maps. The
top layer is the interface layer, implemented in Python and
compliant with the gym standard. It supports customizable
observations and reward functions, and allows for the selection of maps and agents. More details about the architecture
of Unreal-MAP can be found in Appendix C.
Thanks to the hierarchical architecture of Unreal-MAP, users
can easily customize tasks through simple operations via
top layers. Here we provide a detailed explanation of how
each element of a task5
is customized within Unreal-MAP.
Agent Set. Within the interface level of Unreal-MAP, the
agent selection module enables users to specify the types,
numbers, and associated teams of agents.
State Space. The global state is composed of the states of
individual agents and the environmental state. Customization of the environmental state can be achieved by selecting
different maps and modifying them along with related entities. The state of the agents can be customized through the
agent init function in the advanced module layer and the
agent component module in the interface layer.
Observation Space and Observation Function. UnrealMAP transmits global information from the UE side to the
Python side, where the make obs function in the interface
layer is used to construct the agents’ observations. Modif cation of this function allows for the customization of each
agent’s observation space and function. Moreover, modifying agents’ properties, such as the observation range, can
also change their observations. Additionally, Unreal-MAP
supports more sophisticated agent observation simulation
mechanisms, such as masking entities blocked by walls,
which can be implemented through the agent perception
module in the advanced module layer.
Action Space. Unreal-MAP supports continuous actions,
discrete actions, and hybrid actions. Users can assign a
built-in action set to each agent via the agent init function
in the interface layer. Furthermore, a deeper customization
of agent actions can be achieved through the agent actionrelated modules in the advanced module layer.
Transition Function. Similar to the state space, the transition function in Unreal-MAP is comprised of local transitions of all agents and environmental transitions. The latter can be modified through map-related and entity-related
modules. Local transitions of agents can be customized
by modifying the agent init function and the step function,
or more deeply through the agent component modules and
agent controller modules, such as agent kinematics.
Reward Function. Unreal-MAP constructs rewards using
global information and an event system. Users can customize the agents’ rewards by modifying the make reward
function, which supports team and individual rewards, as
well as sparse and dense reward structures.


## 4.3. Other Features of Unreal-MAP
Unreal-MAP connects the UE community and MARL community. It allows users to utilize the extensive, realistic
resources such as models, rendering materials and physical simulations from the UE community to develop scenarios for MARL training. This is the biggest advantage of
Unreal-MAP over other MARL environments in terms of
the highly scalability and realism of the created scenarios.
Furthermore, this section will introduce other features of
Unreal-MAP that are beneficial in the context of MARL.
Computational efficiency. Numerous modifications have
been made to the underlying engine to adapt it for efficient
MARL training. These include optimizations within the
simulation engine and enhancements in the communication
between the simulator and the algorithm side (details are
provided in Appendix C.3). In practical training, UnrealMAP also supports a non-render training mode without
rendering frame computation.
Controllable simulation time flow. Unreal-MAP optimizes
the time flow control mechanism in Unreal Engine (details
are provided in Appendix C.2). Users can easily modify
the time dilation factor to adjust the ratio of simulated time
flow and real time flow. The ability to control the time flow

offers numerous benefits. On one hand, users can accelerate the simulation time for rapid training or decelerate for
debugging. On the other hand, since adjusting the speed
of simulation does not influence memory resources, users
can make fuller use of computational resources by adjusting the time dilation factor, with more detailed information
available in Section 7.2.
Compatibility with multiple systems and computational
backends. Unreal-MAP is natively compatible with creating environments and deploying algorithms on Windows,
Linux, and MacOS. It supports training on pure CPU setups
as well as on hybrid CPU/GPU configurations. Furthermore,
Unreal-MAP supports cross-device real-time rendering6
, allowing users to conduct multi-process training on a Linux
server while performing real-time rendering of specific processes on a Windows host.



## HMAP

To facilitate the deployment of algorithms for Unreal-MAP,
we also develop an experimental framework HMAP. HMAP
is a multi-agent experimental framework with decoupled
Task-Core-Algorithm components. Currently, HMAP not
only integrates Unreal-MAP’s tasks, but also supports other
MARL environments such as SMAC (Samvelyan et al.,
2019) and MPE (Mordatch & Abbeel, 2017). On the algorithm side, HMAP also supports a wide range of algorithms. This includes rule-based algorithms (most of them
are built-in policies for Unreal-MAP example tasks), singleagent RL algorithms like DQN (Mnih et al., 2015) and
SAC (Haarnoja et al., 2018), as well as MARL algorithms
such as MAPPO (Yu et al., 2022) and HAPPO (Zhong et al.,
2024). Furthermore, HMAP is compatible with third-party
frameworks, supporting all algorithms from PyMARL2 (Hu
et al., 2021) and HARL (Zhong et al., 2024).
The unique feature of HMAP is its support for multiteam training. By thoroughly decoupling algorithms from
tasks, HMAP employs its core as a “glue module”, enabling any algorithm module to control teams within any
task module. The highly modular design presents three key
benefits. Firstly, it enables modification of built-in policies
in tasks within Python-based algorithm modules, which can
significantly reduce the workload of building non-learningbased policy7 on the UE side. Secondly, it enables teams
controlled by multiple algorithms to interact within a same
task, facilitating co-training of algorithms from different
frameworks under the same task. Thirdly, it is user-friendly,
as all experimental configurations based on HMAP can be
implemented through a single JSON file. After completing

The configuration, users can initiate the training task with
just one line of code. More details of HMAP can be found
in Appendix D.




## Example Scanarios and Experiments

Unreal-MAP includes a variety of basic scenarios for multiagent systems, each of which is extensible and can be used
to create numerous tasks. This section describes 4 example
scenarios, which are used to develop 15 tasks applied in
Section 7. We use these example scenarios to demonstrate
that Unreal-MAP can be used to construct tasks with distinct
multi-agent characteristics. These characteristics include
heterogeneity, large-scale, multi-team, sparse team rewards,
and multi-agent games. More details of these example scenarios can be referred to Appendix E.
Metal Clash - Scenario designed for heterogeneous and
large-scale multi-agent tasks. It involves an SMAC-style
competition between two teams of agents. Each team can
be controlled by either rule-based or learning-based algorithms. Metal Clash provides three types of basic agents:
missile cars, laser cars and support drones. The properties
of each basic agent, such as maximum speed and health
points (HP), are encapsulated as configurable parameters.
Users can easily modify these parameters, creating a variety of heterogeneous agent types beyond the original three.
The number and types of agents in each team can be freely
changed, altering the features and difficulty of the tasks.
Monster Crisis - Scenario designed for sparse team rewards
in a multi-agent cooperative setting. This is a village-style
scenario where several mushroom agents need to resist the
invasion of a monster. The entire team receives a positive
reward only if the they kill the monster, and there are no
rewards or penalties in other cases. Users can adjust the
difficulty by modifying the monster’s HP and the number of
agents.
Flag Capture - Scenario designed for multi-team gaming
tasks. It involves several robot teams gaming to capture
a flag. The closest robot can pick up the flag, and their
teammates must defend it from other teams. At the end of
each episode, the team that held the flag the longest wins.
The number and team affiliations of agents can be freely
changed to modify the features and difficulty of the tasks.
Navigation Game - Scenario designed for heterogeneous
and two-team zero-sum gaming tasks. It includes two landmarks, a keeper team, and a navigator team. Although they
cannot attack each other, the ground keeper can drive away
the air navigator, while the ground navigator can drive away
the ground keeper. If the air navigator stays over any landmark for a certain period, the navigator team is deemed to
have won. The rewards for the two teams are zero-sum.
The number of agents in each team can be freely changed,
altering the difficulty of the tasks.


## Experiments

To demonstrate the utility of Unreal-MAP, we develop 15
example tasks and deploy several MARL algorithms across


## PLEASE PUT THESE IMAGES HERE /home/hamid/Desktop/Projects/GUI_BDI_RL/docs/1.0_DAY_27/TASK_2/images/table_Description_of_example_tasks_in_the_experiments.png
/home/hamid/Desktop/Projects/GUI_BDI_RL/docs/1.0_DAY_27/TASK_2/images/Figure_4_the_comparison_of_test_win_rate_for_tested_algorithms_across_15_tasks.png



them. We find that by altering the properties of scenarios,
it is capable of developing tasks that challenge current algorithms. Additionally, various algorithms exhibit superior
performance in their areas of strength. It is worth mentioning that we DO NOT intend to develop the example tasks
as benchmarks, but rather use them to show that based on
Unreal-MAP, it is possible to develop extensible tasks that
facilitate the deployment of popular MARL algorithms.
We then evaluate the training efficiency and resource consumption of Unreal-MAP. We find that changing the speed
within a process only affects the CPU utilization of the device, and it is possible to use more CPU resources under
memory-limited conditions by adjusting the time dilation
factor, or vice versa. We also discover that simulation tasks
for a scale of 20 agents developed based on Unreal-MAP,
can achieve physical simulation frames at the 1M level per
second, enabling the corresponding training tasks to be completed within hours.
Finally, we implement a sim2real demo based on the
navigation-game scenario, to demonstrate the potential of
Unreal-MAP in simulating real-world environments and
deploying algorithms in the real world.


### 7.1. Performance in Example Tasks

We develop 15 example tasks based on 4 scenarios from
Unreal-MAP, as detailed in Table 1. Based on HMAP, we
deploy 7 SOTA MARL algorithms on all tasks, including the
actor-critic-based algorithms as MAPPO (Yu et al., 2022),
HATRPO and HAPPO (Zhong et al., 2024), as well as the
value-based algorithms as QMIX (Rashid et al., 2020b),
QTRAN (Son et al., 2019), QPLEX (Wang et al., 2020) and
WQMIX (Rashid et al., 2020a). To ensure a fair comparison,
the main network of each algorithm is preserved uniform,

and hyperparameters are standardized (refer to Appendix I.1
for details). The effectiveness of the training is tested after
every 1280 episodes. The average win rates are calculated
based on 512 episodes per test, across 5 or more random
seeds. The results are illustrated in Figure 4, where the lines
represent the mean values and the shadowed areas indicate
the 95% confidence interval.
The results from Figure 4 show that Unreal-MAP is capable
of developing MARL-compatible multi-agent tasks. By
comparing the test results of different tasks developed from
the same scenario, it is evident that changing 1) the property
of individual entities or agents 2) the types of agents 3)
the number of agents 4) the number of agent teams can
effectively alter the features and difficulty of the tasks.
By comparing the results of different algorithms, we also
discover some interesting findings. Value-based algorithms
significantly outperform actor-critic(AC)-based methods in
tasks with sparse team rewards, but not in other tasks compared to AC’s SOTA algorithms. This is because valuebased algorithms, which focus on value decomposition,
are better at solving hard-to-decompose team reward problems. Within AC algorithms, MAPPO performs better in
large-scale tasks due to synchronous updates and parameter sharing, while HAPPO and HATRPO perform better in
multi-team (unstable environmental changes) tasks due to
asynchronous monotonic updates. Additional experiments
and analysis details can be found in Appendix I.2.


### 7.2. Efficiency and Computational Consumption
We conduct experiments on the efficiency index and resource consumption indices of Unreal-MAP. The efficiency
index adopted is TPS, i.e., the number of virtual Timesteps
Per real Second. The resource consumption indices include
CPU utilization, memory occupancy, and GPU memory
occupancy. Unreal-MAP has two dimensions to control
simulation efficiency and resource consumption: the number of processes and the speed within each process. Hence,
we conduct related experiments, obtaining the relationship
curves between the above two dimensions and four indices,
as shown in Figure 5. All experiments are conducted on
a Linux server equipped with 8 NVIDIA RTX3090 GPUs,
and the tested task is metal clash 5lc 5mc, details of experiments are available in Appendix F.
Through the experimental results, we can see that increasing
the number of parallel processes increases the TPS at the
cost of increasing all resource consumption indices. However, increasing the time dilation factor only increases
CPU utilization, with almost no effect on memory and
GPU memory. The time dilation factor is roughly proportional to TPS and CPU utilization. This means that under
limited memory resources, training efficiency can be improved by increasing the time dilation factor to fully utilize
the CPU; similarly, under limited CPU resources, reducing the time dilation factor and increasing the number of
processes can avoid the waste of computing resources.
Furthermore, according to our experimental results, when
the number of processes is 8 and the time dilation factor is 32, TPS can reach 400. Since the maximum number of episode steps is 100, training 1024 episodes on the
metal clash 5lc 5mc task takes less than 2 minutes. This
means that under such settings, the tested server can simultaneously support 50 such tasks (each with 20 agents)
and complete all training tasks (100k episodes) within 3
hours. When the number of processes reaches 128 and the
time dilation factor is set to 32, the TPS can reach 1000+,
and the training task can be completed in about an hour.
It is important to emphasize that TPS here counts for the
number of virtual Unreal-MAP timesteps per real second.
Since this is a simulation of 20 agents, and each timestep
in Unreal-MAP undergoes 1280 frames of calculations for
environmental dynamics and kinematics to maintain fine
state transitions (details in Appendix I.2), this means that
the speed of simulation physics frame calculation can
reach the 1M level, which is a highly efficient computation.

7.3. Physical Experiment
We conduct this experiment to demonstrate the potential of
Unreal-MAP in bridging the sim-to-real gap. Firstly, we
construct a real-world experimental setup, which consists of

a motion capture system, a communication system, several
autonomous UGVs and UAVs, and a number of physical entities. Subsequently, we develop the landmark conquer scenario through Unreal-MAP, wherein the entities are proportionally replicated from the physical setup, and the kinematics of the unmanned units are also recreated. Ultimately, we
develop an algorithm-Unreal-MAP-hardware framework,
with details presented in Appendix H.
During the training phase, the algorithmic side, represented
by HMAP, interact with Unreal-MAP to train policies within
the simulated scenarios. In the execution phase, the physical system relay global information captured by the motion
capture system and first-person view data from the vehicles’
cameras to Unreal-MAP. Unreal-MAP then update its internal environment with this information and transmit the
filtered observational data to HMAP. The algorithm within
HMAP generate action commands based on these observations, which are conveyed to Unreal-MAP. Unreal-MAP
execute virtual state transitions based on these commands,
and concurrently transmit the decomposed action information to the real-world setup for execution by the autonomous
vehicles/drones.
Figure 6 presents snapshots from both the virtual and the
real-world scenarios. The experimental results indicate that
the whole system can successfully replicate the policies of
the multi-agent system from the virtual environment within
the physical setup.


## 8. Limitations and Future Work
In this paper, we propose Unreal-MAP, an MARL general platform based on the Unreal Engine. We demonstrate the effects of deploying sample tasks developed on
this platform and cutting-edge MARL algorithms. However, Unreal-MAP is not perfect. One limitation is that
Unreal-MAP still requires the CPU to handle scene logic,
physical calculations, network communications, and other
tasks, and its training rate cannot match some purely-GPUimplemented environments. Additionally, although UnrealMAP has made all development tasks achievable solely
through Python and Blueprint, the MARL community may
not yet be accustomed to Blueprint programming.
We plan to further develop Unreal-MAP in two directions.
The first is related to generative AI, where we plan to first
integrate large models at the Python-based interface layer
to assist users in quickly customizing MARL tasks. The
second is focused on sim2real. We will develop a complete,
plug-and-play sim2real toolkit based on UMAP, mapping
real-world demands into the virtual world of Unreal-MAP,
thereby pushing the practical application of MARL to the
next level.
Impact Statement
The further development of MARL requires simulation
environments with high scalability and realistic physical
modelling capabilities. By combining MARL with game
engines, our work will help develop a general platform
and cater to the development trend of generative AI. Our
work also directly bridges the MARL community and the
game development community, with great potential to further enhance the development of the MARL and RL fields.
However, directly adopting content from the game engine
community is double-edged, and we need to ensure that
it does not bring potential negative impacts. To this end,
we have added a discussion on ethical review content in
Appendix G.




## A. Open Source Statement
We list the contents included in our open-source project, which not only encompasses Unreal-MAP and HMAP but also the
corresponding ecosystem, including tutorials, Docker environments, etc. The open-source project is as follows:
1. Regarding code environment configuration: We release a Docker image supporting Unreal-MAP and HMAP services
on Docker Hub. This image includes the HMAP framework, a default version of Unreal-MAP’s compiled binary files,
and a series of environment configurations.
2. Regarding Unreal-MAP: We publish Unreal-MAP’s usage tutorials and one-click deployment scripts on GitHub.
These scripts facilitate the compilation of rendering/training-only binary files for various platforms and automate the
downloading of large files. The Unreal project and the modified Unreal Engine of Unreal-MAP will be available on a
cloud drive, accessible for automatic download via Python scripts.
3. Regarding HMAP: We publish HMAP’s usage tutorials and its entire content to GitHub. This content includes the core
of HMAP, wrappers for all supported environments, built-in algorithms, and algorithms from third-party frameworks.
4. Future Plans for Open Source Work: We will continue to maintain all GitHub repositories, develop new scenarios,
incorporate more algorithms from third-party frameworks, and develop sim-to-real related toolkits.


## B. Related Work
The domain-specific simulation environments for MARL can be further broadly categorized into two types: those with
physics engines and those without. Here, physics engines refer to a suite of tools capable of simulating the physical laws
inherent in real-world tasks (Templet, 2021). Given that game engines also aim at reincarnating the real-world elements
into the digital world (Vohera et al., 2021), environments leveraging game engines are classified under the physics engine
category.
Among the environments without physics engines, MPE (Mordatch & Abbeel, 2017) utilizes a simple rule-based particle
world to simulate multi-agent tasks such as predator-prey and cooperative navigation. MAgent (Zheng et al., 2018), grounded
in a grid world, facilitates simulations involving the aggregation and combat of pixel-block agents, notable for its ability to
support large-scale multi-agent settings. The two environments mentioned above are based on the state transition laws of
particle worlds and particle interactions. Although they are completely open-source and their task elements are relatively
easy to modify, their scenarios are overly simplistic and lack realism.
Hanabi (Bard et al., 2020) provides a multiplayer card game scenario, which is commonly used in MARL research based on
opponent modeling. However, the overly narrow theme prevents it from further simulating tasks involving heterogeneity,
large scale, and mixed strategies. Neural MMO (Suarez et al., 2021) is developed in a 3D grid world derived from massively
multiplayer online games, supporting large-scale multi-agent simulations over long time horizons.

Gobigger (Zhang et al., 2022), based on a ball world concept, stands out for enabling simulations involving collaboration
and competition among multiple teams. However, similar to MPE and MAgent, their particle-based 2D environments fall
significantly short of simulating the real-world complexities of 3D scens.
As for the environments with physics engines, GRF (Kurach et al., 2020) is built upon the GameplayFootball simulator (Schuiling, 2017), creating a highly realistic football match setting that allows agents to simulate the behaviors of human
players. However, it does not support large-scale scenarios, multi-team training, and mixed multi-agent gameplay. Moreover,
although its environment interface and underlying engine are open-source, the underlying engine is not suitable.
SMAC (Samvelyan et al., 2019) and SMACv2 (Ellis et al., 2022) are developed based on the popular video game StarCraft II,
constructing a multi-agent micromanagement environment where each agent controls individual units to complete adversarial
tasks. Despite their widespread use, the fact that their underlying games and engines are not fully open-source limits further
expansion, confining their built-in tasks to battle-type game scenarios only.
Hide-and-Seek (Baker et al., 2019) has set up a series of multi-agent curriculum learning scenarios, such as hide and seek,
based on a 3D engine. However, its theme is too singular, making it impossible to simulate tasks involving heterogeneity,
large scale, multiple teams, etc., and it does not allow for customization of all task elements. Hok3v3 (Liu et al., 2023),
specifically designed for heterogeneous multi-agent tasks, is based on the Honor of Kings engine, with agent action spaces
consistent with those of human players engaging in the real game. However, it only supports heterogeneous multi-agent
scenarios (3VS3) and does not have an open-source underlying game and related engine.
MAMuJoCo (Peng et al., 2021a) is developed using the Mujoco physics engine (Todorov et al., 2012), where multiple agents
each control different joints to collaboratively manage the movements of a single robot. However, all of the multi-agent
scenarios are fully cooperative and do not support large-scale multi-agent tasks.
Marathon Environment (Booth & Booth, 2019) is developed using the Unity3D engine, supporting multiple agents learning
complex movements such as running and backflipping. The built-in tasks are relatively simple and are unable to simulate
large-scale, multi-team, and mixed multi-agent gameplay tasks. Moreover, its underlying engine, Unity3D, is not fully
open-source, thus preventing comprehensive modifications from the bottom to the top layer.
JaxMARL (Rutherford et al., 2024) integrates numerous MARL environments together and has re-implemented these
environments using JAX technology, enabling them to support efficient, GPU-based parallel computing. However, to support
pure GPU parallelism, some environments in JAXMARL have lost their original CPU-based underlying physical engines.
Moreover, as a collection of environments that integrates multiple basic environments, it does not support multi-team
multi-algorithm training, nor does it support controllable time-flow simulation and cross-platform real-time rendering.
It is evident that environments without physics engines are adept at simulating challenging tasks designed to push the
limits of existing algorithms. In contrast, environments equipped with physics engines offer greater potential for real-world
applications but are constrained in terms of academic flexibility. Our goal is to develop an environment that not only has
practical application potential but also fully leverages scalability, ultimately leading to the creation of Unreal-MAP.


C. Unreal-MAP Details
C.1. Architecture of Unreal-MAP
Unreal-MAP utilizes a hierarchical design that consists of five layers, all of which are open source. As shown in Figure 2,
the first layer of Unreal-MAP is the native part of the Unreal Engine, including the physics engine, rendering engine,
AI engine, and a range of 3D assets. We build the entire Unreal-MAP based on the open-source version of UE, making
modifications to some of the native modules. For instance, the original AI detection system for agents in UE was very
inefficient in large-scale scenes. Unreal-MAP optimizes the detection of multiple entities by incorporating tensor operations
and eliminating redundant checks.
The second layer of Unreal-MAP comprises the underlying systems and programming specification, all implemented in C++.
The time control system and task system in this layer ensure the correct initiation and termination of simulation episodes,
guaranteeing the precision of simulation time steps and the reproducibility of experimental results. Other components of this
layer define the specification for all base classes, communication, and debugging within Unreal-MAP.
The third layer consists of three fundamental classes implemented using Blueprints. The agent class defines all entities that
can be controlled by algorithms, while the entity actor corresponds to all environmental entities that do not make decisions.
13
Unreal-MAP: Unreal-Engine-Based General Platform for Multi-Agent Reinforcement Learning
Classes derived from these two form all the entity elements within a task scenario. The abstract class acts as a bridge,
connecting the underlying systems to the highest Python-based layer, facilitating communication, debugging, action updates,
and observation feedback.
The fourth layer of Unreal-MAP consists of advanced functional modules, implemented using Blueprints. These modules
allow for the modification of various attributes of agents, including appearance, perception, action sets, movement, and
navigation, enabling the development of diverse types of agents. Moreover, by leveraging the abundant resources in the
Unreal community, the map construction module facilitates the creation of new maps and even the importation of real-world
maps. The entity construction module aids in developing complex environmental entities, such as altering the kinematic
model of missiles launched by drones.
The fifth layer serves as the interface for interaction between Unreal-MAP and algorithms, all implemented in Python. This
interface adheres to the gym (Brockman, 2016) specification, encompassing basic functions like reset, step, and done, and
supports the customization of agent-level observation and reward functions. Attributes such as agent size, initial position,
detection range, and health are directly encapsulated within the agent initialization function, allowing for easy modification.
As shown in Figure 7, the selection of agents, tasks, and maps in Unreal-MAP are independent. Users can customize the
types, numbers, and teams of agents in a task and switch maps flexibly.
From the perspective of designing and utilizing a MARL simulation environment, users only need to focus on the fourth and
fifth layers of Unreal-MAP. In most cases, users can directly customize MARL tasks by modifying the built-in scenarios and
agent parameters through the fifth layer. If there is a need to develop new scenarios or further develop existing ones, users
can also easily develop through the graphical programming approach provided in the fourth layer. Such hierarchical design
of Unreal-MAP significantly reduces the burden of customizing tasks.
C.2. Time in Unreal-MAP
Time is one of the most important factor in simulations. There are two different type of time in Unreal-MAP:
1. Real World Time treal. The actual time of our world.
2. Simulation Time tsim. The time in the simulated virtual world.
It is inevitable that simulation speed (from the perspective of treal) will be influenced by factors such as CPU frequency,
GPU performance, policy neural network size, machine workload, etc. As a result, Unreal-MAP decouples simulation time
flow therefore has achieved flexible control of simulation time
1. Unreal-MAP allows researchers to slow down simulation time by setting a time dilation factor, extending a second in
the simulation multiple times to render details of agents in slow motion.
2. Unreal-MAP allows researchers to accelerate simulation time by setting the same time dilation factor (before reaching
the hardware limitation). Gathering large amount of samples is necessary in most RL tasks. Accelerating computation
is the primary ways to achieve this goal.
Unreal-MAP guarantees that the simulation results will not be influenced by time dilation factor, hardware or workload. For
instance, as long as the random seed remains identical, same agent trajectories are expected: 1) regardless of whether we
choose to enable GPU to accelerate neural network computation. 2) regardless of whether we choose to simulate agents
slowly or rapidly by setting different time dilation factors.
There are three global time-related settings to adjust in Unreal-MAP.
Decision time interval. From the perspective of agents in the simulated environment, agents will have a chance to act once
every t
step
sim . Alternatively, t
step
sim is also the time interval between each RL step. t
step
sim is usually a short period with a default
value 0.5s. Nevertheless, for tasks such as flights that last hours in a episode, t
step
sim should be increased accordingly.
t
step
sim does NOT has directly relationship with how long a RL step will actually take in the real world. More specifically, a
team can take as long as necessary to compute the next-step action after receiving observation, meanwhile the simulation
time flow freezes until all teams have committed agent actions. In extreme situations, algorithms can spend hours to update
large policy networks and the simulated agents will not be influenced by this delay.
14
Unreal-MAP: Unreal-Engine-Based General Platform for Multi-Agent Reinforcement Learning
Baseline Frame Rate. Baseline Frame Rate t
fr
sim determines how many frames to compute for each simulation second
in Unreal-MAP. As an example, when t
fr
sim = 30, the simulation will proceed (tick) 1
30 s after each frame. Important
computation such as collision detection and agent dynamic update are performed in each of these frames. As an example, let
t
step
sim = 0.5 and t
fr
sim = 30, under this circumstance 15 ticks will be performed between each RL step. Similarly, t
fr
sim does
NOT have direct relationship with the real world time flow.
Time Dilation Factor. In Unreal-MAP, Time Dilation Factor t
df
real is the sole bridge between simulation time flow and real
world time flow. In reinforcement learning, there are three typical cases that involves the control of time in simulation:
1. Task Development and Evaluation. In this case, it is demanded that simulation time flows at a normal speed to observe
the interaction of agents. A dilation factor t
df
real ≈ 1 will synchronize simulation time flow with the real world time flow.
2. Slow Motion. In this case, it is required that the simulation runs slowly to allow human observers to diagnose issues in
multi-agent cooperation. Changing the dilation factor t
df
real < 1 will slow down the simulated world accordingly.
3. Training. In this case, it is demanded that simulation runs as fast as possible to collect training data. Unreal-MAP will
attempt to accelerate the simulation until reaching the t
df
real threshold. If not possible due to hardware, the simulation
will still proceed at the fastest possible simulation speed.


