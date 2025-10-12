Journal of Machine Learning Research 23 (2022) 1-18 Submitted 11/21; Revised 6/22; Published 7/

# CleanRL: High-quality Single-file Implementations of Deep

# Reinforcement Learning Algorithms

Shengyi Huang^1 costa.huang@outlook.com
Rousslan Fernand Julien Dossa^2 doss@ai.cs.kobe-u.ac.jp
Chang Ye^3 c.ye@nyu.edu
Jeff Braga^1 jeffreybraga@gmail.com
Dipam Chakraborty^4 dipamc77@gmail.com
Kinal Mehta^5 kinal.mehta11@gmail.com
Jo ̃ao G.M. Ara ́ujo^6 joaoguilhermearujo@gmail.com

(^1) College of Computing and Informatics, Drexel University, USA
(^2) Graduate School of System Informatics, Kobe University, Japan
(^3) Tandon School of Engineering, New York University, USA
(^4) AIcrowd
(^5) International Institute of Information Technology, Hyderabad
(^6) Cohere
Editor:Joaquin Vanschoren

## Abstract

```
CleanRLis an open-source library that provides high-quality single-file implementations of
Deep Reinforcement Learning (DRL) algorithms. These single-file implementations are self-
contained algorithm variant files such asdqn.py,ppo.py, andppoatari.pythat individ-
ually include all algorithm variant’s implementation details. Such a paradigm significantly
reduces the complexity and the lines of code (LOC) in each implemented variant, which
makes them quicker and easier to understand. This paradigm gives the researchers the most
fine-grained control over all aspects of the algorithm in a single file, allowing them to proto-
type novel features quickly. Despite having succinct implementations, CleanRL’s codebase
is thoroughly documented and benchmarked to ensure performance is on par with reputable
sources. As a result, CleanRL produces a repository tailor-fit for two purposes: 1) under-
standing all implementation details of DRL algorithms and 2) quickly prototyping novel
features. CleanRL’s source code can be found athttps://github.com/vwxyzjn/cleanrl.
Keywords: deep reinforcement learning, single-file implementation, open-source
```
## 1. Introduction

In recent years, Deep Reinforcement Learning (DRL) algorithms have achieved great suc-
cess in training autonomous agents for tasks ranging from playing video games directly from
pixels to robotic control (Mnih et al., 2013; Lillicrap et al., 2016; Schulman et al., 2017). At
the same time, open-source DRL libraries also flourish in the community (Raffin et al., 2021;
Liang et al., 2018; D’Eramo et al., 2020; Fujita et al., 2021; Weng et al., 2021). Many of
them have adopted good modular designs and fostered vibrant development communities.
Nevertheless, understanding all the implementation details of an algorithm remains difficult

©c2022 Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, Jeff Braga, Dipam Chakraborty, Kinal Mehta,
and Jo ̃ao G.M. Ara ́ujo.
License: CC-BY 4.0, seehttps://creativecommons.org/licenses/by/4.0/. Attribution requirements are provided
athttp://jmlr.org/papers/v23/21-1342.html.


```
Huang, Dossa, Ye, Braga, Chakraborty, Mehta, and Araujo ́
```
Figure 1: Filediff in Visual Studio Code: left click selectppoatari.pythencmd/ctrl+ left
click selectppocontinuousaction.pyto highlight neural network architecture
differences of PPO when applying to Atari games and MuJoCo tasks.

because these details are spread to different modules. However, understanding these imple-
mentation details is essential because they could significantly affect performance (Engstrom
et al., 2020).
In this paper, we introduceCleanRL, a DRL library based on single-file implementa-
tions to help researchers understand all the details of an algorithm, prototype new features,
analyze experiments, and scale the experiments with ease. CleanRLis anon-modularli-
brary. Each algorithm variant inCleanRLis self-contained in a single file, in which the
lines of code (LOC) have been trimmed to the bare minimum. Along with succinct im-
plementations, CleanRL’s codebase is thoroughly documented and benchmarked to ensure
performance is on par with reputable sources. For example, our Proximal Policy Opti-
mization (PPO) (Schulman et al., 2017) implementation with Atari games is a single file
ppoatari.pyusing only 337 LOC, yet it closely matchesopenai/baselines’ PPO per-
formance in the game breakout (Appendix A), making it much easier to understand the
algorithm in one go. In contrast, the researchers using modular DRL libraries often need to
understand the modular design (usually 7 to 20 files) which can contain thousands of LOC.
As a result, CleanRL is tailor-fit for two purposes: 1) understanding all implementation
details of DRL algorithms and 2) quickly prototyping novel features.

## 2. Single-file Implementations

Despite the many features modular DRL libraries offer, understanding all the relevant code
of an algorithm is a non-trivial effort. As an example, running the PPO model in Atari
games usingStable Baselines 3 (SB3)with a debugger involves jumping back and forth
between 20 python files that comprise 4000+ LOC (Raffin et al., 2021) (Appendix B). This
makes it difficult to understand how the algorithm works due to the sheer amount of code


```
CleanRL: High-quality Single-file Implementations of DRL Algorithms
```
and its complex structure. This is a problem because even small implementation details
can have a large impact on the performance of deep RL algorithms (Engstrom et al., 2020),
and understanding them has become increasingly important.
CleanRLmakes it much easier to understand implementation details with a simple idea
— putting all implementation details of an algorithm variant into a single file. We call
this practice “single-file implementations.” Single-file implementations allow us to focus on
implementing a specific variant without worrying about handling special cases. Also, for
utilities that are not relevant to the algorithm itself, like logging and plotting, we import
third-part libraries. As a result, CleanRL produces a codebase with an order of magnitude
fewer LOC for each algorithm variant. For example, we have a:

1. ppo.py(321 LOC) for the classic control environments, such asCartPole-v1,
2. ppoatari.py(337 LOC) for the Atari environments (Bellemare et al., 2013),
3. ppocontinuousaction.py(331 LOC) for the robotics environments (e.g., MuJoCo,
    PyBullet) with continuous action spaces (Schulman et al., 2017).

The single-file implementations have the following benefits.
Transparent learning experienceIt becomes easier to recognize all aspects of the
code in one place. By looking atppo.py, it is straightforward to recognize the core imple-
mentation details of PPO. It also becomes easier to identify the difference between algorithm
variants viafilediff. For example, comparingppo.pywithppoatari.pyshows a 30 LOC
difference required to add environment prepossessing and modify neural networks. Mean-
while, another comparison withppocontinuousaction.pyshows a 25 LOC difference
required to use normalization and account for continuous action space. See Figure 1 as an
example. Being able to display the variant’s differences explicitly has helped us explain 37
implementation details of PPO (Huang et al., 2022).
Better debug interactivityEverything is located in a single file, so when debugging,
the user does not need to browse different modules like in modular libraries. Additionally,
most variables in the files exist in theglobal Python name scope. This means the researchers
can useCtrl+Cto stop the program execution and check most variables and their shapes
in the interactive shell (Appendix C). This is more convenient than using the Python’s
debugger, which only shows the variables in a specific name scope like in a function.
Painless performance attributionIf a new version of our algorithm has obtained a
higher performance, we know the exact single file which is responsible for the performance
improvement. To attribute the performance improvement, we can simply do a filediff be-
tween the current and past versions, and every line of code change is made explicit to us.
In comparison, two different versions of modular RL libraries usually involve dozens of file
changes, which are more difficult to compare.
Faster prototyping experienceCleanRL gives researchers fine-grained control to
everything related to the algorithm in a single file, hence making it efficient to develop
prototypes without having to subclass like in other modular RL libraries. As an example,
invalid action masking (Huang and Onta ̃n ́on, 2022) is a common technique used in games
with large, parameterized action spaces. WithCleanRL, it takes about 40 LOC to imple-
ment (Huang et al., 2022, Sec. 4), whereas in other libraries it could take substantially


```
Huang, Dossa, Ye, Braga, Chakraborty, Mehta, and Araujo ́
```
more LOC (e.g., more than 600 LOC, excluding the test cases^1 ) because of overhead such
as re-factoring the functional arguments and making more general classes.
Because of these benefits, we have also implemented single-file implementations for Deep
Q-learning (Mnih et al., 2013), Categorical Deep Q-learning (Bellemare et al., 2017), Deep
Deterministic Policy Gradient (Lillicrap et al., 2016), Twin-delayed Deep Deterministic
Policy Gradient (Fujimoto et al., 2018), Soft Actor-cirtic (Haarnoja et al., 2018a), Phasic
Policy Gradient (Cobbe et al., 2021), and Random Network Distillation (Burda et al., 2019).
Despite of these benefits of single-file implementations, one downside is the excessive
amount of duplicate code. To help reduce the maintenance overhead, we have adopted a
series of developmental tools to automatically format code, pin dependencies, scale experi-
ments with cloud providers, etc (Appendix D).

## 3. Documentation and Benchmark

All CleanRL’s single-file implementations are thoroughly documented and benchmarked in
our main documentation site (https://docs.cleanrl.dev/). For each single-file imple-
mentation, we document the original paper and relevant information, usage, an explanation
of logged metrics, note-worthy implementation details, and benchmark results which include
learning curves, a table comparing performance against reputable sources when applicable,
and links to the tracked experiments. In particular, the benchmark experiments are tracked
with Weights and Biases (Biewald, 2020), which allows the users to interactively explore
other tracked data such as system metrics, hyperparameters, and the agents’ gameplay
videos. For convenience, we have included tables comparing the performance of CleanRL’s
single-file implementations against reputable sources when applicable (Appendix A).

### 3.1. Log and telemetry formats

- **Console output (`stdout`):** every training update emits a single log line of the form
    `global_step=<int>, episodic_return=<float>, episodic_length=<int>, time_elapsed=<sec>,
    total_timesteps=<int>, fps=<float>, wall_time=<HH:MM:SS>`. These keys are stable across
    algorithms, enabling straightforward parsing for dashboards or ingestion pipelines.
- **TensorBoard summaries:** scalar metrics are recorded under namespaces such as
    `charts/*` (learning curves, SPS), `losses/*` (policy/value losses), and `rollout/*`
    (episode statistics). Hyperparameters are rendered as Markdown within the `text/hparams`
    tag, preserving a complete CLI snapshot per run.
- **Weights & Biases tracking:** when `--track` is enabled, CleanRL syncs TensorBoard
    data to W&B and augments it with `system/*` metrics (CPU, GPU, memory) plus
    `media/videos/*` entries that host MP4 gameplay clips. The run configuration mirrors
    the command-line arguments as JSON for reproducibility.
- **Video artifacts:** `--capture-video` produces paired files `videos/<run>/rl-video-episode-<n>.mp4`
    and metadata siblings `rl-video-episode-<n>.meta.json` detailing `episode_id`,
    `episode_return`, `episode_length`, and timestamps. These files feed both local analysis
    and W&B media dashboards.

### 3.2. Execution pipeline (mermaid diagram)

```mermaid
flowchart TD
    A[CLI launch via tyro] --> B[Parse Args dataclass]
    B --> C[Derive runtime sizes
        (batch_size, minibatch_size,
        num_iterations)]
    C --> D[Run naming & trackers
        (TensorBoard, optional W&B)]
    D --> E[Seeding: Python, NumPy,
        Torch, Gym]
    E --> F[Vectorized env factory
        make_env -> SyncVectorEnv]
    F --> G[Rollout loop
        collect obs/actions/rewards]
    G --> H[Compute GAE advantages]
    H --> I[PPO optimization epochs
        (minibatch update,
        clipping, entropy bonus)]
    I --> J[Write scalars/videos
        to TensorBoard & W&B]
    J --> K[Console log
        global_step & SPS]
    I -.-> L{target_kl hit?}
    L -->|Yes| M[Break early
        from update epochs]
    L -->|No| G
    K --> N[Close envs & writers]
```

This flow captures the common structure across CleanRL scripts: lightweight argument parsing,
derived runtime constants, reproducible seeding, vectorized rollouts, PPO-style optimization,
and multi-channel telemetry.

### 3.3. Command-line arguments (ppo.py example)

CleanRL’s scripts expose configuration exclusively through dataclass fields consumed by `tyro.cli`.
The table below summarises the standard flags for `cleanrl/ppo.py` (shared by most PPO variants):

| Category | Flag | Default | Description |
| --- | --- | --- | --- |
| Run identity | `--exp-name` | `ppo` | String used in run naming (`<env>__<exp>__<seed>__<timestamp>`). |
| | `--seed` | `1` | Global seed forwarded to Python, NumPy, Torch, Gym env reset. |
| Determinism | `--torch-deterministic` | `True` | When true, toggles deterministic CuDNN kernels. |
| Hardware | `--cuda` | `True` | Enables CUDA if a GPU is available. |
| Tracking | `--track` | `False` | When set, initializes Weights & Biases with TensorBoard sync. |
| | `--wandb-project-name` | `cleanRL` | Target W&B project slug. |
| | `--wandb-entity` | `None` | Optional W&B entity/team override. |
| Media | `--capture-video` | `False` | Records RGB episodes for env index 0 under `videos/`. |
| Environment | `--env-id` | `CartPole-v1` | Gymnasium environment identifier. |
| Training horizon | `--total-timesteps` | `500000` | Total environment steps targeted. |
| Optimizer | `--learning-rate` | `2.5e-4` | Adam optimizer step size. |
| Parallelism | `--num-envs` | `4` | Number of vectorized environments. |
| Rollout | `--num-steps` | `128` | Steps per environment per rollout window. |
| LR schedule | `--anneal-lr` | `True` | Linear annealing of learning rate over training. |
| Discounting | `--gamma` | `0.99` | Discount factor. |
| GAE | `--gae-lambda` | `0.95` | Lambda parameter for generalized advantage estimation. |
| SGD | `--num-minibatches` | `4` | Number of mini-batches per epoch. |
| | `--update-epochs` | `4` | PPO optimization epochs per rollout. |
| Advantage norm | `--norm-adv` | `True` | Standardises advantages per mini-batch. |
| Clipping | `--clip-coef` | `0.2` | PPO surrogate ratio clipping coefficient. |
| Value loss | `--clip-vloss` | `True` | Applies value function clipping. |
| Entropy bonus | `--ent-coef` | `0.01` | Scales policy entropy regularisation. |
| Value coef | `--vf-coef` | `0.5` | Scales value loss contribution. |
| Gradients | `--max-grad-norm` | `0.5` | Gradient clipping norm. |
| Early stop | `--target-kl` | `None` | Optional KL threshold to end PPO epochs early. |

Runtime-only fields (`batch_size`, `minibatch_size`, `num_iterations`) are computed after parsing and
are not exposed as CLI flags. Other CleanRL scripts amend this dataclass with algorithm-specific
hyperparameters (e.g., convolutional encoders for Atari, continuous-action noise schedules). Tools such
as `cleanrl_utils.tuner` consume the same arguments to automate sweeps.

## 4. When to Use CleanRL

CleanRL has its own set of pros and cons like other popular modular RL libraries. For
example, modular DRL libraries, such as SB3, offer a friendly end user API — if an end
user does not know much about DRL but wants to apply PPO in their tasks, SB3 would
be a great fit. Among many other benefits, SB3 makes it easy to configure different compo-
nents. CleanRL does not have a friendly end user API likeagent.learn(), but it exposes
all implementation details and is easy to read, debug, modify for research, and study RL.
Comparatively, CleanRL is well-suited for researchers who need to understand all imple-
mentation details of DRL algorithms, and prototype novel features quickly.
CleanRL complements the DRL research community with a unique developing experi-
ence. In fact, there is a win-win situation for CleanRL and SB3: “prototype with CleanRL
and port to SB3 for wider adoption in the community.” CleanRL’s codebase often allows
researchers to prototype specialized features much quicker. As shown above, the invalid ac-
tion masking technique with PPO takes∼40 LOC to implement. Once we have rigorously
validated this technique, our results and analysis will provide concrete guidance for porting
this technique to SB3, which enable our technique to reach a wider range of audience given
SB3’s friendly end user APIs.

1. Seehttps://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/25.


```
CleanRL: High-quality Single-file Implementations of DRL Algorithms
```
## Acknowledgments

We thank Santiago Onta ̃n ́on for providing helpful feedback and Angela Steyn for proofread-
ing the paper. We thank Google’s TPU Research Cloud (TRC) for supporting TPU related
experiments. Also, we thank the following people for their open-source contributions:

1. @adamcakg: refactor class arguments (#34)
2. Jordan Terry: fix thebibtexentry (#44).
3. Michael Lu: fix links in the docs (#48)
4. Felipe Martins: fix links in the docs (#54)
5. @chutaklee: refactor TD3 class initialization (#37)
6. Ram Rachum: use correct python exception clause (#205)
7. Alexander Nikulin: fix reward normalization wrapper (#209)
8. Ben Trevett: refactor replay buffer (#36)
9. @ElliotMunro200: specify correct python version requirement (#177)
10. Helge Spieker: fix typos (#45)


```
Huang, Dossa, Ye, Braga, Chakraborty, Mehta, and Araujo ́
```
## Appendix A. Benchmark experiments

Like mentioned in the Section 3, we rigorously benchmark our single-file implementations
to validate their quality. Below are the tables that compare performance against reputable
resources when applicable, where the reported numbers are the final average episodic returns
of at least 3 random seeds. For more detailed information, see the main documentation site
(https://docs.cleanrl.dev/).

A.1 Proximal Policy Optimization Variants and Performance

The following table reports the final episodic returns obtained by the agent in Gym’s classic
control tasks (Brockman et al., 2016):

Environment ppo.py openai/baselies’ PPO (Huang et al., 2022)

CartPole-v1 492.40±13.05 497.54±4.
Acrobot-v1 -89.93±6.34 -81.82±5.
MountainCar-v0 -200.00±0.00 -200.00±0.

The following tables report the final episodic returns obtained by the agent in Gym’s Atari
tasks (Brockman et al., 2016; Bellemare et al., 2013):

Environment ppoatari.py openai/baselies’ PPO [Huang et al., 2022](ˆ1)

BreakoutNoFrameskip-v4 416.31±43.92 406.57±31.
PongNoFrameskip-v4 20.59±0.35 20.512±0.
BeamRiderNoFrameskip-v4 2445.38±528.91 2642.97±670.

Environment ppoatarilstm.py openai/baselies’ PPO (Huang et al., 2022)

BreakoutNoFrameskip-v4 128.92±31.10 138.98±50.
PongNoFrameskip-v4 19.78±1.58 19.79±0.
BeamRiderNoFrameskip-v4 1536.20±612.21 1591.68±372.

Environment ppoatarimultigpu.py(160 mins) ppoatari.py(215 mins)

BreakoutNoFrameskip-v4 429.06±52.09 416.31±43.
PongNoFrameskip-v4 20.40±0.46 20.59±0.
BeamRiderNoFrameskip-v4 2454.54±740.49 2445.38±528.

The following table reports the final episodic returns obtained by the agent in Gym’s Mu-
JoCo tasks (Brockman et al., 2016; Todorov et al., 2012):


```
CleanRL: High-quality Single-file Implementations of DRL Algorithms
```
Environment ppocontinuousaction.py openai/baselies’ PPO (Huang et al., 2022)

Hopper-v2 2231.12±656.72 2518.95±850.
Walker2d-v2 3050.09±1136.21 3208.08±1264.
HalfCheetah-v2 1822.82±928.11 2152.26±1159.

The following table reports the final episodic returns obtained by the agent in EnvPool’s
Atari tasks (Brockman et al., 2016; Bellemare et al., 2013; Weng et al., 2022):

Environment ppoatarienvpool.py(80 mins) ppoatari.py(220 mins)

Breakout 389.57±29.62 416.31±43.
Pong 20.55±0.37 20.59±0.
BeamRider 2039.83±1146.62 2445.38±528.

The following table reports the final episodic returns obtained by the agent in Procgen
tasks (Cobbe et al., 2020):

Environment ppoprocgen.py openai/baselies’ PPO (Huang et al., 2022)

StarPilot 31.40±11.73 33.97±7.
BossFight 9.09±2.35 9.35±2.
BigFish 21.44±6.73 20.06±5.

The following table reports the final episodic returns obtained by the agent in Isaac Gym (Makoviy-
chuk et al., 2021):

Environment ppocontinuousactionisaacgym.py
(160 mins)

```
Denys88/rlgames(215 mins)
```
Cartpole (40s) 413.66±120.93 417.49 (30s)
Ant (240s) 3953.30±667.086 5873.
Humanoid (350s) 2987.95±257.60 6254.
Anymal (317s) 29.34±17.80 62.
BallBalance (160s) 161.92±89.20 319.
AllegroHand (200m) 762.93±427.92 3479.
ShadowHand (130m) 427.16±161.79 5713.

The following table reports the finalepisodic lengthinstead ofepisodic returnobtained by
the agent in PettingZoo (Terry et al., 2021):

Environment ppopettingzoomaatari.py(160 mins)

pongv3 4153.60±190.
surroundv2 3055.33±223.
tennisv3 14538.02±7005.


```
Huang, Dossa, Ye, Braga, Chakraborty, Mehta, and Araujo ́
```
A.2 Deep Deterministic Policy Gradient Variants and Performance

The following tables report the final episodic returns obtained by the agent in Gym’s Mu-
JoCo tasks (Brockman et al., 2016; Todorov et al., 2012):

Environment ddpgcontinuousaction.py OurDDPG.py (Fujimoto
et al., 2018, Tab. 1)

```
DDPG.py using settings
from (Lillicrap et al.,
2016) in (Fujimoto
et al., 2018, Tab. 1)
```
HalfCheetah 9382.32±1395.52 8577.29 3305.
Walker2d 1598.35±862.66 3098.11 1843.
Hopper 1313.43±684.46 1860.02 2020.

Environment ddpgcontinuousactionjax.py
(RTX 3060)

```
ddpgcontinuousactionjax.py
(VM w/ TPU)
```
HalfCheetah 9910.53±673.49 9790.72±1494.
Walker2d 1397.60±677.12 1314.83±689.
Hopper 1603.5±727.281 1602.20±696.

```
ddpgcontinuousaction.py
(RTX 2060)
```
HalfCheetah 9382.32±1395.
Walker2d 1598.35± 862
Hopper 1313.43±684.

A.3 Twin-Delayed Deep Deterministic Policy Gradient Variants and
Performance

The following tables report the final episodic returns obtained by the agent in Gym’s Mu-
JoCo tasks (Brockman et al., 2016; Todorov et al., 2012):

Environment td3continuousaction.py TD3.py(Fujimoto et al., 2018, Tab. 1)

HalfCheetah 9018.31±1078.31 9636.95±859.
Walker2d 4246.07±1210.84 4682.82±539.
Hopper 3391.78±232.21 3564.07±114.


```
CleanRL: High-quality Single-file Implementations of DRL Algorithms
```
Environment td3continuousactionjax.py
(RTX 3060)

```
td3continuousactionjax.py
(VM w/ TPU)
```
HalfCheetah 9099.93±1171.83 9127.81±965.
Walker2d 2874.39±1684.57 3519.38±368.
Hopper 3382.66±242.52 3126.40±558.

```
td3continuousaction.py
(RTX 2060)
```
HalfCheetah 9018.31±1078.
Walker2d 4246.07±1210.
Hopper 3391.78±232.

A.4 Soft Actor-Critic Variant and Performance

The following table reports the final episodic returns obtained by the agent in Gym’s Mu-
JoCo tasks (Brockman et al., 2016; Todorov et al., 2012):

Environment saccontinuousaction.py Haarnoja et al. (2018b)

HalfCheetah-v2 10310.37±1873.21 ∼11,
Walker2d-v2 4418.15±592.82 ∼4,
Hopper-v2 2685.76±762.16 ∼3,

A.5 Phasic Policy Gradient Variant and Performance

The following table reports the final episodic returns obtained by the agent in Procgen
tasks (Cobbe et al., 2020):

Environment ppgprocgen.py ppoprocgen.py openai/phasic-policy-gradient

Starpilot (easy) 35.19±13.07 33.15±11.99 42.01±9.
Bossfight (easy) 10.34±2.27 9.48±2.42 10.71±2.
Bigfish (easy) 27.25±7.55 22.21±7.42 15.94±10.


```
Huang, Dossa, Ye, Braga, Chakraborty, Mehta, and Araujo ́
```
A.6 Deep Q-learning Variants and Performance

The following tables report the final episodic returns obtained by the agent in Gym’s Atari
tasks (Brockman et al., 2016; Bellemare et al., 2013):

Environment dqnatari.py
10M steps

```
Mnih et al. (2015)
50M steps
```
```
Hessel et al. (2018,
Fig. 5)
```
BreakoutNoFrameskip-v4 366.928±39.89 401.2±26.9 ∼230 (10M steps)
∼300 (50M steps)
PongNoFrameskip-v4 20.25±0.41 18.9±1.3 ∼20 (10M steps)
∼20 (50M steps)
BeamRiderNoFrameskip-v4 6673.24±1434.37 6846± 1619 ∼6000 (10M steps)
∼7000 (50M steps)

Environment dqnatarijax.py
10M steps

```
dqnatari.py
10M steps
```
```
Mnih et al. (2015)
50M steps
```
BreakoutNoFrameskip-v4 377.82±34.91 366.928±39.89 401.2±26.
PongNoFrameskip-v4 20.43±0.34 20.25±0.41 18.9±1.
BeamRiderNoFrameskip-v4 5938.13±955.84 6673.24±1434.37 6846± 1619

The following tables report the final episodic returns obtained by the agent in Gym’s classic
control tasks (Brockman et al., 2016):

Environment dqn.py

CartPole-v1 488.69±16.
Acrobot-v1 -91.54±7.
MountainCar-v0 -194.95±8.

Environment dqnjax.py dqn.py

CartPole-v1 499.84±0.24 488.69±16.
Acrobot-v1 -89.17±8.79 -91.54±7.
MountainCar-v0 -173.71±29.14 -194.95±8.


```
CleanRL: High-quality Single-file Implementations of DRL Algorithms
```
A.7 Categorical Deep Q-learning Variants and Performance

The following table reports the final episodic returns obtained by the agent in Gym’s Atari
tasks (Brockman et al., 2016; Bellemare et al., 2013):

Environment c51atari.py
10M steps

```
Bellemare et al.
(2013, Fig. 14)
50M steps
```
```
Hessel et al. (2018,
Fig. 5)
```
BreakoutNoFrameskip-v4 461.86±69.65 748 ∼500 (10M steps)
∼600 (50M steps)
PongNoFrameskip-v4 19.46±0.70 20.9 ∼20 (10M steps)
∼20 (50M steps)
BeamRiderNoFrameskip-v4 9592.90±2270.15 14,074 ∼12000 (10M steps)
∼14000 (50M steps)

The following table reports the final episodic returns obtained by the agent in Gym’s classic
control tasks (Brockman et al., 2016):

Environment c51.py

CartPole-v1 481.20±20.
Acrobot-v1 -87.70±5.
MountainCar-v0 -166.38±27.

A.8 Random Network Distillation

The following table reports the final episodic returns obtained by the agent in Gym’s Atari
tasks (Brockman et al., 2016; Bellemare et al., 2013):

Environment pporndenvpool.py Burda et al. (2019)

MontezumaRevenge-v5 7100 (1 seed) 8152 (3 seeds)


```
Huang, Dossa, Ye, Braga, Chakraborty, Mehta, and Araujo ́
```
## Appendix B. Stepping Through Stable-baselines 3 Code with a Debugger

In this section, we attempt to run the following Stable-baselines 3 (v1.5.0)^2 code with a
debugger to identify the related modules.

```
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=25_000)
```
```
Here is the list of the related python files and their lines of code (LOC):
```
1. stablebaselines3/ppo/ppo.py— 315 LOC, 51 lines of docstring (LOD)
2. stablebaselines3/common/onpolicyalgorithm.py— 280 LOC, 49 LOD
3. stablebaselines3/common/baseclass.py— 819 LOC, 231 LOD
4. stablebaselines3/common/utils.py— 506 LOC, 195 LOD
5. stablebaselines3/common/envutil.py— 157 LOC, 43 LOD
6. stablebaselines3/common/atariwrappers.py— 249 LOC, 84 LOD
7. stablebaselines3/common/vecenv/ init .py— 73 LOC, 24 LOD
8. stablebaselines3/common/vecenv/dummyvecenv.py— 126 LOC, 25 LOD
9. stablebaselines3/common/vecenv/basevecenv.py— 375 LOC, 112 LOD

```
10.stablebaselines3/common/vecenv/util.py— 77 LOC, 31 LOD
```
```
11.stablebaselines3/common/vecenv/vecframestack.py— 65 LOC, 14 LOD
```
```
12.stablebaselines3/common/vecenv/stackedobservations.py— 267 LOC, 74 LOD
```
```
13.stablebaselines3/common/preprocessing.py— 217 LOC, 68 LOD
```
```
14.stablebaselines3/common/buffers.py— 770 LOC, 183 LOD
```
```
15.stablebaselines3/common/policies.py— 962 LOC, 336 LOD
```
```
16.stablebaselines3/common/torchlayers.py— 318 LOC, 97 LOD
```
```
17.stablebaselines3/common/distributions.py— 700 LOC, 228 LOD
2.https://github.com/DLR-RM/stable-baselines3/releases/tag/v1.5.
```

```
CleanRL: High-quality Single-file Implementations of DRL Algorithms
```
```
18.stablebaselines3/common/monitor.py— 240 LOC, 76 LOD
```
```
19.stablebaselines3/common/logger.py— 640 LOC, 201 LOD
```
```
20.stablebaselines3/common/callbacks.py— 603 LOC, 150 LOD
```
The total LOC involved is 7759. Notice we have labeled the popular utilities such as vec-
torized environments, Atari environment pre-processing wrappers, and episode statistics
recording code with the blue color. This means the total LOC related to core PPO imple-
mentationnotcounting the blue color files and lines of docstring is 4498.

## Appendix C. Interactive Shell

InCleanRL, we have put most of the variables in theglobal python name scope. This makes
it easier to inspect the variables and their shapes. The following figure shows a screenshot
of the Spyder editor^3 , where the code is on the left and the interactive shell is on the right.
In the interactive shell, we can easily inspect the variables for debugging purposes without
modifying the code.

```
3.https://www.spyder-ide.org/
```

```
Huang, Dossa, Ye, Braga, Chakraborty, Mehta, and Araujo ́
```
## Appendix D. Maintaining Single-file Implementations

Despite the many benefits that single-file implementations offer, one downside is excessive
amount of duplicate code, which makes them difficult to maintain. To help address this
challenge, we have adopted a series of development tools to reduce maintenance burden.
These tools are:

1. poetry(https://python-poetry.org/): poetry is a dependency management tool
    that helps resolve and pins dependency versions. We use poetry to improve repro-
    ducibility and provide a smooth dependency installation experience. See our instal-
    lation documentation (https://docs.cleanrl.dev/get-started/installation/)
    for more detail.
2. pre-commit(https://pre-commit.com/): pre-commit is a tool that helps us au-
    tomate a sequence of short tasks (called pre-commit “hooks”) such as code for-
    matting. In particular, we always use the following hooks when submitting code
    to the main repository. Seehttps://github.com/vwxyzjn/cleanrl/blob/master/
    CONTRIBUTING.mdfor more information.

```
(a)pyupgrade(https://github.com/asottile/pyupgrade): pyupgrade upgrades
syntax for newer versions of the language.
(b)isort(https://github.com/PyCQA/isort): isort sorts imported dependencies
according to their type (e.g, standard library vs third-party library) and name.
(c) black(https://black.readthedocs.io/en/stable/): black enforces an uni-
form code style across the codebase.
(d)autoflake (https://github.com/PyCQA/autoflake): autoflake helps remove
unused imports and variables.
(e) codespell(https://github.com/codespell-project/codespell): codespell
helps avoid common incorrect spelling.
```
3. Docker(https://www.docker.com/): docker helps us package the code into a con-
    tainer which can be used to orchestrate training in a reproducible way.

```
(a)AWS Batch(https://aws.amazon.com/batch/): Amazon Web Services Batch
could leverage our built containers to run thousands experiments concurrently.
(b) We have built utilities to help package code into a container and submit to AWS
Batch using a few lines of command. In 2020 alone, the authors have run over
50,000+ hours of experiments using this workflow. Seehttps://docs.cleanrl.
dev/cloud/installation/for more documentation.
```

```
CleanRL: High-quality Single-file Implementations of DRL Algorithms
```
## Appendix E. W&B Editing Panel

```
A screenshot of the W&B panel that allows the the users to change smoothing weight, add
panels to show different metrics like losses, visualize the videos of the agents’ gameplay,
filter, group, sort, and search for desired experiments.
```
5. Group experiments by
attributes
3. Filter out experiments
2. Slider to check out videos
in different stages of training
1. Extra settings of the
charts on smoothing weight,
error bars, colors, and
others.
8. Search experiment by
name
6. Select run set from
projects (allow merging
runs from multiple
projects)
9. Runtime of a specific
experiment
4. Add other charts (e.g.
losses)
7. Sort experiments by
attributes

## Appendix F. Author Contributions

- Shengyi Huang and Rousslan Fernand Julien Dossaco-founded CleanRL and
    has led its overall development.
- Chang Yecontributed a prototype with Random Network Distillation (Burda et al.,
    2019).
- Shengyi Huang, Rousslan Fernand Julien Dossa, and Chang Yeare the main
    code reviewers and maintainers.
- Jeff Bragacontributed hundreds of hours of tracked experiments in Weights and
    Biases and submitted various codebase improvements.
- Dipam Chakrabortycontributed the Phasic Policy Gradient implementation.
- Kinal Mehtacontributed the Deep Q-learning implementation with JAX.
- Jo ̃ao G.M. Ara ́ujocontributed the Twin-Delayed Deep Deterministic Policy Gra-
    dient implementation with JAX.


```
Huang, Dossa, Ye, Braga, Chakraborty, Mehta, and Araujo ́
```
- Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, Jeff Braga,
    Dipam Chakraborty, Kinal Mehta, Jo ̃ao G.M. Ara ́ujowrote the paper.

## References

M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. The Arcade Learning Environment:
An Evaluation Platform for General Agents.Journal of Artificial Intelligence Research,
47:253–279, 2013.

Marc G. Bellemare, Will Dabney, and R ́emi Munos. A Distributional Perspective on Rein-
forcement Learning. In Doina Precup and Yee Whye Teh, editors,Proceedings of the 34th
International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-
11 August 2017, volume 70 ofProceedings of Machine Learning Research, pages 449–458.
PMLR, 2017. URLhttp://proceedings.mlr.press/v70/bellemare17a.html.

Lukas Biewald. Experiment tracking with weights and biases, 2020. URLhttps://www.
wandb.com/. Software available from wandb.com.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie
Tang, and Wojciech Zaremba. Openai gym, 2016.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random
network distillation. InInternational Conference on Learning Representations, 2019.
URLhttps://openreview.net/forum?id=H1lJJnR5Ym.

Karl Cobbe, Chris Hesse, Jacob Hilton, and John Schulman. Leveraging procedural gener-
ation to benchmark reinforcement learning. In Hal Daum ́e III and Aarti Singh, editors,
Proceedings of the 37th International Conference on Machine Learning, volume 119 of
Proceedings of Machine Learning Research, pages 2048–2056. PMLR, 13–18 Jul 2020.
URLhttps://proceedings.mlr.press/v119/cobbe20a.html.

Karl W Cobbe, Jacob Hilton, Oleg Klimov, and John Schulman. Phasic policy gradi-
ent. In Marina Meila and Tong Zhang, editors,Proceedings of the 38th International
Conference on Machine Learning, volume 139 ofProceedings of Machine Learning Re-
search, pages 2020–2027. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.
press/v139/cobbe21a.html.

Carlo D’Eramo, Davide Tateo, Andrea Bonarini, Marcello Restelli, and Jan Peters. Mush-
roomrl: Simplifying reinforcement learning research. Journal of Machine Learning Re-
search, 2020.

Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry
Rudolph, and Aleksander Madry. Implementation Matters in Deep RL: A Case Study on
PPO and TRPO. In8th International Conference on Learning Representations, ICLR
2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URLhttps:
//openreview.net/forum?id=r1etN1rtPB.


```
CleanRL: High-quality Single-file Implementations of DRL Algorithms
```
Scott Fujimoto, Herke van Hoof, and David Meger. Addressing Function Approximation
Error in Actor-Critic Methods. In Jennifer G. Dy and Andreas Krause, editors,Pro-
ceedings of the 35th International Conference on Machine Learning, ICML 2018, Stock-
holmsm ̈assan, Stockholm, Sweden, July 10-15, 2018, volume 80 ofProceedings of Ma-
chine Learning Research, pages 1582–1591. PMLR, 2018. URLhttp://proceedings.
mlr.press/v80/fujimoto18a.html.

Yasuhiro Fujita, Prabhat Nagarajan, Toshiki Kataoka, and Takahiro Ishikawa. Chainerrl:
A deep reinforcement learning library. Journal of Machine Learning Research, 22(77):
1–14, 2021.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft Actor-Critic:
Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.
In Jennifer G. Dy and Andreas Krause, editors,Proceedings of the 35th International
Conference on Machine Learning, ICML 2018, Stockholmsm ̈assan, Stockholm, Sweden,
July 10-15, 2018, volume 80 ofProceedings of Machine Learning Research, pages 1856–

1865. PMLR, 2018a. URLhttp://proceedings.mlr.press/v80/haarnoja18b.html.

Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan,
Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al. Soft actor-critic algo-
rithms and applications.arXiv preprint arXiv:1812.05905, 2018b.

Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dab-
ney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining
improvements in deep reinforcement learning. Proceedings of the AAAI Conference on
Artificial Intelligence, 32(1), Apr. 2018. URLhttps://ojs.aaai.org/index.php/AAAI/
article/view/11796.

Shengyi Huang and Santiago Onta ̃n ́on. A closer look at invalid action masking in policy
gradient algorithms. In Roman Bart ́ak, Fazel Keshtkar, and Michael Franklin, editors,
Proceedings of the Thirty-Fifth International Florida Artificial Intelligence Research So-
ciety Conference, FLAIRS 2022, Hutchinson Island, Jensen Beach, Florida, USA, May
15-18, 2022, 2022. doi: 10.32473/flairs.v35i.130584. URLhttps://doi.org/10.32473/
flairs.v35i.130584.

Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, and
Weixun Wang. The 37 implementation details of proximal policy optimization. In
ICLR Blog Track, 2022. URL https://iclr-blog-track.github.io/2022/03/25/
ppo-implementation-details/.

Eric Liang, Richard Liaw, Robert Nishihara, Philipp Moritz, Roy Fox, Ken Goldberg,
Joseph Gonzalez, Michael I. Jordan, and Ion Stoica. RLlib: Abstractions for Distributed
Reinforcement Learning. In Jennifer G. Dy and Andreas Krause, editors,Proceedings of
the 35th International Conference on Machine Learning, ICML 2018, Stockholmsm ̈assan,
Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning
Research, pages 3059–3068. PMLR, 2018. URLhttp://proceedings.mlr.press/v80/
liang18b.html.


```
Huang, Dossa, Ye, Braga, Chakraborty, Mehta, and Araujo ́
```
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval
Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement
learning. In Yoshua Bengio and Yann LeCun, editors,4th International Conference on
Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Confer-
ence Track Proceedings, 2016. URLhttp://arxiv.org/abs/1509.02971.

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles
Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State.
Isaac gym: High performance GPU based physics simulation for robot learning. InThirty-
fifth Conference on Neural Information Processing Systems Datasets and Benchmarks
Track (Round 2), 2021. URLhttps://openreview.net/forum?id=fgFBtYgJQX_.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou,
Daan Wierstra, and Martin Riedmiller. Playing Atari with Deep Reinforcement Learning.
ArXiv preprint, abs/1312.5602, 2013. URLhttps://arxiv.org/abs/1312.5602.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G.
Bellemare, Alex Graves, Martin A. Riedmiller, Andreas Fidjeland, Georg Ostrovski, Stig
Petersen, Charlie Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Ku-
maran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through
deep reinforcement learning.Nature, 518:529–533, 2015.

Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and
Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations.
Journal of Machine Learning Research, 22(268):1–8, 2021. URLhttp://jmlr.org/
papers/v22/20-1364.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
Policy Optimization Algorithms. ArXiv preprint, abs/1707.06347, 2017. URLhttps:
//arxiv.org/abs/1707.06347.

J Terry, Benjamin Black, Nathaniel Grammel, Mario Jayakumar, Ananth Hari, Ryan Sulli-
van, Luis S Santos, Clemens Dieffendahl, Caroline Horsch, Rodrigo Perez-Vicente, et al.
Pettingzoo: Gym for multi-agent reinforcement learning.Advances in Neural Information
Processing Systems, 34:15032–15043, 2021.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based
control. In2012 IEEE/RSJ International Conference on Intelligent Robots and Systems,
pages 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012.6386109.

Jiayi Weng, Huayu Chen, Dong Yan, Kaichao You, Alexis Duburcq, Minghao Zhang, Hang
Su, and Jun Zhu. Tianshou: A highly modularized deep reinforcement learning library.
arXiv preprint arXiv:2107.14171, 2021.

Jiayi Weng, Min Lin, Shengyi Huang, Bo Liu, Denys Makoviichuk, Viktor Makoviychuk,
Zichen Liu, Yufan Song, Ting Luo, Yukun Jiang, Zhongwen Xu, and Shuicheng Yan.
Envpool: A highly parallel reinforcement learning environment execution engine.arXiv
preprint arXiv:2206.10558, 2022.


