# MOSAIC Before and After Unreal-MAP

This note captures a contrarian assessment of the MOSAIC research claims and the concrete gaps that prevent the current paper from matching its narrative. It also outlines how the Unreal-MAP integration can close those gaps if executed rigorously.

## Summary Snapshot

| Lens | Today (pre-Unreal) | What Unreal-MAP enables |
| --- | --- | --- |
| **Scope of evidence** | Control plane already supports Atari, Box2D, MuJoCo, MiniGrid, and ViZDoom adapters, but only ToyText RL (FrozenLake, CliffWalking, Taxi) has been exercised end-to-end in the paper’s quantitative section. | High-fidelity MARL scenarios (Metal Clash, navigation, etc.) with heterogeneous agents and real sim2real assets. |
| **Heterogeneity** | Architecture discusses RL + BDI + LLM, but empirical section only runs CleanRL baselines. | HMAP lets us deploy multiple algorithm families per team; Unreal maps provide actual multi-team tasks to showcase BDI/LLM oversight. |
| **Multi-agent** | UI plumbing for per-agent tabs landed after Day-15 (game IDs and frame refs now flow), but the paper still lacks a published MARL evaluation exercising those views. | Unreal-MAP delivers multi-team tasks out of the box, so we can finally showcase the multi-agent UI with real data and stress fast-lane rendering + resource governance. |
| **Novelty story** | “GUI-first, IPC-governed lab” — solid engineering but incremental vs. RLCoach/Simion Zoo/RLlib dashboards. | “Process-supervised MARL lab that mixes UE-grade simulations with heterogenous workers” — genuinely new if we deliver the case study. |

## Good (Already in Shape)

1. **Control-plane architecture is solid.** Qt6 GUI, versioned IPC, heartbeats, deterministic replay, resource knobs—this is thoughtful engineering that stands up in software-review venues. The adapter catalog already covers MiniGrid, Atari, Box2D, MuJoCo, and ViZDoom even though the published experiments only benchmark ToyText.
2. **Telemetry pipeline works.** JSONL → gRPC → SQLite with WAL, embedded TensorBoard/W&B views, structured logs. Once multi-agent metadata lands, this becomes a competitive observability story.
3. **Documentation depth.** The research proposal already has polished diagrams, tables, and references; once the story matches the data, packaging for a software-style publication or demo track is straightforward.

## Bad (Gaps Blocking the Claims)

1. **No MARL evidence.** The current paper never shows multiple agents, teams, or competitive/cooperative settings. Yet the narrative leans heavily on heterogeneous, multi-agent orchestration.
2. **Heterogeneous worker story is aspirational.** BDI and LLM workers are described but not benchmarked. All quantitative tables involve CleanRL on ToyText tasks.
3. **Theory vs. practice mismatch.** The FSM/safety/liveness write-up is effectively protocol documentation. Reviewers will call this “math dressing” unless we leverage it to analyze a real heterogeneous experiment (e.g., deterministically replaying an Unreal battle).
4. **UI wiring remains single-agent centric.** Even though `game_id`/`frame_ref` flow, most Qt widgets/presenters (TensorBoard tabs, W&B tabs, worker catalogs) assume one agent per tab (naming like `WAND-Agent-{agent_id}`). File layout mirrors this bias (`ui/environments/single_agent_env`, a shallow `multi_agent_env` stub). Until we refactor tab factories, presenters, and naming conventions, we cannot convincingly say the GUI is multi-agent ready.

## Ugly (Credibility Risks)

1. **Over-claiming reproducibility/heterogeneity.** Phrases like “transcends trade-offs between reproducibility, observability, and composability” ring hollow when the experiments are single-agent ToyText runs.
2. **“Zero instrumentation” needs nuance.** CleanRL stays untouched thanks to the sitecustomization harness in `3rd_party/cleanrl_worker`, but every new framework still needs a shim that emits JSONL and speaks the IPC verbs. We should document that lift instead of implying literally zero work.
3. **Venue mismatch.** The current manuscript aims for full research-paper prestige while delivering demo-level evidence. Without a flagship heterogeneous case study, reviewers will treat it as a software note regardless of venue.
4. **Identifier confusion.** Even with Unreal-MAP, MOSAIC still needs `game_id`, `agent_id`, and `frame_ref` because the control plane, database schema, and fast-lane consumer key everything off those identifiers. Unreal will supply rich telemetry, but we still must map UE teams/actors into our IDs for the GUI/SQLite/rerun tooling to work.

## What Unreal-MAP Brings — If We Execute

1. **Ready-made MARL scenarios.** The Unreal-MAP paper provides four scenario families (Metal Clash, navigation, etc.) with multiple teams, event systems, and map/task decoupling. Wiring even one of them through MOSAIC instantly upgrades the experimental section.
2. **HMAP as algorithm glue.** HMAP already supports MAPPO, HAPPO, QMIX, DQN, etc., with JSON configs per team. That aligns perfectly with MOSAIC’s “workers as unmodified subprocesses” story—if we wrap HMAP as a worker and stream telemetry through our pipeline.
3. **UE-grade rendering forces fast-lane completion.** Unreal frames are high-bandwidth. To display them we must finish the FastLane consumer/QML tab (per `fastlane_slowlane.md`). Doing so proves the GUI can handle realistic feeds, not just grid worlds.

## Must-Haves Before Claiming “After Unreal-MAP”

1. **Refactor the UI for real multi-agent tabs.** Establish naming conventions and directory structure for multi-agent widgets/presenters (e.g., `ui/environments/multi_agent_env`, per-agent TensorBoard/W&B panes) so adding a second team isn’t bespoke wiring.
2. **Exercise multi-agent telemetry at scale.** We now have `game_id`, `agent_id`, and `frame_ref` propagation; the missing piece is running a real MARL scenario to prove the per-agent tabs, database, and fast lane stay stable under multiple teams.
3. **Implement Unreal adapter + HMAP worker bridge.** Treat Unreal-MAP’s Gym-compatible Python interface as a MOSAIC adapter and HMAP as the worker. The worker publishes slow-lane telemetry; the adapter streams fast-lane frames/metrics.
4. **Produce a flagship case study.** Example: “Metal Clash 4v4 with MAPPO vs. rule-based defenders under BDI supervision.” Report orchestration overhead, fault recovery, and deterministic replay on that scenario. Show the GUI controlling pause/resume and capturing per-team telemetry.
5. **Document installation + reproducibility.** The new install section in `unreal-map-integration-assesment.md` is a start, but we need step-by-step scripts, hardware requirements, and sample configs checked into the repo.
6. **Take Unreal-MAP for a test drive.** Actually build/run `/home/hamid/Desktop/Projects/GUI_BDI_RL/unreal-map` using the upstream instructions so we understand its telemetry, IDs, and multi-agent idioms before designing the bridge.

## Positive Narrative Once Gaps Are Closed

> “MOSAIC turned from a ToyText orchestration demo into a UE-backed MARL lab. By integrating Unreal-MAP + HMAP through the existing IPC/control-plane, we now demonstrate multi-team reinforcement learning, deterministic replay across heterogeneous agents, and fast-lane rendering of photorealistic scenes, all supervised from the same GUI.”

Delivering that narrative requires the four must-haves above. Without them, the paper remains a well-engineered single-agent lab that over-claims heterogeneity. Unreal-MAP is the lever that can make the claims real—if we use it to ship the first genuine heterogeneous experiment instead of just referencing it aspirationally.
