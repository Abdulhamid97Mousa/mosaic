# Unreal-MAP ⇄ MOSAIC Integration Assessment

## 1. Objective

- Capture the current state of MOSAIC (gym_gui) streaming infrastructure, especially the fast-lane shared-memory path, and how it enables high-fidelity external simulators.
- Summarize Unreal-MAP capabilities that are relevant to MOSAIC’s process-supervised RL lab.
- Lay down a concrete implementation plan with milestones, owners, and blockers before wiring Unreal-MAP tasks into MOSAIC.

## 2. How to Install Unreal-MAP

The upstream project lives under `/home/hamid/Desktop/Projects/GUI_BDI_RL/unreal-map`. Before wiring it into MOSAIC we need a reproducible setup flow. The upstream team documents two supported installation paths; both are captured here so future contributors can replicate the environment.

### 2.1 Professional / Source Build

1. **Build Unreal Engine from source.** Follow the official Epic guide: <https://docs.unrealengine.com/4.27/zh-CN/ProductionPipelines/DevelopmentSetup/BuildingUnrealEngine/>. This yields a source-built UE toolchain that Unreal-MAP (UHMP) expects.
2. **Clone the UHMP repo.** `git clone https://github.com/binary-husky/unreal-hmp.git` (store it alongside this repo, e.g., `/home/hamid/Desktop/Projects/GUI_BDI_RL/unreal-map`).
3. **Fetch large assets.** Run `python Please_Run_This_First_To_Fetch_Big_Files.py` inside the cloned repo to download the binaries Git cannot host.
4. **Bind the project to your source-built UE.** Right-click `UHMP.uproject`, choose “Switch Unreal Engine version…”, point it at the UE source build, then open the generated `UHMP.sln` in Visual Studio and compile.
5. **Launch the editor.** Double-click `UHMP.uproject` to open the Unreal Editor; verify sample scenes load.

> Steps 1 and 4 are the most error-prone. The upstream author recommends this walkthrough video (0:00–1:46 covers step 1; 1:46–end covers step 4): <https://ageasga-my.sharepoint.com/:v:/g/personal/fuqingxu_yiteam_tech/EawfqsV2jF5Nsv3KF7X1-woBH-VTvELL6FSRX4cIgUboLg?e=Vmp67E>.

### 2.2 Prebuilt Binary (hmp2g)

If you only need the compiled runtime, refer to `https://github.com/binary-husky/hmp2g/blob/master/ZDOCS/use_unreal_hmap.md`. That guide walks through downloading the packaged binaries, configuring paths, and launching UE scenes without compiling the engine yourself.

## 3. MOSAIC Ingestion State (2025-11-15)

### 3.1 Fast Lane (shared memory)

- Reference: `gym_gui/fastlane/fastlane_slowlane.md`, `gym_gui/fastlane/buffer.py`.
- Architecture already documented and implemented:
  - Workers/telemetry proxy publish RGB frames + HUD metrics into a single-writer/shared-memory ring via `FastLaneWriter` (SPSC, lossy, Disruptor-inspired).
  - GUI attaches `FastLaneReader` to the ring, consumes freshest frame without touching gRPC/SQLite.
  - Phase-A tasks remaining: `FastLaneConsumer` QObject, Qt Quick renderer shim, QML shell, FastLane tab wiring. (Plan enumerated in `fastlane_slowlane.md`.)
- Current integration: CleanRL worker writes to fast lane; diagnostic `[FASTLANE] …` logs confirm activation.

### 3.2 Slow Lane (telemetry)

- Existing gRPC → `TelemetryAsyncHub` → `TelemetryService` → `TelemetryDBSink` → SQLite path remains authoritative for replay and analytics.
- Multi-agent telemetry regression still open (missing `game_id`, `frame_ref` propagation). Reference: `docs/1.0_DAY_12/EXPLANATION_WHAT_CHANGED.md`.

## 4. Unreal-MAP / HMAP Capabilities Snapshot

- Source summary: `Unreal-Engine-Based-General_Platform_for_multi_agent_reinforcement_learning.md`.
- Key attributes:
  - UE-based five-layer architecture with Gym-compatible Python interface (maps, agents, events can be scripted without touching C++/Blueprints).
  - Decoupled tasks, maps, and teams; supports heterogeneous multi-agent scenarios and team-level reward shaping.
  - HMAP framework lets algorithms (rule-based, DQN/SAC, MAPPO/HAPPO, PyMARL2, HARL) control specific teams via JSON configuration.
  - Supports controllable time dilation, CPU/GPU mixes, cross-device rendering.

## 5. Integration Plan

| # | Milestone | Description | Owners / Touchpoints |
|---|-----------|-------------|-----------------------|
| 1 | **Document readiness** | Keep this assessment live; link fast-lane doc, telemetry regression doc, Unreal reference. | Docs team (Day 27) |
| 2 | **Finish FastLane UI** | Implement `FastLaneConsumer`, Qt Quick renderer, QML tab, and presenter wiring per Phase-A plan so shared-memory frames render in the GUI. | UI team (`gym_gui/ui/*`) |
| 3 | **Fix multi-agent telemetry gaps** | Enforce `game_id`, `agent_id`, `frame_ref` emission across workers, trainers, and UI to unblock multi-agent tabs. | Telemetry + worker owners |
| 4 | **Unreal adapter** | Build new `EnvironmentAdapter` subclass that instantiates Unreal-MAP tasks via Python interface, maps UE teams to MOSAIC agent IDs, exposes Gym-like `step/reset/render`. | Adapter team (`gym_gui/core/adapters`) |
| 5 | **HMAP worker bridge** | Package Unreal/HMAP training as a MOSAIC worker: start HMAP scenario, stream RunStep/RunEpisode protobufs (slow lane), publish frames to FastLaneWriter. | Worker integration (e.g., `cleanrl_worker` patterns) |
| 6 | **End-to-end validation** | Run demo scenario (e.g., Metal Clash) from MOSAIC GUI: confirm live fast-lane video, slow-lane telemetry persistence, replay, multi-agent tabs. | QA + docs |

## 6. Risks / Open Questions

- **Telemetry completeness:** Without game/agent IDs and frame references, dynamic tabs and replays will mislabel Unreal teams. Must close the Day-12 regression before milestone 4.
- **Resource budgeting:** Unreal frames are larger than Gym toy-text/Box2D; FastLaneConfig must reflect resolution, and Qt Quick rendering should downsample/limit FPS to keep Qt responsive.
- **Worker lifecycle:** Need clear handoff between MOSAIC trainer daemon and Unreal/HMAP processes (start/stop, crash recovery, run metadata). Align with existing `trainer_telemetry_proxy.py` patterns.
- **Cross-platform packaging:** Unreal tooling differs per OS; document host requirements (e.g., UE runtime on Windows/Linux, GPU expectations) before shipping adapters to users.

## 7. Next Actions (Week of 2025-11-17)

1. Land FastLane UI scaffolding (Phase-A tasks).
2. Patch telemetry schema enforcement + adapter frame_ref generation.
3. Draft Unreal adapter skeleton + worker CLI stub.
4. Update this doc with progress notes + blockers.
