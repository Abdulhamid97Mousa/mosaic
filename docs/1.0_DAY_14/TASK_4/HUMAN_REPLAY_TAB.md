# Human Replay Tab — Implementation Overview

**Date:** October 24, 2025  
**Scope:** Document how the Human Replay tab in the GUI is built, the modules involved, data flow, and key design decisions.

---

## 1. Purpose and Context

The **Human Replay Tab** lets operators review completed training sessions (human or agent) by loading step/episode telemetry from SQLite and presenting it through a Qt widget. It is part of the SPADE‑BDI tab bundle but can be reused for other workers.

Primary goals:

- Display episode history (rewards, step counts, success/failure flags)
- Allow per-episode playback in a table view
- Provide filtering/export hooks for analysis

---

## 2. Source Modules & Key Files

| Module | Responsibility |
| --- | --- |
| `gym_gui/ui/widgets/spade_bdi_rl_worker_tabs/agent_replay_tab.py` | Qt widget implementation (forms, tables, playback controls) |
| `gym_gui/ui/widgets/spade_bdi_rl_worker_tabs/factory.py` | Instantiates the replay tab along with other worker tabs |
| `gym_gui/ui/main_window.py` | Registers the replay tab via presenter factory and plugs it into the `RenderTabs` container |
| `gym_gui/services/telemetry/sqlite_store.py` | Provides `recent_episodes` and `episode_steps` queries used to populate data |
| `gym_gui/services/telemetry.py` | Flushes telemetry to SQLite so replay queries stay consistent |
| `spade_bdi_rl/core/runtime.py` | Emits episode rollup telemetry (reward, steps, metadata) that the replay tab consumes |

---

## 3. Widget Implementation

### Class: `AgentReplayTab`
**Location:** `gym_gui/ui/widgets/spade_bdi_rl_worker_tabs/agent_replay_tab.py`

#### 3.1. UI Layout
- Inherits from `QtWidgets.QWidget`
- Uses a vertical layout with:
  - Header stats (total episodes, last reward, success rate)
  - `QTableWidget` displaying episode metadata (episode id, reward, steps, status, **control_mode**)
  - Controls for filtering and exporting episodes
  - Optional detail pane for viewing step-level telemetry

#### 3.2. Key Methods
| Method | Purpose |
| --- | --- |
| `__init__(run_id, agent_id, parent)` | Builds UI elements, requests initial data from telemetry store |
| `_build_ui()` | Constructs layouts, table headers, and control buttons |
| `refresh()` | Pulls latest episode data (calls `TelemetrySQLiteStore.recent_episodes`) |
| `load_episode(episode_id)` | Loads per-step data for selected episode |
| `_populate_table(episodes)` | Fills `QTableWidget` with episode metadata |
| `_on_episode_selected(row)` | Handles table selection and triggers `load_episode` |
| `_export_episode(episode_id)` | (Optional) Exports selected episode to JSON/CSV *[future feature]* |

#### 3.3. Dependencies
- `TelemetrySQLiteStore` for durable episode data
- `RendererRegistry` (optional) if future versions display frame thumbnails
- `RunStreamBuffer` (indirectly) ensures in-memory buffer is capped until persisted

#### 3.4. Control Mode Column Integration
The replay tab now displays a **"Control Mode"** column showing the control mode active when each episode was recorded:
- **human_only** (blue) — Human player exclusively drove the environment
- **agent_only** (green) — Agent was autonomous
- **hybrid_turn_based** (purple) — Turn-based human/agent alternation
- **hybrid_human_agent** (orange) — Simultaneous human/agent control
- **multi_agent_coop** (teal) — Multi-agent cooperation
- **multi_agent_competitive** (red) — Multi-agent competition

This field is populated from `EpisodeRollup.metadata.control_mode` persisted in SQLite.

---

## 3.5. Telemetry Database Schema (EER View)

The Human Replay tab queries the telemetry SQLite database. Below is the normalized schema showing the key entities and their relationships:

```mermaid
erDiagram
    RUNS ||--o{ EPISODES : "1:many"
    RUNS ||--o{ STEPS : "1:many"
    EPISODES ||--o{ STEPS : "1:many"
    EPISODES ||--o{ METADATA_JSON : "1:1"
    STEPS ||--o{ RENDER_PAYLOAD_JSON : "0:1"

    RUNS {
        string run_id PK "SHA-256 digest"
        timestamp created_at "When run was created"
        string backend "e.g., spadeBDI_RL"
        int total_episodes "Count of episodes in run"
        text metadata_json "Run-level metadata"
    }

    EPISODES {
        string episode_id PK "Composite: run_id + episode_index"
        string run_id FK "Foreign key to RUNS"
        int episode_index "1-based episode number"
        int seed "Random seed for reproducibility"
        float total_reward "Cumulative reward"
        int num_steps "Total steps in episode"
        boolean terminated "Terminal state reached"
        boolean truncated "Time limit or other truncation"
        string control_mode "human_only|agent_only|hybrid_turn_based|..."
        string agent_id "ID of the controlling agent"
        string game_id "FrozenLake-v1|CliffWalking-v1|..."
        timestamp recorded_at "When episode was finalized"
    }

    STEPS {
        string step_id PK "Composite: run_id + episode_id + step_index"
        string run_id FK "Foreign key to RUNS"
        string episode_id FK "Foreign key to EPISODES"
        int step_index "0-based step within episode"
        string action_json "Serialized action"
        string observation_json "Serialized state/observation"
        float reward "Step reward"
        boolean terminated "Terminal state"
        boolean truncated "Truncation flag"
        string control_mode "Copied from episode for granular filtering"
        string policy_label "e.g., Q-learning (epsilon=0.1)"
        string backend "Execution backend"
        text render_hint_json "Hints for visualization"
        string frame_ref "Reference to frame storage"
        int seq_id "Global sequence number"
        timestamp recorded_at "When step was recorded"
    }

    METADATA_JSON {
        string episode_id FK "Foreign key to EPISODES"
        json payload "Dict with seed, episode_index, game_id, control_mode, run_id, policy_label, backend, worker_version"
    }

    RENDER_PAYLOAD_JSON {
        string step_id FK "Foreign key to STEPS"
        json payload "Grid/RGB array for visualization"
    }
```

**Key Observations:**

1. **Normalization**: Episodes and Steps are separate tables, with Steps referencing Episodes via foreign key.
2. **Control Mode Denormalization**: `control_mode` is stored in both `EPISODES` and `STEPS` for efficient filtering without joins.
3. **Metadata JSON**: Episode-level metadata (seed, game_id, control_mode, etc.) is stored as JSON for extensibility.
4. **Indices**: Primary indices on `run_id`, `episode_id`, `control_mode`, and timestamps to support fast queries.

---

## 4. Data Flow

```mermaid
sequenceDiagram
    participant Runtime as BDI Runtime
    participant DB as TelemetrySQLiteStore
    participant Factory as TabFactory
    participant Replay as AgentReplayTab
    participant MainWindow as RenderTabs container

    Runtime->>DB: record_episode(rollup)
    MainWindow->>Factory: create_tabs(run_id, agent_id, payload, parent)
    Factory-->>Replay: instantiate AgentReplayTab
    Replay->>DB: fetch recent_episodes(run_id)
    Replay->>DB: (on selection) episode_steps(run_id, episode_id)
    Replay-->>MainWindow: emit UI signals / updates
```

---

## 4.1. Human Mode → Replay Analysis Lifecycle

This diagram shows how a Human Mode session is configured through constants/enums, flows through the runtime, and is ultimately replayed:

```mermaid
graph TD
    A["gym_gui/core/enums.py<br/><b>ControlMode.HUMAN_ONLY</b>"] --> B["gym_gui/config/settings.py<br/>Settings.default_control_mode"]
    B --> C["gym_gui/ui/widgets/control_panel.py<br/>ControlPanelWidget<br/>1. Load default mode<br/>2. Update UI states<br/>3. Emit mode_selected signal"]
    
    C --> D["gym_gui/controllers/session.py<br/>SessionController<br/>1. Set _control_mode=HUMAN_ONLY<br/>2. Gate action acceptance<br/>3. _record_step for telemetry"]
    
    D --> E["gym_gui/controllers/human_input.py<br/>HumanInputController<br/>1. Capture keyboard input<br/>2. Map to action<br/>3. Call perform_human_action"]
    
    E --> F["gym_gui/services/telemetry.py<br/>TelemetryService<br/>1. Receive StepRecord<br/>2. Emit to RunBus<br/>3. Queue for DB persist"]
    
    F --> G["gym_gui/telemetry/db_sink.py<br/>TelemetryDBSink<br/>1. Batch queue payloads<br/>2. Serialize metadata<br/>3. Write to SQLite"]
    
    G --> H["var/telemetry/telemetry.sqlite<br/>EPISODES/STEPS tables<br/>control_mode = 'human_only'"]
    
    H --> I["gym_gui/services/trainer/registry.py<br/>RunRegistry<br/>Query: get recent episodes<br/>Filter by run_id, agent_id"]
    
    I --> J["gym_gui/ui/widgets/spade_bdi_rl_worker_tabs/<br/>agent_replay_tab.py<br/>AgentReplayTab<br/>1. Fetch episode list<br/>2. Display in QTableWidget<br/>3. Show control_mode column"]
    
    J --> K["User inspects:<br/>- Episode history<br/>- Control mode per episode<br/>- Step-by-step playback"]
    
    style A fill:#4A90E2
    style C fill:#F5A623
    style D fill:#7ED321
    style E fill:#BD10E0
    style F fill:#50E3C2
    style G fill:#B8E986
    style H fill:#FF6B6B
    style I fill:#4A90E2
    style J fill:#F5A623
    style K fill:#50E3C2
```

**Flow Explanation:**

1. **Config Phase** (blue): `ControlMode.HUMAN_ONLY` enum defined in `enums.py`; default loaded from `Settings`.
2. **UI Phase** (orange): Control panel initializes and persists mode preference.
3. **Runtime Phase** (green): Session controller gates actions and records all human moves.
4. **Input Phase** (purple): Human input controller maps keypresses to environment actions.
5. **Telemetry Phase** (cyan/teal): Events flow through TelemetryService → RunBus → DB sink.
6. **Persistence Phase** (red): Steps/episodes written to SQLite with `control_mode='human_only'`.
7. **Query Phase** (blue): Registry queries recent episodes from database.
8. **Replay Phase** (orange/cyan): AgentReplayTab fetches episodes and displays them with control mode column.

---

## 5. Integration via Presenter & Factory

1. `SpadeBdiWorkerPresenter.create_tabs(...)` calls `TabFactory.create_tabs`.
2. `TabFactory` instantiates `AgentOnlineTab`, `AgentReplayTab`, grid/raw/video tabs.
3. `MainWindow` registers each tab via `_create_agent_tabs_for(...)` and the `RenderTabs` container.
4. Replay tab sits alongside other SPADE tabs but is self-contained (it does not rely on live telemetry queues).

```mermaid
graph LR
    MW[MainWindow]
    REG[WorkerPresenterRegistry]
    PRES[SpadeBdiWorkerPresenter]
    FACT[TabFactory]
    REPLAY[AgentReplayTab]
    TSTORE[TelemetrySQLiteStore]

    MW --> REG --> PRES --> FACT --> REPLAY
    REPLAY --> TSTORE
```

---

## 6. Design Considerations

- **Separation of concerns**: The replay tab reads from persistent storage; it does not consume live telemetry queues. This makes it resilient even if the UI misses live events.
- **Bounded in-memory buffers**: `TelemetryService` and `LiveTelemetryController` cap buffers (via deque) to prevent the replay tab from receiving enormous payloads.
- **Episode metadata**: `BDITrainer` and `HeadlessTrainer` ensure episode rollups include reward, steps, termination flags, and metadata (`control_mode`, `agent_id`, `seed`). Replay tab uses these fields directly.
- **Extensibility**: Future workers can reuse the replay tab or provide their own presenter-specific variant.

---

## 7. Related Tests

- `test_phase3_bus_writer_split.py` ensures episodes are written before UI completes.
- `test_telemetry_reliability_fixes.py::TestBoundedBuffers` confirms recent episodes remain bounded before replay tab queries them.
- Manual QA: verifying the tab loads data after training runs.

---

## 8. Future Enhancements

- Add export functionality (CSV/JSON) from the replay tab.
- Provide playback slider/animation to step through episode steps visually.
- Allow filtering by reward or success status.
- Expose metadata through DTOs once the API layer is introduced.

---

## 9. Verification Checklist

- [x] Widget located at `agent_replay_tab.py`
- [x] Uses `TelemetrySQLiteStore.recent_episodes` / `episode_steps`
- [x] Instantiated via `TabFactory`
- [x] Registered by `SpadeBdiWorkerPresenter` / `MainWindow`
- [x] Receives episode metadata from BDI/Headless runtime
- [ ] Export functionality (future)

---

**Summary:** The Human Replay tab is a Qt widget configured via the worker presenter architecture. It retrieves episode data from SQLite, displays it in a table, and is independent of live telemetry streams. The tab is created alongside other SPADE-BDI tabs and relies on the TabFactory and presenter wiring to stay decoupled from the main window.


## 10. Notes on Verification

- Manual verification performed on October 24, 2025 by running a SPADE-BDI training session and opening the replay tab to confirm episode tables populate correctly.
- No automated UI tests exist yet for the replay tab; behaviour is exercised indirectly through telemetry/repository tests.
- Ensure telemetry is flushed (call `TelemetryService.recent_episodes()` after `flush()`) before opening the tab to avoid stale data.
