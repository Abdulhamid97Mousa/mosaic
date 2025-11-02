# Fast Analytics Path — Worker-First Artifact Streaming

_This note defines the "fast" analytics lane that bypasses the Telemetry Live Path and Durable Path entirely. It captures the expectations for workers that push metrics straight into TensorBoard and Weights & Biases, and the GUI features required to surface those artifacts without relying on RunBus step events._

## 1. Context and Motivation

Day 18 Task 1 documented two telemetry lanes:

- **Live Path** — `TelemetryAsyncHub` consumes gRPC protos, normalises payloads, and drives Qt widgets in real time.
- **Durable Path** — RunBus events feed `TelemetryDBSink` and `TelemetrySQLiteStore` for replay and archival.

Analytics-focused workers (CleanRL, upcoming JAX pipelines) already generate rich dashboards through TensorBoard and Weights & Biases without emitting per-step telemetry. Historically we routed them through the live pipeline anyway, which created noise (empty Live tabs, redundant JSON churn). This Fast Analytics Path formalises the third lane: **worker-managed artifacts + analytics tabs only.**

## 2. Architecture Snapshot

| Lane | Primary Producer | Transport | GUI Consumer | Storage Contract |
|------|------------------|-----------|--------------|------------------|
| Live Path | Telemetry-enabled workers (SPADE-BDI) | gRPC → RunBus → Qt | Live telemetry tabs, replay widgets | `telemetry/steps` & `telemetry/episodes` tables |
| Durable Path | Trainer daemon | RunBus batched events | SQLite store, run archive export | `var/telemetry/*.db` |
| **Fast Analytics Path** | Worker process itself | Local artifact directories, hosted dashboards | TensorBoard tab, W&B tab, analytics manifests | `var/trainer/runs/<run_id>/artifacts.json` |

The Fast Path assumes the telemetry bus only needs lifecycle events (start, heartbeat, stop). All metric payloads live in worker-owned storage.

## 3. Data Flow Diagram

```mermaid
directive topMargin=12
flowchart LR
    subgraph Worker Host
        W[CleanRL / Custom Worker]
        TB[(TensorBoard events)]
        WB[(W&B run)]
        CK[(Checkpoints)]
    end
    subgraph Trainer Daemon
        D1[Launch & monitor process]
        D2[Lifecycle events on RunBus]
        D3[Artifact scanner]
    end
    subgraph GUI
        A1[Analytics Manifest]
        T[TensorBoard Tab]
        B[Weights & Biases Tab]
        L[Live Tab]
    end

    W --> TB
    W --> WB
    W --> CK
    D1 --> W
    D1 --> D2
    D1 --> D3
    D3 --> A1
    A1 --> T
    A1 --> B
    A1 -. disabled notice .-> L
    D2 -->|RunLifecycleEvent| GUI
```

Key characteristics:

- The trainer only surfaces lifecycle events (start, progress heartbeat, exit status).
- Artifact scanner registers TensorBoard directories, W&B IDs, checkpoint paths in an analytics manifest.
- GUI renders analytics tabs based on the manifest and displays a banner in the Live tab explaining that telemetry is disabled for this worker.

## 4. Worker Responsibilities

1. **Local layout** — Workers must emit artifacts under `var/trainer/runs/<run_id>/` with well-known subdirectories:
   - `tensorboard/` → event files (`events.out.tfevents.*`)
   - `wandb/` → offline W&B run data (`wandb/latest-run.json`, media)
   - `checkpoints/` → model snapshots (optional)
   - `stdout.log`, `stderr.log` → streaming logs for the GUI tail panel
2. **Manifest hints** — Optionally write `artifacts_manifest.json` containing:

   ```json
   {
     "tensorboard_dir": "tensorboard",
     "wandb_run_id": "user/project/run",
     "wandb_mode": "online",
     "checkpoints": ["checkpoints/ckpt_500k.pt"]
   }
   ```

   The trainer daemon merges this into its own manifest after validation.
3. **Heartbeat** — Emit lightweight lifecycle metrics (elapsed steps, last checkpoint) via stdout or a sidecar JSON file so the GUI can update progress without RunBus telemetry.

## 5. Trainer Daemon Adjustments

- **Artifact scanner** — After launch and on graceful exit, scan the worker run directory for TensorBoard and W&B assets, generating `artifacts.json` that the GUI can query.
- **Lifecycle events** — Use existing `RunLifecycleEvent` topics to report state transitions (`STARTED`, `RUNNING`, `FAILED`, `COMPLETED`) plus progress counters.
- **SSE bridge (optional)** — Offer a lightweight Server-Sent Events stream to push artifact updates to the GUI when long runs add checkpoints or new TensorBoard files.
- **Error surfacing** — Pipe `stderr` to the log tab and flag the run as `FAILED` if the worker exits non-zero, even though no telemetry records exist.

## 6. TensorBoard Tab Integration

1. **Launch strategy** — Run `tensorboard --logdir <tensorboard_dir>` inside a sandboxed process when the tab opens, binding to an ephemeral port.
2. **Embedding** — Use a Qt WebEngine view to display the TensorBoard UI, injecting the authenticated URL.
3. **Index caching** — Before launching TensorBoard, run `tensorboard data-server` to pre-index large event files and minimise startup latency.
4. **Offline fallback** — If the directory is missing, display actionable guidance ("Worker did not emit TensorBoard logs; check `var/trainer/runs/<run_id>/stdout.log`.").

## 7. Weights & Biases Tab Integration

- **Online runs** — If `wandb_run_id` and `WANDB_API_KEY` are available, embed the W&B dashboard via web view or open the run URL externally.
- **Offline runs** — Parse `wandb/latest-run.json` and summarise metrics (latest scalars, config) in a table; offer export button to sync when the user authenticates.
- **Credential guardrails** — Prompt the operator for API credentials only when needed; never store tokens in repo-managed files.

## 8. GUI Presenter Wiring

- Extend `SpadeBdiWorkerPresenter` (and the forthcoming analytics presenter) with `bootstrap_fast_path_tabs(manifest)` that:
  - Populates TensorBoard and W&B tabs based on the manifest.
  - Disables the Live Telemetry tab and displays the analytics banner.
  - Hooks lifecycle events to update progress labels, run duration, and checkpoint listings.
- Update the worker selection form so analytics workers (CleanRL, future JAX) automatically select the Fast Path routing.

## 9. Testing Strategy

| Scenario | Expected Outcome |
|----------|------------------|
| Worker emits TensorBoard + W&B artifacts | TensorBoard tab launches embedded server; W&B tab links to dashboard; Live tab shows "Analytics-only" notice. |
| Worker emits only TensorBoard | W&B tab displays warning; TensorBoard tab remains functional. |
| Worker emits no artifacts | Both analytics tabs present guidance; trainer still records lifecycle events. |
| Worker fails mid-run | Lifecycle status flips to FAILED; analytics tabs remain accessible for post-mortem. |

Automated tests should stub the manifest and ensure presenters toggle tabs correctly. Integration tests can launch a dummy worker script that writes minimal TensorBoard files every few seconds.

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| TensorBoard server lingering after run closure | Track process handle and terminate on tab close or app shutdown. |
| Large event files degrade UI responsiveness | Compress or prune events periodically; expose "archive" button to move old events out of the active directory. |
| W&B API outages | Fallback to offline JSON summary and queue sync when service recovers. |
| Operators expect live telemetry despite fast path | Display prominent notice in Live tab and training form tooltip clarifying analytics-only behaviour. |

## 11. Next Actions

1. Finalise artifact manifest schema and implement scanner in trainer daemon.
2. Wire analytics presenters to read manifest and initialise TensorBoard/W&B tabs.
3. Update training form copy to explain the three-lane architecture (Live, Durable, Fast).
4. Add automated test fixtures covering manifest permutations.
5. Document operational playbook for cleaning up large artifact directories (`var/trainer/runs/`).
