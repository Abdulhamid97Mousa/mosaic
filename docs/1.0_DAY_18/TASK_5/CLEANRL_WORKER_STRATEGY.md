# CleanRL Worker Strategy — Dual Path Alignment

## Actor Roster and Lane Separation

We now treat the GUI’s actor selector as the routing switch for telemetry vs. analytics lanes. The roster must read:

- **Human-Only** — stays on the telemetry path (manual play, relies on live tab).
- **Spade BDI Worker** — renamed from “BDI-Q-Agent”; continues to publish full RunBus telemetry.
- **LLM Multi-Step Worker** — hybrid lane (limited telemetry for prompts plus background analytics).
- **CleanRL Worker** — new analytics-only lane; the Live-Agent tab must remain disabled, with TensorBoard/W&B tabs surfaced instead.

Action items:

- Update widget enums, presenter wiring, and trainer dispatcher mappings from the legacy `BDI-Q-Agent` label to `Spade BDI Worker`.
- Add `CleanRL Worker` to the selector with logic that bypasses Live tab setup and jumps straight to analytics manifests.
- Make the lane decision explicit in the presenter: telemetry workers call `bootstrap_live_tabs()`, analytics workers call `bootstrap_analytics_tabs()`.
- Audit copy/docs so the worker is consistently called `spade_bdi_rl_worker` (not `spade_bdi_rl`).

## 1. Why CleanRL Needs a Different Lane

CleanRL optimises for fast experimentation and post-run analytics. Its core scripts stream metrics to TensorBoard, Weights & Biases, and Optuna, but they do not emit the rich per-step telemetry that our Live-Agent tabs expect. The project README reinforces this single-file, analytics-heavy workflow (`cleanrl/README.md`). Trying to shoehorn CleanRL runs into the Live Telemetry pipeline would add overhead and duplicate what CleanRL already excels at. Instead, we should honour the **dual-path architecture**:

- **Telemetry Path:** Real-time UI, SQLite step/episode storage, replay features (best for SPADE-BDI workers and other agents instrumented for live playback).
- **Analytics Path:** Post-training dashboards, scalar metrics, hyperparameter sweeps (the native CleanRL experience).

Design goal: launch CleanRL workers from gym_gui while keeping their metrics on the analytics path, then surface the resulting logs inside dedicated analytics tabs.

---

## 2. CleanRL Data Flow Snapshot

| Source | Format | Update cadence | Current consumer |
|--------|--------|----------------|------------------|
| TensorBoard event files (`events.out.tfevents.*`) | Scalars, histograms, optional videos | Buffered during run, read post-run | TensorBoard UI |
| Weights & Biases (`wandb/` dir or remote project) | Scalars, configs, media artifacts | Near real time via W&B agent | W&B dashboard |
| Optuna study DB (`optuna.db`) | Trial params and objectives | Updated at trial boundaries | `cleanrl_utils.tuner` CLI |
| Checkpoints (`runs/<exp>/checkpoints`) | PyTorch weights | On schedule (e.g., every N steps) | Manual reloads |

None of these assets need to enter the RunBus stream. We only require run lifecycle events (started, finished, failed) so the GUI can manage worker state.

Legacy worker profiling shows why keeping telemetry lightweight matters: CPU-only FrozenLake loops finish 5 000 episodes in under a second (`spadeBDI_RL/docs/LEGACY_TRAINING_BENCHMARKS.md`), while the JAX variant saturates a 16 GB GPU for the same workload (`spadeBDI_RL/docs/JAX_GPU_PROFILING.md`). CleanRL’s analytics-first path maps to the latter scenario—heavy artifact generation but not stepwise UI streaming.

---

## 3. Proposed Integration Model

1. **Launcher shim:** Treat each CleanRL script as a managed worker. The GUI collects CLI arguments (env id, seed, total timesteps, CUDA flag, W&B toggles) and spawns the script via the trainer daemon. Only lifecycle events go through RunBus (e.g., `RunLifecycleEvent` with status and stdout tail).
2. **Log cataloguing:** Upon run completion (or periodically for long runs), the daemon registers artifact paths (TensorBoard dir, W&B run ID, Optuna DB, checkpoints) in run metadata stored alongside existing telemetry manifests.
3. **Analytics tabs:** The GUI offers dedicated views:
   - TensorBoard tab embedding the cleaned event directory.
   - W&B tab showing the hosted dashboard or a summary if offline.
   - Trial browser reading Optuna SQLite to list best hyperparameters.
4. **Replay compatibility:** Optional. If CleanRL captures videos (`--capture-video`), expose them through the replay tab by referencing the artifact path; otherwise leave the Live tab empty for these runs.

This approach respects CleanRL’s strengths and keeps live telemetry lightweight for agents designed for it.

---

## 4. Implementation Steps

| Phase | Tasks | Owners |
|-------|-------|--------|
| **A. Worker orchestration** | Extend trainer dispatcher to accept `cleanrl/<algo>.py` entrypoints, map GUI form fields to tyro flags, stream stdout/stderr into lifecycle logs. | Trainer daemon |
| **B. Artifact registration** | After process exit, walk the run directory to enumerate TensorBoard events, W&B run metadata, Optuna DB, checkpoints. Persist a manifest in run metadata JSON and tag storage paths. | Trainer services |
| **C. GUI analytics tabs** | Add TensorBoard and W&B tabs (webview or embedded server). Provide Optuna trial table with sorting by objective. Show fallback message in Live tab explaining that this worker relies on analytics path. | UI team |
| **D. Metadata sync** | Extend run archive/export workflow to include analytics assets (compress TB events, link W&B run). Ensure deletion routines clean up these directories. | Storage/UX |
| **E. Documentation & toggles** | Update training form help text to clarify CleanRL runs skip live telemetry. Offer checkbox to force minimal RunBus telemetry (e.g., periodic scalar summary) if future features require it. | Docs + UI |
| **F. Actor roster rename** | Replace legacy `BDI-Q-Agent` strings with `Spade BDI Worker` and wire `CleanRL Worker` to analytics-only routines. | UI + Trainer |

---

## 5. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large TensorBoard directories slow UI embedding | High | Launch an internal TB server only when tab opened; stream gzip archives when exporting. |
| Offline W&B runs cannot render dashboards | Medium | Provide summary view (JSON metrics table) and link to local artifacts; prompt for API key when available. |
| Optuna DB contention during live sweeps | Low | Mount Optuna DB read-only in GUI; rely on CleanRL CLI for write access. |
| Operator confusion about missing Live tab data | Medium | Display banner in Live tab: “CleanRL worker streams metrics via TensorBoard/W&B; live telemetry not available.” |
| GPU-bound jobs starve other workers | Medium | Reference JAX profiling guidance (`spadeBDI_RL/docs/JAX_GPU_PROFILING.md`) and default CleanRL worker to CPU unless CUDA resources are explicitly reserved. |

---

## 6. Next Actions

1. Scaffold CleanRL worker launcher and confirm lifecycle events appear in RunBus logs.
2. Define artifact manifest schema (`tensorboard_dir`, `wandb_run_id`, `optuna_db_path`, `checkpoints`).
3. Prototype TensorBoard webview integration pointing at a static run directory.
4. Document the workflow in Day 18 Task 5 log and update training form tooltips.
5. Cross-check run manifests against `spadeBDI_RL/REALITY_CHECK.md` throughput expectations so CleanRL launches expose the same capacity guardrails.

---

## 7. Dependency Alignment & Packaging

**Implemented Solution:** Modular requirements architecture (Day 18, Task 5)

### Directory Structure

```text
requirements/
├── base.txt              # Core GUI + shared infrastructure (PyQt6, grpcio, gymnasium)
├── spade_bdi_worker.txt  # SPADE-BDI dependencies (spade, agentspeak, slixmpp)
└── cleanrl_worker.txt    # CleanRL dependencies (torch, tensorboard, wandb, tyro)
```

### Installation Paths

- **GUI only:** `pip install -r requirements.txt` (references `requirements/base.txt`)
- **SPADE-BDI worker:** `pip install -r requirements/spade_bdi_worker.txt` (includes base + SPADE libs)
- **CleanRL worker:** `pip install -r requirements/cleanrl_worker.txt` (includes base + CleanRL libs)

### Implementation Notes

- Root `requirements.txt` now references `requirements/base.txt` as entry point, keeping GUI lean (~15 deps vs ~200+)
- Worker-specific bundles use `-r base.txt` to inherit shared dependencies
- CleanRL bundle references `cleanrl/requirements/requirements.txt` for full compatibility
- The `cleanrl_worker` directory currently contains scaffolding only. First engineering step: sync the real contents from the `cleanrl` submodule and prune training assets we do not redistribute.
- Keep worker dependencies isolated: either ship a dedicated virtualenv for the trainer daemon or vendor a `pip install -r requirements/cleanrl_worker.txt` step during worker bootstrap.
- Respect CleanRL's optional extras (`requirements-atari.txt`, `requirements-jax.txt`, etc.) by exposing checkboxes in the GUI form so the trainer can decide which extras to install before spawning runs.
- Ensure trainer manifests capture which requirement sets were applied so we can diff future updates.

### Future Enhancements

- Integrate dependency bundle selection into trainer dispatcher metadata schema
- Add pre-flight dependency checks before worker launch
- Cache installed bundles in `var/trainer/venvs/<bundle_name>` to avoid reinstalls

## 8. Supporting Evidence

- `cleanrl/README.md`: Confirms CleanRL’s single-file philosophy, built-in TensorBoard/W&B/Optuna instrumentation, and lack of GUI telemetry.
- `spadeBDI_RL/docs/LEGACY_TRAINING_BENCHMARKS.md`: Benchmarks the legacy NumPy worker, highlighting CPU-only efficiency that favours lightweight lifecycle events over dense RunBus streams.
- `spadeBDI_RL/docs/JAX_GPU_PROFILING.md`: Details the GPU profiling checklist and metrics (≈95 % utilisation, 15.8 GB VRAM) that motivate GPU safety toggles for CleanRL workers.
- `spadeBDI_RL/REALITY_CHECK.md`: Establishes current worker throughput (≈4 200 episodes/s) and resource envelopes that our CleanRL manifests should respect.
- `docs/CLEANRL_VS_JUMANJI.md`: Captures architectural differences between CleanRL analytics and gym_gui live telemetry, reinforcing the dual-path design and upcoming analytics tab integration.
