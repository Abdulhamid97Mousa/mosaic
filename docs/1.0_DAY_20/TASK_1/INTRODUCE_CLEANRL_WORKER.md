# Task 1 — Introduce CleanRL Worker (Single-Agent Mode)

## Objective

Land a CleanRL-based worker alongside the existing SPADE-BDI and headless flows so that the trainer daemon can execute any of CleanRL’s single-file algorithms, stream lifecycle telemetry through the MOSAIC FSM (`INIT → HSHK → RDY → EXEC ↔ PAUSE → FAULT → TERM`), and surface analytics artifacts (TensorBoard, Weights & Biases, Optuna) inside the GUI.

The implementation must keep the SPADE-BDI path intact, honour the JSON run-config contract enforced by `validate_train_run_config`, and respect the existing telemetry throttling/backpressure story.

## Integration Architecture

### Worker runtime (`cleanrl_worker/`)

- Add a first-class package entry point `cleanrl_worker/cli.py` that accepts `--config`, `--algo`, `--env-id`, resource flags, and analytics toggles. The CLI loads the JSON config written by the trainer, resolves the requested CleanRL script under `cleanrl_worker/cleanrl/`, and launches the training loop.
- Normalise runtime concerns inside `cleanrl_worker/runtime.py`:
  - translate config payloads into tyro/argparse arguments expected by each algorithm script,
  - set up telemetry/analytics emitters (TensorBoard, W&B, stdout lifecycle events),
  - publish a handshake to the trainer via `TrainerService.RegisterWorker` before the main loop starts, storing the returned `session_token`.
- Implement `cleanrl_worker/telemetry.py` to emit minimal JSONL lifecycle events (`run_started`, `heartbeat`, `run_completed`, `run_failed`). Per-step telemetry stays disabled; analytics assets remain the primary output channel.

### Trainer daemon (`gym_gui/services/trainer`)

- Extend `TrainerDispatcher._build_worker_command` to recognise `worker_meta.type == "cleanrl"`, generate the worker config file under `var/trainer/configs/`, and spawn `python -m cleanrl_worker.cli` with the correct flags.
- Ensure `TrainerService.SubmitRun` accepts the CleanRL-specific resource envelope (default CPU, optional CUDA). GPU reservations continue to use `GPUAllocator`.
- `TrainerService.RegisterWorker` already generates `session_token` values; store CleanRL capability metadata (`algo`, `env_id`, `extras`) so the GUI can render analytics tabs with context.
- Telemetry ingestion (`PublishRunSteps`, `PublishRunEpisodes`) stays gated. CleanRL workers emit lifecycle-only events, so the daemon promotes runs from `HSHK` to `EXEC` once the handshake completes and the first `run_started` lifecycle event is observed.

### GUI surface (`gym_gui/ui/workers`)

- Keep the worker catalog authoritative: `gym_gui/ui/workers/catalog.py` lists `CleanRL Worker` as analytics-first (`requires_live_telemetry=False`, `provides_fast_analytics=True`).
- Presenter work (Day 14 refactor) ensures analytics workers call `bootstrap_analytics_tabs()` instead of `bootstrap_live_tabs()`. Update the train dialog to surface CleanRL algorithm/extra selections and to document the absence of live telemetry.
- Live telemetry widgets remain disabled for CleanRL runs; instead, add TensorBoard/W&B/Optuna tabs that read manifests produced after each run.

#### CleanRL Train Form Decoupling Plan

The current SPADE-BDI dialog temporarily acts as the entry point for CleanRL submissions, which forces operators to skip XMPP credentials and leaks SPADE validation states into analytics-first runs. The CleanRL worker needs a dedicated form/presenter pairing so the UI can stay declarative.

- Build a dedicated `CleanRlTrainForm` under `gym_gui/ui/widgets/cleanrl_train_form.py` that inherits from a lean base form, surfaces algorithm/seed/extras controls, and never renders SPADE-BDI inputs.
- Create a `CleanRlWorkerPresenter` registered in `gym_gui/ui/presenters/workers/registry.py`; bind the form to it so policy serialization, metadata stitching, and analytics bootstrapping stay isolated from `SpadeBdiWorkerPresenter`.
- Update `MainWindow` and the train dialog factory to select the CleanRL form via the worker catalog entry, ensuring SPADE-only fields (JID, password, behaviour tree path) are removed from CleanRL submissions.
- Adjust unit and integration tests to target the new form and presenter, including fixtures that mimic analytics-only telemetry so regression suites stop relying on SPADE-specific mocks.

Document the split once the new widget ships so future workers can follow the same presenter registry flow.

## Supported Algorithm Matrix

The CleanRL repository shipped with this project already contains the single-file implementations required by the worker. The table below captures the minimum launch matrix we must support on Day 20; each row must have a working CLI integration, analytics output, and MOSAIC lifecycle tracking.

| Algorithm family | Script (under `cleanrl_worker/cleanrl/`) | Action space | Extra dependencies | Notes |
| --- | --- | --- | --- | --- |
| PPO (classic control) | `ppo.py` | Discrete/continuous | Base bundle (`requirements/cleanrl_worker.txt`) | Baseline smoke test (`CartPole-v1`, CPU) |
| PPO Atari | `ppo_atari.py` | Discrete image | `-r cleanrl_worker/requirements/requirements-atari.txt` | Requires AutoROM ROMs; verify headless launch |
| PPO Continuous | `ppo_continuous_action.py` | Continuous | Base bundle | Validates mujoco-style control when mujoco extras installed |
| DQN | `dqn.py` | Discrete | Base bundle | Ensures replay-buffer telemetry is relayed through analytics artifacts |
| C51 | `c51.py` | Discrete | Base bundle | Confirms categorical output is logged into TensorBoard |
| SAC | `sac_continuous_action.py` | Continuous | Base bundle | Requires `torch` + `mujoco` extras when targeting locomotion envs |
| TD3 | `td3_continuous_action.py` | Continuous | Base bundle | Smoke test with `Pendulum-v1` |
| PPG Procgen | `ppg_procgen.py` | Discrete | `-r cleanrl_worker/requirements/requirements-procgen.txt` | Validates heavier asset downloads; ensure var directory has disk space |
| RND + EnvPool | `ppo_rnd_envpool.py` | Discrete | `-r cleanrl_worker/requirements/requirements-envpool.txt` | Confirms high-throughput path; flag optional on CI |
| QDagger | `qdagger_dqn_atari_impalacnn.py` | Discrete | Atari + EnvPool extras | Exercise expert-trajectory loading via config manifest |

Document any algorithm not yet validated in `docs/1.0_DAY_20/TASK_3/` so the coverage matrix stays truthful.

## Worker Config Schema

CleanRL runs continue to use the standard JSON payload persisted by the trainer. The CleanRL worker adds an analytics manifest but otherwise reuses existing fields.

```json
{
   "run_name": "cleanrl-ppo-cartpole",
   "resources": {
      "gpus": { "requested": 0, "mandatory": 0 },
      "cpus": { "requested": 2 }
   },
   "metadata": {
      "worker": {
         "type": "cleanrl",
         "module": "cleanrl_worker.cli",
         "config": {
            "run_id": "${RUN_ID}",
            "algo": "ppo",
            "env_id": "CartPole-v1",
            "seed": 42,
            "total_timesteps": 50000,
            "extras": {
               "track_wandb": false,
               "tensorboard_dir": "${VAR_TENSORBOARD_DIR}/${RUN_ID}",
               "requirements": ["base"],
               "notes": "Day20 smoke test"
            }
         }
      }
   }
}
```

`TrainerDispatcher` writes the `metadata.worker.config` blob to `var/trainer/configs/worker-${RUN_ID}.json`. `cleanrl_worker.cli` accepts the path via `--config` and merges CLI overrides supplied in the GUI train form (e.g., algorithm dropdown, seed input, extra requirement toggles).

## Execution Lifecycle

1. **GUI submission** — The train dialog serialises the CleanRL config, including selected algorithm, extras, and analytics toggles. The config is validated by `gym_gui.services.trainer.validation.train_config.validate_train_run_config`.
2. **Dispatcher spawn** — `TrainerDispatcher._dispatch_run` reserves resources, writes the worker config, and spawns `python -m cleanrl_worker.cli --config <path> --grpc --grpc-target 127.0.0.1:50055`.
3. **Handshake** — TODO: wire `cleanrl_worker.cli` into the trainer client so the worker performs `RegisterWorker` before emitting lifecycle events. Until then, the telemetry proxy performs the handshake on behalf of analytics runs.
4. **Training loop** — The runtime imports the requested CleanRL script, builds tyro arguments, starts TensorBoard logging, and periodically emits lifecycle JSONL messages (`run_started`, `heartbeat`, `run_completed`/`run_failed`). These events feed the telemetry proxy so the daemon can update the FSM to `EXEC` and `TERM`.
5. **Analytics ingestion** — On process exit, the worker persists a manifest describing TensorBoard/W&B/Optuna artifacts under `var/trainer/runs/<run_id>/analytics.json`. The GUI analytics tabs use this manifest to render dashboards.
6. **Cleanup** — The dispatcher releases GPU/CPU slots, updates `RunRegistry`, and archives worker stdout/stderr under `var/logs/trainer/<run_id>/`.

## Implementation Checklist

1. **Runtime scaffolding**
   - [x] Create `cleanrl_worker/cli.py` with tyro-style argument parsing and config loading helpers.
   - [x] Implement `cleanrl_worker/runtime.py` with dynamic import logic for CleanRL scripts, lifecycle events, and analytics manifest writing.
   - [x] Add `cleanrl_worker/telemetry.py` and `cleanrl_worker/analytics.py` for lifecycle events and manifest construction.
2. **Trainer wiring**
   - [ ] Extend `_build_worker_command` to handle CleanRL metadata and requirement extras.
   - [ ] Update `_build_worker_env` to export `CLEANRL_RUN_DIR`, `WANDB_RUN_ID`, etc., inside the worker process.
   - [ ] Ensure dispatcher log streaming re-emits CleanRL structured logs via `log_constant` when available.
3. **GUI workflow**
   - [x] Build dedicated CleanRL train form widget (`gym_gui/ui/widgets/cleanrl_train_form.py`) that inherits from a shared base and exposes only algorithm, seed, and analytics toggles.
   - [x] Register a `CleanRlWorkerPresenter` in `gym_gui/ui/presenters/workers/registry.py` and ensure the CleanRL form resolves it while SPADE-BDI continues using `SpadeBdiWorkerPresenter`.
   - [x] Update the main window train dialog flow to choose the form via the worker catalog, removing SPADE-BDI-only inputs (JID, password, behaviour tree) from CleanRL submissions.
   - [x] Wire analytics tabs (TensorBoard, WAB) to manifests produced by the worker; see `docs/1.0_DAY_20/TASK_5/WAB_TAB.md` for details.
4. **Dependencies**
   - [ ] Verify `requirements/cleanrl_worker.txt` installs base dependencies plus optional extras (Atari, Procgen, EnvPool, JAX).
   - [ ] Provide helper scripts to bootstrap these extras (`python -m cleanrl_worker.tools.install_extras --set atari,procgen`).
5. **Documentation**
   - [ ] Refresh this plan once initial runs succeed, capturing validated algorithms in the coverage matrix.
   - [ ] Update root `README.md` and Day 18 strategy docs to reference the new worker path.

## Validation Strategy

| Test tier | Purpose | Command / location |
| --- | --- | --- |
| Unit | Config parsing, manifest building, telemetry handshake mocks | `pytest cleanrl_worker/tests/test_cli.py` (new) |
| Algo smoke tests | Ensure each algorithm in the coverage matrix launches, handshakes, and terminates on a short horizon | `python -m cleanrl_worker.cli --algo <name> --env-id <env> --total-timesteps 2048 --grpc-target 127.0.0.1:50055 --dry-run` |
| Integration | End-to-end trainer submission via existing registry/dispatcher | `pytest gym_gui/tests/test_trainer_cleanrl_worker.py` (to be authored) |
| GUI acceptance | Verify analytics tabs render after run completion, live telemetry stays disabled | Manual run documented in `docs/1.0_DAY_20/TASK_3/` |

Each smoke test must record:

- Submission payload (stored in `var/trainer/configs/`),
- Handshake log snippet proving `RegisterWorker` succeeded,
- Confirmation that analytics artifacts were generated under `var/trainer/runs/<run_id>/`,
- Final run status (`RunStatus.TERMINATED` with `outcome="completed"`).

## Risks & Mitigations

- **Dependency weight** — CleanRL extras (Atari, Procgen, EnvPool) increase install time. Mitigate by caching wheels under `var/cache/pip/` and allowing operators to opt-in via GUI toggles.
- **Long-running GPU jobs** — CleanRL scripts can monopolise GPUs. Honour `resources.gpus.requested` and expose a `max_wallclock_minutes` guard so the dispatcher terminates runaway jobs.
- **Analytics drift** — TensorBoard paths must remain stable. Persist manifests referencing `VAR_TENSORBOARD_DIR` and validate paths during GUI load, falling back to a “not available yet” banner.
- **FSM divergence** — Lifecycle-only telemetry may delay the `EXEC` transition. Ensure the worker sends an explicit `run_started` event immediately after handshake so the dispatcher advances the FSM reliably.
- **ROM/licence management** — Atari experiments require AutoROM ROM downloads. Provide a preflight checklist in `docs/1.0_DAY_20/TASK_3/` and guard smoke tests behind feature flags.

## Follow-up Work

1. Implement backpressure integration (`TelemetryAsyncHub` credit loop) so analytics-heavy runs can still signal throttling if stdout fills.
2. Add checkpoint/resume hooks once CleanRL scripts expose checkpoint CLI flags; document how they interact with MOSAIC `PAUSE`/`FAULT` transitions.
3. Investigate embedding a lightweight metric bridge (e.g., periodic scalar snapshots) so the GUI can display headline numbers even without full telemetry.
4. Publish operator runbooks covering dependency installation, ROM management, and analytics access.

Updates to this document should capture algorithm validation results, dependency changes, and any deviations from the coverage matrix above.
