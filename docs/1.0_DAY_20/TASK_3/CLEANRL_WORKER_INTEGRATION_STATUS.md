# Task 3 — CleanRL Worker Integration Status (2025-11-03)

## Objective

Document the current state of the `cleanrl_worker` integration, confirm which follow-up items remain outstanding, and outline the concrete engineering steps required to expose CleanRL as a first-class worker alongside the existing SPADE-BDI path.

## Repository Findings

- `cleanrl_worker/` now exposes a proper shim (config loader, runtime, telemetry, analytics) under `MOSAIC_CLEANRL_WORKER/`. Thin wrappers remain at the legacy import paths (`cleanrl_worker.cli`, etc.) for compatibility.
- `cleanrl_worker/pyproject.toml` has been replaced with a local build definition that installs the shim and vendored modules together.
- The upstream test suite under `cleanrl_worker/tests/` has been replaced by focused unit coverage (CLI, config parsing, runtime dry-run/execution). GUI-side analytics integration now lives in `gym_gui/tests`.
- `gym_gui/ui/main_window.py` delegates analytics-tab wiring to `AnalyticsTabManager`, which creates both TensorBoard and W&B tabs when the worker uploads manifests.

## Validation of Follow-up Items

| Item | Current Status | Evidence |
| --- | --- | --- |
| TelemetryAsyncHub credit/backpressure loop | **Implemented**. The drain loop initializes the credit manager, consumes credits, and emits STARVED/RESUMED control events before publishing to the RunBus. Regression tests document the behaviour. | `gym_gui/services/trainer/streams.py:543-706`; `gym_gui/tests/test_telemetry_credit_backpressure_gap.py` |
| CleanRL runtime & manifest wiring | **Implemented**. The shim now spawns CleanRL scripts via subprocess, emits lifecycle events/heartbeats, writes `analytics.json`, and surfaces TensorBoard/W&B tabs via `AnalyticsTabManager`. | `cleanrl_worker/runtime.py`, `cleanrl_worker/cli.py`, `gym_gui/ui/panels/analytics_tabs.py`, runtime/CLI pytest suites |
| Lightweight metric bridge for analytics-first workers | **Missing**. The GUI analytics path has not been implemented, and the telemetry hub only emits STEP/EPISODE events. | No references to a metric bridge under `gym_gui/ui/widgets` or `cleanrl_worker/` |
| Operator runbooks (dependencies, ROM management, analytics access) | **Missing**. Documentation only records the intention in Task 1; there are no operator-facing runbooks yet. | `docs/1.0_DAY_20/TASK_1/INTRODUCE_CLEANRL_WORKER.md` (follow-up list), no runbook files under `docs/` |

## Restructure & Integration Plan

1. **Package Layout** — ✅
   - Shim modules (`cleanrl_worker/__init__.py`, `cli.py`, `runtime.py`, `telemetry.py`, `analytics.py`, `config.py`) now live alongside the vendored CleanRL sources.
   - Local `pyproject.toml` installs the shim while exposing `cleanrl`/`cleanrl_utils` for imports.

2. **Runtime & Telemetry Shim** — ✅
   - CLI loads trainer configs, applies overrides, and emits lifecycle JSONL.
   - Runtime spawns the requested CleanRL algorithm as a subprocess, emits heartbeats, captures logs, and writes `analytics.json` into `var/trainer/runs/<run_id>/`.

3. **Analytics Manifest** — ✅
   - `cleanrl_worker/analytics.py` assembles manifests; the runtime persists them; `AnalyticsTabManager` reads metadata to create TensorBoard/W&B tabs.

4. **Tests** — ✅ (unit coverage in place, integration suite still to come)
   - Targeted pytest modules cover CLI parsing, runtime execution (stubbed subprocess), and form/presenter wiring.
   - A dedicated trainer integration test remains on the backlog once the dispatcher flow is exercised end-to-end.

5. **Trainer & GUI Wiring** — ✅
   - Worker catalog, forms, presenters, and `AnalyticsTabManager` surface CleanRL runs through analytics tabs. Trainer promotes runs to `EXECUTING` when the first episode arrives, even without step telemetry.

6. **Documentation & Runbooks** — 🚧
   - Operator guidance (dependencies, ROM management, analytics access) is still pending.

7. **Validation** — ⏳
   - Smoke: `source .venv/bin/activate && pytest cleanrl_worker/tests`
   - GUI wiring: `source .venv/bin/activate && pytest gym_gui/tests/test_cleanrl_train_form.py gym_gui/tests/test_worker_presenter_and_tabs.py`
   - Integration test for dispatcher ↔ CleanRL worker remains to be authored.

## Immediate Next Actions

1. Design the lightweight metric bridge so analytics-first workers can stream headline numbers to the GUI without full telemetry.
2. Draft operator runbooks (dependencies, ROM/W&B access, troubleshooting) and link them from Task 1.
3. Scope checkpoint/resume hooks once CleanRL exposes stable CLI flags.

This document will be updated after each milestone with the commands executed and pytest results to maintain traceability.

## Update — Direct Handshake & Trainer Proxy Coverage (2025-11-03)

### Summary of Modifications

- Authored `gym_gui/tests/test_trainer_cleanrl_worker.py`, an integration-focused suite that stubs a CleanRL worker process and asserts:
  - The telemetry proxy issues a `RegisterWorker` request before streaming.
  - RunStep/RunEpisode JSONL payloads are forwarded into the gRPC client.
  - Analytics manifests spawn TensorBoard and W&B tabs via `AnalyticsTabManager`.
- Enhanced `cleanrl_worker/MOSAIC_CLEANRL_WORKER/runtime.py` so the CleanRL worker performs its own gRPC `RegisterWorker` handshake (schema metadata, capability flags, session token retention) before launching the algorithm subprocess.
- Extended `cleanrl_worker/tests/test_runtime.py` with a handshake regression test that validates schema defaults and session token capture while keeping the existing dry-run and subprocess stubs intact.

### Updated Roadmap

- ✅ **Complete:** End-to-end trainer proxy test (`gym_gui/tests/test_trainer_cleanrl_worker.py`).
- ✅ **Complete:** Worker-initiated `RegisterWorker` handshake in `CleanRLWorkerRuntime`.
- 🚧 **Pending:** Lightweight metric bridge, operator runbooks, checkpoint/resume hooks.

### Validation (2025-11-03)

- `source .venv/bin/activate && pytest gym_gui/tests/test_trainer_cleanrl_worker.py` → ✅ 2 passed (proxy handshake + analytics tab coverage).
- `source .venv/bin/activate && pytest cleanrl_worker/tests/test_runtime.py` → ✅ 5 passed (runtime handshake, dry-run summary, manifest write).
- `source .venv/bin/activate && pytest cleanrl_worker/tests` → ✅ 9 passed overall (config parsing, CLI, runtime).
- `source .venv/bin/activate && pytest gym_gui/tests/test_cleanrl_train_form.py gym_gui/tests/test_worker_presenter_and_tabs.py` → ✅ 31 passed (form payload, presenter registry integration, analytics tab wiring).

CleanRL runs now execute via the shim, emit lifecycle events/heartbeats, perform the MOSAIC handshake directly, and write manifests that the GUI renders through TensorBoard/W&B tabs.

## Update — MuJoCo Documentation & Catalog Hooks (2025-11-03)

### Summary of Modifications

- Authored `gym_gui/game_docs/Gymnasium/MuJuCo/__init__.py` with HTML blurbs for eleven MuJoCo benchmarks (Ant, HalfCheetah, Hopper, Walker2d, Humanoid, HumanoidStandup, Inverted{Pendulum,DoublePendulum}, Reacher, Pusher, Swimmer) plus a shared requirements footer covering `pip install gymnasium[mujoco]`, engine installation, and the absence of human controls.
- Wired the new snippets into `gym_gui/game_docs/__init__.py` and `gym_gui/game_docs/game_info.py`, ensuring the Game Info panel surfaces MuJoCo context alongside existing ToyText/Box2D/MiniGrid entries.
- Extended `gym_gui/core/enums.GameId` to include the MuJoCo suite, mapped them to the `EnvironmentFamily.MUJOCO`, defaulted render mode to RGB, and limited control modes to `AGENT_ONLY` so the UI avoids advertising unsupported human play.
- Updated the CleanRL train form (`gym_gui/ui/widgets/cleanrl_train_form.py`) so operators can pick MuJoCo environments directly from the dropdown without memorising IDs.

### Validation (2025-11-03)

- `source .venv/bin/activate && pytest gym_gui/tests/test_cleanrl_train_form.py` → ✅ 2 passed (form still serialises configs after expanding the environment list).
- `source .venv/bin/activate && pytest gym_gui/tests/test_worker_presenter_and_tabs.py` → ✅ 29 passed (analytics presenter registry unaffected by the new GameIds and docs linkage).

MuJoCo environments now appear in the CleanRL workflow with contextual documentation, while the enums/family maps keep idle-tick logic and agent-only constraints intact.

## Update — CLI Overrides & Dynamic Catalog (2025-11-03)

### Summary of Modifications

- Added a general CLI override path in `CleanRLWorkerRuntime.build_cleanrl_args`; any extra fields (e.g., `cuda`, `capture_video`, `wandb_project_name`) now flow directly into Tyro arguments. This allows GPU toggles and metadata updates without cracking open the runtime again.
- Backfilled unit coverage in `cleanrl_worker/tests/test_runtime.py` to assert that overrides produce `--cuda=false`, `--capture-video=true`, and other flags alongside the existing tensorboard/W&B knobs.
- Replaced the static environment dropdown in the CleanRL train form with a dynamic list sourced from `GameId` + family metadata. Newly added MuJoCo and classic-control IDs appear automatically, and operator defaults can still fall back to custom entries when needed.
- Extended `GameId` to cover CartPole, Acrobot, MountainCar, Atari (Pong/Breakout), and Procgen CoinRun/Maze so the catalog stays in sync with the worker’s supported benchmarks.
- Added an explicit GPU toggle to the CleanRL train form; it writes `extras["cuda"]` into the worker config so the runtime emits `--cuda=true/false`. The form honours the same field when auto-generating environment manifests.
- Introduced optional W&B override fields (project, entity, run name, API key) in the CleanRL train form. These hydrate CleanRL CLI flags and the worker environment (`WANDB_API_KEY`) so operators can switch projects or credential scopes per run without touching `.env`.
- Documented the clean pass-through of additional extras. The runtime now handles at least the following operational knobs without code changes:
  - `cuda` → `--cuda=<bool>`
  - `capture_video` → `--capture-video=<bool>`
  - `wandb_project_name` / `wandb_entity` → `--wandb-project-name=<value>` / `--wandb-entity=<value>`
  - `tensorboard_dir`, `track_wandb`, and structured `algo_params` (unchanged behaviour)
  - Any numeric/string/boolean sequence extras (e.g., `total_frames`, `num_envs`) surfaced from future UI updates

### Validation (2025-11-03)

- `source .venv/bin/activate && pytest cleanrl_worker/tests` → ✅ 4 passed (CLI summary, arg building with overrides, dry-run summary).
- `source .venv/bin/activate && pytest gym_gui/tests/test_cleanrl_train_form.py` → ✅ 2 passed (dynamic dropdown + config serialization).

Operators can now flip GPU usage, capture settings, project metadata, and W&B credentials via the CleanRL form, with all fields flowing through to the worker CLI and environment overrides.

Operators can now flip GPU usage, capture settings, and project names via extras, and the UI automatically reflects new `GameId` entries without manual widget edits.

## Update — Operator Guide & Flag Reference (2025-11-04)

### UI Controls → Runtime / CLI Mapping

| UI control | Trainer payload field(s) | CleanRL CLI flag(s) / env | Notes |
| --- | --- | --- | --- |
| Algorithm dropdown | `metadata.worker.module` (resolved via registry) | _module selection_ | `cleanrl_worker.runtime.DEFAULT_ALGO_REGISTRY` maps the label to the canonical module (e.g., `ppo` → `cleanrl.ppo`). |
| Environment dropdown/custom | `config.env_id` | `--env-id=<id>` | Catalog pulls from `GameId`; custom toggle lets operators paste non-catalog IDs. |
| Total Timesteps | `config.total_timesteps` | `--total-timesteps=<int>` | Default step of 2,048 for quick smoke tests. |
| Seed | `config.seed` (only when >0) | `--seed=<int>` | Leave at 0 to let CleanRL randomise. |
| Agent ID | `config.agent_id` + `metadata.ui.agent_id` + `extras.agent_id` | propagated as metadata | Keeps telemetry/run manifests consistent with SPADE naming. |
| Worker ID override | `config.worker_id` + `metadata.ui.worker_id` | propagated as metadata | Displayed in the run list and used during direct `RegisterWorker`. |
| GPU toggle | `extras.cuda` | `--cuda=true/false` | Default is on; disable when running on CPU-only hosts. |
| TensorBoard checkbox | `extras.tensorboard_dir="tensorboard"` | `--tensorboard-dir=<abs path>` | Runtime resolves the relative path inside `var/trainer/runs/<run_id>/tensorboard` and enables the GUI tab. |
| Track W&B checkbox | `extras.track_wandb=true` | `--track` + env `TRACK_WANDB=1` | Enables manifest entry for W&B and requests login-less API usage when a key is supplied. |
| W&B Project | `extras.wandb_project_name` | `--wandb-project-name=<value>` | Optional – falls back to CleanRL defaults if left blank. |
| W&B Entity | `extras.wandb_entity` | `--wandb-entity=<value>` | Set to `abdulhamid-m-mousa-beijing-institute-of-technology` for the current workspace. |
| W&B Run Name | `extras.wandb_run_name` | `--wandb-run-name=<value>` | Lets operators reuse the same run naming scheme across workers. |
| W&B API Key | `environment.WANDB_API_KEY` | exported before launch | Overrides `.env`; stored only in the trainer payload (not persisted to disk). |
| W&B Email | `extras.wandb_email` + `environment.WANDB_EMAIL` | exported before launch | Helps identify the operator account in wandb run metadata. |
| Notes | `extras.notes` | analytics manifest only | Rendered inside the Analytics panel for quick operator annotations. |
| Dry Run toggle | `metadata.worker.arguments += ["--dry-run", "--emit-summary"]` | `--dry-run --emit-summary` | Use this path to confirm CLI compatibility without launching the algorithm. |
| Algorithm Parameters (per algo) | `extras.algo_params` | `--<param>=<value>` | The runtime fans out dict entries into CLI flags; defaults come from `_ALGO_PARAM_SPECS`. |

Current algorithm parameter presets (see `gym_gui/ui/widgets/cleanrl_train_form.py`):

- `ppo` — `learning_rate`, `num_envs`, `num_steps`
- `ppo_atari` — `learning_rate`, `total_frames`
- `dqn` — `learning_rate`, `batch_size`, `buffer_size`

Any manual edits in the UI flow through to the CleanRL Tyro parser, so operators can tune horizon, learning rate, or replay sizes without rebuilding the worker.

### MuJoCo Coverage & Requirements

- **Environments exposed in the catalog:** Ant, HalfCheetah, Hopper, Walker2d, Humanoid, HumanoidStandup, InvertedPendulum, InvertedDoublePendulum, Reacher, Pusher, Swimmer (all `*-v5`).
- **Documentation:** curated blurbs live under `docs/1.0_DAY_19/TASK_2/mujoco_contrarian_analysis.md` and `gym_gui/game_docs/Gymnasium/MuJuCo/`; the Game Info panel now links to these entries.
- **Dependencies:** ensure `pip install gymnasium[mujoco]` has been executed inside `.venv`, MuJoCo engine binaries are installed, and set `MUJOCO_GL=egl` (already committed to `.env`).
- **Usage recommendation:** the GUI presents these environments in agent-only mode to avoid implying keyboard control; CleanRL worker remains the primary consumer.

### Weights & Biases Setup (Headless-Friendly)

1. Run `source .venv/bin/activate && wandb login --relogin --key <API_KEY>` once per operator account to populate `~/.netrc`. Use the same key that is stored in `.env` (`WANDB_API_KEY`).
2. In the CleanRL (or SPADE-BDI) train form, tick “Track Weights & Biases” and fill **Project**, **Entity**, optional **Run Name**, and (if necessary) override **API Key**/**Email**. Leaving the key blank defers to the global login.
3. When a run starts, the runtime exports `WANDB_API_KEY`, `WANDB_EMAIL`, and `TRACK_WANDB=1`. If the key is valid, the Analytics pane will materialise a `Weights & Biases` tab with the embedded web session (`wandb_artifact_tab.py`).
4. If the GUI reports `Invalid TrainRunConfig: 'artifacts' is a required property`, double-check that at least one analytics toggle (TensorBoard or W&B) was enabled; the form always emits the block now, so stale configs should be deleted from `var/trainer/configs/`.

### Launch Checklist for Operators

1. **Prepare dependencies** — `.venv/bin/python -m pip install -r requirements.txt` (ensure the optional `gymnasium[mujoco]` and `wandb` extras are present).
2. **Set environment** — keep `.env` up to date (`MUJOCO_GL=egl`, `WANDB_API_KEY`, optional `WANDB_EMAIL`). Source it before launching the GUI (`set -a; source .env; set +a`).
3. **Open the GUI** — run `python -m gym_gui` (or the project launcher script) and choose **CleanRL Worker** from the Workers dropdown.
4. **Configure the form** — pick the algorithm, environment, timesteps, and fill Agent/Worker IDs plus analytics toggles as described above. Adjust algorithm parameters in the “Algorithm Parameters” box as required by the experiment.
5. **Dry run first** — leave “Dry run only” checked to validate CLI arguments, then relaunch without dry run once verification succeeds.
6. **Monitor analytics** — after execution starts, TensorBoard artifacts appear under `var/trainer/runs/<run_id>/tensorboard`. The W&B tab will embed the project at `https://wandb.ai/<entity>/<project>/runs/<run>` once the SDK confirms authentication.
7. **Troubleshoot** — consult `var/trainer/runs/<run_id>/logs/cleanrl.stderr.log` for CleanRL errors, and `var/logs/trainer_daemon.log` for handshake or analytics manifest issues. Deadline errors in the GUI typically mean the worker blocked on W&B authentication; confirm step 1 above.

### Follow-on Work (Tracked)

- Headless W&B login: evaluate using `wandb.sdk.wandb_login.use_api_key` inside the worker runtime to remove the CLI dependency on `~/.netrc`.
- Lightweight metric bridge: still pending (see Immediate Next Actions above) so headline metrics can render even if TensorBoard/W&B are disabled.
- Checkpoint/resume hooks: waiting on upstream CleanRL CLI support; runtime already surfaces `supports_checkpoint` extras for when flags land.
