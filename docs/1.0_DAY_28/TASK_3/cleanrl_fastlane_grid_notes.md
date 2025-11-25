# CleanRL FastLane Grid Notes (Task 3)

## What we just verified
- **FastLane grid mode rendered 8 concurrent envs** for run `01KATF461V3681NE4E0T3TX33Z`; the GUI showed the expected 2×4 mosaic while the worker logs reported slots 0–7 contributing frames (`FastLane frame published` repeated for each slot).
- **Model artifact saved** at `/home/hamid/Desktop/Projects/GUI_BDI_RL/var/trainer/runs/01KATF461V3681NE4E0T3TX33Z/runs/Walker2d-v5__ppo_continuous_action__1__1763972364/ppo_continuous_action.cleanrl_model` (TensorBoard, logs, and videos live next to it).
- **Tyro warnings** about `wandb-entity` and `target-kl` are expected: the CleanRL scripts annotate those fields as `str`/`float` but keep `None` defaults. Tyro prints a warning yet still passes the values through; no action is required unless we want to fork CleanRL to add Optional typing.

## Why two run directories exist
Every accepted submission from the CleanRL train form becomes a trainer run ID:

| Run ID | Source config | Notes |
| --- | --- | --- |
| `01KATF461V3681NE4E0T3TX33Z` | `var/trainer/configs/config-01KATF461V3681NE4E0T3TX33Z.json` + `worker-01KATF461V3681NE4E0T3TX33Z-21.json` | The grid-mode training you just ran (Walker2d-v5). All artifacts live under `var/trainer/runs/01KATF461…`. |
| `01KAS9ZEXF2KG28RDD3SHQ4Z4J` | `var/trainer/configs/config-01KAS9ZEXF2KG28RDD3SHQ4Z4J.json` + `worker-01KAS9ZEXF2KG28RDD3SHQ4Z4J-12.json` | An earlier submission (either a dry-run or an abandoned launch). The trainer allocates the run ID as soon as you accept the form, so even cancelled/failed attempts leave a directory for auditing. |

The naming scheme is:

- `config-<RUN>.json` – full trainer payload (environment, telemetry paths, worker metadata).
- `worker-<RUN>-<slot>.json` – the per-worker CleanRL config handed to `cleanrl_worker.cli`. The slot suffix increments globally so you can correlate to `logs/cleanrl.*` inside the run folder.
- `var/trainer/runs/<RUN>/runs/<CleanRL run name>` – the CleanRL-generated `runs/<env>__<algo>__<seed>__<timestamp>` folder where checkpoints & eval videos land.

If a run is obsolete you can archive/delete its `var/trainer/runs/<RUN>` folder once you are certain the trainer daemon is stopped.

## Launching the trained policy for evaluation today
The GUI now supports "📦 Load Trained Policy" for CleanRL via the policy form. To evaluate manually via CleanRL's own eval helper:

```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL
source .venv/bin/activate
# Packages installed in editable mode:
#   pip install -e .                           # gym-gui
#   pip install -e 3rd_party/cleanrl_worker    # cleanrl + cleanrl_worker
python - <<'PY'
from cleanrl.ppo_continuous_action import Agent, make_env
from cleanrl_utils.evals.ppo_eval import evaluate
MODEL = "var/trainer/runs/01KATF461V3681NE4E0T3TX33Z/runs/Walker2d-v5__ppo_continuous_action__1__1763972364/ppo_continuous_action.cleanrl_model"
print("Evaluating", MODEL)
evaluate(
    MODEL,
    make_env,
    env_id="Walker2d-v5",
    eval_episodes=5,
    run_name="Walker2d-v5__ppo_continuous_action__eval",
    Model=Agent,
    capture_video=True,
)
PY
```

- The snippet reuses CleanRL’s built-in evaluator so you get a short set of rollout returns and (if `capture_video=True`) mp4 files under `runs/<eval name>/`.
- To visualize behaviour inside the GUI, we need a proper policy form + worker presenter that marshals `load_model` into a trainer request. That work is still pending (Enable `supports_policy_load` for CleanRL, add a policy-selection dialog, and have the worker presenter build an “evaluation” config that points at your saved checkpoint).

### Why videos appear even when `capture_video=false`
CleanRL only looks at `capture_video` while the vectorized training envs are running. When `Save Model?` is enabled, the script always calls `cleanrl_utils.evals.ppo_eval.evaluate(...)` with `capture_video=True`, so evaluation rollouts generate MP4s under `videos/<run_name>-eval/` even if training skipped recording. This is expected for now. If we want to suppress those MP4s we need to (a) pass an explicit `eval_capture_video` toggle through the Extras map and (b) patch the vendored CleanRL scripts to respect it before running `evaluate`.

## Headless Training buttons (current behaviour)
- **Configure Agent…** and **Train Agent** both call `_on_agent_form_requested`, so right now they are aliases. “Train” should ideally reuse the last-saved config and submit immediately, but until we implement that caching layer the two buttons behave the same.
- **Load Trained Policy** is intentionally disabled for CleanRL because no policy form is registered. Enabling it requires the policy workflow described above.

### Suggested UX follow-ups
1. Persist the last submitted CleanRL config to disk and let “🤖 Train Agent” resubmit it without reopening the form.
2. Register a CleanRL policy form (`_factory.register_policy_form("cleanrl_worker", …)`) so the load button becomes actionable.
3. When policy evaluation is supported, mark `supports_policy_load=True` in `gym_gui/ui/workers/catalog.py` so the control panel enables the button.

## CleanRL policy-evaluation plan (UI + worker changes)

| Area | File(s) | Action |
| --- | --- | --- |
| UI dialog | `gym_gui/ui/widgets/cleanrl_policy_form.py` (new) | Same skeleton as `SpadeBdiPolicySelectionForm`, but smarter defaults: when the user picks a `.cleanrl_model`, the form reads the sibling CleanRL `config.json` (or trainer `config-<RUN>.json`) to auto-detect `env_id`, seed, algo, and telemetry settings. Environment-family/game combos start filled + disabled; the user can tick “Override environment” to change them. Telemetry/video-mode controls, grid limit, `eval_capture_video`, and seed override remain exposed so operators can tweak behavior. |
| Factory registration | `gym_gui/ui/widgets/cleanrl_train_form.py` | After registering the train form, call `_factory.register_policy_form("cleanrl_worker", CleanRlPolicyForm)` if no policy form exists. |
| Worker catalog | `gym_gui/ui/workers/catalog.py` | Once the policy form ships, flip `supports_policy_load=True` for CleanRL so Control Panel enables the “Load Trained Policy” button in agent-only mode. |
| Presenter | `gym_gui/ui/workers/cleanrl_presenter.py` (new) | Provide `build_train_request` + `build_policy_eval_request`. The presenter receives the auto-detected metadata (env/algo/seed) from the policy form, merges any overrides, and builds the trainer payload with extras `{"mode": "policy_eval", "policy_path": str(path), "eval_capture_video": checkbox, "fastlane_only": bool, "video_mode": ..., "grid_limit": ...}`. |
| Main window hookup | `gym_gui/ui/main_window.py` | `_on_trained_agent_requested` asks the presenter for a policy config then calls `_submit_training_config`. Update log statements to use `LOG_UI_MAINWINDOW_INFO` / `LOG_WORKER_POLICY_EVENT` so runs are traceable. |
| Worker config schema | `3rd_party/cleanrl_worker/cleanrl_worker/config.py` | Allow `extras.mode` and optional `extras.policy_path`. Raise `ValueError` if evaluation mode lacks a readable checkpoint. Persist the auto-detected env/algo metadata so the runtime doesn't rely on manual input. |
| Runtime behaviour | `3rd_party/cleanrl_worker/cleanrl_worker/runtime.py` | Branch on `mode`. For `policy_eval`: spin up envs, attach FastLane, load the checkpoint into the CleanRL agent class, call the appropriate `cleanrl_utils.evals.*.evaluate` helper (it already streams to RecordVideo & prints episodic returns), and emit a lifecycle `run_completed` once evaluation finishes. No PPO training loop should execute. |
| CLI passthrough | `3rd_party/cleanrl_worker/cleanrl_worker/cli.py` | No major change—`--config … --extras …` already handles custom extras; just document that policy configs set `mode` + `policy_path`. |
| FastLane/HUD | `gym_gui/ui/fastlane_consumer.py` / `gym_gui/ui/widgets/fastlane_tab.py` | If `mode=policy_eval`, append “(Eval)” to HUD text and plot aggregated episodic returns in the Analytics tab so operators can differentiate training vs. replay. |
| Docs | this note + `docs/1.0_DAY_28/TASK_1/telemetry_standardization_plan.md` + future “How to load CleanRL policies” page | Keep naming conventions + workflow documented.

### Configuration naming
- Policy configs may be saved as `config-<RUN>-policy.json` to make grep easier; trainer daemons still copy the payload into `config-<RUN>.json` internally, so both naming patterns point to the same content.
- Worker extras keys: `mode`, `policy_path`, `eval_capture_video`, `video_mode`, `grid_limit`, `fastlane_only`, `seed_override`.
- FastLane telemetry continues to use the `runbus.*` / `fastlane.*` semantic conventions defined in `gym_gui/telemetry/semconv.py`.

### Logging strategy
- UI: emit `LOG_UI_TRAIN_FORM_INFO` when the policy form opens, `LOG_UI_MAINWINDOW_INFO` when a policy config is submitted, and add a dedicated `LOG_UI_POLICY_FORM_TRACE` / `LOG_UI_POLICY_FORM_ERROR` pair (new constants) if we want per-form diagnostics akin to the train form. Control Panel should keep using `LOG_UI_MAINWINDOW_EVENT` when auto-subscribing to the run.
- Worker: reuse `LOG_WORKER_POLICY_EVENT`, `LOG_WORKER_POLICY_WARNING`, and `LOG_WORKER_POLICY_ERROR` when the runtime enters evaluation mode, loads the checkpoint, or encounters I/O issues. If we need finer granularity, add `LOG_WORKER_POLICY_EVAL_STARTED` / `LOG_WORKER_POLICY_EVAL_COMPLETED` constants near the existing policy log block in `gym_gui/logging_config/log_constants.py`.
- Telemetry: FastLane consumer already logs `LOG_FASTLANE_CONNECTED` / `LOG_FASTLANE_UNAVAILABLE`; in addition we can emit `LOG_FASTLANE_QUEUE_DEPTH` snapshots tagged with `mode=eval` so operators can confirm the evaluation stream is healthy.

### Why this will work
1. **Reuse proven components:** The SPADE-BDI stack already ships a policy form and presenter (`gym_gui/ui/widgets/spade_bdi_policy_selection_form.py`). Copying that pattern ensures we keep the same wiring (Control Panel → form factory → presenter → trainer client) that the rest of the GUI relies on.
2. **Worker parity:** CleanRL workers already parse trainer configs and expose FastLane wrappers. By adding an evaluation mode inside `runtime.py` we keep the exact same CLI/process isolation the trainer expects—only the CleanRL inner loop changes.
3. **Telemetry contract:** FastLane + RunBus naming stays aligned with `gym_gui/telemetry/semconv.py`, so analytics dashboards and log filters keep working regardless of whether a run is training or evaluation.
4. **Lifecycle isolation:** Evaluation uses the same `MOSAIC_CLEANRL_WORKER/cli.py` entry point, so run directories (`var/trainer/runs/<run_id>`) still collect logs, TensorBoard, and videos; we simply distinguish the purpose in metadata and via the new log codes.

### Broader references
- `gym_gui/ui/widgets/spade_bdi_policy_selection_form.py` demonstrates how we build policy pickers today (family dropdown + file browser + seed fields). The CleanRL form can follow the same UX so users don’t have to learn a second workflow.
- `cleanrl_worker.cleanrl_utils.evals.*.evaluate` already handles loading checkpoints, running inference, and generating videos/metrics. We simply expose this capability through our worker runtime instead of requiring users to run it manually from the shell.

## Debugging breadcrumbs
- FastLane grid health now logs `LOG_FASTLANE_HEADER_INVALID` and `LOG_FASTLANE_FRAME_READ_ERROR` whenever the UI sees an empty/invalid shared-memory header or a read exception.
- Run lifecycles stay traceable through `var/trainer/configs/config-<RUN>.json` and the per-worker configs listed above; grep for the run ID in `var/logs/gym_gui.log` to see when the trainer accepted/submitted the job.

## Open questions tracked from this run
- Decide whether we want Tyro warning suppression (by patching the vendored CleanRL dataclasses to use `Optional[str]` for WandB fields) or simply document that they’re benign.
- Determine UX for showing which config created a given run (e.g., hyperlink from control panel history to `config-<RUN>.json`).
- Implement the CleanRL policy evaluation path so the GUI can load checkpoints without manual scripts.

## Latest integration work (PPO checkpoints + FastLane)

**Why:** `ppo.py` (discrete action script) doesn’t ship with `--save-model`, so the GUI couldn’t produce a checkpoint for Load Trained Policy. Adding flags directly to the vendor file caused merge headaches, so we wrapped the script instead of patching it.

**What changed:**

- Added `3rd_party/cleanrl_worker/cleanrl_worker/algorithms/ppo_with_save.py`, a verbatim copy of upstream PPO with three extra args: `save_model`, `upload_model`, `hf_entity`. When `save_model=True` it writes `runs/<run>/<exp>.cleanrl_model`, runs the built-in `cleanrl_utils.evals.ppo_eval.evaluate`, logs `eval/episodic_return` scalars, and optionally pushes to Hugging Face if `upload_model` is on.
- Pointed `DEFAULT_ALGO_REGISTRY['ppo']` at that wrapper so trainer payloads automatically launch the checkpoint-enabled version while other algorithms still use the stock modules.
- Regenerated `metadata/cleanrl/0.1.0/schemas.json` so the PPO schema advertises the wrapper module and exposes the new toggles in the train form. Algorithms that truly lack `save_model` remain untouched to avoid tyro failures.

**FastLane impact:** nothing changed in the telemetry plumbing—FastLane still streams from the PPO wrapper exactly as the original discrete script did, so "Fast lane: connected" appears as soon as the worker publishes frames. The only new artifact is the saved checkpoint and its evaluation metrics, which unlock the Load Trained Policy flow once we finish the policy form work above.

### CleanRL evaluation registry

- Added `3rd_party/cleanrl_worker/cleanrl_worker/eval_registry.py`. Each supported algorithm now has a declarative evaluation entry (agent class, env factory, CleanRL eval helper, and whether the factory accepts the extra `gamma` parameter). This removes the runtime signature introspection we were doing before.
- `runtime._run_policy_eval` consults the registry first; if an entry exists, the worker uses those adapters and falls back to the legacy import path only when necessary. That guarantees that Load Trained Policy launches the right env/agent pair and keeps FastLane online.
- Discrete PPO is the first client: its entry points to `ppo_with_save.Agent`, automatically adapts the four-argument `make_env` to the five-argument eval helper, and still reuses `cleanrl_utils.evals.ppo_eval.evaluate` for the actual rollouts.
- `ppo_continuous_action` was added to the registry (2025-11-26) to support MuJoCo continuous action environments like Walker2d-v5, Swimmer-v5, Ant-v5, etc.

## Nov 25 updates: policy evaluation loop + HUD

### UI knobs
- The CleanRL policy form now exposes **“Eval episodes per batch”** (defaults to 50) and a **“Repeat evaluation until stopped”** toggle. The spinner feeds `extras.eval_batch_size` (and `eval_episodes` for backwards compatibility); the checkbox sets `extras.eval_repeat` so operators can keep FastLane streaming until they manually cancel the run.
- Metadata emitted by the form includes these values (`ui.eval_batch_size`, `ui.eval_repeat`) so presenters/tests can assert the intended behaviour.

### Worker/runtime
- `_run_policy_eval` now routes through `run_batched_evaluation()` which repeatedly calls the registered CleanRL `evaluate()` helper.
  - Each batch writes TensorBoard scalars under `var/trainer/runs/<RUN_ID>/tensorboard_eval/` so the GUI can open a **TensorBoard-Agent-cleanrl_eval** tab without clobbering the original training logs.
  - Batch summaries (episodes, avg/min/max, std, wall-clock duration) are appended to `<run_dir>/eval_summary.json` and streamed via `LifecycleEmitter` heartbeats.
  - The analytics manifest is updated after every batch to reflect the new returns + tensorboard directory, which means the TensorBoard tab can load even while the evaluation job is still running.
- Extras returned in `RuntimeSummary` now surface `tensorboard_dir`, `eval_batch_size`, and `eval_repeat` so downstream tooling (tests, docs, inspectors) can reason about evaluation runs.

### FastLane HUD
- `FastLaneTab` watches each run’s `eval_summary.json`. When `run_mode=policy_eval` it appends a one-line status (e.g., `eval batch 3 | episodes=50 avg=112.3 min=90.1 max=142.6`) under the live reward/return HUD so you can see aggregate results while the frames play.
- New log constants `LOG_UI_FASTLANE_EVAL_SUMMARY_UPDATE` / `LOG_UI_FASTLANE_EVAL_SUMMARY_WARNING` record when the HUD picks up a new summary or fails to parse the file (e.g., during partial writes).

### Practical workflow
1. Open **Load CleanRL Policy…**, select a checkpoint, set *Eval episodes per batch* (e.g., 75) and optionally check *Repeat evaluation* if you want FastLane to loop indefinitely until you cancel the run.
2. Submit the dialog – the trainer writes a policy-eval config, launches the worker, and immediately creates:
   - `var/trainer/runs/<RUN_ID>/tensorboard_eval/` (TB scalars per batch),
   - `var/trainer/runs/<RUN_ID>/eval_summary.json` (latest HUD stats),
   - `var/trainer/runs/<RUN_ID>/videos/<RUN_ID>-eval/` (if “Capture evaluation video” is enabled).
3. In the GUI you’ll see both the TensorBoard tab (pointing at `tensorboard_eval`) and the FastLane tab with the new HUD overlay showing the rolling summary from `eval_summary.json`.

If you leave **Repeat evaluation** enabled, the worker keeps running batches until you cancel the run from the GUI or CLI. The manifest and HUD update after every batch so you still get telemetry even though the run never reaches “completed”.
