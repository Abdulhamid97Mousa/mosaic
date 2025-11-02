# Embedded TensorBoard Viewer Is Looking At The Wrong Run Folder

## What We Observed
- On 2025-11-01 at 23:16:44 the *TensorBoard-Agent-1* tab rendered the default “No dashboards are active for the current data set” warning even though training had just finished.
- The tab reported its log directory as `var/trainer/runs/frozenlake-v2-q-learning-20251101-151440/tensorboard`, and the embedded browser kept polling `http://127.0.0.1:6006/` without finding any scalars.
- Even after we pointed TensorBoard at the correct ULID directory via the CLI, the GUI still showed a blank orange shell because the absolute path never flowed back to the widget.

## What The Files Say
- The trainer actually wrote events under the ULID-based run directory:
  ```bash
  ls var/trainer/runs/01K8ZZR95R8B07568RNQQMSRVH/tensorboard
  # → events.out.tfevents.1762010084.hamidOnUbuntu.34400.0
  ```
- There is **no** directory at the slug path referenced by the UI:
  ```bash
  test -d var/trainer/runs/frozenlake-v2-q-learning-20251101-151440/tensorboard || echo "missing"
  # → missing
  ```
- The trainer config saved for this run still points the TensorBoard artifact at the slug path:
  ```bash
  jq '.metadata.artifacts.tensorboard.relative_path' \
    var/trainer/configs/config-01K8ZZR95R8B07568RNQQMSRVH.json
  # → "var/trainer/runs/frozenlake-v2-q-learning-20251101-151440/tensorboard"
  ```

## Root Cause
1. During submission, `gym_gui/services/trainer/config.validate_train_run_config` replaces the human-readable `run_name` with a ULID (`01K8ZZR95R8B07568RNQQMSRVH`) for `worker.config.run_id`, but it leaves `metadata.artifacts.tensorboard.relative_path` pointing at the old slug. The GUI trusts this path, so it opens a directory that never gets created.
2. When the worker starts, `spade_bdi_worker/core/tensorboard_logger.py` emits an `artifact` telemetry event with `kind="tensorboard"` and the real `log_dir`. The validation layer (`gym_gui/validations/validations_pydantic.ArtifactEvent`) only allows kinds `{"policy", "video", "checkpoint", "log"}`. The `tensorboard` event is rejected, so the UI never receives the corrected absolute path.

## Fixes Applied (2025-11-02)
1. `validate_train_run_config` now reuses any ULID provided in the worker block during resubmission and rewrites `metadata.artifacts.tensorboard` so both `relative_path` and `log_dir` reference the ULID folder (`var/trainer/runs/<ulid>/tensorboard`).
2. `ArtifactEvent` validation allows `kind="tensorboard"`, so the worker’s artifact telemetry delivers the real absolute log directory to the GUI.
3. The TensorBoard artifact tab logs status changes (`LOG724` / `LOG725`) and honours the emitted path via its new signal, making debugging easier. The UI also exposes a *Show/Hide Details* toggle so the embedded viewer can grab all vertical real estate once logs arrive.
4. Splitter minimums/layout defaults in the main window were relaxed and made collapsible so the TensorBoard tab and runtime log columns can expand into unused space.

With these changes, pointing TensorBoard at the ULID path either via CLI (`tensorboard --logdir var/trainer/runs/<ulid>/tensorboard`) or through the GUI now surfaces the scalars immediately once the worker flushes data.
