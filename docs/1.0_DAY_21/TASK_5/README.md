# PPO script differences (absolute paths)

This note compares the original CleanRL PPO script and the locally adjusted worker copy, with absolute file paths for traceability.

- Original (vendor copy): `/home/hamid/Desktop/Projects/GUI_BDI_RL/cleanrl/cleanrl/ppo.py`
- Modified (worker copy): `/home/hamid/Desktop/Projects/GUI_BDI_RL/cleanrl_worker/cleanrl/ppo.py`

## Summary of changes

### 1) Imports and typing

- Added `from typing import Optional` and `import sys` in the worker copy.
- Worker copy initially imported `wandb` at the module top; the vendor version imports `wandb` lazily only when `args.track` is enabled. The worker copy then wraps `wandb.init` in try/except to degrade gracefully if initialization fails.

### 2) CLI args and runtime initialization

- Worker copy supports `CLEANRL_TENSORBOARD_DIR` to override the TensorBoard writer directory (falls back to `runs/{run_name}`).
- Worker copy allows environment overrides for Weights & Biases:
  - `WANDB_MONITOR_GYM` (interprets "0/false/no" as false; otherwise true)
  - `WANDB_INIT_TIMEOUT` (seconds; default 120)
- On `wandb.init` failure, the worker copy prints a warning to stderr and continues with `args.track = False`.

### 3) Environment and shape handling robustness

- Worker copy resolves shapes defensively:
  - Observation shape: `obs_shape = tuple(getattr(envs.single_observation_space, "shape", ()) or ())`
  - Action shape: `act_shape = tuple(getattr(envs.single_action_space, "shape", ()) or ())`
- Storage tensors and the model input dimension use these shapes, guarding against rare cases where shapes can be `None` or atypical.

### 4) Logging and metrics differences

- Vendor version prints episodic returns to stdout and logs them to TensorBoard; worker copy logs them but does not print (quieter CI output).
- Vendor version logs both `losses/old_approx_kl` and `losses/approx_kl`.
  - Worker copy keeps `losses/approx_kl` but omits `losses/old_approx_kl` in its final metrics set (reduces diagnostic detail slightly).
- Both versions log: learning rate, value/policy losses, entropy, clipfrac, explained variance, and SPS to TensorBoard; worker copy omits the console SPS print.

### 5) Algorithmic/training semantics

- Core PPO logic (advantage computation, ratio/clipping, value clipping, entropy, gradient clipping) remains the same.
- Early stop on target KL is semantically unchanged, only refactored with a nested `if` in the worker copy.
- Step loop uses slightly different variable names for termination flags (`terminated`/`truncated` vs `terminations`/`truncations`), but logic is equivalent.

### 6) Error handling and resilience

- Worker copy is more robust in environments without WANDB access: `wandb.init` is protected with try/except; runs proceed without tracking if initialization fails.
- Defensive shape handling reduces the chance of attribute errors when using non-standard wrappers or vector envs.

### 7) Notable unchanged section

- The section marked in the vendor script as:
  
  `# bootstrap value if not done`
  
  is present in both files and functionally identical (GAE/returns computation).

## Practical impact

- **Reproducibility:** Unchanged by the listed modifications; seeds and device selection match vendor defaults.
- **CI/Headless usage:** Improved in the worker copy (quieter stdout; resilient to WANDB/network hiccups; configurable TensorBoard dir).
- **Diagnostics:** Slightly reduced if you rely on `losses/old_approx_kl`; consider re-enabling if needed.
- **Compatibility:** Worker copy is a drop-in for discrete action spaces (same assertion). Robust shape guards help with unusual envs.

## Recommendations

- If you want to preserve optional tracking without requiring WANDB to be installed, prefer lazy import:
  - Remove the top-level `import wandb` and keep `import wandb` inside the `if args.track:` block.
- If KL diagnostics matter during tuning, restore the `losses/old_approx_kl` scalar in the worker copy.
- Keep the `CLEANRL_TENSORBOARD_DIR` behavior; it is useful for routing logs in multi-run setups and CI.

## File references

- Vendor script: `/home/hamid/Desktop/Projects/GUI_BDI_RL/cleanrl/cleanrl/ppo.py`
- Worker script: `/home/hamid/Desktop/Projects/GUI_BDI_RL/cleanrl_worker/cleanrl/ppo.py`

## Run outputs: tensorboard and wandb directories

During a trainer-launched run, two output folders were created under the run root:

- TensorBoard: `/home/hamid/Desktop/Projects/GUI_BDI_RL/var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/tensorboard`
- Weights & Biases: `/home/hamid/Desktop/Projects/GUI_BDI_RL/var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/wandb`

### How the TensorBoard folder is created

- The worker script uses `torch.utils.tensorboard.SummaryWriter(writer_dir)`.
- In the worker copy, `writer_dir` is either taken from the environment variable `CLEANRL_TENSORBOARD_DIR` or falls back to `runs/{run_name}`.
- The trainer service likely exports `CLEANRL_TENSORBOARD_DIR=/home/hamid/Desktop/Projects/GUI_BDI_RL/var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/tensorboard` before launching the process, so SummaryWriter places event files there.

### How the W&B folder is created

- The W&B Python client writes a `wandb` directory in the current working directory (or `WANDB_DIR` if set). Because the trainer runs the process with the run root as CWD, you see `.../runs/<RUN_ID>/wandb` being created.
- Inside that folder, W&B also launches a co-located "Local Service" and stores its own internal state under a nested `wandb` subdirectory, yielding `.../wandb/wandb` (see next section).

## Why there is both `wandb` and `wandb/wandb`

- Top-level `.../wandb` is the per-run artifacts directory (history, summary, metadata, etc.).
- Nested `.../wandb/wandb` is W&B's Local Service working directory. It contains internal logs (e.g., `debug-internal.log`), temporary files, and runtime metadata.

This duplication is expected: one directory for run files, and a nested one for the background service.

## What are the "sockets" mentioned in logs?

The W&B Local Service uses a Unix Domain Socket for IPC (inter-process communication) with the client library. From your logs:

```json
{"time":"2025-11-06T19:49:52.633377402+08:00","level":"INFO","msg":"main: starting server","port-filename":"/tmp/tmpodcdl0_j/port-2988680.txt","pid":2988680,"log-level":0,"disable-analytics":false,"shutdown-on-parent-exit":false,"enable-dcgm-profiling":false}
{"time":"2025-11-06T19:49:52.633917843+08:00","level":"INFO","msg":"server: will exit if parent process dies","ppid":2988680}
{"time":"2025-11-06T19:49:52.633912933+08:00","level":"INFO","msg":"server: accepting connections","addr":{"Name":"/tmp/wandb-2988680-2989134-193683278/socket","Net":"unix"}}
{"time":"2025-11-06T19:49:52.820599124+08:00","level":"INFO","msg":"connection: ManageConnectionData: new connection created","id":"1(@)"}
{"time":"2025-11-06T19:49:52.82950308+08:00","level":"INFO","msg":"handleInformInit: received","streamId":"1075","id":"1(@)"}
{"time":"2025-11-06T19:55:54.387759384+08:00","level":"INFO","msg":"server: parent process exited, terminating service process"}
```

- `addr.Name` shows the Unix socket path under `/tmp/` used for client-service communication.
- When the parent process exits, the Local Service notices and terminates.

## W&B network retries and offline behavior

From `.../wandb/wandb/debug-internal.log`:

```json
{"time":"2025-11-06T19:49:52.829723977+08:00","level":"INFO","msg":"stream: starting","core version":"0.22.3"}
{"time":"2025-11-06T19:49:52.941990972+08:00","level":"INFO","msg":"api: retrying error","error":"Post \"https://api.wandb.ai/graphql\": proxyconnect tcp: EOF"}
{"time":"2025-11-06T19:49:55.138772933+08:00","level":"INFO","msg":"api: retrying error","error":"Post \"https://api.wandb.ai/graphql\": proxyconnect tcp: EOF"}
{"time":"2025-11-06T19:49:59.55702645+08:00","level":"INFO","msg":"api: retrying error","error":"Post \"https://api.wandb.ai/graphql\": proxyconnect tcp: EOF"}
...
```

These indicate the Local Service is attempting to contact `api.wandb.ai` but cannot (proxy or egress is blocking). In your stderr:

```text
wandb: Network error (ProxyError), entering retry loop.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
[cleanrl][warning] wandb.init failed: Run initialization has timed out after 120.0 sec. Please try increasing the timeout with the `init_timeout` setting: `wandb.init(settings=wandb.Settings(init_timeout=120))`.
```

The worker copy already sets a default `WANDB_INIT_TIMEOUT=120`, catching the init failure and continuing with `track=False`.

## Gym RecordVideo + W&B monitor close crash

From `.../logs/cleanrl.stderr.log`:

```text
AttributeError: 'RecordVideo' object has no attribute 'enabled'
  File "/home/hamid/Desktop/Projects/GUI_BDI_RL/.venv/lib/python3.11/site-packages/wandb/integration/gym/__init__.py", line 74, in close
    if not self.enabled:
```

Cause:
- When `monitor_gym=True`, W&B's Gym integration hooks environment closing/monitoring. It expects wrapper objects with an `enabled` attribute (e.g., its own Monitor wrapper), but `gymnasium.wrappers.RecordVideo` does not define `enabled`.
- Combined with a failed/partial `wandb.init` due to network timeouts, the integration can leave patched hooks that call into W&B close logic on a `RecordVideo` instance, producing this error at `envs.close()`.

Workarounds:
- Disable W&B Gym monitoring for these runs. In the worker copy you can set the environment variable before launch:
  - `WANDB_MONITOR_GYM=0` (interpreted as false by the worker).
- Or update the code to pass `monitor_gym=False` by default (only enable when you are not using `RecordVideo`).
- Alternatively, turn off tracking (`--track=False`) for fully offline runs.

Note: You can activate the environment and verify paths yourself:

```bash
source .venv/bin/activate
# inspect TB events
ls -lah /home/hamid/Desktop/Projects/GUI_BDI_RL/var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/tensorboard
# inspect W&B internals
ls -lah /home/hamid/Desktop/Projects/GUI_BDI_RL/var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/wandb
ls -lah /home/hamid/Desktop/Projects/GUI_BDI_RL/var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/wandb/wandb
```
