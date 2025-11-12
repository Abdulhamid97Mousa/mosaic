# CleanRL Run Artifacts and Logs

This note documents how the trainer generates the TensorBoard/W&B directories for run `01K9CG05J5Y0FFQMT1T920EZK2`, why you see nested `wandb/wandb` folders, what the socket/debug logs mean, and the origin of the `RecordVideo`/W&B crash in `cleanrl.stderr.log`.

## 1. Where the directories come from

### TensorBoard (`var/trainer/runs/<RUN_ID>/tensorboard`)

- `cleanrl_worker` writes TensorBoard events wherever `CLEANRL_TENSORBOARD_DIR` points. In `runtime.py`, once `extras["tensorboard_dir"]` is set by the UI checkbox, the runtime expands it under the run folder and exports the absolute path via `env["CLEANRL_TENSORBOARD_DIR"] = str(tb_path)`. (See `cleanrl_worker/MOSAIC_CLEANRL_WORKER/runtime.py`, lines ~310-340.)
- The worker then instantiates `torch.utils.tensorboard.SummaryWriter(writer_dir)` inside `cleanrl/ppo.py`, so all event files land in `/home/hamid/Desktop/Projects/GUI_BDI_RL/var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/tensorboard`.
- Verification: `source .venv/bin/activate && ls -lah var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/tensorboard`.

### W&B (`var/trainer/runs/<RUN_ID>/wandb`)

- When the form toggles “Export WANDB artifacts”, `extras["track_wandb"] = True` and the runtime allocates a run-local `wandb` folder before launching (`wandb_root = run_dir / "wandb"`). It populates cache/config/log directories and then sets `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_CONFIG_DIR`, etc. in the child environment. (runtime.py lines ~320-370.)
- As soon as `wandb.init()` runs, the SDK writes run metadata (history.jsonl, config.yaml, media, etc.) directly under this top-level `wandb` directory.

### Why there is both `wandb` and `wandb/wandb`

- The W&B SDK also launches a "Local Service" sidecar that keeps its own working directory under `WANDB_DIR/wandb`. That nested folder holds service logs (`debug-internal.log`, `debug.log`), SQLite caches, and IPC metadata. The top-level folder belongs to the Python process; the nested folder belongs to the background service.

```
/home/hamid/.../runs/01K9CG05J5Y0FFQMT1T920EZK2/wandb/
└── wandb/   # Local Service home (logs + sockets bookkeeping)
```

## 2. What the socket log is telling you

Excerpt from `var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/wandb/wandb/debug.log`:

```json
{"time":"2025-11-06T19:49:52.633377402+08:00","level":"INFO","msg":"main: starting server","port-filename":"/tmp/tmpodcdl0_j/port-2988680.txt","pid":2988680,"log-level":0,"disable-analytics":false,"shutdown-on-parent-exit":false,"enable-dcgm-profiling":false}
{"time":"2025-11-06T19:49:52.633917843+08:00","level":"INFO","msg":"server: will exit if parent process dies","ppid":2988680}
{"time":"2025-11-06T19:49:52.633912933+08:00","level":"INFO","msg":"server: accepting connections","addr":{"Name":"/tmp/wandb-2988680-2989134-193683278/socket","Net":"unix"}}
{"time":"2025-11-06T19:49:52.820599124+08:00","level":"INFO","msg":"connection: ManageConnectionData: new connection created","id":"1(@)"}
{"time":"2025-11-06T19:49:52.82950308+08:00","level":"INFO","msg":"handleInformInit: received","streamId":"1075","id":"1(@)"}
{"time":"2025-11-06T19:55:54.387759384+08:00","level":"INFO","msg":"server: parent process exited, terminating service process"}
```

- The local service exposes a Unix Domain Socket (here `/tmp/wandb-2988680-2989134-193683278/socket`) so the client library can stream metrics without blocking the training loop.
- "port-filename" is a temp file containing the socket endpoint; the GUI/CLI polls it to attach.
- When the parent process exits, the service shuts down (`server: parent process exited...`).

## 3. Why `debug-internal.log` keeps saying `proxyconnect tcp: EOF`

`var/trainer/runs/.../wandb/wandb/debug-internal.log` shows:

```json
{"time":"2025-11-06T19:49:52.829723977+08:00","level":"INFO","msg":"stream: starting","core version":"0.22.3"}
{"time":"2025-11-06T19:49:52.941990972+08:00","level":"INFO","msg":"api: retrying error","error":"Post \"https://api.wandb.ai/graphql\": proxyconnect tcp: EOF"}
{"time":"2025-11-06T19:49:55.138772933+08:00","level":"INFO","msg":"api: retrying error","error":"Post \"https://api.wandb.ai/graphql\": proxyconnect tcp: EOF"}
...
{"time":"2025-11-06T19:55:01.825416434+08:00","level":"INFO","msg":"api: retrying error","error":"Post \"https://api.wandb.ai/graphql\": proxyconnect tcp: EOF"}
```

- The SDK attempted to sync with `api.wandb.ai` but your trainer host blocks outbound HTTPS unless a proxy is configured. Because `WANDB_HTTP_PROXY`/`WANDB_HTTPS_PROXY` were not set (or the proxy was unreachable), every attempt hit `EOF` after the proxy handshake.
- The worker catches this by setting `WANDB_INIT_TIMEOUT=120` and, upon repeated failures, flips the run into offline mode (`[cleanrl][warning] wandb.init failed...`). The retries continue in the background until the process exits.

## 4. `cleanrl.stderr.log` crash: W&B monitor vs. `RecordVideo`

`var/trainer/runs/.../logs/cleanrl.stderr.log` contains:

```
Traceback (most recent call last):
  ...
  File "/home/hamid/Desktop/Projects/GUI_BDI_RL/cleanrl_worker/cleanrl/ppo.py", line 335, in <module>
    envs.close()
  ...
  File "/home/hamid/Desktop/Projects/GUI_BDI_RL/.venv/lib/python3.11/site-packages/wandb/integration/gym/__init__.py", line 74, in close
    if not self.enabled:
AttributeError: 'RecordVideo' object has no attribute 'enabled'
```

- When `track_wandb=True`, the W&B Gym integration wraps the environment to capture videos/metrics. It expects wrappers with an `enabled` attribute (e.g., its own Monitor), but Gymnasium’s `RecordVideo` wrapper does not expose that attribute.
- During `envs.close()` the patched `close()` tries to read `self.enabled`, hits `AttributeError`, and the run terminates with an exception. Mitigations:
  1. Disable gym monitoring when you also use `RecordVideo` (set `WANDB_MONITOR_GYM=0` or keep the analytics checkbox off).
  2. Patch `RecordVideo` or vendor the W&B integration to guard `hasattr(self, "enabled")`.
  3. Skip `RecordVideo` whenever `track_wandb` is on.

## 5. Reproducing/inspecting locally

```bash
source .venv/bin/activate
# Inspect TensorBoard events
ls -lah var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/tensorboard
# Inspect W&B layout (note the nested wandb/wandb service home)
ls -lah var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/wandb
ls -lah var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/wandb/wandb
# Tail recent WANDB retries
sed -n '1,40p' var/trainer/runs/01K9EZHKGHYFARCTWY1J279NHE/wandb/wandb/debug-internal.log
# Inspect the env-close crash
sed -n '1,80p' var/trainer/runs/01K9CG05J5Y0FFQMT1T920EZK2/logs/cleanrl.stderr.log
```

If you still see `proxyconnect tcp: EOF` in fresher runs (e.g., `01K9EZHKGHYFARCTWY1J279NHE` on 2025‑11‑07), the host cannot reach `api.wandb.ai` through the proxy you configured (`http(s)://127.0.0.1:7890`). Verify that the proxy actually listens on that port (for example `curl --proxy http://127.0.0.1:7890 https://api.wandb.ai/graphql`). Until that succeeds, all WANDB uploads will remain local under `var/trainer/.../wandb` even though TensorBoard continues to write to disk.
