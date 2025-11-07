# CleanRL Worker Form Submission Precedence

This note describes how the CleanRL training form (`gym_gui/ui/widgets/cleanrl_train_form.py`) combines UI inputs with environment variables and worker defaults when launching `cleanrl_worker`. Understanding the priority order helps explain why tensorboard/W&B toggles behave the way they do.

## 1. Submission pipeline overview

1. The form collects state in `_collect_state()` and builds a trainer payload via `_build_config()`.
2. That payload is handed to the trainer runtime (`cleanrl_worker/MOSAIC_CLEANRL_WORKER/runtime.py`), which prepares run directories and launches `python -m cleanrl_worker.MOSAIC_CLEANRL_WORKER.launcher ...` inside `var/trainer/runs/<run_id>`.
3. The launcher imports the selected CleanRL algorithm (e.g., `cleanrl_worker/cleanrl/ppo.py`).

Each stage respects overrides from the previous one; earlier layers win when both attempt to set the same knob.

## 2. Priority order

1. **Host/Trainer environment variables** – The runtime copies the parent environment (`env = os.environ.copy()`) and only calls `env.setdefault(...)` when applying form-driven defaults. Anything you export before opening the form (e.g., `export WANDB_DIR=/tmp/custom`) survives unchanged.
2. **Form fields** – Checkboxes and text inputs write into the payload’s `environment` and `extras` maps. When the runtime merges these into `env`, they only win if the same variable was not already set.
3. **Worker defaults** – If neither the host env nor the form supplies a value, the CleanRL entry point falls back to its own defaults (e.g., `wandb` disabled, tensorboard writer under `runs/<run_name>`).

## 3. Feature gates driven by form checkboxes

| Form field | Injected variables / extras | Result when unchecked |
|------------|-----------------------------|-----------------------|
| `Track TensorBoard` | `TRACK_TENSORBOARD=1` and `extras["tensorboard_dir"]="tensorboard"`; runtime expands this to `CLEANRL_TENSORBOARD_DIR=var/trainer/runs/<id>/tensorboard` | No `TRACK_TENSORBOARD`; runtime skips tensorboard dir creation; `SummaryWriter` falls back to default logdir and GUI never watches for TB artifacts |
| `Track WANDB` | `TRACK_WANDB=1`, `extras["track_wandb"]=True`, optional project/entity/run name/api key/email | Runtime keeps WANDB entirely disabled: no `WANDB_DIR`, no probes, no `wandb.init` attempt, analytics tabs show WANDB as “disabled” |
| `Route WANDB traffic through VPN proxy` | Populates `extras["wandb_http_proxy"]` / `_https_` from inputs or `WANDB_VPN_*` env and sets `extras["wandb_use_vpn_proxy"]=True` | `_resolve_wandb_proxies` ignores the empty extras entry and consults `WANDB_VPN_*`, `WANDB_*`, generic `HTTP[S]_PROXY` in that order |

Because the runtime only creates tensorboard/W&B directories when their checkboxes are checked, leaving them unchecked guarantees there is nothing for `analytics_tabs.py` or `render_tabs.py` to display.

## 4. WANDB proxy precedence

`_resolve_wandb_proxies()` (runtime.py) implements:

1. `extras["wandb_http_proxy"]` / `_https_proxy` coming from the form.
2. `WANDB_VPN_HTTP_PROXY` / `_HTTPS_PROXY` from the trainer shell.
3. `WANDB_HTTP_PROXY` / `_HTTPS_PROXY`.
4. Generic `HTTP_PROXY`/`http_proxy` and `HTTPS_PROXY`/`https_proxy`.

The first non-empty string wins. `_apply_proxy_env()` then mirrors the resolved value into all related env vars so the child process sees a consistent view.

## 5. Practical scenarios

- **Hosting constraints override the form:** Setting `TRACK_WANDB=0` in the trainer shell forces WANDB off even if the UI box is checked—the runtime sees the pre-set “0” and keeps it.
- **TensorBoard reroute:** Export `CLEANRL_TENSORBOARD_DIR=/mnt/logs/tb` before launching the GUI. When you enable the checkbox, the runtime notices the var is already set and does not rewrite it, so CleanRL writes into `/mnt/logs/tb` instead of `var/trainer/...`.
- **Offline analytics:** Leaving all analytics checkboxes unchecked means `analytics.json` won’t contain `tensorboard` or `wandb` sections, so `AnalyticsTabManager` short-circuits without scheduling probes.

Use these rules when building automation around the forms: set host env vars for fleet-wide defaults, rely on the UI for per-run overrides, and remember unchecked boxes fully disable their downstream features.
