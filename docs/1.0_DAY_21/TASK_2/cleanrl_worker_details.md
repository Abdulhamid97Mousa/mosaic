# CleanRL Worker — Flags, Arguments, and Parameters (outside MOSAIC_CLEANRL_WORKER)

This document lists the command-line flags, config fields, extras, environment variables, and behavior used by the CleanRL worker wrapper in this repo, focusing only on code outside the vendor directory `cleanrl_worker/MOSAIC_CLEANRL_WORKER`.

Applies to:

- `cleanrl_worker/__init__.py` (facade)
- `cleanrl_worker/cli.py` (entrypoint wrapper)
- `cleanrl_worker/config.py` (config facade)
- `cleanrl_worker/runtime.py` (runtime facade)
- `cleanrl_worker/telemetry.py` (telemetry facade)
- `cleanrl_worker/analytics.py` (analytics facade)

Note: Implementation of CLI/config/runtime/telemetry lives under `cleanrl_worker/MOSAIC_CLEANRL_WORKER`. This page documents the effective interface and parameters that the wrapper exposes and uses, with concrete behavior sourced from the vendor modules.

## CLI entrypoint (cleanrl_worker.cli)

Wrapper: `python -m cleanrl_worker.cli` delegates to `MOSAIC_CLEANRL_WORKER.cli:main`.

Supported flags (from vendor CLI):

- --config PATH (required)
  - Path to trainer-issued worker config JSON.
- --algo NAME
  - Override algorithm identifier (e.g., ppo, dqn). Overrides config.algo.
- --env-id ENV_ID
  - Override environment id (e.g., CartPole-v1). Overrides config.env_id.
- --total-timesteps INT
  - Override total timesteps. Overrides config.total_timesteps.
- --seed INT
  - Override algorithm seed. Overrides config.seed.
- --worker-id ID
  - Override worker id used in RegisterWorker and logs.
- --extras JSON
  - JSON object merged into config.extras; see Extras mapping below.
- --grpc (flag)
  - Enable gRPC RegisterWorker handshake with the trainer daemon.
- --grpc-target HOST:PORT (default: 127.0.0.1:50055)
  - Target for gRPC if --grpc is set.
- --dry-run (flag)
  - Resolve algorithm module and args; do not execute training.
- --emit-summary (flag)
  - With --dry-run, print resolved summary JSON to stdout.

Exit codes:

- 0 on success (including dry-run).
- Non-zero if the underlying CleanRL process fails (propagated return code) or on unexpected errors.

Example:

```bash
python -m cleanrl_worker.cli \
  --config /path/to/worker_config.json \
  --grpc --grpc-target 127.0.0.1:50055 \
  --extras '{"track_wandb": true, "tensorboard_dir": "tb", "algo_params": {"num_envs": 8, "learning_rate": 0.0003}}'
```

## Config schema (WorkerConfig)

Effective config class: `MOSAIC_CLEANRL_WORKER.config.WorkerConfig`

Required fields:

- run_id: string
- algo: string
- env_id: string
- total_timesteps: integer

Optional fields:

- seed: integer | null
- worker_id: string | null
- extras: object (dict)

Loading:

- `load_worker_config(path)` accepts either:
  - A direct worker config dict; or
  - A trainer payload that contains `metadata.worker.config`.
- `parse_worker_config(payload)` validates required fields.
- `WorkerConfig.with_overrides(...)` merges CLI overrides and merges the `extras` dict (CLI extras win).

## Runtime orchestration (CleanRLWorkerRuntime)

Constructor parameters:

- config: WorkerConfig
- use_grpc: bool (from `--grpc`)
- grpc_target: str (from `--grpc-target`)
- dry_run: bool (from `--dry-run`)
- algo_registry: mapping[name, module] (defaults to built-in registry)

Default algorithm registry (name -> module):

- ppo -> cleanrl.ppo
- ppo_continuous_action -> cleanrl.ppo_continuous_action
- ppo_atari -> cleanrl.ppo_atari
- dqn -> cleanrl.dqn
- c51 -> cleanrl.c51
- ppg_procgen -> cleanrl.ppg_procgen
- ppo_rnd_envpool -> cleanrl.ppo_rnd_envpool

Resolution rules:

- Prefer the canonical module (e.g., `cleanrl.ppo`).
- If it starts with `cleanrl.`, also try `cleanrl_worker.<canonical>` to support bundled copies.
- The module must export a callable `main()`.

### CleanRL script arguments (built by runtime)

Base args:

- --env-id=ENV_ID
- --total-timesteps=INT
- --seed=INT (if provided)

Extras mapping:

- track_wandb: bool
  - If true, adds `--track` to the underlying CleanRL script.
- tensorboard_dir: str
  - Adds `--tensorboard-dir=<dir>`; also materialized under the run directory.
- algo_params: object (mapping)
  - Each key/value becomes a CLI flag with underscores converted to hyphens:
    - bool true -> `--key`
    - bool false -> omitted
    - numbers/strings -> `--key=value`
- Other extras (not in reserved set): any
  - Converted to CLI flags via `--key=value` with underscore-to-hyphen, value-typed:
    - bool -> `--key=true|false`
    - number/string -> `--key=value`
    - sequence -> repeated `--key=value` entries

Reserved extras (not forwarded except as noted):

- tensorboard_dir, track_wandb, algo_params, notes

### RegisterWorker handshake (when --grpc)

Channel options:

- grpc.max_receive_message_length = 64 MiB
- grpc.max_send_message_length = 64 MiB

Request fields:

- run_id: config.run_id
- worker_id: config.worker_id or `cleanrl-worker-<run_id[:8]>`
- worker_kind: "cleanrl"
- proto_version: "MOSAIC/1.0"
- schema_id: `extras.schema_id` or `cleanrl.<algo>`
- schema_version: `int(extras.schema_version)` or 1
- supports_pause: `bool(extras.supports_pause)`
- supports_checkpoint: `bool(extras.supports_checkpoint)`

On success, a session_token is stored (internal use).

### Process execution, logging, and heartbeats

- Working directories:
  - `var/trainer/runs/<run_id>/` (created if missing)
  - Logs written to `logs/cleanrl.stdout.log` and `logs/cleanrl.stderr.log`
- Environment:
  - Sets `PYTHONUNBUFFERED=1` (ensures timely stdout flushing)
- Heartbeats:
  - Emits a heartbeat lifecycle event every 30s while the process runs.
- Exit handling:
  - Non-zero return codes raise `CalledProcessError`.

## Telemetry events (LifecycleEmitter)

Printed to stdout as JSON lines (consumed by the GUI trainer’s Telemetry Proxy):

- run_started(run_id, payload)
- heartbeat(run_id, payload)
- run_completed(run_id, payload)
- run_failed(run_id, payload)

Payloads typically include algo/env_id; completion includes analytics manifest.

## Analytics manifest (build_manifest)

Reads extras and the run directory to produce `analytics.json` under the run path.

Extras keys consumed:

- tensorboard_dir: str -> artifacts.tensorboard
- wandb_run_path: str -> artifacts.wandb.run_path
- wandb_entity: str
- wandb_project_name | wandb_project: str
- optuna_db: str -> absolute path under run dir
- checkpoints_dir: str -> absolute path under run dir
- notes: str

## Environment variables

Set by runtime:

- PYTHONUNBUFFERED=1

Underlying CleanRL scripts may support their own env vars (not documented here).

## Worked examples

Dry-run with summary printed:

```bash
python -m cleanrl_worker.cli \
  --config ./worker_config.json \
  --dry-run --emit-summary
```

Run with gRPC handshake and WANDB/TensorBoard:

```bash
python -m cleanrl_worker.cli \
  --config ./worker_config.json \
  --grpc --grpc-target 127.0.0.1:50055 \
  --extras '{"track_wandb": true, "tensorboard_dir": "tb", "wandb_entity": "myteam", "wandb_project": "mosaic"}'
```

Custom CleanRL algorithm params via extras.algo_params:

```bash
python -m cleanrl_worker.cli \
  --config ./worker_config.json \
  --extras '{"algo_params": {"num_envs": 8, "learning_rate": 0.0003, "gamma": 0.99}}'
```

RegisterWorker capability hints via extras:

```bash
python -m cleanrl_worker.cli \
  --config ./worker_config.json \
  --grpc --grpc-target 127.0.0.1:50055 \
  --extras '{"supports_pause": true, "supports_checkpoint": true, "schema_version": 2}'
```

## GUI Validation Workflow

- The CleanRL Train dialog now performs `cleanrl_worker.cli --dry-run --emit-summary` automatically before launching a run. The validation output is surfaced under the form, and the run is only scheduled when the dry-run succeeds.
- Selecting **Validate only (skip training)** runs the dry-run and keeps the dialog open so you can adjust configuration without submitting to the trainer.
- WANDB VPN proxy fields from the form feed the same extras/environment keys documented above (`wandb_use_vpn_proxy`, `wandb_http_proxy`, `wandb_https_proxy`).
