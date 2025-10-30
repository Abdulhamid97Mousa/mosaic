# SPADE-BDI RL (Refactored Worker)

This package restructures the legacy `spadeBDI_RL` prototype into a library-first
layout geared towards trainer subprocess execution. The worker consumes a single
JSON configuration, streams newline-delimited telemetry (JSONL) to stdout, and
saves policy artifacts under `var/trainer/policies/<env>/<agent>.json` by default.

## Highlights

- **Headless worker** – `python -m spadeBDI_RL_refactored.worker --config run.json`
  runs training/evaluation without GUI bindings.
- **Adapter factory** – maps logical environment IDs (e.g. `FrozenLake-v2`) onto
  lightweight adapters that conform to Gymnasium-style semantics.
- **Policy storage** – standardised JSON snapshots with Q-table payloads and
  metadata for trainer/GUI usage.
- **JSONL telemetry** – compatible with the existing `TelemetryService` and
  trainer daemon expectations.
- **Bundled prerequisites** – the original `main_agent.asl` is shipped under
  `assets/asl/`, and a ready-to-run ejabberd manifest lives in
  `infrastructure/docker-compose.yaml`. This keeps the worker self-contained
  even when the legacy repository layout changes.

## Configuration Contract

A minimal worker configuration looks like:

```json
{
  "run_id": "demo-001",
  "env_id": "FrozenLake-v2",
  "seed": 123,
  "max_episodes": 5,
  "max_steps_per_episode": 100,
  "policy_strategy": "train",
  "agent_id": "bdi_rl"
}
```

Submit the payload via `--config <file>` or stdin. The worker emits events such
as `run_started`, `step`, `episode`, `artifact`, and `run_completed`.

## SPADE/XMPP prerequisites

The BDI layer still relies on an ejabberd XMPP broker. Start it with the
bundled manifest before launching the worker if you expect the agent to connect
over XMPP:

```bash
docker compose -f spadeBDI_RL_refactored/infrastructure/docker-compose.yaml up -d
```

The default credentials (`agent@localhost` / `secret`) match the values used by
`core.agent.DEFAULT_JID` and `DEFAULT_PASSWORD`. The packaged AgentSpeak source
at `assets/asl/main_agent.asl` mirrors the legacy plans and is used whenever a
BDI agent is instantiated via `core.create_agent()`.

## Legacy Demos

Existing visual demos (`visual_demo.py`, `bdi_policy_cli.py`) now count as
legacy clients. They can import from this package to stay functional, but they
are no longer wired into the runtime dependency chain used by the GUI trainer.
