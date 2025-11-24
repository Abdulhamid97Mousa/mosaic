# gui_jason_grpc_cleanrl

Lightweight bridge between Jason MAS agents, CleanRL policies, and upcoming symbolic (ILP/BDI) components—built without the heavy Gym GUI stack so everything runs headless, testable, and self-contained.

## Motivation & Scope
- **Initial plan**: Replace the heuristic Jason⇄Python bridge with a ToyText-aware runtime that wraps CleanRL policies and exposes richer metadata (`docs/initial_plan.md`, `docs/2.0_DAY_1.md`).
- **Async readiness**: Even though the current smoke tests are synchronous, we’re designing the RPC clients/servers to migrate to `grpc.aio` once throughput demands it (`docs/2.0_DAY_2.md`).
- **Symbolic RL path**: Zahra’s proposal (`docs/dev_zahra/2025-01-13-...` and the updated version in this repo) outlines how ILP-learned BDI plans will piggyback on the CleanRL data spine (`docs/dev_zahra/UPDATED-SYMBOLIC-RL-VIA-BDI-ILP-RESEARCH-PROPOSAL.md`). This repo hosts the Python and Jason plumbing needed to prove that idea without firing up the full GUI.

## Repository Layout

| Path | Description |
|------|-------------|
| `cleanrl_service/` | Policy server, ToyText adapters, future ILP exporters. |
| `bridge/` | Vendored proto + Jason-facing bridge (WIP). |
| `cleanrl_worker/` | Copy of the canonical CleanRL worker (untouched); `sitecustomize.py` exposes it on `PYTHONPATH`. |
| `jason_worker/` | Jason CLI/interpreter + demo MAS (vendored). |
| `shared/` | Vendored log constants + logging helpers. |
| `configs/` | Path helpers (`var/policies`, etc.). |
| `tests/` | gRPC smoke tests (more coming). |
| `var/` | Runtime artifacts (policies, episodes, ILP outputs). |
| `docs/` | Initial plan + day-by-day logs + symbolic RL proposals. |

## Quick Start

```bash
cd 3rd_party/gui_jason_grpc_cleanrl
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run policy server (Terminal A)
python -m cleanrl_service.policy_server --host 127.0.0.1 --port 50051

# Smoke test (Terminal B)
pytest tests/request_action_smoke.py
```

## Current Status
- Policy server returns ToyText metadata with vendored log constants (random policy placeholder).
- `tests/request_action_smoke.py` hits the gRPC server to ensure ToyText stepping works.
- `configs/paths.py` + `var/policies/` scaffold PPO checkpoints; `cleanrl_worker/sitecustomize.py` mirrors MOSAIC’s pattern so CleanRL scripts can be imported untouched.
- `docs/dev_zahra/UPDATED-...` captures how the symbolic/ILP layer will plug into this repo.

## Roadmap (from plan + day logs)
1. Integrate real FrozenLake PPO checkpoint so `_choose_action()` uses CleanRL inference (async-ready wrapper).
2. Port `bridge/jason_server.py` to proxy Jason RPCs to the policy server; evaluate `grpc.aio` for throughput.
3. Logging & storage: emit trajectories under `var/episodes/…` to feed ILP, then build predicate extraction + ILP runner.
4. Extend Jason MAS to consume symbolic hints and compare symbolic vs neural policies.
5. Optional ASCII viewer once core loop is proven.

For deeper context, see:
- `docs/initial_plan.md` — canonical ToyText integration plan.
- `docs/2.0_DAY_1.md`, `docs/2.0_DAY_2.md` — rolling progress/decisions.
- `docs/dev_zahra/*` — symbolic RL proposal evolution tailored to this repo.
