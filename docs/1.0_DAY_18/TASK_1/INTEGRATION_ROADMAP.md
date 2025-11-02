# Day 18 Task 1 – Schema Integration Roadmap

_Source context: [`INITIAL_COMPARISON.md`](./INITIAL_COMPARISON.md)_

This plan sequences the work required to close the data-contract gaps identified in the comparison document. It emphasises integration touchpoints across adapters, telemetry, storage, trainer proto, and UI so the new schema can roll out safely.

## Guiding Principles

- **Single source of truth** – define the schema once and fan it out through adapters, telemetry, trainer, storage, and UI.
- **Backwards-compatible evolution** – version payloads and use feature flags so existing runs, trainers, and telemetry continue to function during migration.
- **Incremental rollout** – land thin slices that keep the system runnable; avoid big-bang merges.
- **Visibility** – log schema mismatches, add validation gates, and instrument upgrades with telemetry dashboards.

## Phase 0 – Discovery & Alignment

| Objective | Deliverables | Integration Notes |
|-----------|--------------|--------------------|
| Baseline current contracts | Adapter payload audit, telemetry field inventory, trainer proto usage map | Required for negotiating schema versions with remote workers. |
| Draft schema proposal | JSON Schema + Proto extensions, glossary of canonical keys | Review with adapter, trainer, and UI owners; identify migration blockers. |
| Migration playbook | Compatibility matrix, rollout checkpoints, rollback strategy | Ensure trainer services and GUI releases coordinate version bumps. |

Dependencies: sign-off from Schema Council (adapters, telemetry, trainer, UI leads).

## Phase 1 – Schema Foundations

| Objective | Deliverables | Integration Notes |
|-----------|--------------|--------------------|
| Implement `SpaceSerializer` & payload dataclasses | `gym_gui/core/schema/step_payload.py`, `spaces/serializer.py` | Provide Python helpers consumed by adapters & validation layers. |
| Introduce schema validators | `gym_gui/services/telemetry_schema.py`, reusable JSON Schema bundle | Validators must degrade gracefully when legacy payloads arrive. |
| Add new log constants & telemetry keys | `LOG_SCHEMA_MISMATCH`, `LOG_VECTOR_AUTORESET_MODE`, etc. | Enables real-time detection of non-compliant payloads. |

Versioning: publish schema version `v1` using `payload_version` field; reserve `v0` for legacy.

## Phase 2 – Adapter Integration

| Objective | Deliverables | Integration Notes |
|-----------|--------------|--------------------|
| Retrofit existing adapters (toy-text, Box2D) | Emit structured render payloads + space descriptors | Guard dev flags to fall back to legacy payloads until downstream ready. |
| Add schema compliance tests | `gym_gui/tests/test_adapter_contract.py` | Use golden fixtures in `tests/data/schema_samples/`. |
| Establish adapter API for new envs | `EnvironmentAdapter.describe_space()` | Document contract so future adapters start compliant. |

Adapters should publish `schema_version = 1` and include `space_descriptor`, `vector_metadata`, and `normalization_stats` fields when relevant.

## Phase 3 – Telemetry & Trainer Pipeline

| Objective | Deliverables | Integration Notes |
|-----------|--------------|--------------------|
| Update telemetry ingestion | `TelemetryService.record_step`, `TelemetryDBSink._process_step_queue` | Validate payloads, capture schema version, enrich logs on mismatch. |
| Extend gRPC contract | Update `trainer.proto` to carry new fields (`space_signature_json`, `vector_metadata_json`, `normalization_stats_json`) | Bump proto version; regenerate stubs; maintain backwards compatibility by keeping legacy fields optional. |
| Trainer runtime compatibility | Trainer clients/daemons accept both `v0` and `v1` payloads | Feature flag to switch to schema-enforced mode once all adapters updated. |

Coordination: release new trainer binaries before GUI flips the enforcement flag.

## Phase 4 – Storage & Migration

| Objective | Deliverables | Integration Notes |
|-----------|--------------|--------------------|
| Add DB columns / JSON blobs | Telemetry SQLite schema migration (Alembic or custom) | Write migration script; ensure WAL integrity. |
| Snapshot handling | Replace raw `snapshot_path` with embedded blobs or signed URLs | Update storage service and adjust disk retention policies. |
| Seed registry | Persist per-env seeds, autoreset flags | Enables deterministic replays and analytics. |

Test: run migration against staging telemetry DBs; capture before/after metrics.

## Phase 5 – UI Integration

| Objective | Deliverables | Integration Notes |
|-----------|--------------|--------------------|
| Update render tabs | Consume `RENDER_PAYLOAD_GRID`, `RENDER_PAYLOAD_RGB`, `RENDER_PAYLOAD_GRAPH` | Remove ANSI-based fallbacks once schema enforced. |
| Telemetry visualisation | Panels for vector metadata, normalization stats | Ensure responsive layout and lazy loading for large descriptors. |
| Backwards compatibility UI mode | Toggle to display legacy payloads during transition | Offers safe fallback if schema rollout slips. |

Smoke tests: run GUI with toy-text + Box2D + dummy graph adapter sample payloads.

## Phase 6 – Validation & Tooling

| Objective | Deliverables | Integration Notes |
|-----------|--------------|--------------------|
| Contract test suite | End-to-end tests verifying adapters → telemetry → storage → UI | Integrate into CI; gate merges on schema compliance. |
| Developer utilities | `python -m gym_gui.tools.dump_space_signature`, fixture generator | Simplifies onboarding of new environments. |
| Observability dashboards | Grafana/Kibana views for schema version adoption | Track rollout progress across workers and runs. |

## Phase 7 – Rollout & Cleanup

| Objective | Deliverables | Integration Notes |
|-----------|--------------|--------------------|
| Staged deployment | Feature flag timeline, training cluster rollout plan | Start with canaries, collect telemetry, expand gradually. |
| Deprecate legacy paths | Remove ANSI parsers, fallback renderers, redundant DTOs | Ensure all teams migrated before deleting. |
| Documentation updates | Dev guide, adapter cookbook, telemetry schema reference | Archive legacy docs; publish migration FAQ. |

## Risk & Mitigation Matrix

| Risk | Impact | Mitigation |
|------|--------|------------|
| Trainer GUI drift during rollout | Broken telemetry, replay failures | Versioned proto with capability negotiation; feature flags. |
| Partial adapter adoption | Mixed payloads leading to inconsistent UI | Schema validator logs + metrics; enforce adapter checklist before merge. |
| Storage migration downtime | Loss of telemetry data | Pre-flight backups, staggered migrations, WAL checkpoint validation. |
| Performance regressions | Larger payloads stress run bus and storage | Benchmark new payload sizes; compress if needed. |

## Success Metrics

- ≥95% of production runs emitting schema version `v1` within two releases.
- Zero schema mismatch errors in telemetry logs for compliant adapters.
- UI supports at least one graph/sequence environment without ad-hoc hacks.
- Telemetry database retains vector seeds and normalization stats for replays.


## Next Actions

1. Schedule design review for schema proposal (Phase 0 deliverable).
2. Assign owners and timelines for each phase; update project tracker.
3. Stand up staging environment mirroring trainer + telemetry pipeline for integration testing.

This document should evolve alongside implementation; log decisions, status, and deviations inline so Task 1 remains the integration source of truth.
