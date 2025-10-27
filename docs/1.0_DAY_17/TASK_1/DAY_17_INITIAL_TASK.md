# Day 17 – Initial Task Checklist (October 27, 2025)

## Objective
Restore the Live-Agent tab closure workflow so that operators can delete or archive specific `run_id` entries directly from the UI.

## Current State Recap
- ✅ Close buttons on dynamic Live-Agent tabs now surface `TabClosureDialog` before any teardown.
- ✅ `TabClosureDialog` is invoked at runtime and wired to delete/archive/keep decisions.
- ✅ Persistence helpers (`TelemetryService`, `TelemetrySQLiteStore`) expose delete/archive/status SUMMARY APIs consumed by the dialog flow.
- ✅ SPADE worker telemetry sanitises Taxi `action_mask` arrays via `_make_json_safe` to prevent JSON serialization failures, with new runtime logging (`LOG916`).

## Required Outcomes
- ✅ Reintroduced `_prompt_tab_closure()` inside `RenderTabs` so users pick keep/archive/delete before closing.
- ✅ DELETE/ARCHIVE decisions propagate through `TelemetryService` and `TelemetrySQLiteStore`; deleted runs no longer respawn.
- ✅ Added regression coverage (`TestRenderTabsClosureFlow`) and ran the full `gym_gui/tests` suite (149 green).
- ✅ SPADE worker gained JSON sanitisation logging constant `LOG_WORKER_RUNTIME_JSON_SANITIZED` for telemetry introspection.

## Constraints / Notes
- Maintain compatibility with existing run data; new persistence flags must migrate safely.
- Respect the logging constants added in Day 16 for dialog telemetry.
- Prioritize minimal UI disruption—default KEEP behavior should remain one click.

## Next Steps
1. Backfill architecture diagrams / sequence notes for the completed dialog flow (for doc parity).
2. Extend SPADE-BDI regression tests to cover Taxi telemetry sanitisation and headless completion.
3. Investigate outstanding SPADE suite failures (adapter loading & legacy tests) now that telemetry serialization is resolved.

---
Updated by Codex on October 27, 2025 (evening).

---
Prepared by Codex on October 27, 2025.
