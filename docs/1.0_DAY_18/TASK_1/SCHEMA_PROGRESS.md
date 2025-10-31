# Schema & Telemetry Contract Progress (October 31, 2025)

## What Changed
- Migrated vector metadata constants to `gym_gui/constants/constants_vector.py` and aligned all imports.
- Added the telemetry schema scaffold (`gym_gui/core/schema/`) to formalise step payload contracts and vector metadata descriptors.
- Updated `SessionController` to enforce schema expectations, emit structured log signals, and guard unsupported autoreset modes.
- Extended validation services so callers can retrieve JSON schema snapshots for downstream tooling or remote workers.
- Registered Atari and MiniGrid schema overlays via `TelemetrySchemaRegistry`, paving the way for family-specific validation without breaking the default contract.
- Headless SPADE-BDI worker now advertises schema metadata (ID/version), space signatures, vector metadata, and per-step `time_step` counters in emitted telemetry.
- Training configurations handed off to the trainer now embed `schema_id`, `schema_version`, and the JSON `schema_definition` so remote bootstrap scripts can reject mismatched payloads before launch.
- Fast-training/TensorBoard-only runs log the same schema identifiers, keeping analytics in sync even when per-step telemetry is disabled.
- Embedded TensorBoard viewer launches automatically in the GUI (auto-start, WebEngine-backed) with optional log cleanup when the tab is closed.
- Expanded base requirements with MiniGrid/Atari runtime dependencies (`requirements/base.txt`).

## Verification
- Added `spade_bdi_worker/tests/test_schema_contract.py` covering schema registry registration, vector metadata fragments, validation service exports, and the Atari/MiniGrid overlays.
- Extended GUI integration tests to assert schema hints on emitted steps and run-start payloads.
- Updated UI logging tests and worker presenter tests to expect the new schema identifiers in training configurations.
- Ran `pytest spade_bdi_worker/tests/test_schema_contract.py` inside the project virtual environment to confirm the new scaffolding behaves as expected.

## Next Up
- After schema-backed behaviour proves stable, onboard MiniGrid and Atari families by registering specialised schema overlays in the registry.
- Wire schema retrieval into trainers/workers so remote components validate payloads before publishing telemetry.
