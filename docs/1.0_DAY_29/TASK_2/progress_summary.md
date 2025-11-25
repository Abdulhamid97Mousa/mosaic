# Jason Worker Log Refactor Summary

## Assigned Scope

- Remove the old "supervisor" logging vocabulary so Jason integrations are described as "worker" interactions.
- Ensure dependent services (Jason supervisor shim and gRPC bridge) keep functioning while emitting the new log constants.

## Touched Files & Rationale

| File | Change | Reason |
| --- | --- | --- |
| `gym_gui/logging_config/log_constants.py` | Replaced `LOG_SERVICE_SUPERVISOR_*` symbols with `LOG_SERVICE_JASON_WORKER_{EVENT,WARNING,ERROR}` and updated the exported registry. | Establishes the new worker-centric structured log identifiers used across the GUI services. |
| `gym_gui/services/jason_supervisor/service.py` | Updated imports and log calls to the new worker constants. | Keeps the interim supervisor shim aligned with the renamed constants so existing UI overlays continue to receive activity logs. |
| `gym_gui/services/jason_bridge/server.py` | Swapped bridge logging to the worker constants and mapped rejects/failures to warning/error severities. | Prevents gRPC bridge telemetry from referencing the removed supervisor symbols while preserving diagnostics. |

## Validation

- Ran `codacy_cli_analyze` for every modified file (`log_constants.py`, `jason_supervisor/service.py`, `jason_bridge/server.py`).
  - No new lint findings; only the pre-existing Lizard warnings on `log_constants.py` (file size and complexity) remain.

## Follow-Up Considerations

- Fold the remaining `JasonSupervisorService` usage into the new `JasonWorkerService` implementation so the service locator and bridge no longer depend on the supervisor shim.
- Revisit `log_constants.py` to split modules and reduce cyclomatic complexity if we keep extending the structured log catalog.
