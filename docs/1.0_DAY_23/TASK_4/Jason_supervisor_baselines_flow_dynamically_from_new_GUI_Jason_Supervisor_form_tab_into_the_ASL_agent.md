## Jason Supervisor baselines: make them flow dynamically from the GUI form into the ASL agent

This note upgrades the earlier plan with concrete code grounding, crisp acceptance criteria, and a contrarian rationale for each key decision. The goal is to eliminate hard‑coded baselines in the Jason agent and drive them from a GUI form, end‑to‑end, without touching trainer wiring.

---

## Current state (grounded in code)

- Protobuf contract (`gym_gui/services/jason_bridge/bridge.proto`) defines `SupervisorStatus` with fields:
	- `active`, `safety_on`, `last_action`, `actions_emitted`, `last_error` (no configuration payload).
- Bridge server returns the above snapshot in `GetSupervisorStatus` (`gym_gui/services/jason_bridge/server.py`, method `GetSupervisorStatus`).
- Supervisor keeps only lightweight state (`gym_gui/services/jason_supervisor/service.py`, `SupervisorState`, `snapshot()`); there’s no config storage.
- Java bridge environment (`3rd_party/jason_worker/.../GymGuiBridgeEnvironment.java`) builds a `supervisor_status(...)` percept from `GetSupervisorStatus` but has no config parsing or belief injection.
- The Jason agent (`3rd_party/jason_worker/.../supervisor_agent.asl`) declares static baselines like `epsilon_base(0.15).`, `intrinsic_on(false).`, `shaping_enabled(false).`, `monitor_period_ms(1000).` which violates the requirement that these values must come from the GUI form.

Conclusion: we need a small contract extension plus plumbing to carry GUI config → service → gRPC → Java → ASL beliefs.

---

## Minimal, compatible contract change

Add an optional configuration payload to `SupervisorStatus`:

- `string config_json = 6;` (field numbers 1–5 remain unchanged).
- Empty string = “no config yet”.
- Regenerate stubs with `tools/generate_protos.sh`.

Why this and not a new RPC? See “Contrarian rationale” below.

---

## Service and server enhancements

- Add `_config: dict[str, Any] = {}` to `JasonSupervisorService` with:
	- `set_config(config: dict[str, Any])` (type guard: float/int/bool/str; ignore invalid keys, log warnings).
	- `get_config()` returns a defensive copy.
	- `config_json()` returns a compact JSON string.
- Extend `snapshot()` to include the current config (for UI inspection), but the public wire format remains `SupervisorStatus`.
- Update `GetSupervisorStatus` in the bridge server to populate `config_json` from the service. All other RPCs remain unchanged.

Why not push beliefs directly from Python? See “Contrarian rationale”.

---

## GUI: “Jason Supervisor” form/tab

- Add a new tab under the control panel (kept orthogonal to game modes) with fields:
	- Epsilon Base (0–1), Epsilon Boost Value (0–1), Monitor Period (ms), Intrinsic On (bool), Shaping Enabled (bool).
- “Apply” pushes a dict `{epsilon_base, epsilon_boost_value, monitor_period_ms, intrinsic_on, shaping_enabled}` to `JasonSupervisorService.set_config()`.
- “Reset” is optional (clears config).
- This tab does not affect trainer wiring or control modes.

---

## Java environment: parse config and inject beliefs

- After `getSupervisorStatus`, read `config_json`.
- If present, parse (prefer a stable lib such as Gson) and emit/refresh beliefs:
	- `epsilon_base(Value)`, `epsilon_boost_value(Value)`, `monitor_period_ms(Int)`, `intrinsic_on(Bool)`, `shaping_enabled(Bool)`.
- Remove existing versions of those beliefs before adding new ones to avoid duplicates.
- Malformed JSON → log and skip; do not spam the agent.

---

## ASL: drop hard‑coded baselines, rely on injected beliefs

- Remove the static declarations for baselines in `supervisor_agent.asl`.
- Bootstrap remains: `get_supervisor_status; !monitor.`
- If a plan depends on a value (e.g., `?monitor_period_ms(P)`), and it isn’t present yet, wait briefly and re‑pull status (handles “Apply” race).
- Adaptive plans stay the same; they now operate on live, GUI‑driven beliefs.

---

## Contrarian rationale: why do it this way?

1) In‑band `config_json` vs a new `GetConfig` RPC
- Pro: One round‑trip, fewer client branches, and atomic view of status+config (consistent with UI overlays in `control_panel.py`).
- Pro: Backward compatibility—old clients ignore the extra field; new clients gain config with zero new methods.
- Con: Mixed concerns in one message. We accept this for operational simplicity and versioning stability.

2) JSON blob vs a typed proto message
- Pro: Extensible without schema churn; the GUI form can add knobs without breaking the Java MAS.
- Pro: We already carry JSON (`params_json`) for updates; aligns with existing patterns.
- Con: Fewer type guarantees. We mitigate with strict runtime checks, logs, and small, well‑defined keys.

3) Pull from MAS (Java) and inject beliefs vs pushing beliefs from Python
- Pro: Keeps supervisor and bridge server ignorant of Jason’s belief base—clean separation and fault isolation, as originally required.
- Pro: MAS owns its own lifecycle; retries and partial failures are local, not cross‑process.
- Con: Slightly more client code; acceptable for the isolation benefits.

4) A dedicated “Supervisor” GUI tab vs mixing into existing training dialogs
- Pro: Clear boundary; avoids accidental coupling with trainer/worker paths (a hard constraint in this repo).
- Pro: Easier operator story: tune supervision parameters without re‑triggering training pipelines.
- Con: Another tab to maintain; worth it for clarity and safety.

---

## Data model (v1)

`config_json` keys and types:
- `epsilon_base: float [0,1]`
- `epsilon_boost_value: float [0,1]`
- `monitor_period_ms: int (>0)`
- `intrinsic_on: bool`
- `shaping_enabled: bool`

Precedence and defaults:
- If `config_json` is empty or a key is missing, the Java side does not emit that belief; ASL waits/retries.
- Avoid injecting defaults from the bridge; values must come from the GUI form (explicit operator intent).

---

## End‑to‑end flow (happy path)

1) Operator opens “Jason Supervisor” tab and applies values.
2) `JasonSupervisorService.set_config()` stores the validated dict.
3) `GetSupervisorStatus` returns the usual snapshot + `config_json`.
4) Java MAS parses `config_json`, clears prior beliefs, injects new ones.
5) ASL plans read the beliefs and execute (e.g., boosting epsilon on plateau).

---

## Migration & compatibility

- Old clients: unchanged; `config_json` is ignored.
- New clients: must tolerate empty `config_json` and retry until the operator applies config.
- Stubs regeneration: required for both Python and Java after the proto change.

---

## Acceptance criteria

- With no GUI apply, `supervisor_status(...)` arrives without injected baseline beliefs in MAS.
- After GUI apply, MAS beliefs reflect the applied values within one poll cycle.
- Removing a belief key from the GUI config causes the MAS belief to disappear after the next poll (no stale duplication).
- No changes to trainer wiring or CleanRL worker paths.

---

## Work items (scoped, file‑level)

1) Proto: add `config_json = 6;` in `gym_gui/services/jason_bridge/bridge.proto`; run `tools/generate_protos.sh`.
2) Service: add config store and JSON helpers in `gym_gui/services/jason_supervisor/service.py`.
3) Server: include `config_json` in `GetSupervisorStatus` in `gym_gui/services/jason_bridge/server.py`.
4) GUI: create the “Jason Supervisor” tab and call `set_config()`.
5) Java: parse `config_json` and inject beliefs in `GymGuiBridgeEnvironment.java`.
6) ASL: remove static baselines and rely on injected beliefs in `supervisor_agent.asl`.
7) Tests: extend `test_jason_supervisor_service.py` and `test_jason_bridge_server.py` for the new field.

---

## Risks & mitigations

- Malformed JSON → log and skip; keep prior beliefs until next good snapshot.
- UI drift → tab is isolated; no changes to existing modes.
- Version skew → extra field is optional; old clients function as before.

---

## Optional dev tips

- Generate stubs headlessly: use `tools/generate_protos.sh` (requires `grpcio-tools`).
- The bridge server is env‑gated; set `JASON_BRIDGE_ENABLED=1` to auto‑start with the GUI.

---

By keeping the change surface tiny (one new proto field, one new GUI tab) and pushing complexity to the edges (JSON parsing at the MAS boundary), we satisfy the “values come from the GUI” requirement, preserve fault isolation, and avoid destabilizing trainer code paths.

