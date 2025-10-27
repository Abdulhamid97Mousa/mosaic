# Initial Plan — Day 16 Task 2  
**Topic:** Multi-Agent Capable Logging Pipeline

## 1. Situation Assessment

- The existing structured logging stack (log constants + `log_constant()` helper + `CorrelationIdAdapter`) already carries `run_id`/`agent_id`, but assumes a single active agent per UI session.
- Multi-agent training (e.g., two agents facing off) introduces **three simultaneous scopes**:
  1. **Run scope** — shared context (match id, arena).
  2. **Agent scope** — per agent logs, telemetry, and rendering.
  3. **Interaction scope** — events involving two+ agents (collisions, rewards transfer).
- Current UI filters group by component/subcomponent only; agent-specific slicing is not first-class. Worker stdout is also single-stream, making it hard to emit agent-prefixed lines.

## 2. Goals

1. **Zero-cost agent filtering** in the UI log viewer (dropdown / quick filter on `agent_id`).
2. **Dynamic agent logger adapters** so each agent can emit logs through its own context without hand-building `extra` dicts.
3. **Extendable log constant schema** to declare whether a log is `run`, `agent`, or `interaction` scoped.
4. **Backwards compatibility** for current single-agent scenarios.

## 3. Proposed Enhancements

### 3.1 LogConstant Schema

- Add optional fields:
  - `scope: Literal["run","agent","interaction"]`
  - `default_agent: str | None` (used for constants always tied to a specific agent stream)
  - `dynamic_fields: tuple[str, ...]` to document required `extra` keys (e.g., `("agent_id",)`).
- Example:
  ```python
  LOG_AGENT_OBSERVED_REWARD = _constant(
      "LOG950",
      "INFO",
      "Agent observed reward tick",
      component="Agent",
      subcomponent="Telemetry",
      scope="agent",
      dynamic_fields=("agent_id", "opponent_id"),
      tags=_tags("reward", "multi_agent"),
  )
  ```

### 3.2 Helpers & Context Managers

- Extend `log_constant()` helper with:
  - `agent_id: str | None` kwarg; merges into payload when provided.
  - Validation: if constant declares `scope="agent"`, enforce `agent_id` in payload.
  - Optionally return `logging.LogRecord` for advanced piping (future).
- Introduce `AgentLogAdapter`:
  ```python
  class AgentLogAdapter(CorrelationIdAdapter):
      def __init__(self, base_logger, run_id, agent_id):
          super().__init__(base_logger, {"run_id": run_id, "agent_id": agent_id})
  ```
  - Factory utility: `get_agent_logger(run_id, agent_id)` returning adapter cached per `(run_id, agent_id)`.

### 3.3 Logger & Handler Configuration

- Update `CorrelationIdFilter` to set `agent_id="unresolved"` instead of `"unknown"` when missing, enabling UI to surface “unassigned” logs.
- Extend formatters/filters:
  - Add `scope` and optional `opponent_agent` fields to `_DEFAULT_FORMAT`.
  - Ensure `ComponentFilter` registers `scope` in the registry for UI consumption (e.g., `component=Agent`, `subcomponent=Telemetry`, `scope=agent`).
- Introduce `AgentRouteHandler` (optional future):
  - Fan-out records into per-agent rotating files under `var/logs/agents/{run_id}/{agent_id}.log`.

### 3.4 Dispatcher & Worker Output

- Modify worker stdout structured line to include `agent_id` key. Dispatcher bridge re-emits with constant scope metadata.
- For multi-agent workers, spawn a per-agent telemetry emitter that wraps `AgentLogAdapter` so each environment step logs against the right agent context.

## 4. UI / Filtering Impact

- Update runtime log viewer:
  - Add agent filter dropdown populated from observed `agent_id`s (pluck from log stream or metadata).
  - When user selects agent, apply combined filter `component in ("Agent","Adapter") AND agent_id=<selected>`.
- Provide “interaction” view that shows logs with `scope="interaction"` (both agents).

## 5. Implementation Steps

1. **Schema Update**
   - Extend `LogConstant` dataclass + `ALL_LOG_CONSTANTS` list; add validation for new fields.
   - Update docs referencing log constant structure.
2. **Helper Enhancements**
   - Adjust `log_constant()` signature, add validation + tests.
   - Provide `AgentLogAdapter` + factory, update worker/runtime to use it.
3. **Logger Configuration**
   - Extend formatter string and filters; ensure tests cover missing agent defaults.
   - Optional: configure per-agent file handler.
4. **Dispatcher/Worker Alignment**
   - Ensure worker structured logs include agent metadata; update dispatcher re-emit to preserve.
5. **UI Filtering**
   - Update runtime log handler UI to surface agent filters and apply them to log table queries.
6. **Testing**
   - Unit: helper validation, adapter caching, formatter output.
   - Integration: multi-agent simulated telemetries verifying per-agent log separation.

## 6. Risks / Considerations

- **Performance:** Additional file handlers per agent may add I/O overhead; start with in-memory filtering only.
- **Telemetry Sync:** Need to keep `agent_id` consistent across telemetry events and logs to avoid ambiguous records.
- **Backward Compatibility:** Ensure tests confirm single-agent flows behave identically (agent defaults to `unknown`).

## 7. Deliverables (Day 16 Scope)

- Updated logging schema + helper.
- Agent log adapter utilities.
- Plan + stubs for UI filter alterations.
- Documentation & tests covering the new behavior.
