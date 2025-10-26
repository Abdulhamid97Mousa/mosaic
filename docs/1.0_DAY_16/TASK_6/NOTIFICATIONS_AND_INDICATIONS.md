# Task 6 — Notifications & Indicators Plan (Contrarian Strategy)

**Scope:** Reimagine notifications, indicators, and tab-closure affordances for live agent training and telemetry views. Focus on how the UI signals state, prompts users when closing tabs, and reconciles database retention for each run.

---

## 1. Problem Statement (Contrarian Framing)

- **Status opacity:** Live tabs give no at-a-glance signal about run health (buffer overflow, pending replay, unsynced DB state). Users only discover problems through logs.
- **All-or-nothing closure:** Closing a tab silently discards UI context and leaves the run data intact without asking whether to purge, archive, or keep it. We never differentiate between intentional clean-up and accidental closures.
- **Blocking bias:** Current UX leans on modal busy indicators (_BusyDialog) which freeze interaction. Overusing them for warnings would induce dialog fatigue and encourage reflexive "OK" clicks.
- **Missed escalation paths:** Errors funnel straight to logs instead of surfacing as tiered indicators (badge → inline message → modal) that escalate only when necessary.
- **Fragmented state tracking:** Telemetry widgets track dropped frames/episodes but do not bubble those metrics up to the tab header, leaving the user blind to silent data loss.

Contrarian stance: **Start from non-disruptive, persistent indicators and treat modals as the exception.** The UI should continuously reflect run status, making confirmation dialogs a last-mile guardrail when the user attempts destructive actions (like closing an active tab).

---

## 2. Current State Audit

| Area | Observations | Risks |
|------|--------------|-------|
| `gym_gui/ui/widgets/render_tabs.py` | `RenderTabs._close_dynamic_tab` removes tabs immediately, calls `cleanup()`, deletes widget, no user confirmation. | Accidental tab closure loses live context; user cannot choose whether to purge run data. |
| `gym_gui/ui/widgets/live_telemetry_tab.py` | Tracks `_dropped_steps`, `_dropped_episodes`, `_pending_render_timer_id`, but indicators remain internal. | Silent quality degradation; closing tab gives no hint that run had issues needing follow-up. |
| `gym_gui/ui/widgets/busy_indicator.py` | Provides modal progress overlay, but no reusable confirmation dialog or toast infra. | Any new prompt risks overloading busy indicator, creating blocking UX. |
| `gym_gui/ui/widgets/control_panel.py` & `main_window.py` | No global notice surface (status bar, badge) to show outstanding runs or unsaved state. | Users cannot see whether closing the window will drop data or terminate runs. |
| Telemetry storage | No notion of soft-delete vs archive; closing tab is decoupled from DB lifecycle. | Inconsistent retention policies; data lingers without user intent. |

---

## 3. Indicator & Notification Principles

1. **Persistent before modal:** Default to inline badges, colored tab chips, or status bar notices. Invoke modal confirmation only when the user initiates a destructive action (close tab, terminate run).
2. **Run-centric signals:** Indicators track run_id rather than widget state, enabling consistent messaging across Telemetry, Replay, and Worker tabs.
3. **Graduated severity:**
   - Info → badge with tooltip
   - Warning → badge + inline banner inside tab
   - Critical → banner + modal confirmation (only on user action)
4. **Explicit retention choices:** Closing a tab offers three explicit paths: Keep data, Archive snapshot (freeze run data for future replay), Delete run data.
5. **Telemetry provenance:** Every indicator must cite the metric triggering it (e.g., "Dropped 120 steps (24%)"). No generic "unsaved changes" phrasing.

---

## 4. Indicator Taxonomy & UI Placement

| Indicator Type | Trigger | Visual Treatment | Delivery Surface |
|----------------|---------|------------------|------------------|
| **Run Active** | Live stream connected, receiving steps | Tab label badge "LIVE" (green), pulsing dot in corner | `RenderTabs` tab bar |
| **Stale Stream** | No payload for N seconds | Badge switches to amber "PAUSED", tooltip explains last timestamp | Tab bar + inline banner |
| **Data Loss** | `_dropped_steps` / `_dropped_episodes` exceeds threshold | Red counter chip `Dropped: 34` next to badge, inline banner with escalation link to logs | Tab header + TelemetryTab header |
| **Pending Replay** | Telemetry DB still flushing (as reported by service) | Blue badge "Pending Flush"; disable deletion option until flush completes | Tab bar + confirmation dialog state |
| **Unsynced Annotations** | User-added notes (future) not saved | Grey badge "Draft"; confirm on close | Tab bar |

Implementation notes:

- Use `QTabBar.setTabData` to store run state and drive custom close button painting.
- Introduce `IndicatorState` dataclass (run_id, severity, badges) to be shared between LiveTelemetryTab and RenderTabs.

---

## 5. Tab Closure Decision Tree (Keep vs Archive vs Delete)

```text
Close request → Is run still active?
  ├─ Yes → Show confirmation dialog
  │    ├─ Keep & Close Tab → Stop streaming, keep data, leave run metadata untouched
  │    ├─ Archive Snapshot → Trigger TelemetryService snapshot/export; mark run archived
  │    └─ Delete Run Data → Call TelemetryService to purge run_id (steps + episodes)
  └─ No → Check outstanding indicators
       ├─ Indicators present → Inline banner warns (badge persists), allow quick archive/delete
       └─ Clean state → Close immediately (optional toast summarising run)
```

User-visible dialog content (contrarian focus):

- Title: "Close Live Training Tab"
- Body: "Run `agent-A` collected 3 episodes (2 dropped). Choose what happens to its data."
- Options (radio buttons + descriptions):

  1. Keep data (default): "Retain telemetry and keep run history accessible."
  2. Archive snapshot: "Seal the run for replay; move it to archived run list."
  3. Delete data: "Remove telemetry from storage and worker cache." (warning icon)
- Checkbox: "Apply to other tabs from this run" (batch close)
- Secondary link: "View dropped items" opens telemetry log in panel.

---

## 6. Notification Delivery Strategy

1. **Inline banners** (non-modal `QFrame`) inside LiveTelemetryTab for warnings/errors. Automatically appear when indicator severity rises.
2. **Status bar summary** in `MainWindow`: "Runs needing action: 2 (1 pending deletion, 1 flushing)."
3. **Contextual toasts** (non-blocking `QSystemTrayIcon` or custom popover) when background events complete (archives done, deletions finished).
4. **Modal confirmation** only at tab close or run termination, reusing a new reusable dialog class (extend busy indicator file or introduce `confirm_action.py`).

---

## 7. Implementation Phases

| Phase | Focus | Key Changes |
|-------|-------|-------------|
| **Phase 0 – State Plumbing** | Track indicator data | Add `IndicatorState` dataclass; extend LiveTelemetryTab to emit state via Qt signal; update RenderTabs to subscribe and adjust tab badges. |
| **Phase 1 – Visual Indicators** | Surface badges/banners | Custom tab bar painting for badges; inline banner component; status bar integration. |
| **Phase 2 – Tab Closure Flow** | Confirmation dialog + DB choices | Introduce reusable confirmation dialog; modify `_close_dynamic_tab` to request decision; integrate with TelemetryService for keep/archive/delete actions. |
| **Phase 3 – Persistence Actions** | Implement archive/delete | Add TelemetryService APIs (`archive_run`, `delete_run`), ensure worker/trainer updates; handle SQLite cleanup; log audit trail. |
| **Phase 4 – Testing & Telemetry** | Automated coverage | UI tests for indicator transitions; smoke tests for archive/delete paths; instrumentation to log user decisions. |

---

## 8. File & Module Impact

| Component | Action |
|-----------|--------|
| `docs/1.0_DAY_16/TASK_6/NOTIFICATIONS_AND_INDICATIONS.md` | Planning artefact (this file). |
| `gym_gui/ui/widgets/render_tabs.py` | Badge rendering, close workflow, confirmation dialog invocation, batch operations. |
| `gym_gui/ui/widgets/live_telemetry_tab.py` | Emit indicator updates, inline banner injection, maintain dropped metrics thresholds. |
| `gym_gui/ui/widgets/busy_indicator.py` or new dialog module | Add reusable confirmation dialog (non-blocking variant). |
| `gym_gui/services/telemetry.py` & `storage/` modules | Archive/delete API surface; ensure WAL checkpoints and retention. |
| `gym_gui/ui/main_window.py` | Status bar summary, cross-tab indicator coordination. |
| `gym_gui/ui/widgets/control_panel.py` | Optional: display live indicator icons alongside run list (future). |
| Tests under `gym_gui/tests/ui/` & `services/` | Add regression coverage for decision flows and persistence outcomes. |

---

## 9. Open Questions (for validation)

1. What constitutes an "archive"? (Option A: retain telemetry tables with archived flag; Option B: export to dedicated replay store.)
2. Should delete immediately purge from SQLite or schedule via maintenance task? Consider WAL and concurrent readers.
3. How to signal when archive/delete completes asynchronously? (Proposed: toast + status bar update.)
4. Do we cascade decisions to Agent Replay tabs referencing the same run? Need coordination to avoid stale UI elements.
5. How should headless trainer sessions advertise indicator state (CLI parity)?

---

## 10. Next Steps Checklist

- [ ] Align with telemetry persistence owners on archive vs delete semantics.
- [ ] Prototype `IndicatorState` emission in LiveTelemetryTab and record threshold heuristics.
- [ ] Design confirmation dialog wireframe (copy deck + button order) with UX review.
- [ ] Draft TelemetryService API changes and evaluate migration impact.
- [ ] Plan automated UI tests (PySide/Qt fuzzing) for close-tab workflow.
