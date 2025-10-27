# Day 15 — Task 2 Checklist

- [x] Rename SPADE-specific dialogs/forms to worker-scoped modules (`*_spade_bdi_form.py`).
- [x] Remove duplicate `policy_selection_dialog.py`; ensure single authoritative implementation.
- [x] Introduce worker form factory or presenter hook so MainWindow requests UI by `worker_id`.
- [x] Update MainWindow to use the new abstraction (no direct dialog imports).
- [x] Adjust presenter exports / registries (e.g., `spade_bdi_rl_worker_tabs`) to depend on the
      factory instead of re-exporting dialogs.
- [x] Update documentation to reflect new naming/DI strategy.
- [x] Run focused tests (`python -m pytest gym_gui/tests/test_worker_presenter_and_tabs.py` and
      UI-related checks if applicable).
- [x] Rename legacy signal/handler names (e.g., `agent_loadout_requested`) to form-centric naming.
