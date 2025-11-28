# Environment Selector & Telemetry Widget Impact Assessment

## Current UI Pain Points (resolved)

1. **Human Control → Environment group**
   - `ControlPanelWidget.populate_games()` (`gym_gui/ui/widgets/control_panel.py`, lines 320+) simply iterates whatever `GameId` list it receives and calls `_game_combo.addItem(get_game_display_name(game), game)`. No metadata is attached, so the combo becomes ~50 entries on nightly builds.
   - The `_create_environment_group()` helper never consults `ENVIRONMENT_FAMILY_BY_GAME`, so there is no visual grouping by `EnvironmentFamily` (ToyText, MiniGrid, ALE, etc.). Operators must remember which IDs belong together even though `gym_gui/core/enums.py` already exposes the mapping.
   - Because `_current_game` is the only stored selection, there is no state slot for “family,” making it impossible to default to a preferred subset without rebuilding the widget.

2. **Single-Agent Mode → CleanRL worker widget**
   - `_create_training_group()` (`control_panel.py`, lines 760–820) instantiates a “CleanRL Environment” combo that is populated by `get_cleanrl_environment_choices()` but then disabled/hidden until a worker is selected, so users see a blank widget that duplicates the modal form.
   - `CleanRlTrainForm.__init__` (`gym_gui/ui/widgets/cleanrl_train_form.py`, ~230–330) builds a *second* combo by iterating the same `CLEANRL_ENVIRONMENT_CHOICES` global. Any change to the environment roster requires touching both files, creating drift risk.
   - The CleanRL form also mixes ToyText, Procgen, ALE, etc., within a single combo because `_build_environment_choices()` only appends the family in parentheses (e.g., `"frozen_lake (Toy Text)"`). There is no filtering UI even though the form already imports `EnvironmentFamily` and `ENVIRONMENT_FAMILY_BY_GAME`.

3. **Telemetry Mode context**
   - `_build_ui()` lays out the Human Control column as a bare `QVBoxLayout` containing `_create_environment_group()`, `_create_game_config_scroll()`, `_create_mode_group()`, `_create_control_group()`, `_create_telemetry_mode_group()`, and `_create_status_group()`—none of which live inside a `QScrollArea`.
   - After the “Fast Lane Only” radio buttons moved into their own group (`_create_telemetry_mode_group()`), the stack already fills a 1080p screen. Adding an “Environment Family” selector will push the Status group below the fold unless we wrap the column in a scroll area or collapse existing group boxes.

## Final UX Behavior

- **Split selection into two widgets:**
  1. `Environment Family` combo shows `EnvironmentFamily` enum values (ToyText, MiniGrid, Classic Control, etc.).
  2. `Environment` combo displays only games belonging to the selected family.
  3. Default family = ToyText so FrozenLake/CliffWalking remain front-and-center for newcomers.

- **Apply the same pattern to CleanRL forms:** both the inline widget (Single-Agent Mode) and the modal `CleanRlTrainForm` should reuse a shared helper (`build_family_to_game_map()`) so the list of games stays in sync.

- **Telemetry mode widget remains where it is**, but we need to ensure the added Environment Family combo does not push Status too far down; consider giving the Human Control column a scroll area if necessary.

## Code evidence snapshot

| Concern | Source location | Why it matters |
| --- | --- | --- |
| Flat game combo | `ControlPanelWidget.populate_games()` & `_create_environment_group()` (`gym_gui/ui/widgets/control_panel.py`, lines 320–430) | Populates `_game_combo` with every `GameId` without consulting `ENVIRONMENT_FAMILY_BY_GAME`, so there is no built-in grouping or filtering. |
| CleanRL duplication | `_create_training_group()` (`control_panel.py`, lines 760–820) and `CleanRlTrainForm.__init__` (`gym_gui/ui/widgets/cleanrl_train_form.py`, lines 230–330) | Both widgets rebuild the same environment list from `get_cleanrl_environment_choices()`; keeping them in sync is manual work. |
| Metadata already exists | `EnvironmentFamily` + `ENVIRONMENT_FAMILY_BY_GAME` (`gym_gui/core/enums.py`, lines 18–230) | Provides all family labels needed for the new selector, so no schema changes are required—only UI wiring. |
| Column sprawl | `_build_ui()` layout (`control_panel.py`, lines 500–620) | Human Control column stacks six groups without a scroll area, so inserting another combo will push telemetry/status widgets below the viewport on 1080p monitors. |

## Impacted Modules (completed)

| Area | Files | Notes |
| --- | --- | --- |
| Human Control tab | `gym_gui/ui/widgets/control_panel.py` (environment group, state handling, `set_game` logic) | Need to store both selected family and game, update `_game_combo` when family changes, persist defaults. |
| Environment metadata | `gym_gui/config/game_configs.py`, `gym_gui/config/game_config_builder.py` | Already expose `EnvironmentFamily`; no schema changes expected, but helper functions to build per-family lists will likely live near these configs. |
| CleanRL inline widget | `gym_gui/ui/widgets/control_panel.py::_create_training_group` | Currently shows “CleanRL Environment” combo; should adopt the family+game selector or remove the redundant combo and defer to the modal form. |
| CleanRL train form | `gym_gui/ui/widgets/cleanrl_train_form.py` | Needs the same split selection plus reuse of helper data to avoid double maintenance. |
| Docs | `docs/1.0_DAY_28/TASK_1/README.md` and new `TASK_2` notes | Must explain the new Environment Family selector so operators know how filtering works. |

## Validation & Testing considerations

- Update existing tests (e.g., `test_cleanrl_fastlane_wrapper`) if helper signatures change.
- GUI smoke tests should cover selecting different families and ensuring only relevant games appear.
- Ensure telemetry/logging (e.g., run metadata) still records the correct `game_id` even though the user now selects via two combos.

## Status

- [x] Implement shared helper & UI splits for Environment Family selectors.
- [x] Update ControlPanel & CleanRL widgets to use the helper.
- [x] Documented behavior in TASK_1 briefs; screenshots pending later release notes.
