# Day 20 — Task 4: MiniGrid Integration Plan (Drafted 2025-11-03)

## Objective

Expose MiniGrid environments inside `gym_gui` so operators can launch reinforcement-learning experiments against the same grid-world tasks already supported in the `xuance` fork (`xuance/environment/single_agent_env/minigrid.py`). The integration must feel native to the GUI workflow: environments appear in the catalog, training runs hydrate the correct wrappers, and telemetry/contracts stay compatible with the existing trainer/worker pipeline.

## Source Anchors & Audit Targets

- **Reference implementation:** `xuance/environment/single_agent_env/minigrid.py` (baseline wrappers, reward shaping, observation conversions).
- **Current GUI environment plumbing:**
  - Environment registry: `gym_gui/core/env_registry.py`.
  - Worker entry points: `gym_gui/services/trainer/dispatcher.py`, `gym_gui/workers/*`.
  - GUI selectors/forms: `gym_gui/ui/widgets` (environment dropdowns, metadata panels).
- **Logging contract:** `gym_gui/logging_config/log_constants.py` (+ new MiniGrid-specific constants), `_log_constant` helper usages across workers.
- **Dependency manifests:** `requirements/minigrid.txt?` (to be created if absent) and top-level `requirements.txt` consolidation.
  - Take inventory of optional extra groups (e.g., `requirements/cleanrl_worker.txt`) to decide where MiniGrid lives.

## Xuance Feature Parity Checklist

| Capability | Xuance Baseline (\* = from `xuance/environment/single_agent_env/minigrid.py`) | gym_gui Target | Gaps / Notes |
| --- | --- | --- | --- |
| Reward scaling | `reward_scale = 10` (\*) | `MiniGridConfig.reward_multiplier` default 10 | Need toggle in UI + adapter to respect per-env overrides. |
| Observation wrappers | `RGBImgPartialObsWrapper` + `ImgObsWrapper` (\*) | Adapter must compose wrappers conditionally | Confirm telemetry schema (RGB vs. flatten) before enabling analytics. |
| Seed discipline | `env.reset(seed, options)` (\*) | `SessionController` should forward seeds; config exposes `seed` | Ensure CleanRL worker passes the same seed to maintain reproducibility. |
| Action mapping | Hard-coded 7 discrete actions (\*) | `human_input` maps keyboard → 7 actions | Add validation so games with fewer actions raise clear error. |
| Logging | Xuance prints reward + done to stdout | GUI must use structured log constants (`LOG518`–`LOG521`) | Align log levels with telemetry ingestion expectations. |
| Config surface | Xuance uses YAML + CLI args | GUI exposes `MiniGridConfig` + control panel overrides | Document parameter equivalence for operators transitioning from Xuance. |
| Rendering | Xuance headless training (no Qt) | GUI requires replay thumbnails and trainer render payloads | Verify render hints survive new wrapper chain. |

**Guideline Alignment:** Xuance emphasizes thin wrappers and reproducibility. Our plan mirrors that ethos by keeping adapters lightweight, exposing reward scaling explicitly, and reusing the same wrapper order. Deviations (e.g., structured logging, Qt UI plumbing) are necessary for GUI integration and will be documented so contributors recognise the delta from Xuance guidelines.

## Impacted Files & Ownership Map

| Area | File(s) / Module(s) | Owner Notes |
| --- | --- | --- |
| Adapter layer | `gym_gui/core/adapters/minigrid.py`, `gym_gui/core/adapters/__init__.py` | Ensure interfaces stay consistent with other adapters. |
| Config & registry | `gym_gui/config/game_configs.py`, `gym_gui/config/game_config_builder.py`, `gym_gui/core/factories/adapters.py`, `gym_gui/core/enums.py` | Update dataclasses, defaults, and factory wiring. |
| Constants & logging | `gym_gui/logging_config/log_constants.py`, `gym_gui/constants/__init__.py`, `gym_gui/constants/game_constants.py`, `gym_gui/constants/constants_wandb.py` | Reserve LOG codes (518–521) and surface reward multiplier defaults. |
| UI composition | `gym_gui/ui/widgets/control_panel.py`, `gym_gui/ui/environments/gym/config_panel.py`, `gym_gui/ui/environments/minigrid/config_panel.py`, `gym_gui/ui/panels/control_panel_container.py`, `gym_gui/ui/main_window.py` | Refactor control panel into family-specific helpers as per TODO. |
| Human input | `gym_gui/controllers/human_input.py` | Map MiniGrid actions, guard when space size ≠ 7. |
| Session lifecycle | `gym_gui/controllers/session.py`, `gym_gui/services/trainer/service.py` | Handle `MiniGridConfig`, type hints, and settings overrides. |
| Docs & game info | `docs/1.0_DAY_20/TASK_4/`, `gym_gui/game_docs/game_info.py` | Maintain operator guides and mission text per environment. |
| Tests | `gym_gui/tests/test_minigrid_adapter.py`, regression suites touching toy-text/Box2D | Expand coverage for reward scaling + config round-trips. |

Include these paths in PR description and ensure ownership sign-off for adapter + UI teams.

## Work Breakdown Structure (Expanded)

1. **Baseline Reconnaissance**
   - Diff the xuance MiniGrid wrapper against upstream `gymnasium_minigrid` to catalog features we must keep (e.g., seed handling, frame stacking, flattened observations, reward scaling ×10, partial obs toggles).
   - Inventory existing environment adapters in `gym_gui` (CartPole, Procgen, GridWorld if present) to mirror structure/abstractions.
   - Record log codes used by comparable adapters so we can add MiniGrid-specific log constants in the same range (e.g., `LOG_ENV_MINIGRID_LOAD_ERROR`).

2. **Core Environment Adapter**
   - Create `gym_gui/environments/minigrid.py` that wraps MiniGrid tasks with the GUI’s observation/action interface contract.
   - Implement spec translation helpers (state shape, discrete vs. multi-discrete actions) to feed telemetry schemas.
   - Align behaviour with xuance’s wrapper: optional `RGBImgPartialObsWrapper`/`ImgObsWrapper`, deterministic `env.reset(seed=...)`, flattened observations appended with `direction`, reward scaling factor (configurable constant, default 10 to parity with xuance).
   - Emit structured logs for lifecycle: use new constants `LOG_ENV_MINIGRID_BOOT`, `LOG_ENV_MINIGRID_STEP`, `LOG_ENV_MINIGRID_ERROR` (codes TBD in controller/service ranges) with metadata `{"env_id": ..., "render_mode": ...}`.

3. **Environment Registry Wiring**
   - Register MiniGrid variants (e.g., `MiniGrid-Empty-5x5-v0`, `MiniGrid-DoorKey-8x8-v0`, `MiniGrid-LavaGapS7-v0`) in the environment catalog with human-readable labels, difficulty tags, and default configs.
   - Surface config knobs (max steps, seed, agent start, partial obs toggle, render mode, reward multiplier constant) and ensure serialization to run configs.
   - Introduce constants enumerating supported environment IDs (`MINIGRID_ENV_IDS`) so API and UI stay synchronized.

4. **GUI Updates**
   - Update environment dropdowns to group MiniGrid tasks under a dedicated section.
   - Add contextual info panel (difficulty, grid size, reward key) for MiniGrid selections.
   - Display log-code aware status banners (e.g., if `LOG_ENV_MINIGRID_ERROR` occurs, surface actionable hint).
   - Ensure replay and rendering subsystems (`gym_gui/replays`, `gym_gui/rendering`) can consume MiniGrid frame payloads without additional wrappers.

5. **Trainer & Worker Compatibility**
   - Ensure trainer-side validation accepts MiniGrid observation/action spaces (flattened `Box` high=255, dtype uint8, discrete action space size 7).
   - Extend worker bootstrap (CleanRL/SPADE/headless as applicable) to instantiate MiniGrid gym environments via the new adapter.
   - Route adapter lifecycle logs through `TrainerService` log bridge so MiniGrid events appear alongside existing environment telemetry.

6. **Dependency & Build System Adjustments**
   - Add `minigrid` (Gymnasium MiniGrid) to appropriate requirement sets; gate optional deps if necessary (`pip install gymnasium-minigrid` + `pygame`).
   - Document installation/testing implications (headless flag for CI, `SDL_VIDEODRIVER=dummy`).
   - Ensure extra requirement file declares pinned versions consistent with xuance (`gymnasium>=0.28`, `gymnasium-minigrid>=2.2.2` when validated).

7. **Testing & Verification**
   - Author unit tests for the MiniGrid adapter (observation encoding, reset/step contract, reward scaling toggles, log code emissions via caplog fixture).
   - Add integration smoke test launching a MiniGrid environment through the trainer (short horizon, verifying run registry, log codes, config serialization).
   - Provide manual validation checklist for GUI interaction (screenshot or log capture) referencing log code expectations.

8. **Documentation & Release Notes**
   - Draft operator doc outlining available MiniGrid scenarios and known limitations, including table of supported env IDs, reward multipliers, observation formats.
   - Update changelog/README sections referencing environment coverage.
   - Add snippet demonstrating reading MiniGrid-specific logs (e.g., `LOG_ENV_MINIGRID_STEP` code) and debugging steps.

## Logging & Constants Matrix

| Domain | Identifier(s) | Purpose | Action Items |
| --- | --- | --- | --- |
| Log constants | `LOG518` (`LOG_ENV_MINIGRID_BOOT`), `LOG519` (`LOG_ENV_MINIGRID_STEP`), `LOG520` (`LOG_ENV_MINIGRID_ERROR`), `LOG521` (`LOG_ENV_MINIGRID_RENDER_WARNING`) | Structured lifecycle and diagnostic coverage for MiniGrid adapter | Emit from adapter lifecycle, surface in trainer telemetry, add doc examples. |
| Reward scaling | `MiniGridConfig.reward_multiplier` (default `10.0`) | Maintain Xuance parity for sparse-reward visibility | Expose in control panel, persist in session snapshots, test override path. |
| Wrapper toggles | `MiniGridConfig.partial_observation`, `MiniGridConfig.image_observation` | Control wrapper composition (`RGBImgPartialObsWrapper`, `ImgObsWrapper`) | Bind to UI checkboxes; default to `True`. |
| View limits | `MiniGridConfig.agent_view_size`, `MiniGridConfig.max_episode_steps` | Manage curriculum difficulty and truncation | Map UI spin boxes; treat `0` as `None` before dispatch. |
| Seed/render | `MiniGridConfig.seed`, `MiniGridConfig.render_mode` (`rgb_array`) | Determinism + replay pipeline compatibility | Thread through `SessionController`, `TrainerService`; warn if render payload missing (`LOG521`). |
| Telemetry keys | `"minigrid_variant"`, `"reward_multiplier"`, `"partial_obs"`, `"max_steps"` | Analytics tagging | Extend telemetry payload builder; document schema diff. |

Track new constants in `gym_gui/constants/__init__.py` and ensure `validate_log_constants()` is updated when adding log codes.

## Test & Validation Stack

- **Unit tests:** `gym_gui/tests/test_minigrid_adapter.py`, extend to cover reward scaling, wrapper toggles, and telemetry snapshots; add new UI-focused tests once control panel refactor lands.
- **Integration tests:** Expand `tests/test_worker_presenter_and_tabs.py` (or new suite) to assert MiniGrid entries appear in worker presenters and emit log constants during loader lifecycle.
- **GUI smoke:** Use Qt bot harness to select each MiniGrid game, flip toggles, and verify overrides propagate to `_game_overrides` mapping and `SessionController.load_environment` call.
- **Manual QA:** Checklist: verify keyboard mappings, ensure replay thumbnails render, confirm analytics dashboards ingest `LOG518`–`LOG521`, run CleanRL worker end-to-end.
- **Regression:** Re-run FrozenLake/Taxi/Box2D tests after gym helper extraction to ensure no regressions from `ui/environments/gym` delegation.

## Xuance Baseline Implementation Notes

- **Wrapper chain:** Xuance instantiates `gym.make(env_id, render_mode="rgb_array")`, then conditionally applies `RGBImgPartialObsWrapper` and `ImgObsWrapper` before exposing flattened observations.
- **Reward policy:** Rewards multiplied by `config.reward_scale` (default 10) to combat sparsity; we reuse this via `MiniGridConfig.reward_multiplier`.
- **Seed discipline:** `reset(seed, options)` and `env.action_space.seed(seed)` guard determinism; replicate the behaviour so trainers and CleanRL worker stay reproducible.
- **Logging:** Xuance largely prints to stdout; GUI will convert these lifecycle events into structured log constants while preserving the same touchpoints (boot, step, error, render).
- **Config surface:** Xuance uses YAML/CLI flags; our control panel surfaces equivalent toggles (partial obs, reward multiplier, view size, max steps) but defaults remain aligned.

## Alignment With Xuance Guidelines

MiniGrid integration remains faithful to Xuance’s minimalist wrapper philosophy—single adapter, optional observation wrappers, explicit reward scaling—while layering GUI-specific requirements (structured logging, Qt controls, telemetry tagging). Deviations (e.g., new constants, UI grouping, trainer wiring) are deliberate platform adaptations and are documented so contributors can trace differences back to Xuance’s baseline expectations.

## Open Questions / Assumptions

- Which worker(s) will launch MiniGrid first? (Assumption: Headless worker and CleanRL worker both need support; SPADE-BDI optional.)
- Do we need curriculum sequences (e.g., multiple environments per run), or is single-environment selection sufficient for Day 20?
- Should we port xuance-specific wrappers verbatim or reimplement minimal glue using upstream MiniGrid classes? (Current lean is reimplement + borrow reward multiplier constant.)
- How will reward visualization behave given sparse reward structure? (Might require GUI hints or scaling toggle.)
- Is reward scaling (×10) desirable for all MiniGrid tasks, or should it be part of per-environment metadata? (Need stakeholder confirmation.)

## Risks & Mitigations

- **Dependency gridlock:** MiniGrid relies on `gymnasium` and `pygame`; ensure versions align with existing GUI requirements. Mitigate by pinning compatible versions, smoke-testing import order, and adding dependency conflict detection (log with `LOG_ENV_MINIGRID_ERROR`).
- **Observation format drift:** MiniGrid can expose RGB arrays or symbolic encodings. Decide on a canonical representation (likely image observation) and convert consistently; add unit tests verifying shape/dtype constants.
- **Performance concerns:** Rendering grids at high FPS may strain GUI threads. Consider optional headless mode or capped frame rates for previews.
- **Logging volume:** Excessive per-step logging could flood log sinks; scope structured logs to lifecycle and guard with rate limiting constants (e.g., `MINIGRID_STEP_LOG_FREQUENCY = 100`).
- **Reward scaling mismatch:** Downstream analytics might expect raw rewards; document multiplier constant and allow opt-out via config.
- **Wrapper drift:** Upstream `gymnasium-minigrid` occasionally changes dict observation keys (`image`, `direction`). Guard flatten logic with assertions and log `LOG_ENV_MINIGRID_RENDER_WARNING` when payloads deviate.

## Things to Watch Out For

- **Render modes:** MiniGrid defaults to `human` rendering; headless contexts require `render_mode="rgb_array"` and `pygame` dummy drivers. Always propagate render mode from config and log `LOG_ENV_MINIGRID_RENDER_WARNING` on failure.
- **Seed discipline:** Gym GUI expects deterministic resets when seeds are provided; ensure we thread `env_seed` through workers and confirm via unit test.
- **Action validation:** Some MiniGrid tasks support fewer actions (e.g., 3). Validate action space size and expose mismatch via clear error logs to prevent agent misconfiguration.
- **Telemetry payload size:** Flattened observations can exceed default JSON payload sizes; check telemetry batching thresholds and enable compression if needed.
- **Packaging conflicts:** `gymnasium` vs. legacy `gym` namespaces may coexist. Confirm import order and avoid mixing wrappers from different packages.
- **Reward units in analytics:** Document that scaled rewards may appear larger than baseline grid-world tasks; consider UI hint or normalization toggle.

## Implementation Notes (2025-11-03)

- Registered MiniGrid adapters and logging constants; new `MiniGridAdapter` flattens RGB observations, emits `LOG518`–`LOG521`, and reuses xuance reward scaling defaults.
- Control panel exposes MiniGrid overrides (partial obs, image wrapper, reward multiplier, view size, step limit); environment catalog now lists Empty 5x5, DoorKey 5×5/6×6/8×8/16×16, and LavaGap S7.
- SPADE-BDI worker tabs treat MiniGrid runs as grid-based; game info panel documents MiniGrid variants.
- Added regression tests (`gym_gui/tests/test_minigrid_adapter.py`) skipping gracefully when MiniGrid is not installed; compile-time smoke tests pass.
- Gym-family controls remain inline in `control_panel.py`; next milestone is to delegate to `ui/environments/gym/config_panel.py` helpers (see TODO section 3).

## Exit Criteria

- MiniGrid environments selectable and launchable via GUI as of 2025-11-03.
- Trainer run configs serialize MiniGrid metadata without schema validation failures.
- At least one automated test and one manual run log demonstrating successful MiniGrid episode execution.
- Documentation stored under `docs/1.0_DAY_20/TASK_4/` reflects final behavior and operator guidance.
- New log constants validated via `validate_log_constants()` and referenced by tests.
