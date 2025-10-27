# Day 17 – Task 2: Taxi-v3 Training Failure Investigation (October 27, 2025)

## Summary
Taxi-v3 looks healthy in the static configuration tables, but training it through the Train Agent form fails immediately. The SPADE-BDI worker aborts because the telemetry pipeline tries to JSON-encode NumPy arrays (the Taxi action mask) without converting them to plain lists first. FrozenLake and CliffWalking do not expose these arrays, so they continue to work.

## Evidence & Repro
1. Loaded the GUI taxi adapter and ran a headless training loop:
   ```bash
   source .venv/bin/activate && python - <<'PY'
   from gym_gui.core.adapters.toy_text import TaxiAdapter
   from spade_bdi_rl.core.config import RunConfig, PolicyStrategy
   from spade_bdi_rl.core.telemetry_worker import TelemetryEmitter
   from spade_bdi_rl.core.runtime import HeadlessTrainer

   adapter = TaxiAdapter(); adapter.load()
   config = RunConfig(
       run_id="test-taxi",
       game_id="Taxi-v3",
       seed=1,
       max_episodes=3,
       max_steps_per_episode=50,
       policy_strategy=PolicyStrategy.TRAIN,
       agent_id="agent",
   )
   emitter = TelemetryEmitter()
   trainer = HeadlessTrainer(adapter, config, emitter)
   result = trainer.run()
   print("result", result)
   PY
   ```
2. The run emits one step and then aborts with:
   ```
   {"type":"run_completed", ..., "status":"failed","error":"Object of type ndarray is not JSON serializable"}
   ```

## Root Cause Analysis
- `gym_gui/constants/game_constants.py` already lists `GameId.TAXI` with a 5×5 grid skeleton; nothing is missing there.
- `gym_gui/config/game_configs.py` defines `TaxiConfig` but its `to_gym_kwargs()` intentionally returns `{}` (Gymnasium’s Taxi ignores custom kwargs). Again, no blocking issue.
- `spade_bdi_rl/constants.py` only provides worker-wide defaults (step delay, epsilon, etc.) and does not perform per-game logic.
- The failure originates inside `HeadlessTrainer._build_observation_dict()` (`spade_bdi_rl/core/runtime.py:257` onward). The method copies `info` from the adapter directly into the telemetry payload. Taxi’s `info` includes `action_mask` as a NumPy array, so the subsequent call to `TelemetryEmitter.step()` (`core/runtime.py:215`) hands that array to `json.dump`, raising the serialization error.
- FrozenLake/CliffWalking succeed because their Gymnasium `info` dicts are scalar-only (ints/floats/bools) and therefore JSON-friendly.

## Impact
- Any Taxi training launched via the Train Agent form (or the worker CLI) terminates on the first step.
- No Taxi telemetry/episodes persist, so the GUI never populates Live Agent tabs for Taxi.
- Worker exit code is `1` (failure), so orchestration thinks the run crashed.

## Fix Ideas
1. **Sanitize Taxi observations before emitting telemetry** – in `HeadlessTrainer._build_observation_dict`, convert NumPy arrays to Python lists (e.g., `observation_dict[key] = value.tolist()` when `hasattr(value, "tolist")`).
2. Alternatively, **extend `TelemetryEmitter`** to deep-convert arrays, but doing it earlier keeps the payloads small and consistent with other adapters.
3. Add a regression test under `spade_bdi_rl/tests` that makes one Taxi step and asserts telemetry events are JSON serializable.

Once the action mask (and any other NumPy payloads) are coerced to lists, Taxi should behave like the other toy-text grids during training.
