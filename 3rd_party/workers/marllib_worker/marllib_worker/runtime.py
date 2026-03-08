"""Runtime orchestration for MARLlib multi-agent RL training.

Wraps the MARLlib Python API::

    marl.make_env -> marl.algos.X -> marl.build_model -> algo.fit

MARLlib internally manages Ray (``ray.init`` / ``ray.shutdown``), so
the runtime calls ``algo.fit()`` in-process rather than spawning a
subprocess.
"""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .config import MARLlibWorkerConfig

try:
    from gym_gui.core.worker import TelemetryEmitter
    from gym_gui.config.paths import VAR_TRAINER_DIR, ensure_var_directories

    _HAS_GYM_GUI = True
except ImportError:
    TelemetryEmitter = None  # type: ignore[assignment,misc]
    VAR_TRAINER_DIR = Path("var/trainer")
    _HAS_GYM_GUI = False

    def ensure_var_directories() -> None:  # type: ignore[misc]
        pass


LOGGER = logging.getLogger(__name__)


class MARLlibWorkerRuntime:
    """Execute MARLlib training runs for MOSAIC.

    Implements the ``WorkerRuntime`` protocol (``run() -> Dict``).
    """

    def __init__(
        self,
        config: MARLlibWorkerConfig,
        *,
        dry_run: bool = False,
    ) -> None:
        self._config = config
        self._dry_run = dry_run

        self._emitter: Any = None
        if TelemetryEmitter is not None:
            self._emitter = TelemetryEmitter(run_id=config.run_id, logger=LOGGER)

    @property
    def config(self) -> MARLlibWorkerConfig:
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute MARLlib training or return a dry-run summary."""

        if self._dry_run:
            return self._dry_run_summary()

        run_dir = self._setup_run_dir()
        self._emit_started()

        try:
            marl = self._import_marllib()

            # 1. Environment
            env = marl.make_env(
                environment_name=self._config.environment_name,
                map_name=self._config.map_name,
                force_coop=self._config.force_coop,
                **self._config.env_params,
            )

            # 2. Algorithm
            algo_factory = getattr(marl.algos, self._config.algo, None)
            if algo_factory is None:
                raise ValueError(
                    f"Algorithm '{self._config.algo}' not found in marl.algos"
                )
            algo = algo_factory(
                hyperparam_source=self._config.hyperparam_source,
                **self._config.algo_params,
            )

            # 3. Model
            model = marl.build_model(
                env,
                algo,
                {
                    "core_arch": self._config.core_arch,
                    "encode_layer": self._config.encode_layer,
                },
            )

            # 4. Stop conditions
            stop = {
                "timesteps_total": self._config.stop_timesteps,
                "episode_reward_mean": self._config.stop_reward,
                "training_iteration": self._config.stop_iters,
            }

            # 5. Running params (passed as **kwargs to algo.fit)
            running_params: Dict[str, Any] = {
                "local_mode": self._config.local_mode,
                "num_gpus": self._config.num_gpus,
                "num_workers": self._config.num_workers,
                "share_policy": self._config.share_policy,
                "checkpoint_freq": self._config.checkpoint_freq,
                "checkpoint_end": self._config.checkpoint_end,
                "framework": self._config.framework,
                "seed": self._config.seed if self._config.seed is not None else 321,
                "local_dir": str(run_dir),
            }

            if self._config.restore_model_path:
                running_params["restore_path"] = {
                    "model_path": self._config.restore_model_path,
                    "params_path": self._config.restore_params_path,
                }

            # 6. Launch with heartbeat
            cancel = self._start_heartbeat()

            LOGGER.info(
                "Starting MARLlib training | algo=%s env=%s map=%s",
                self._config.algo,
                self._config.environment_name,
                self._config.map_name,
            )

            try:
                algo.fit(env, model, stop=stop, **running_params)
            finally:
                if cancel is not None:
                    cancel()

            # 7. Discover output and build manifest
            ray_tune_dir = self._find_ray_tune_output(run_dir)
            self._write_manifest(run_dir, ray_tune_dir)
            self._emit_completed(ray_tune_dir)

            return {
                "status": "completed",
                "algo": self._config.algo,
                "environment": self._config.environment_name,
                "map_name": self._config.map_name,
                "config": self._config.to_dict(),
                "ray_tune_dir": str(ray_tune_dir) if ray_tune_dir else None,
            }

        except Exception as exc:
            self._emit_failed(exc)
            raise

    # ------------------------------------------------------------------
    # MARLlib import (sys.argv guard)
    # ------------------------------------------------------------------

    @staticmethod
    def _import_marllib() -> Any:
        """Import MARLlib with a clean ``sys.argv``.

        MARLlib captures ``sys.argv`` at import time
        (``SYSPARAMs = deepcopy(sys.argv)`` in ``marllib/marl/__init__.py``).
        Without patching, MOSAIC CLI flags leak into MARLlib's config
        parser.
        """
        saved = sys.argv[:]
        sys.argv = ["marllib_worker"]
        try:
            from marllib import marl  # type: ignore[import-untyped]

            return marl
        finally:
            sys.argv = saved

    # ------------------------------------------------------------------
    # Run directory
    # ------------------------------------------------------------------

    def _setup_run_dir(self) -> Path:
        ensure_var_directories()
        run_dir = (VAR_TRAINER_DIR / "runs" / self._config.run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        return run_dir

    # ------------------------------------------------------------------
    # Ray Tune output discovery
    # ------------------------------------------------------------------

    def _find_ray_tune_output(self, run_dir: Path) -> Optional[Path]:
        """Locate the Ray Tune trial directory inside *run_dir*.

        Ray Tune writes output as::

            run_dir/<experiment_name>/<trial_name>/
                progress.csv
                params.json
                events.out.tfevents.*
                checkpoint_<N>/
        """
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(".") or child.name == "logs":
                continue
            # Check nested trial directories
            for trial_dir in sorted(child.iterdir()):
                if trial_dir.is_dir() and (trial_dir / "progress.csv").exists():
                    return trial_dir
            # Check if child is itself a trial dir
            if (child / "progress.csv").exists():
                return child
        return None

    # ------------------------------------------------------------------
    # Analytics manifest
    # ------------------------------------------------------------------

    def _write_manifest(
        self, run_dir: Path, ray_tune_dir: Optional[Path]
    ) -> None:
        try:
            from .analytics import build_manifest

            manifest = build_manifest(run_dir, ray_tune_dir, self._config)
            manifest.save(run_dir / "analytics.json")
            LOGGER.info("Analytics manifest written to %s", run_dir / "analytics.json")
        except ImportError:
            LOGGER.debug("gym_gui not available — skipping analytics manifest")
        except Exception:
            LOGGER.warning("Failed to write analytics manifest", exc_info=True)

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def _emit_started(self) -> None:
        if self._emitter is None:
            return
        self._emitter.run_started(
            {
                "worker_type": "marllib",
                "algo": self._config.algo,
                "environment": self._config.environment_name,
                "map_name": self._config.map_name,
                "seed": self._config.seed,
                "share_policy": self._config.share_policy,
                "stop_timesteps": self._config.stop_timesteps,
            }
        )

    def _emit_completed(self, ray_tune_dir: Optional[Path]) -> None:
        if self._emitter is None:
            return
        self._emitter.run_completed(
            {
                "status": "completed",
                "algo": self._config.algo,
                "environment": self._config.environment_name,
                "map_name": self._config.map_name,
                "ray_tune_dir": str(ray_tune_dir) if ray_tune_dir else None,
            }
        )

    def _emit_failed(self, exc: Exception) -> None:
        if self._emitter is None:
            return
        self._emitter.run_failed(
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "algo": self._config.algo,
                "environment": self._config.environment_name,
            }
        )

    def _start_heartbeat(self) -> Optional[Any]:
        """Start a daemon thread emitting heartbeats every 30 s.

        Returns a callable that stops the heartbeat, or ``None``.
        """
        if self._emitter is None:
            return None

        stop_event = threading.Event()
        emitter = self._emitter
        cfg = self._config

        def _loop() -> None:
            while not stop_event.is_set():
                emitter.heartbeat(
                    {
                        "status": "running",
                        "algo": cfg.algo,
                        "environment": cfg.environment_name,
                        "map_name": cfg.map_name,
                    }
                )
                stop_event.wait(30.0)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

        def _cancel() -> None:
            stop_event.set()
            t.join(timeout=5.0)

        return _cancel

    # ------------------------------------------------------------------
    # Dry-run
    # ------------------------------------------------------------------

    def _dry_run_summary(self) -> Dict[str, Any]:
        from .registries import get_algo_type

        LOGGER.info(
            "Dry-run | algo=%s env=%s map=%s",
            self._config.algo,
            self._config.environment_name,
            self._config.map_name,
        )
        return {
            "status": "dry-run",
            "algo": self._config.algo,
            "algo_type": get_algo_type(self._config.algo),
            "environment": self._config.environment_name,
            "map_name": self._config.map_name,
            "share_policy": self._config.share_policy,
            "config": self._config.to_dict(),
        }
