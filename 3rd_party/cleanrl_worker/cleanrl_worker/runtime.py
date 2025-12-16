

"""Runtime orchestration helpers for launching CleanRL algorithms."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Iterator

import grpc
from gym_gui.config.paths import VAR_TRAINER_DIR, ensure_var_directories
from gym_gui.telemetry.semconv import TelemetryEnv, VideoModes
from . import fastlane as fastlane_module
from .eval_registry import get_eval_entry
from .eval import EvalBatchSummary, EvalRunResult, run_batched_evaluation
from gym_gui.services.trainer.proto import trainer_pb2, trainer_pb2_grpc
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_POLICY_EVAL_STARTED,
    LOG_WORKER_POLICY_EVAL_COMPLETED,
    LOG_WORKER_POLICY_EVAL_BATCH_STARTED,
    LOG_WORKER_POLICY_EVAL_BATCH_COMPLETED,
)

from .analytics import build_manifest
from .config import WorkerConfig
from .telemetry import LifecycleEmitter
from contextlib import contextmanager, redirect_stdout, redirect_stderr

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")
_CMD_COMPONENT_PATTERN = re.compile(r"^[^\n\r\x00]*$")


LOGGER = logging.getLogger(__name__)


class _EvalStepLoggingWrapper:
    """Lightweight wrapper that logs every evaluation step for debugging."""

    def __init__(self, env, run_id: str, slot: int):
        self.env = env
        self._run_id = run_id
        self._slot = slot
        self._step = 0

    def reset(self, *args, **kwargs):  # pragma: no cover - env specific
        self._step = 0
        LOGGER.debug(
            "Policy eval env reset | run_id=%s slot=%s",
            self._run_id,
            self._slot,
        )
        return self.env.reset(*args, **kwargs)

    def step(self, action):  # pragma: no cover - env specific
        result = self.env.step(action)
        self._step += 1
        obs, reward, terminated, truncated, info = result
        try:
            reward_value = float(reward[0] if hasattr(reward, "__getitem__") else reward)
        except Exception:  # pragma: no cover - defensive conversion
            reward_value = 0.0
        terminated_flag = bool(terminated[0] if hasattr(terminated, "__getitem__") else terminated)
        truncated_flag = bool(truncated[0] if hasattr(truncated, "__getitem__") else truncated)
        LOGGER.debug(
            "Policy eval env step | run_id=%s slot=%s step=%s reward=%.4f done=%s truncated=%s",
            self._run_id,
            self._slot,
            self._step,
            reward_value,
            terminated_flag,
            truncated_flag,
        )
        if terminated_flag or truncated_flag:
            LOGGER.info(
                "Policy eval episode finished | run_id=%s slot=%s steps=%s done=%s truncated=%s",
                self._run_id,
                self._slot,
                self._step,
                terminated_flag,
                truncated_flag,
            )
        return result

    def __getattr__(self, name):  # pragma: no cover - proxy helpers
        return getattr(self.env, name)

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter as _TensorBoardWriter
except Exception:  # pragma: no cover - optional dependency
    _TensorBoardWriter = None

AlgoRegistry = Mapping[str, str]


DEFAULT_ALGO_REGISTRY: AlgoRegistry = {
    # PPO family
    "ppo": "cleanrl_worker.algorithms.ppo_with_save",
    "ppo_continuous_action": "cleanrl.ppo_continuous_action",
    "ppo_atari": "cleanrl.ppo_atari",
    "ppo_atari_multigpu": "cleanrl.ppo_atari_multigpu",
    "ppo_atari_lstm": "cleanrl.ppo_atari_lstm",
    "ppo_atari_envpool": "cleanrl.ppo_atari_envpool",
    "ppo_atari_envpool_xla_jax": "cleanrl.ppo_atari_envpool_xla_jax",
    "ppo_atari_envpool_xla_jax_scan": "cleanrl.ppo_atari_envpool_xla_jax_scan",
    "ppo_procgen": "cleanrl.ppo_procgen",
    "ppo_rnd_envpool": "cleanrl.ppo_rnd_envpool",
    "ppo_pettingzoo_ma_atari": "cleanrl.ppo_pettingzoo_ma_atari",
    # Policy optimization variants
    "ppg_procgen": "cleanrl.ppg_procgen",
    "pqn": "cleanrl.pqn",
    "pqn_atari_envpool": "cleanrl.pqn_atari_envpool",
    "pqn_atari_envpool_lstm": "cleanrl.pqn_atari_envpool_lstm",
    "rpo_continuous_action": "cleanrl.rpo_continuous_action",
    # Q-learning family
    "dqn": "cleanrl.dqn",
    "dqn_atari": "cleanrl.dqn_atari",
    "dqn_atari_jax": "cleanrl.dqn_atari_jax",
    "dqn_jax": "cleanrl.dqn_jax",
    "rainbow_atari": "cleanrl.rainbow_atari",
    "qdagger_dqn_atari_impalacnn": "cleanrl.qdagger_dqn_atari_impalacnn",
    "qdagger_dqn_atari_jax_impalacnn": "cleanrl.qdagger_dqn_atari_jax_impalacnn",
    "c51": "cleanrl.c51",
    "c51_jax": "cleanrl.c51_jax",
    "c51_atari": "cleanrl.c51_atari",
    "c51_atari_jax": "cleanrl.c51_atari_jax",
    # Continuous control
    "ddpg_continuous_action": "cleanrl.ddpg_continuous_action",
    "ddpg_continuous_action_jax": "cleanrl.ddpg_continuous_action_jax",
    "td3_continuous_action": "cleanrl.td3_continuous_action",
    "td3_continuous_action_jax": "cleanrl.td3_continuous_action_jax",
    "sac_continuous_action": "cleanrl.sac_continuous_action",
    "sac_atari": "cleanrl.sac_atari",
}


_RESERVED_EXTRA_KEYS: frozenset[str] = frozenset(
    {
        "tensorboard_dir",
        "track_wandb",
        "algo_params",
        "notes",
        "cuda",
        "use_cuda",
        "gpu",
        "wandb_run_name",
        "wandb_email",
        "wandb_api_key",
        "wandb_http_proxy",
        "wandb_https_proxy",
        "wandb_use_vpn_proxy",
        "agent_id",
        "fastlane_only",
        "fastlane_slot",
        "fastlane_video_mode",
        "fastlane_grid_limit",
        "fastlane_enabled",
        "mode",
        "policy_path",
        "checkpoint_path",
        "eval_capture_video",
        "eval_episodes",
        "eval_gamma",
        "eval_max_episode_steps",
        "eval_max_episode_seconds",
        "eval_batch_size",
        "eval_repeat",
    }
)


@dataclass(frozen=True)
class RuntimeSummary:
    """Lightweight summary returned from runtime dry-runs."""

    status: str
    module: str
    callable: str
    config: Dict[str, Any]
    extras: Dict[str, Any]


class CleanRLWorkerRuntime:
    """Resolve CleanRL algorithms and coordinate launch parameters."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        use_grpc: bool,
        grpc_target: str,
        dry_run: bool = False,
        algo_registry: AlgoRegistry = DEFAULT_ALGO_REGISTRY,
    ) -> None:
        self._config = config
        self._use_grpc = use_grpc
        self._grpc_target = grpc_target
        self._dry_run = dry_run
        self._algo_registry = dict(algo_registry)
        self._session_token: Optional[str] = None

    @property
    def config(self) -> WorkerConfig:
        return self._config

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def _allowed_entry_modules(self) -> set[str]:
        canonical = {str(value) for value in self._algo_registry.values()}
        aliased = {f"cleanrl_worker.{value}" for value in canonical}
        return canonical.union(aliased)

    def _assert_whitelisted_module(self, module_name: str) -> None:
        if not _MODULE_NAME_PATTERN.fullmatch(module_name):
            raise ValueError(f"Refusing to import unexpected module name '{module_name}'")
        if module_name not in self._allowed_entry_modules():
            raise ValueError(f"Module '{module_name}' is not registered in the CleanRL registry")

    def resolve_entrypoint(self) -> tuple[str, str]:
        """Resolve the module and callable implementing the requested algorithm."""

        canonical_module = self._algo_registry.get(self._config.algo)
        if canonical_module is None:
            raise ValueError(f"Algorithm '{self._config.algo}' is not registered")

        candidates = [canonical_module]
        if canonical_module.startswith("cleanrl."):
            candidates.append(f"cleanrl_worker.{canonical_module}")

        resolved_name: Optional[str] = None
        for candidate in candidates:
            try:
                spec = importlib.util.find_spec(candidate)
            except (ImportError, AttributeError):
                spec = None

            if spec is not None:
                resolved_name = candidate
                break

        if resolved_name is None:
            raise ModuleNotFoundError(
                f"Unable to locate CleanRL module for algorithm '{self._config.algo}' "
                f"(tried {', '.join(candidates)})"
            )

        self._assert_whitelisted_module(resolved_name)

        return resolved_name, f"{resolved_name}.main"

    def _import_entrypoint(self, module_name: str):
        """Import module and fetch its main entrypoint."""

        self._assert_whitelisted_module(module_name)
        module = importlib.import_module(module_name)
        entrypoint = getattr(module, "main", None)
        if entrypoint is None or not callable(entrypoint):
            LOGGER.warning(
                "Module %s does not expose a main() entrypoint; relying on CLI execution",
                module_name,
            )
            return None
        return entrypoint

    def build_cleanrl_args(self) -> list[str]:
        """Construct CLI arguments for the underlying CleanRL script."""

        args = [
            f"--env-id={self._config.env_id}",
            f"--total-timesteps={self._config.total_timesteps}",
        ]
        if self._config.seed is not None:
            args.append(f"--seed={self._config.seed}")

        extras: Mapping[str, Any] = self._config.extras
        wandb = extras.get("track_wandb")
        if isinstance(wandb, bool) and wandb:
            args.append("--track")

        cuda_flag = extras.get("cuda") or extras.get("use_cuda") or extras.get("gpu")
        if isinstance(cuda_flag, bool) and cuda_flag:
            args.append("--cuda")

        algo_params = extras.get("algo_params")
        if isinstance(algo_params, Mapping):
            for key, value in algo_params.items():
                flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool):
                    if value:
                        args.append(flag)
                else:
                    args.append(f"{flag}={value}")

        for key, value in extras.items():
            if key in _RESERVED_EXTRA_KEYS:
                continue
            args.extend(self._format_cli_override(key, value))

        return args

    def _format_cli_override(self, key: str, value: Any) -> list[str]:
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            return []
        if isinstance(value, bool):
            return [f"{flag}={'true' if value else 'false'}"]
        if isinstance(value, (int, float)):
            return [f"{flag}={value}"]
        if isinstance(value, str):
            if not value.strip():
                return []
            return [f"{flag}={value}"]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            formatted: list[str] = []
            for item in value:
                if isinstance(item, bool):
                    formatted.append(f"{flag}={'true' if item else 'false'}")
                elif isinstance(item, (int, float, str)):
                    formatted.append(f"{flag}={item}")
            return formatted
        return []

    def _prepare_summary(self, module_name: str) -> RuntimeSummary:
        return RuntimeSummary(
            status="dry-run" if self._dry_run else "pending",
            module=module_name,
            callable=f"{module_name}.main",
            config=asdict(self._config),
            extras=dict(self._config.extras),
        )

    def _register_with_trainer(self) -> None:
        """Perform RegisterWorker handshake directly with the trainer daemon."""

        if not self._use_grpc:
            return

        options = (
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        )

        worker_id = self._config.worker_id or f"cleanrl-worker-{self._config.run_id[:8]}"
        schema_id = self._config.extras.get("schema_id") or f"cleanrl.{self._config.algo}"
        supports_pause = bool(self._config.extras.get("supports_pause", False))
        supports_checkpoint = bool(self._config.extras.get("supports_checkpoint", False))
        schema_version_raw = self._config.extras.get("schema_version", 1)
        try:
            schema_version = int(schema_version_raw)
        except (TypeError, ValueError):
            schema_version = 1

        # gRPC core only supports HTTP proxies (CONNECT). Some environments export
        # HTTPS_PROXY/https_proxy with an https:// URI, which leads to noisy
        # 'https scheme not supported in proxy URI' errors even for local targets.
        # Suppress HTTPS proxy variables during the handshake and ensure localhost
        # is in no_proxy to avoid proxy resolution altogether for the trainer.
        with _suppress_https_proxy_for_grpc():
            with grpc.insecure_channel(self._grpc_target, options=options) as channel:
                stub = trainer_pb2_grpc.TrainerServiceStub(channel)
                request = trainer_pb2.RegisterWorkerRequest(  # type: ignore[attr-defined]
                    run_id=self._config.run_id,
                    worker_id=worker_id,
                    worker_kind="cleanrl",
                    proto_version="MOSAIC/1.0",
                    schema_id=str(schema_id),
                    schema_version=schema_version,
                    supports_pause=supports_pause,
                    supports_checkpoint=supports_checkpoint,
                )
                try:
                    response = stub.RegisterWorker(request)
                except grpc.RpcError as exc:  # pragma: no cover - defensive logging
                    detail = exc.details() if hasattr(exc, "details") else str(exc)
                    code = exc.code().name if hasattr(exc, "code") else "UNKNOWN"
                    raise RuntimeError(
                        f"RegisterWorker handshake failed ({code}): {detail}"
                    ) from exc

        self._session_token = getattr(response, "session_token", None)

    def run(self, emitter: Optional[LifecycleEmitter] = None) -> RuntimeSummary:
        """Execute (or dry-run) the configured CleanRL algorithm."""

        module_name, _ = self.resolve_entrypoint()
        if self._dry_run:
            return self._prepare_summary(module_name)

        ensure_var_directories()
        run_dir = (VAR_TRAINER_DIR / "runs" / self._config.run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._import_entrypoint(module_name)
        except ModuleNotFoundError:
            if module_name.startswith("cleanrl."):
                LOGGER.debug(
                    "Deferring import of %s to launcher bootstrap (vendored CleanRL expected)",
                    module_name,
                )
            else:
                raise

        self._register_with_trainer()

        extras: Dict[str, Any] = dict(self._config.extras)
        mode = str(extras.get("mode") or "train")
        if mode == "policy_eval":
            return self._run_policy_eval(module_name, run_dir, extras, emitter)

        args = self.build_cleanrl_args()
        cmd = _sanitize_launch_command(
            [
                sys.executable,
                "-m",
                "cleanrl_worker.launcher",
                module_name,
                *args,
            ]
        )

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        pythonpath_entries = []
        site_dir = Path(__file__).resolve().parent
        pythonpath_entries.append(str(site_dir))
        repo_path = str(REPO_ROOT)
        pythonpath_entries.append(repo_path)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        tb_path: Optional[Path] = None
        tensorboard_dir = extras.get("tensorboard_dir")
        # Preserve the original relative dirname for analytics manifest
        tensorboard_dirname_relative: Optional[str] = None
        if isinstance(tensorboard_dir, str) and tensorboard_dir:
            tensorboard_dirname_relative = tensorboard_dir.strip()
            tb_path = (run_dir / tensorboard_dir).resolve()
            tb_path.mkdir(parents=True, exist_ok=True)
            extras["tensorboard_dir"] = str(tb_path)
        else:
            extras.pop("tensorboard_dir", None)

        wandb_enabled = bool(extras.get("track_wandb"))
        wandb_root: Optional[Path] = None
        if wandb_enabled:
            wandb_root = (run_dir / "wandb").resolve()
            (wandb_root / "cache").mkdir(parents=True, exist_ok=True)
            (wandb_root / "config").mkdir(parents=True, exist_ok=True)
            (wandb_root / "logs").mkdir(parents=True, exist_ok=True)

        stdout_path = logs_dir / "cleanrl.stdout.log"
        stderr_path = logs_dir / "cleanrl.stderr.log"

        capture_flag = extras.get("algo_params", {}).get("capture_video") if extras.get("algo_params") else None
        if capture_flag:
            # Only the CleanRL CLI flag controls recording; the global MOSAIC_CAPTURE_VIDEO
            # shim stays disabled to avoid duplicate video artifacts. We still ensure the
            # videos directory exists so downstream analytics can surface files written by
            # CleanRL (which uses cwd/videos/<run_name>/...).
            (run_dir / "videos").mkdir(parents=True, exist_ok=True)
        else:
            env.pop("MOSAIC_CAPTURE_VIDEO", None)
            env.pop("MOSAIC_VIDEOS_DIR", None)

        if tensorboard_dir and isinstance(tensorboard_dir, str):
            env["CLEANRL_TENSORBOARD_DIR"] = str(tb_path)
        def _ensure_no_proxy(var: str) -> None:
            current = env.get(var, "")
            entries = [entry.strip() for entry in current.split(",") if entry.strip()]
            for target in ("127.0.0.1", "localhost"):
                if target not in entries:
                    entries.append(target)
            env[var] = ",".join(entries)

        _ensure_no_proxy("no_proxy")
        _ensure_no_proxy("NO_PROXY")

        if wandb_enabled and wandb_root is not None:
            env.setdefault("WANDB_DIR", str(wandb_root))
            env.setdefault("WANDB_CACHE_DIR", str(wandb_root / "cache"))
            env.setdefault("WANDB_CONFIG_DIR", str(wandb_root / "config"))
            env.setdefault("WANDB_NETRC_PATH", str(wandb_root / "netrc"))
            env.setdefault("WANDB_START_METHOD", "thread")
            env.setdefault("WANDB__SERVICE", "disabled")
            env.setdefault("WANDB_SKIP_SERVICE", "1")
            env.setdefault("WANDB_DISABLE_SERVICE", "true")
            env.setdefault("WANDB_DISABLE_GYM", "true")
            env.setdefault("WANDB_MODE", "online")

        # Resolve WANDB proxies, preferring VPN vars if present, then extras and generic fallbacks.
        http_proxy, https_proxy = _resolve_wandb_proxies(extras, os.environ)
        _apply_proxy_env(env, http_proxy, https_proxy)

        wandb_run_name = extras.get("wandb_run_name")
        if isinstance(wandb_run_name, str) and wandb_run_name:
            env.setdefault("WANDB_NAME", wandb_run_name)
            env.setdefault("WANDB_RUN_ID", wandb_run_name)

        wandb_email = extras.get("wandb_email")
        if isinstance(wandb_email, str) and wandb_email:
            env.setdefault("WANDB_EMAIL", wandb_email)

        wandb_entity = extras.get("wandb_entity")
        if isinstance(wandb_entity, str) and wandb_entity:
            env.setdefault("WANDB_ENTITY", wandb_entity)

        wandb_project = extras.get("wandb_project_name") or extras.get("wandb_project")
        if isinstance(wandb_project, str) and wandb_project:
            env.setdefault("WANDB_PROJECT", wandb_project)

        wandb_api_key = extras.get("wandb_api_key")
        if isinstance(wandb_api_key, str) and wandb_api_key:
            env.setdefault("WANDB_API_KEY", wandb_api_key)

        if wandb_enabled:
            if not _preflight_wandb_login(env):
                LOGGER.warning(
                    "WANDB preflight login failed; forcing offline mode for this run",
                )
                env.setdefault("WANDB_MODE", "offline")
        else:
            env.setdefault("WANDB_MODE", "offline")

        with stdout_path.open("w", encoding="utf-8", buffering=1) as out, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as err:
            proc = subprocess.Popen(
                cmd,
                cwd=run_dir,
                stdout=out,
                stderr=err,
                env=env,
            )

            last_heartbeat = time.monotonic()
            heartbeat_interval = 30.0
            while True:
                rc = proc.poll()
                if rc is not None:
                    return_code = rc
                    break
                now = time.monotonic()
                if emitter is not None and now - last_heartbeat >= heartbeat_interval:
                    emitter.heartbeat(
                        self._config.run_id,
                        {
                            "status": "running",
                            "algo": self._config.algo,
                            "env_id": self._config.env_id,
                        },
                    )
                    last_heartbeat = now
                time.sleep(1.0)

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        # Build manifest with relative paths for portability
        manifest_extras = dict(extras)
        if tensorboard_dirname_relative:
            manifest_extras["tensorboard_dir"] = tensorboard_dirname_relative
        manifest = build_manifest(
            run_dir, extras=manifest_extras, run_id=self._config.run_id
        )
        manifest_path = run_dir / "analytics.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        return RuntimeSummary(
            status="completed",
            module=module_name,
            callable=f"{module_name}.main",
            config=asdict(self._config),
            extras=extras,
        )

    # ---------------------------------------------------------------------------
    # Local helpers
    # ---------------------------------------------------------------------------
    def _run_policy_eval(
        self,
        module_name: str,
        run_dir: Path,
        extras: Dict[str, Any],
        emitter: Optional[LifecycleEmitter],
    ) -> RuntimeSummary:
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        policy_path = extras.get("policy_path")
        if not policy_path:
            raise ValueError("Policy evaluation requested without policy_path extra")
        policy_file = Path(policy_path).expanduser()
        if not policy_file.exists():
            LOGGER.warning("Policy checkpoint missing: %s", policy_file)
            raise FileNotFoundError(policy_file)

        stdout_path = logs_dir / "cleanrl.stdout.log"
        stderr_path = logs_dir / "cleanrl.stderr.log"
        videos_dir = run_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dirname = "tensorboard_eval"
        tensorboard_log_dir = run_dir / tensorboard_dirname
        # Create tensorboard directory early so TensorBoard tab can start watching it
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "eval_summary.json"
        manifest_path = run_dir / "analytics.json"
        collected_returns: list[float] = []

        entry = get_eval_entry(self._config.algo)
        
        if entry is not None:
            agent_cls = entry.agent_cls
            make_env = entry.make_env
            evaluate_fn = entry.evaluate
            accepts_gamma = entry.accepts_gamma
        else:
            module = importlib.import_module(module_name)
            evaluate_fn = self._resolve_eval_helper()
            agent_cls = getattr(module, "Agent", None)
            make_env = getattr(module, "make_env", None)
            accepts_gamma = True
            if evaluate_fn is None or agent_cls is None or make_env is None:
                raise RuntimeError(
                    f"Algorithm '{self._config.algo}' does not expose evaluation helpers"
                )

        capture_video = bool(extras.get("eval_capture_video"))
        eval_episodes = extras.get("eval_episodes", 5)
        try:
            eval_episodes = max(1, int(eval_episodes))
        except (TypeError, ValueError):
            eval_episodes = 5
        episodes_per_batch = extras.get("eval_batch_size", eval_episodes)
        try:
            episodes_per_batch = max(1, int(episodes_per_batch))
        except (TypeError, ValueError):
            episodes_per_batch = eval_episodes
        repeat_eval = bool(extras.get("eval_repeat"))
        device_name = "cuda" if extras.get("cuda") else "cpu"
        gamma = extras.get("algo_params", {}).get("gamma", 0.99)
        try:
            gamma = float(gamma)
        except (TypeError, ValueError):
            gamma = 0.99
        eval_gamma = extras.get("eval_gamma")
        if isinstance(eval_gamma, (int, float)):
            gamma = float(eval_gamma)

        LOGGER.info(
            "Policy evaluation starting | run_id=%s env=%s batch_episodes=%s repeat=%s",
            self._config.run_id,
            self._config.env_id,
            episodes_per_batch,
            repeat_eval,
        )

        try:
            import torch

            device = torch.device("cuda") if device_name == "cuda" else torch.device("cpu")
        except Exception:  # pragma: no cover - torch optional during tests
            device = device_name

        algo_params = extras.get("algo_params", {}) if isinstance(extras, dict) else {}

        def _coerce_positive(value, fallback):
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                return fallback

        num_envs = _coerce_positive(algo_params.get("num_envs"), 1)
        grid_limit_value = extras.get("fastlane_grid_limit", num_envs)
        grid_limit = _coerce_positive(grid_limit_value, num_envs)
        fastlane_env = {
            TelemetryEnv.FASTLANE_ONLY: "1" if extras.get("fastlane_only") else "0",
            TelemetryEnv.FASTLANE_SLOT: str(extras.get("fastlane_slot", 0)),
            TelemetryEnv.FASTLANE_VIDEO_MODE: extras.get("fastlane_video_mode", VideoModes.SINGLE),
            TelemetryEnv.FASTLANE_GRID_LIMIT: str(grid_limit),
        }
        core_env = {
            "CLEANRL_RUN_ID": self._config.run_id,
            "CLEANRL_AGENT_ID": extras.get("agent_id", "cleanrl_eval"),
            "CLEANRL_NUM_ENVS": str(num_envs),
        }
        env_updates = {
            "MOSAIC_VIDEOS_DIR": str(videos_dir),
            "MOSAIC_CAPTURE_VIDEO": "1" if capture_video else "0",
            **fastlane_env,
            **core_env,
        }
        max_steps_override = extras.get("eval_max_episode_steps")
        max_seconds_override = extras.get("eval_max_episode_seconds")

        def _write_manifest_payload(returns_snapshot: Sequence[float]) -> None:
            manifest = build_manifest(
                run_dir,
                extras={
                    "mode": "policy_eval",
                    "returns": list(returns_snapshot),
                    "tensorboard_dir": tensorboard_dirname,
                },
                run_id=self._config.run_id,
            )
            manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        _write_manifest_payload([])
        log_constant(
            LOGGER,
            LOG_WORKER_POLICY_EVAL_STARTED,
            extra={
                "run_id": self._config.run_id,
                "algo": self._config.algo,
                "env_id": self._config.env_id,
                "policy_path": str(policy_file),
                "batch_size": episodes_per_batch,
                "repeat": repeat_eval,
                "max_steps": max_steps_override,
                "max_seconds": max_seconds_override,
            },
        )

        def _record_batch(summary: EvalBatchSummary) -> None:
            collected_returns.extend(summary.returns)
            log_constant(
                LOGGER,
                LOG_WORKER_POLICY_EVAL_BATCH_COMPLETED,
                extra={
                    "run_id": self._config.run_id,
                    "batch_index": summary.index,
                    "episodes": summary.episodes,
                    "avg_return": summary.avg_return,
                    "min_return": summary.min_return,
                    "max_return": summary.max_return,
                    "duration_sec": summary.duration_sec,
                },
            )
            try:
                _write_manifest_payload(collected_returns)
            except Exception:  # pragma: no cover - manifest best effort
                pass
            if emitter is not None:
                emitter.heartbeat(
                    self._config.run_id,
                    {
                        "status": "policy_eval_batch",
                        "algo": self._config.algo,
                        "env_id": self._config.env_id,
                        "mode": "policy_eval",
                        "batch_index": summary.index,
                        "episodes": summary.episodes,
                        "avg_return": summary.avg_return,
                        "min_return": summary.min_return,
                        "max_return": summary.max_return,
                        "median_return": summary.median_return,
                        "std_return": summary.std_return,
                        "duration_sec": summary.duration_sec,
                    },
                )

        def _record_batch_start(batch_index: int) -> None:
            log_constant(
                LOGGER,
                LOG_WORKER_POLICY_EVAL_BATCH_STARTED,
                extra={
                    "run_id": self._config.run_id,
                    "batch_index": batch_index,
                    "episodes": episodes_per_batch,
                },
            )

        returns: list[float] = []
        with stdout_path.open("w", encoding="utf-8", buffering=1) as out, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as err, redirect_stdout(out), redirect_stderr(err):
            with _temporary_environ(env_updates), _working_directory(run_dir):
                fastlane_module.reload_fastlane_config()
                try:
                    import cleanrl_worker.sitecustomize as sc  # noqa: WPS433

                    importlib.reload(sc)
                except Exception:  # pragma: no cover - defensive reload
                    pass

                target_gamma = gamma
                if accepts_gamma:
                    base_factory = make_env
                else:

                    def base_factory(env_id, idx, capture_video, run_name, _gamma):
                        return make_env(env_id, idx, capture_video, run_name)

                def adapted_make_env(env_id, idx, capture_video, run_name, _gamma_unused):
                    # Return a factory function that creates the wrapped environment.
                    # CleanRL's ppo_eval.py passes this to gym.vector.SyncVectorEnv which
                    # expects a callable that returns an env, not an env directly.
                    def env_factory():
                        # base_factory returns a factory function, so we call it twice:
                        # base_factory(...) -> returns factory function
                        # base_factory(...)() -> calls factory to create actual env
                        env = base_factory(env_id, idx, capture_video, run_name, target_gamma)()
                        if max_seconds_override is not None:
                            from gym_gui.core.wrappers.time_limits import TimeLimitSeconds

                            try:
                                env = TimeLimitSeconds(env, float(max_seconds_override))
                            except Exception:
                                pass
                        if max_steps_override is not None:
                            from gym_gui.core.wrappers.time_limits import configure_step_limit

                            try:
                                env = configure_step_limit(env, int(max_steps_override))
                            except Exception:
                                pass
                        env = _EvalStepLoggingWrapper(env, self._config.run_id, idx)
                        return env
                    return env_factory

                result: Optional[EvalRunResult] = run_batched_evaluation(
                    evaluate_fn,
                    policy_path=str(policy_file),
                    make_env=adapted_make_env,
                    env_id=self._config.env_id,
                    agent_cls=agent_cls,
                    device=device,
                    capture_video=capture_video,
                    gamma=gamma,
                    episodes_per_batch=episodes_per_batch,
                    repeat=repeat_eval,
                    log_dir=tensorboard_log_dir,
                    summary_path=summary_path,
                    run_name_prefix=self._config.run_id,
                    on_batch=_record_batch,
                    on_batch_start=_record_batch_start,
                )
                if result is not None:
                    returns = self._coerce_return_values(result.returns)
                else:
                    returns = self._coerce_return_values(collected_returns)
        fastlane_module.reload_fastlane_config()

        avg = sum(returns) / len(returns) if returns else 0.0
        log_constant(
            LOGGER,
            LOG_WORKER_POLICY_EVAL_COMPLETED,
            extra={
                "run_id": self._config.run_id,
                "episodes": len(returns),
                "avg_return": avg,
            },
        )
        _write_manifest_payload(returns)

        if emitter is not None:
            emitter.heartbeat(
                self._config.run_id,
                {
                    "status": "completed",
                    "algo": self._config.algo,
                    "env_id": self._config.env_id,
                    "mode": "policy_eval",
                },
            )

        return RuntimeSummary(
            status="completed",
            module=module_name,
            callable="policy_eval",
            config=asdict(self._config),
            extras={
                **extras,
                "evaluation_returns": returns,
                "tensorboard_dir": tensorboard_dirname,
                "eval_batch_size": episodes_per_batch,
                "eval_repeat": repeat_eval,
            },
        )

    def _resolve_eval_helper(self):
        algo = self._config.algo
        if algo not in self._algo_registry:
            return None
        module_candidates = [
            f"cleanrl_worker.cleanrl_utils.evals.{algo}_eval",
            f"cleanrl_utils.evals.{algo}_eval",
        ]
        if algo.startswith("ppo"):
            module_candidates.append("cleanrl_worker.cleanrl_utils.evals.ppo_eval")
            module_candidates.append("cleanrl_utils.evals.ppo_eval")
        allowed_candidates = {name for name in module_candidates if _MODULE_NAME_PATTERN.fullmatch(name)}
        for module_name in module_candidates:
            if module_name not in allowed_candidates:
                continue
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            evaluate = getattr(module, "evaluate", None)
            if callable(evaluate):
                return evaluate
        return None

    def _coerce_return_values(self, values: Any) -> list[float]:
        if isinstance(values, (int, float)):
            return [float(values)]
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            coerced: list[float] = []
            for item in values:
                try:
                    coerced.append(float(item))
                except (TypeError, ValueError):
                    LOGGER.debug("Skipping non-numeric evaluation return: %s", item)
            return coerced
        return []

    def _write_eval_tensorboard(self, log_dir: Path, returns: Sequence[float], avg: float) -> None:
        if not returns or _TensorBoardWriter is None:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            writer = _TensorBoardWriter(log_dir=str(log_dir))
        except Exception:
            return
        with writer:
            for idx, value in enumerate(returns):
                writer.add_scalar("eval/episode_return", value, idx)
            writer.add_scalar("eval/avg_return", avg, len(returns))
            writer.flush()


@contextmanager
def _temporary_environ(overrides: Dict[str, str]) -> Iterator[None]:
    saved: dict[str, Optional[str]] = {k: os.environ.get(k) for k in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    current = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(current)


@contextmanager
def _suppress_https_proxy_for_grpc() -> Iterator[None]:
    """Temporarily remove HTTPS proxy env vars to prevent gRPC errors.

    gRPC only respects HTTP CONNECT proxies. If an https:// proxy is present in
    HTTPS_PROXY/https_proxy, gRPC will emit errors like:
    'https scheme not supported in proxy URI'. We temporarily remove these
    variables and ensure localhost is listed in NO_PROXY while establishing the
    channel/performing the handshake. Variables are restored afterwards.
    """

    keys = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
    saved: dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            if k in os.environ:
                os.environ.pop(k, None)

        def _append_no_proxy(var: str) -> None:
            current = os.environ.get(var, "")
            entries = [e.strip() for e in current.split(",") if e.strip()]
            for host in ("localhost", "127.0.0.1", "::1"):
                if host not in entries:
                    entries.append(host)
            os.environ[var] = ",".join(entries)

        _append_no_proxy("NO_PROXY")
        _append_no_proxy("no_proxy")
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _resolve_wandb_proxies(
    extras: Mapping[str, Any], base_env: Mapping[str, str]
) -> tuple[Optional[str], Optional[str]]:
    """Resolve WANDB HTTP/HTTPS proxy values with sensible precedence.

    Precedence (first non-empty string wins):
    - Extras overrides: wandb_http_proxy / wandb_https_proxy
    - VPN-specific env vars: WANDB_VPN_HTTP_PROXY / WANDB_VPN_HTTPS_PROXY
    - WANDB-specific env vars: WANDB_HTTP_PROXY / WANDB_HTTPS_PROXY
    - Generic env vars: HTTP_PROXY/http_proxy and HTTPS_PROXY/https_proxy
    """

    def _pick_str(value: Any) -> Optional[str]:
        return value if isinstance(value, str) and value.strip() else None

    http_proxy = _pick_str(extras.get("wandb_http_proxy")) or _pick_str(
        base_env.get("WANDB_VPN_HTTP_PROXY")
    ) or _pick_str(base_env.get("WANDB_HTTP_PROXY")) or _pick_str(base_env.get("HTTP_PROXY")) or _pick_str(
        base_env.get("http_proxy")
    )

    https_proxy = _pick_str(extras.get("wandb_https_proxy")) or _pick_str(
        base_env.get("WANDB_VPN_HTTPS_PROXY")
    ) or _pick_str(base_env.get("WANDB_HTTPS_PROXY")) or _pick_str(base_env.get("HTTPS_PROXY")) or _pick_str(
        base_env.get("https_proxy")
    )

    return http_proxy, https_proxy


def _apply_proxy_env(env: Dict[str, str], http_proxy: Optional[str], https_proxy: Optional[str]) -> None:
    """Apply resolved proxies to child environment variables consistently."""
    if http_proxy:
        for key in ("WANDB_HTTP_PROXY", "HTTP_PROXY", "http_proxy"):
            env[key] = http_proxy
    if https_proxy:
        for key in ("WANDB_HTTPS_PROXY", "HTTPS_PROXY", "https_proxy"):
            env[key] = https_proxy


def _preflight_wandb_login(env: Mapping[str, str]) -> bool:
    """Attempt to authenticate with Weights & Biases using provided environment."""

    try:
        import wandb
    except Exception as exc:  # pragma: no cover - wandb optional
        LOGGER.warning("WANDB SDK unavailable during preflight: %s", exc)
        return False

    api_key = env.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY")

    overlay = {k: v for k, v in env.items() if isinstance(v, str)}
    with _temporary_env(overlay):
        try:
            if api_key:
                result = wandb.login(key=api_key, relogin=True)
            else:
                result = wandb.login(relogin=True)
        except Exception as exc:  # pragma: no cover - network/proxy issues
            LOGGER.warning("WANDB login failed during preflight: %s", exc)
            return False
        finally:
            try:
                wandb.finish()
            except Exception:
                pass

    return bool(result)


@contextmanager
def _temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Temporarily overlay environment variables for the duration of a block."""

    saved: dict[str, Optional[str]] = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if isinstance(value, str):
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sanitize_launch_command(cmd: Sequence[Any]) -> list[str]:
    """Ensure CLI components are strings without control characters."""

    sanitized: list[str] = []
    for index, component in enumerate(cmd):
        if component is None:
            raise ValueError(f"Command component at index {index} is None")
        if not isinstance(component, str):
            component = str(component)
        if not _CMD_COMPONENT_PATTERN.fullmatch(component):
            raise ValueError(
                f"Command component at index {index} contains unsupported characters"
            )
        sanitized.append(component)
    return sanitized
