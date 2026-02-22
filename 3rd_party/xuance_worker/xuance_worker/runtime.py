"""Runtime orchestration for XuanCe training.

This module provides the XuanCeWorkerRuntime class which wraps XuanCe's
get_runner() API to execute training and benchmark runs.

Also provides InteractiveRuntime for GUI step-by-step policy evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from .config import XuanCeWorkerConfig

# Method name normalization: UI names -> XuanCe config folder names
# XuanCe's get_runner() looks up configs in ./xuance/configs/{method}/{env}.yaml
# The config folder names are lowercase (ppo, dqn, sac, etc.)
# But the agent names in configs are like "PPO_Clip", "DQN", "SAC", etc.
_METHOD_NAME_MAP: dict[str, str] = {
    # Policy Optimization (single-agent)
    "PPO_Clip": "ppo",
    "PPO_KL": "ppo",
    "PPG": "ppg",
    "A2C": "a2c",
    "PG": "pg",
    "NPG": "npg",
    "TRPO": "trpo",
    # Value-based (single-agent)
    "DQN": "dqn",
    "DDQN": "ddqn",
    "DuelDQN": "dueldqn",
    "NoisyDQN": "noisydqn",
    "C51": "c51",
    "QRDQN": "qrdqn",
    "PerDQN": "perdqn",
    "DRQN": "drqn",
    # Continuous control (single-agent)
    "SAC": "sac",
    "DDPG": "ddpg",
    "TD3": "td3",
    "TD3BC": "td3bc",
    # Parameterized action
    "PDQN": "pdqn",
    "MPDQN": "mpdqn",
    "SPDQN": "spdqn",
    # Model-based
    "DreamerV2": "dreamerv2",
    "DreamerV3": "dreamerv3",
    # Multi-agent
    "MAPPO_Clip": "mappo",
    "MAPPO_KL": "mappo",
    "IPPO_Clip": "ippo",
    "IPPO_KL": "ippo",
    "QMIX": "qmix",
    "VDN": "vdn",
    "WQMIX": "wqmix",
    "QTRAN": "qtran",
    "MADDPG": "maddpg",
    "MASAC": "masac",
    "ISAC": "isac",
    "MATD3": "matd3",
    "IDDPG": "iddpg",
    "IAC": "iac",
    "COMA": "coma",
    "MFQ": "mfq",
    "MFAC": "mfac",
    "DCG": "dcg",
    "VDAC": "vdac",
    "IC3Net": "ic3net",
    "CommNet": "commnet",
    "TARMAC": "tarmac",
    # Random baseline
    "Random": "random",
}


# Environments that use RunnerCompetition and their group counts
# These require passing method as a list to get_runner()
_COMPETITION_ENV_GROUPS: dict[str, dict[str, int]] = {
    "multigrid": {
        "soccer": 2,          # 2v2: 2 teams (Green vs Blue)
        # soccer_1vs1 uses RunnerMARL (not RunnerCompetition):
        # RunnerCompetition uses the off-policy store_experience signature
        # which is incompatible with on-policy agents like MAPPO.
        # RunnerMARL delegates to the agent's own train() loop which
        # correctly handles log_pi_a, values, and finish_path for GAE.
        "collect": 3,         # 3 independent agents
    },
}


def _get_competition_num_groups(env: str, env_id: str) -> int | None:
    """Get number of groups for competition environments.

    Returns the number of separate policies needed for adversarial training.
    Returns None if the environment doesn't use competition mode.

    Args:
        env: Environment family (e.g., "multigrid")
        env_id: Specific environment ID (e.g., "soccer")

    Returns:
        Number of groups/teams, or None if not a competition environment.
    """
    env_lower = env.lower()
    env_id_lower = env_id.lower()

    if env_lower in _COMPETITION_ENV_GROUPS:
        return _COMPETITION_ENV_GROUPS[env_lower].get(env_id_lower)
    return None


# Directory containing custom YAML configs shipped with xuance_worker
_WORKER_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# Gymnasium ID → XuanCe short env_id mapping.
# The GUI uses full gymnasium IDs; XuanCe configs/environments use short names.
# Covers all MosaicMultiGrid and IniMultiGrid registered environments.
_GYMNASIUM_TO_XUANCE: dict[str, str] = {
    # MosaicMultiGrid — legacy (no IndAgObs)
    "MosaicMultiGrid-Soccer-v0": "soccer",
    "MosaicMultiGrid-Collect-v0": "collect",
    "MosaicMultiGrid-Collect-2vs2-v0": "collect_2vs2",
    "MosaicMultiGrid-Collect-1vs1-v0": "collect_1vs1",
    # MosaicMultiGrid — IndAgObs
    "MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0": "soccer_2vs2_indagobs",
    "MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0": "soccer_1vs1",
    "MosaicMultiGrid-Collect-IndAgObs-v0": "collect_indagobs",
    "MosaicMultiGrid-Collect-2vs2-IndAgObs-v0": "collect_2vs2_indagobs",
    "MosaicMultiGrid-Collect-1vs1-IndAgObs-v0": "collect_1vs1",
    "MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0": "basketball_3vs3_indagobs",
    # MosaicMultiGrid — TeamObs
    "MosaicMultiGrid-Soccer-2vs2-TeamObs-v0": "soccer_2vs2_teamobs",
    "MosaicMultiGrid-Collect-2vs2-TeamObs-v0": "collect_2vs2_teamobs",
    "MosaicMultiGrid-Basketball-3vs3-TeamObs-v0": "basketball_3vs3_teamobs",
}


def _gymnasium_to_xuance_env_id(gym_id: str) -> str | None:
    """Convert a gymnasium environment ID to the XuanCe short env_id.

    Returns None if no mapping exists (non-multigrid environments).
    """
    return _GYMNASIUM_TO_XUANCE.get(gym_id)


def _resolve_custom_config_path(
    method: str,
    env: str,
    env_id: str,
    num_groups: int | None,
    config_path: str | None,
) -> str | list[str] | None:
    """Resolve config_path for custom environments that live in xuance_worker.

    XuanCe's get_runner() looks for YAML configs in its own ``xuance/configs/``
    directory.  Environments added by xuance_worker (e.g. soccer_1vs1) keep
    their YAML in ``xuance_worker/configs/`` instead.  This function checks
    there first and returns an absolute path so XuanCe can find it.

    For competition mode (num_groups is not None), config_path must be a
    **list** of paths -- one per group -- as required by get_arguments().

    Returns:
        Resolved config_path (str, list[str], or None if no custom config).
    """
    if config_path is not None:
        return config_path  # User already provided an explicit path

    yaml_path = _WORKER_CONFIGS_DIR / method / env / f"{env_id}.yaml"
    if not yaml_path.exists():
        # Fallback: convert gymnasium ID to XuanCe short env_id.
        # The GUI passes full gymnasium IDs (e.g. "MosaicMultiGrid-Collect-1vs1-v0")
        # but YAML config files use XuanCe's short names (e.g. "collect_1vs1").
        short_id = _gymnasium_to_xuance_env_id(env_id)
        if short_id:
            yaml_path = _WORKER_CONFIGS_DIR / method / env / f"{short_id}.yaml"
            if yaml_path.exists():
                _logger = logging.getLogger(__name__)
                _logger.info(
                    "Mapped gymnasium ID '%s' -> xuance env_id '%s' -> config '%s'",
                    env_id, short_id, yaml_path.name,
                )
        if not yaml_path.exists():
            return None  # Fall back to XuanCe's built-in config lookup

    resolved = str(yaml_path)
    _logger = logging.getLogger(__name__)
    _logger.info("Resolved custom config: %s", resolved)

    if num_groups is not None:
        # Competition mode: replicate path for each group
        return [resolved] * num_groups
    return resolved


def _normalize_method_name(method: str) -> str:
    """Normalize method name for XuanCe config lookup.

    XuanCe's get_runner() expects lowercase method names matching
    the config folder names (ppo, dqn, sac, etc.).

    Args:
        method: Method name from UI (e.g., "PPO_Clip", "MAPPO_Clip").

    Returns:
        Normalized lowercase method name for config lookup.
    """
    # Try exact match first
    if method in _METHOD_NAME_MAP:
        return _METHOD_NAME_MAP[method]

    # Try case-insensitive match
    method_lower = method.lower()
    for key, value in _METHOD_NAME_MAP.items():
        if key.lower() == method_lower:
            return value

    # Fallback: just lowercase and remove common suffixes
    normalized = method_lower.replace("_clip", "").replace("_kl", "")
    return normalized


# Import standardized telemetry from gym_gui
try:
    from gym_gui.core.worker import TelemetryEmitter as StandardTelemetryEmitter
    from gym_gui.logging_config.helpers import log_constant
    from gym_gui.logging_config.log_constants import (
        LOG_WORKER_XUANCE_RUNTIME_STARTED,
        LOG_WORKER_XUANCE_RUNTIME_STOPPED,
        LOG_WORKER_XUANCE_RUNTIME_ERROR,
        LOG_WORKER_XUANCE_TRAINING_STARTED,
        LOG_WORKER_XUANCE_TRAINING_COMPLETED,
        LOG_WORKER_XUANCE_CONFIG_LOADED,
        LOG_WORKER_XUANCE_AGENT_CREATED,
        LOG_WORKER_XUANCE_DEBUG,
    )
    from gym_gui.config.paths import VAR_TRAINER_DIR
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    StandardTelemetryEmitter = None
    log_constant = None

# Register MOSAIC custom environments with XuanCe
# This adds MultiGrid and other environments to XuanCe's registry
try:
    from xuance_worker.environments import register_mosaic_environments
    register_mosaic_environments()
except ImportError:
    pass  # Environments module not available or XuanCe not installed
    VAR_TRAINER_DIR = None

# Import analytics manifest writer
try:
    from .analytics import write_analytics_manifest
    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False
    write_analytics_manifest = None

LOGGER = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """Handle numpy types for json.dumps."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass(frozen=True)
class XuanCeRuntimeSummary:
    """Summary returned from XuanCe training runs.

    Attributes:
        status: Run status ("completed", "dry-run", "error").
        method: Algorithm that was executed.
        env_id: Environment that was used.
        runner_type: XuanCe runner class name (RunnerDRL, RunnerMARL, etc.).
        config: Dictionary representation of the run configuration.
        mode: Optional execution mode ("benchmark", etc.).
        analytics_manifest: Optional path to analytics manifest file.
    """

    status: str
    method: str
    env_id: str
    runner_type: str
    config: Dict[str, Any]
    mode: Optional[str] = None
    analytics_manifest: Optional[str] = None


class XuanCeWorkerRuntime:
    """Orchestrate XuanCe algorithm execution.

    This class provides a clean interface to XuanCe's training infrastructure,
    supporting both single-agent (RunnerDRL) and multi-agent (RunnerMARL,
    RunnerPettingzoo) runners.

    Example:
        >>> config = XuanCeWorkerConfig(
        ...     run_id="test_run",
        ...     method="ppo",
        ...     env="classic_control",
        ...     env_id="CartPole-v1",
        ...     running_steps=10000,
        ... )
        >>> runtime = XuanCeWorkerRuntime(config)
        >>> summary = runtime.run()
        >>> print(summary.status)
        'completed'
    """

    def __init__(
        self,
        config: XuanCeWorkerConfig,
        *,
        dry_run: bool = False,
    ) -> None:
        """Initialize the runtime.

        Args:
            config: XuanCe worker configuration.
            dry_run: If True, validate configuration without executing.
        """
        self._config = config
        self._dry_run = dry_run

        # Log config loaded
        if _HAS_GYM_GUI and log_constant:
            log_constant(
                LOGGER,
                LOG_WORKER_XUANCE_CONFIG_LOADED,
                extra={
                    "run_id": config.run_id,
                    "method": config.method,
                    "env": config.env,
                    "env_id": config.env_id,
                },
            )

        # Create standardized telemetry emitter for lifecycle events
        if _HAS_GYM_GUI:
            self._lifecycle_emitter = StandardTelemetryEmitter(run_id=config.run_id)
        else:
            self._lifecycle_emitter = None

    @property
    def config(self) -> XuanCeWorkerConfig:
        """Return the current configuration."""
        return self._config

    def _build_parser_args(self) -> SimpleNamespace:
        """Build argument namespace for XuanCe.

        Converts our config to a SimpleNamespace that XuanCe's
        get_runner() function expects for parser_args.

        Returns:
            SimpleNamespace with XuanCe-compatible arguments.
        """
        args = SimpleNamespace()

        # Deep learning backend
        args.dl_toolbox = self._config.dl_toolbox

        # Core parameters
        args.device = self._config.device
        args.parallels = self._config.parallels
        args.running_steps = self._config.running_steps

        # IMPORTANT: Include env_id in parser_args to override config file defaults
        # XuanCe's get_runner() only uses the env_id parameter if it's NOT in the config,
        # so we must pass it via parser_args to ensure our env_id is used
        args.env_id = self._config.env_id

        # Seed configuration
        if self._config.seed is not None:
            args.seed = self._config.seed
            args.env_seed = self._config.seed

        # Apply extras
        for key, value in self._config.extras.items():
            setattr(args, key, value)

        return args

    def run(self) -> XuanCeRuntimeSummary:
        """Execute the configured XuanCe algorithm.

        This method creates a XuanCe runner and calls its run() method,
        which handles both training and testing based on the test_mode flag.

        Returns:
            XuanCeRuntimeSummary with execution results.

        Raises:
            RuntimeError: If XuanCe is not installed.
            Exception: Propagated from XuanCe runner on failure.
        """
        # Emit run_started lifecycle event
        if self._lifecycle_emitter:
            self._lifecycle_emitter.run_started(
                {
                    "method": self._config.method,
                    "env": self._config.env,
                    "env_id": self._config.env_id,
                    "dl_toolbox": self._config.dl_toolbox,
                    "running_steps": self._config.running_steps,
                    "device": self._config.device,
                    "parallels": self._config.parallels,
                },
                constant=LOG_WORKER_XUANCE_RUNTIME_STARTED,
            )

        if self._dry_run:
            LOGGER.info(
                "Dry-run mode | method=%s env=%s env_id=%s",
                self._config.method,
                self._config.env,
                self._config.env_id,
            )
            summary = XuanCeRuntimeSummary(
                status="dry-run",
                method=self._config.method,
                env_id=self._config.env_id,
                runner_type="unknown",
                config=self._config.to_dict(),
            )
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed({
                    "status": summary.status,
                    "method": summary.method,
                    "env_id": summary.env_id,
                    "runner_type": summary.runner_type,
                    "config": summary.config,
                })
            return summary

        # Create run directory for outputs (tensorboard, checkpoints, analytics)
        # This MUST happen before XuanCe tries to write any files
        # Follow CleanRL's pattern for directory structure
        run_dir = None
        tensorboard_dir = None
        tensorboard_dirname_relative = None
        if VAR_TRAINER_DIR is not None:
            # Custom scripts set MOSAIC_RUN_DIR to custom_scripts/{ULID}/.
            # Without this check, logs/tensorboard/checkpoints scatter to runs/
            # even though the script writes everything to custom_scripts/.
            mosaic_run_dir = os.environ.get("MOSAIC_RUN_DIR")
            if mosaic_run_dir:
                run_dir = Path(mosaic_run_dir).resolve()
            else:
                subdir = "evals" if self._config.test_mode else "runs"
                run_dir = (VAR_TRAINER_DIR / subdir / self._config.run_id).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create logs directory
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Create tensorboard directory (use "tensorboard" as default, or from extras)
            tensorboard_dirname_relative = self._config.extras.get("tensorboard_dir", "tensorboard")
            if tensorboard_dirname_relative:
                tensorboard_dir = run_dir / tensorboard_dirname_relative
                tensorboard_dir.mkdir(parents=True, exist_ok=True)

            # Create checkpoints directory
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

            # Create videos directory (for potential video capture)
            videos_dir = run_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)

            LOGGER.info("Created run directory: %s", run_dir)
            LOGGER.info("  - logs: %s", logs_dir)
            LOGGER.info("  - tensorboard: %s", tensorboard_dir)
            LOGGER.info("  - checkpoints: %s", checkpoints_dir)
            LOGGER.info("  - videos: %s", videos_dir)

        try:
            # Import XuanCe here to avoid import issues when not installed
            LOGGER.info("Importing XuanCe get_runner...")
            try:
                from xuance import get_runner
                LOGGER.info("XuanCe get_runner imported successfully")
            except ImportError as e:
                raise RuntimeError(
                    "XuanCe is not installed. Install with: pip install -e 3rd_party/xuance_worker"
                ) from e

            # Apply compatibility shims before get_runner (redirects dirs to var/, etc.)
            from .xuance_shims import apply_shims
            apply_shims()

            # Normalize method name for XuanCe config lookup
            # UI sends "PPO_Clip" but XuanCe expects "ppo" for config folder lookup
            normalized_method = _normalize_method_name(self._config.method)

            LOGGER.info(
                "Starting XuanCe training | method=%s (normalized=%s) env=%s env_id=%s backend=%s steps=%d",
                self._config.method,
                normalized_method,
                self._config.env,
                self._config.env_id,
                self._config.dl_toolbox,
                self._config.running_steps,
            )

            parser_args = self._build_parser_args()

            # CRITICAL: Set log_dir in parser_args to redirect TensorBoard to our directory
            # XuanCe constructs its SummaryWriter path using log_dir, so we must override it
            # This ensures TensorBoard events appear where the GUI expects them
            if tensorboard_dir is not None:
                parser_args.log_dir = str(tensorboard_dir)
                LOGGER.info("Set parser_args.log_dir = %s", parser_args.log_dir)

            # Also set model_dir to keep checkpoints organized in our run directory
            if run_dir is not None:
                parser_args.model_dir = str(run_dir / "checkpoints")
                LOGGER.info("Set parser_args.model_dir = %s", parser_args.model_dir)

            # =================================================================
            # DEBUG: Trace get_runner() call parameters
            # =================================================================
            LOGGER.info("=" * 60)
            LOGGER.info("DEBUG: Preparing get_runner() call")
            LOGGER.info("DEBUG: normalized_method = %s (type=%s)", normalized_method, type(normalized_method).__name__)
            LOGGER.info("DEBUG: env = %s", self._config.env)
            LOGGER.info("DEBUG: env_id = %s", self._config.env_id)
            LOGGER.info("DEBUG: config_path = %s", self._config.config_path)
            LOGGER.info("DEBUG: parser_args type = %s", type(parser_args).__name__)
            LOGGER.info("DEBUG: parser_args.__dict__ = %s", vars(parser_args) if hasattr(parser_args, '__dict__') else 'N/A')
            LOGGER.info("DEBUG: is_test = %s", self._config.test_mode)

            # Check if this is a competition environment requiring multiple policies
            # RunnerCompetition expects method as a list (one per team/group)
            num_groups = _get_competition_num_groups(self._config.env, self._config.env_id)
            LOGGER.info("DEBUG: _get_competition_num_groups returned: %s", num_groups)

            if num_groups is not None:
                # Competition mode: pass method as list for separate policies per team
                # e.g., ["mappo", "mappo"] for 2-team Soccer
                method_for_runner = [normalized_method] * num_groups
                LOGGER.info(
                    "DEBUG: Competition mode - method_for_runner = %s (type=%s)",
                    method_for_runner,
                    type(method_for_runner).__name__,
                )
            else:
                # Standard mode: single method string
                method_for_runner = normalized_method
                LOGGER.info(
                    "DEBUG: Standard mode - method_for_runner = %s (type=%s)",
                    method_for_runner,
                    type(method_for_runner).__name__,
                )

            # Resolve custom config path for environments in xuance_worker
            resolved_config_path = _resolve_custom_config_path(
                method=normalized_method,
                env=self._config.env,
                env_id=self._config.env_id,
                num_groups=num_groups,
                config_path=self._config.config_path,
            )
            LOGGER.info("DEBUG: resolved_config_path = %s", resolved_config_path)

            LOGGER.info("DEBUG: Calling get_runner() now...")
            LOGGER.info("=" * 60)

            try:
                runner = get_runner(
                    algo=method_for_runner,
                    env=self._config.env,
                    env_id=self._config.env_id,
                    config_path=resolved_config_path,
                    parser_args=parser_args,
                )
                LOGGER.info("DEBUG: get_runner() returned successfully")
                LOGGER.info("DEBUG: runner type = %s", type(runner).__name__)
            except Exception as e:
                LOGGER.error("=" * 60)
                LOGGER.error("DEBUG: get_runner() FAILED with exception:")
                LOGGER.error("DEBUG: Exception type: %s", type(e).__name__)
                LOGGER.error("DEBUG: Exception message: %s", str(e))
                import traceback
                LOGGER.error("DEBUG: Full traceback:\n%s", traceback.format_exc())
                LOGGER.error("=" * 60)
                LOGGER.error("DEBUG: Parameters that caused failure:")
                LOGGER.error("DEBUG:   method = %s (type=%s)", method_for_runner, type(method_for_runner).__name__)
                LOGGER.error("DEBUG:   env = %s", self._config.env)
                LOGGER.error("DEBUG:   env_id = %s", self._config.env_id)
                LOGGER.error("DEBUG:   config_path = %s", self._config.config_path)
                LOGGER.error("DEBUG:   parser_args = %s", vars(parser_args) if hasattr(parser_args, '__dict__') else parser_args)
                raise

            runner_type = type(runner).__name__
            LOGGER.info("Created XuanCe runner: %s", runner_type)

            # Log agent/runner created
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_AGENT_CREATED,
                    extra={
                        "run_id": self._config.run_id,
                        "runner_type": runner_type,
                        "method": self._config.method,
                    },
                )

            # Log training started
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_TRAINING_STARTED,
                    extra={
                        "run_id": self._config.run_id,
                        "method": self._config.method,
                        "env_id": self._config.env_id,
                        "running_steps": self._config.running_steps,
                    },
                )


            # ── Pretrained weight loading (curriculum transfer) ──────────
            pretrained_dir = self._config.extras.get("pretrained_model_dir")
            if pretrained_dir and not self._config.test_mode:
                pretrained_path = Path(pretrained_dir)
                if pretrained_path.exists():
                    LOGGER.info(
                        "Loading pretrained weights from: %s", pretrained_path
                    )
                    runner.agent.load_model(str(pretrained_path))
                    LOGGER.info("Pretrained weights loaded successfully")
                else:
                    LOGGER.warning(
                        "pretrained_model_dir does not exist: %s",
                        pretrained_path,
                    )
            # Execute training
            runner.run()

            LOGGER.info(
                "XuanCe training completed | method=%s runner=%s",
                self._config.method,
                runner_type,
            )

            # Log training completed
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_TRAINING_COMPLETED,
                    extra={
                        "run_id": self._config.run_id,
                        "method": self._config.method,
                        "env_id": self._config.env_id,
                        "runner_type": runner_type,
                    },
                )

            # Generate analytics manifest with relative paths
            manifest_path = None
            if _HAS_ANALYTICS and run_dir is not None:
                try:
                    manifest_path = write_analytics_manifest(
                        self._config,
                        notes=f"XuanCe {self._config.method} training on {self._config.env_id}",
                        tensorboard_dir=tensorboard_dirname_relative,
                        checkpoints_dir="checkpoints",
                        logs_dir="logs",
                        videos_dir="videos",
                        run_dir=run_dir,
                    )
                    LOGGER.info("Analytics manifest written to: %s", manifest_path)
                except Exception as e:
                    LOGGER.warning("Failed to write analytics manifest: %s", e)

            summary = XuanCeRuntimeSummary(
                status="completed",
                method=self._config.method,
                env_id=self._config.env_id,
                runner_type=runner_type,
                config=self._config.to_dict(),
                analytics_manifest=str(manifest_path) if manifest_path else None,
            )

            # Emit run_completed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(
                    {
                        "status": summary.status,
                        "method": summary.method,
                        "env_id": summary.env_id,
                        "runner_type": summary.runner_type,
                        "config": summary.config,
                    },
                    constant=LOG_WORKER_XUANCE_RUNTIME_STOPPED,
                )

            return summary

        except Exception as e:
            LOGGER.error("XuanCe training failed: %s", e, exc_info=True)

            error_summary = {
                "status": "failed",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "error": str(e),
                "config": self._config.to_dict(),
            }

            # Emit run_failed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_failed(
                    error_summary,
                    constant=LOG_WORKER_XUANCE_RUNTIME_ERROR,
                )

            raise


# ===========================================================================
# Interactive Runtime for GUI step-by-step control
# ===========================================================================


@dataclass
class InteractiveConfig:
    """Configuration for interactive (step-by-step) policy evaluation.

    Attributes:
        run_id: Unique identifier for the run.
        env_id: Environment ID (e.g., "CartPole-v1", "MiniGrid-Empty-8x8-v0").
        method: Algorithm name (e.g., "ppo", "dqn", "sac").
        policy_path: Path to trained policy checkpoint.
        device: Computing device ("cpu", "cuda").
        dl_toolbox: Deep learning backend ("torch", "tensorflow", "mindspore").
        env: Environment family (e.g., "classic_control", "atari").
    """

    run_id: str
    env_id: str
    method: str
    policy_path: str
    device: str = "cpu"
    dl_toolbox: str = "torch"
    env: str = "classic_control"


class InteractiveRuntime:
    """Interactive runtime for step-by-step XuanCe policy evaluation.

    Enables GUI-controlled stepping for scientific comparison with LLM operators.
    Follows the same IPC protocol as cleanrl_worker.InteractiveRuntime.

    Protocol:
        Input (stdin):
            {"cmd": "reset", "seed": 42}  - Reset environment with seed
            {"cmd": "step"}               - Execute one step using loaded policy
            {"cmd": "stop"}               - Terminate gracefully
            {"cmd": "ping"}               - Health check

        Output (stdout):
            {"type": "init", ...}         - Initialization message
            {"type": "ready", ...}        - Environment reset, ready for steps
            {"type": "step", ...}         - Step result with render_payload
            {"type": "episode_done", ...} - Episode completed
            {"type": "error", ...}        - Error message
            {"type": "stopped"}           - Graceful shutdown complete
            {"type": "pong"}              - Health check response
    """

    def __init__(self, config: InteractiveConfig):
        """Initialize interactive runtime.

        Args:
            config: Interactive configuration with policy path.
        """
        self._config = config
        self._policy_path = config.policy_path
        self._env_id = config.env_id
        self._method = config.method
        self._device = config.device
        self._dl_toolbox = config.dl_toolbox

        # State (initialized on reset)
        self._envs = None  # Vector env for model compatibility
        self._agent = None
        self._obs = None
        self._step_idx = 0
        self._episode_reward = 0.0
        self._episode_count = 0

        # Action-selector mode state (set by init_agent, used by select_action)
        self._player_id: Optional[str] = None
        self._n_agents: int = 2  # default for soccer 1v1; used for MAPPO one-hot

        LOGGER.info(
            "InteractiveRuntime initialized | env=%s method=%s policy=%s",
            self._env_id,
            self._method,
            self._policy_path,
        )

    def _load_policy(self, action_selector: bool = False) -> None:
        """Load trained XuanCe policy from checkpoint.

        Args:
            action_selector: When True, skip SyncVectorEnv creation.
                The GUI owns the shared environment in action-selector mode,
                so we only need to load the policy network — no own env needed.
        """
        import gymnasium as gym

        if not self._policy_path:
            raise ValueError("policy_path is required for interactive mode")

        policy_file = Path(self._policy_path).expanduser()
        if not policy_file.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_file}")

        LOGGER.info("Loading XuanCe policy from %s (action_selector=%s)", policy_file, action_selector)

        # Auto-import environment packages to register their environments
        env_id = self._env_id
        is_minigrid = env_id.startswith("MiniGrid") or env_id.startswith("BabyAI")
        if is_minigrid:
            try:
                import minigrid  # noqa: F401 - registers MiniGrid/BabyAI envs
                LOGGER.debug("Imported minigrid to register environments")
            except ImportError:
                LOGGER.warning("minigrid package not installed")

        if not action_selector:
            # Own-environment mode: create SyncVectorEnv for _handle_step
            # Not used in action-selector mode — the GUI manages the shared env.
            def make_env():
                env = gym.make(env_id, render_mode="rgb_array")
                if is_minigrid:
                    try:
                        from minigrid.wrappers import ImgObsWrapper
                        env = ImgObsWrapper(env)
                        env = gym.wrappers.FlattenObservation(env)
                    except ImportError:
                        pass
                env = gym.wrappers.RecordEpisodeStatistics(env)
                return env

            self._envs = gym.vector.SyncVectorEnv([make_env])

        # Load XuanCe agent
        try:
            from xuance import get_runner
        except ImportError as e:
            raise RuntimeError(
                "XuanCe is not installed. Install with: pip install xuance"
            ) from e

        # Apply compatibility shims before get_runner (redirects dirs to var/)
        from .xuance_shims import apply_shims
        apply_shims()

        # Determine environment family from env_id
        env_family = self._config.env
        if is_minigrid:
            env_family = "minigrid"
        elif "CartPole" in env_id or "MountainCar" in env_id or "Pendulum" in env_id:
            env_family = "classic_control"
        elif "Pong" in env_id or "Breakout" in env_id:
            env_family = "atari"
        elif "MosaicMultiGrid" in env_id:
            env_family = "mosaic_multigrid"
        elif "IniMultiGrid" in env_id:
            env_family = "ini_multigrid"
        elif "multigrid" in env_id.lower():
            env_family = "multigrid"

        # XuanCe's get_runner uses "multigrid" for all multigrid variants.
        # Our internal names ("mosaic_multigrid", "ini_multigrid") identify the
        # specific implementation, but YAML configs live under configs/{method}/multigrid/.
        xuance_env = env_family
        if env_family in ("mosaic_multigrid", "ini_multigrid"):
            xuance_env = "multigrid"

        # Build parser args for XuanCe
        parser_args = SimpleNamespace()
        parser_args.dl_toolbox = self._dl_toolbox
        parser_args.device = self._device
        parser_args.parallels = 1
        parser_args.running_steps = 1  # Not training, just loading

        # Resolve config YAML from the worker's own configs directory.
        # XuanCe's get_runner() looks in its vendored package configs/ by default,
        # which doesn't contain our multigrid configs.
        # Also convert gymnasium ID to XuanCe short env_id for config lookup
        # and runner creation (XuanCe's env factory expects short names).
        xuance_env_id = _gymnasium_to_xuance_env_id(self._env_id) or self._env_id
        if xuance_env_id != self._env_id:
            LOGGER.info("Mapped gymnasium ID '%s' -> xuance env_id '%s'", self._env_id, xuance_env_id)

        config_path = _resolve_custom_config_path(
            method=self._method,
            env=xuance_env,
            env_id=xuance_env_id,
            num_groups=None,
            config_path=None,
        )
        if config_path:
            LOGGER.info("Using worker config: %s", config_path)

        # Create runner to get agent — raises on failure (no silent fallback).
        runner = get_runner(
            algo=self._method,
            env=xuance_env,
            env_id=xuance_env_id,
            config_path=config_path,
            parser_args=parser_args,
        )

        if not hasattr(runner, 'agent'):
            raise RuntimeError(
                f"XuanCe runner ({type(runner).__name__}) has no 'agent' attribute. "
                "Cannot load policy."
            )

        self._agent = runner.agent

        # Update n_agents from the loaded agent (used for MAPPO one-hot sizing)
        if hasattr(self._agent, 'n_agents'):
            self._n_agents = self._agent.n_agents
            LOGGER.info("n_agents set to %d from loaded agent", self._n_agents)

        # Load the model weights
        if hasattr(self._agent, 'load_model'):
            self._agent.load_model(str(policy_file))
            LOGGER.info("XuanCe agent model loaded from %s", policy_file)
        else:
            # Try loading directly via torch
            import torch
            state_dict = torch.load(str(policy_file), map_location=self._device)
            if hasattr(self._agent, 'policy'):
                self._agent.policy.load_state_dict(state_dict)
                LOGGER.info("Loaded model weights via torch.load")
            else:
                raise RuntimeError(
                    f"Cannot load model weights: agent {type(self._agent).__name__} "
                    "has neither load_model() nor .policy attribute."
                )

        LOGGER.info("Policy loading completed (action_selector=%s)", action_selector)

    def _get_action(self, obs) -> int:
        """Get action from loaded policy.

        Used by _handle_step (own-environment mode) and indirectly by
        _handle_select_action (action-selector mode, after obs preparation).

        Raises:
            RuntimeError: If agent is not loaded or action selection fails.
                Callers must catch this and emit {"type": "error"}.
                Never silently falls back to random — that hides real failures.
        """
        if self._agent is None:
            raise RuntimeError("Agent not loaded. Send reset or init_agent first.")

        if hasattr(self._agent, 'action'):
            action = self._agent.action(obs)
        elif hasattr(self._agent, 'act'):
            action = self._agent.act(obs)
        elif hasattr(self._agent, 'policy'):
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self._device)
                action = self._agent.policy(obs_tensor)
                if hasattr(action, 'cpu'):
                    action = action.cpu().numpy()
        else:
            raise RuntimeError(
                f"Agent has no recognised action method. "
                f"Type: {type(self._agent).__name__}"
            )

        # Extract scalar if needed
        if hasattr(action, '__len__'):
            if len(action) == 1:
                action = action[0]
            elif len(action) > 1 and hasattr(action[0], '__len__'):
                action = action[0][0]

        return int(action)

    def _append_agent_onehot(self, obs: "np.ndarray", player_id: str) -> "np.ndarray":
        """Append one-hot agent index to obs for MAPPO parameter-sharing mode.

        MAPPO with use_parameter_sharing=True expects:
            input_dim = obs_dim + n_agents
        The one-hot tells the shared network which agent is requesting the action.

        Args:
            obs: Raw observation, shape (obs_dim,).
            player_id: Agent identifier, e.g. "agent_0" or "agent_1".

        Returns:
            Concatenated observation, shape (obs_dim + n_agents,).

        Raises:
            RuntimeError: If player_id cannot be parsed or agent_idx is out of range.
        """
        import numpy as np
        try:
            agent_idx = int(player_id.split("_")[-1])  # "agent_0" → 0
        except (ValueError, IndexError):
            raise RuntimeError(
                f"Cannot parse agent index from player_id='{player_id}'. "
                f"Expected format: 'agent_N' (e.g. 'agent_0')."
            )
        if agent_idx >= self._n_agents:
            raise RuntimeError(
                f"agent_idx={agent_idx} >= n_agents={self._n_agents}. "
                f"Policy was trained with {self._n_agents} agents "
                f"but player_id='{player_id}' implies agent index {agent_idx}. "
                f"Check that the checkpoint matches the game being played."
            )
        one_hot = np.zeros(self._n_agents, dtype=np.float32)
        one_hot[agent_idx] = 1.0
        return np.concatenate([obs, one_hot])

    def _get_marl_action(self, obs: "np.ndarray", player_id: str) -> int:
        """Get action from a MARL policy (IPPO or MAPPO) for a specific agent.

        Calls the policy's forward pass directly with the correct observation
        format, rather than going through the agent's high-level action() method
        which expects full multi-env observation dicts.

        For IPPO (use_parameter_sharing=False):
            - Each agent has its own network keyed by agent_key (e.g. "agent_0").
            - obs is passed as {agent_key: tensor}, no one-hot needed.

        For MAPPO (use_parameter_sharing=True):
            - One shared network; agent identity conveyed via agents_id one-hot.
            - agents_id is built from agent_idx and passed to policy as a separate
              argument — it is NOT concatenated to obs before the call.
              (The policy concatenates it internally after the representation layer.)

        Args:
            obs: Raw observation array, shape (obs_dim,).
            player_id: Agent identifier, e.g. "agent_0" or "agent_1".

        Returns:
            Integer action selected by the policy.

        Raises:
            RuntimeError: If agent is not loaded, player_id is invalid, or
                the policy forward pass fails.
        """
        import torch
        import numpy as np

        if self._agent is None:
            raise RuntimeError("Agent not loaded. Send init_agent first.")

        is_parameter_sharing = getattr(self._agent, 'use_parameter_sharing', False)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self._device)  # (1, obs_dim)

        if is_parameter_sharing:
            # MAPPO: shared network, differentiated by agent_ids one-hot.
            # The one-hot is passed as agents_id — the policy concatenates it
            # with the representation output internally. Do NOT pre-concatenate.
            agent_key = self._agent.model_keys[0]  # Single shared model key
            try:
                agent_idx = int(player_id.split("_")[-1])
            except (ValueError, IndexError):
                raise RuntimeError(
                    f"Cannot parse agent index from player_id='{player_id}'. "
                    f"Expected format: 'agent_N' (e.g. 'agent_0')."
                )
            if agent_idx >= self._n_agents:
                raise RuntimeError(
                    f"agent_idx={agent_idx} >= n_agents={self._n_agents}. "
                    f"Checkpoint was trained with {self._n_agents} agents but "
                    f"player_id='{player_id}' implies index {agent_idx}."
                )
            agents_id = torch.zeros(1, self._n_agents, dtype=torch.float32, device=self._device)
            agents_id[0, agent_idx] = 1.0
            obs_input = {agent_key: obs_tensor}
            LOGGER.debug(
                "MAPPO action: agent_key=%s agent_idx=%d agents_id=%s",
                agent_key, agent_idx, agents_id.tolist()
            )
            with torch.no_grad():
                _, pi_dists = self._agent.policy(
                    observation=obs_input,
                    agent_ids=agents_id,
                    agent_key=agent_key,
                )
            action = pi_dists[agent_key].stochastic_sample()

        else:
            # IPPO: each agent has its own network keyed by agent_key.
            # No one-hot needed — the correct network is selected by agent_key.
            agent_key = player_id  # e.g. "agent_0", "agent_1"
            agent_keys = getattr(self._agent, 'agent_keys', [])
            if agent_key not in agent_keys:
                raise RuntimeError(
                    f"player_id='{player_id}' (agent_key='{agent_key}') not in "
                    f"agent_keys={agent_keys}. "
                    f"Check that player_id matches the environment's agent naming."
                )
            obs_input = {agent_key: obs_tensor}
            LOGGER.debug("IPPO action: agent_key=%s", agent_key)
            with torch.no_grad():
                _, pi_dists = self._agent.policy(
                    observation=obs_input,
                    agent_ids=None,
                    agent_key=agent_key,
                )
            action = pi_dists[agent_key].stochastic_sample()

        return int(action.cpu().item())

    def _handle_init_agent(self, cmd: dict) -> None:
        """Initialize in action-selector mode (no env management).

        The GUI owns the shared environment and will send individual agent
        observations via select_action commands. This handler only needs to
        load the policy network — it does not create its own environment.

        Args:
            cmd: Command dict with 'game_name' and 'player_id'.
        """
        game_name = cmd.get("game_name", "")
        player_id = cmd.get("player_id", "")
        self._player_id = player_id  # stored for use in _handle_select_action

        try:
            if self._agent is None:
                self._load_policy(action_selector=True)
            self._emit({
                "type": "agent_initialized",
                "game_name": game_name,
                "player_id": player_id,
            })
            LOGGER.info("Agent initialized for %s as %s", game_name, player_id)
        except Exception as e:
            LOGGER.exception("init_agent failed for %s as %s", game_name, player_id)
            self._emit({"type": "error", "message": f"init_agent failed: {e}"})

    def _handle_select_action(self, cmd: dict) -> None:
        """Select action given an external observation (action-selector mode).

        The GUI manages the shared environment and sends the individual agent's
        observation here. Policy inference runs and the action is returned.

        For MARL policies (IPPO/MAPPO), _get_marl_action() is used to call the
        policy's forward pass directly with the correct format:
          - IPPO: obs dict keyed by agent_key, no one-hot (each agent has own net)
          - MAPPO: obs dict keyed by shared model_key, agents_id one-hot passed
                   as a separate argument (policy concatenates it internally)

        On ANY failure, emits {"type": "error"} — never falls back to random.
        The GUI's _query_worker_for_action raises RuntimeError on error responses,
        which surfaces immediately rather than silently corrupting the episode.

        Args:
            cmd: Command dict with 'observation' (list) and 'player_id' (str).
        """
        import numpy as np

        observation = cmd.get("observation")
        player_id = cmd.get("player_id") or self._player_id or ""

        if observation is None:
            self._emit({"type": "error", "message": "select_action missing 'observation'"})
            return

        if self._agent is None:
            try:
                self._load_policy(action_selector=True)
            except Exception as e:
                self._emit({"type": "error", "message": f"Policy not loaded: {e}"})
                return

        try:
            obs = np.array(observation, dtype=np.float32)

            # Use MARL-aware action selection for multi-agent policies.
            # _get_marl_action calls the policy's forward() directly, handling:
            #   IPPO: selects the correct per-agent network via agent_key=player_id
            #   MAPPO: builds agents_id one-hot; policy concatenates it internally
            # _get_action is kept only for single-agent (DRL) policies.
            is_marl = hasattr(self._agent, 'use_parameter_sharing')
            if is_marl and player_id:
                action = self._get_marl_action(obs, player_id)
            else:
                action = self._get_action(obs)

            self._emit({
                "type": "action_selected",
                "player_id": player_id,
                "action": int(action),
            })
        except Exception as e:
            LOGGER.exception("select_action failed for player_id=%s", player_id)
            self._emit({"type": "error", "message": f"select_action failed: {e}"})

    def _handle_reset(self, seed: Optional[int] = None) -> None:
        """Handle reset command - initialize environment with seed."""
        try:
            # Load policy on first reset
            if self._envs is None:
                self._load_policy()

            # Reset environment
            self._obs, info = self._envs.reset(seed=seed)
            self._step_idx = 0
            self._episode_reward = 0.0

            # Get initial render frame
            render_payload = None
            try:
                frame = self._envs.call("render")
                if frame is not None and len(frame) > 0:
                    rgb_frame = frame[0]
                    if rgb_frame is not None and hasattr(rgb_frame, 'shape'):
                        render_payload = {
                            "mode": "rgb_array",
                            "rgb": rgb_frame.tolist() if hasattr(rgb_frame, 'tolist') else rgb_frame,
                            "width": int(rgb_frame.shape[1]),
                            "height": int(rgb_frame.shape[0]),
                        }
            except Exception:
                pass

            ready_response = {
                "type": "ready",
                "run_id": self._config.run_id,
                "env_id": self._env_id,
                "method": self._method,
                "seed": seed,
                "observation_shape": list(self._obs.shape) if hasattr(self._obs, 'shape') else None,
                # Include stats for GUI reset
                "step_index": 0,
                "episode_index": self._episode_count,
                "episode_reward": 0.0,
            }
            if render_payload is not None:
                ready_response["render_payload"] = render_payload

            self._emit(ready_response)

            LOGGER.debug("Environment reset with seed=%s", seed)

        except Exception as e:
            LOGGER.exception("Reset failed")
            self._emit({"type": "error", "message": str(e)})

    def _handle_step(self) -> None:
        """Execute one step using the loaded policy."""
        if self._obs is None:
            self._emit({"type": "error", "message": "Environment not initialized. Send reset first."})
            return

        try:
            # Get action from policy
            action = self._get_action(self._obs)

            # Step environment
            obs_new, reward, terminated, truncated, info = self._envs.step([action])

            # Handle reward (may be array from vector env)
            if hasattr(reward, '__len__'):
                reward_scalar = float(reward[0])
            else:
                reward_scalar = float(reward)

            # Handle terminated/truncated
            if hasattr(terminated, '__len__'):
                term = bool(terminated[0])
                trunc = bool(truncated[0])
            else:
                term = bool(terminated)
                trunc = bool(truncated)

            done = term or trunc

            self._episode_reward += reward_scalar
            self._step_idx += 1

            # Get RGB frame for rendering
            render_payload = None
            try:
                frame = self._envs.call("render")
                if frame is not None and len(frame) > 0:
                    rgb_frame = frame[0]
                    if rgb_frame is not None and hasattr(rgb_frame, 'shape'):
                        render_payload = {
                            "mode": "rgb_array",
                            "rgb": rgb_frame.tolist() if hasattr(rgb_frame, 'tolist') else rgb_frame,
                            "width": int(rgb_frame.shape[1]),
                            "height": int(rgb_frame.shape[0]),
                        }
            except Exception:
                pass

            # Emit step telemetry
            step_data = {
                "type": "step",
                "step_index": self._step_idx,
                "episode_index": self._episode_count,
                "action": action,
                "reward": reward_scalar,
                "terminated": term,
                "truncated": trunc,
                "episode_reward": self._episode_reward,
            }

            if render_payload is not None:
                step_data["render_payload"] = render_payload

            self._emit(step_data)

            # Check for episode end
            if done:
                self._episode_count += 1
                self._emit({
                    "type": "episode_done",
                    "total_reward": self._episode_reward,
                    "episode_length": self._step_idx,
                    "episode_number": self._episode_count,
                })
                LOGGER.info(
                    "Episode %d completed | reward=%.3f steps=%d",
                    self._episode_count,
                    self._episode_reward,
                    self._step_idx,
                )
                # Reset counters for next episode (SyncVectorEnv auto-resets)
                self._step_idx = 0
                self._episode_reward = 0.0

            self._obs = obs_new

        except Exception as e:
            LOGGER.exception("Step failed")
            self._emit({"type": "error", "message": str(e)})

    def _emit(self, data: dict) -> None:
        """Emit JSON line to stdout."""
        print(json.dumps(data, default=_json_default), flush=True)

    def run(self) -> None:
        """Main loop - read commands from stdin, execute, respond."""
        # Emit init message
        self._emit({
            "type": "init",
            "run_id": self._config.run_id,
            "env_id": self._env_id,
            "method": self._method,
            "policy_path": self._policy_path,
            "version": "1.0",
        })

        LOGGER.info("XuanCe Interactive runtime started, waiting for commands...")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as e:
                self._emit({"type": "error", "message": f"Invalid JSON: {e}"})
                continue

            cmd_type = cmd.get("cmd")
            LOGGER.debug("Received command: %s", cmd_type)

            if cmd_type == "reset":
                self._handle_reset(cmd.get("seed"))
            elif cmd_type == "step":
                self._handle_step()
            elif cmd_type == "init_agent":
                self._handle_init_agent(cmd)
            elif cmd_type == "select_action":
                self._handle_select_action(cmd)
            elif cmd_type == "stop":
                self._emit({"type": "stopped"})
                LOGGER.info("Stop command received, shutting down")
                break
            elif cmd_type == "ping":
                self._emit({"type": "pong"})
            else:
                self._emit({"type": "error", "message": f"Unknown command: {cmd_type}"})

        # Cleanup
        if self._envs is not None:
            try:
                self._envs.close()
            except Exception:
                pass

        LOGGER.info("XuanCe Interactive runtime stopped")


__all__ = ["XuanCeWorkerRuntime", "XuanCeRuntimeSummary", "InteractiveConfig", "InteractiveRuntime"]
