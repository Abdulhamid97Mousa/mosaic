"""Runtime orchestration for XuanCe training.

This module provides the XuanCeWorkerRuntime class which wraps XuanCe's
get_runner() API to execute training and benchmark runs.

Also provides InteractiveRuntime for GUI step-by-step policy evaluation.
"""

from __future__ import annotations

import json
import logging
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
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    StandardTelemetryEmitter = None
    log_constant = None

# Import analytics manifest writer
try:
    from .analytics import write_analytics_manifest
    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False
    write_analytics_manifest = None

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class XuanCeRuntimeSummary:
    """Summary returned from XuanCe training runs.

    Attributes:
        status: Run status ("completed", "dry-run", "error").
        method: Algorithm that was executed.
        env_id: Environment that was used.
        runner_type: XuanCe runner class name (RunnerDRL, RunnerMARL, etc.).
        config: Dictionary representation of the run configuration.
    """

    status: str
    method: str
    env_id: str
    runner_type: str
    config: Dict[str, Any]


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

    def run(self) -> Dict[str, Any]:
        """Execute the configured XuanCe algorithm.

        This method creates a XuanCe runner and calls its run() method,
        which handles both training and testing based on the test_mode flag.

        Returns:
            Dictionary with execution results (compatible with standardized interface).

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
            summary = {
                "status": "dry-run",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "runner_type": "unknown",
                "config": self._config.to_dict(),
            }
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(summary)
            return summary

        try:
            # Import XuanCe here to avoid import issues when not installed
            try:
                from xuance import get_runner
            except ImportError as e:
                raise RuntimeError(
                    "XuanCe is not installed. Install with: pip install -e 3rd_party/xuance_worker"
                ) from e

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

            runner = get_runner(
                method=normalized_method,
                env=self._config.env,
                env_id=self._config.env_id,
                config_path=self._config.config_path,
                parser_args=parser_args,
                is_test=self._config.test_mode,
            )

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

            # Generate analytics manifest
            manifest_path = None
            if _HAS_ANALYTICS:
                try:
                    manifest_path = write_analytics_manifest(
                        self._config,
                        notes=f"XuanCe {self._config.method} training on {self._config.env_id}",
                    )
                    LOGGER.info("Analytics manifest written to: %s", manifest_path)
                except Exception as e:
                    LOGGER.warning("Failed to write analytics manifest: %s", e)

            summary = {
                "status": "completed",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "runner_type": runner_type,
                "config": self._config.to_dict(),
                "analytics_manifest": str(manifest_path) if manifest_path else None,
            }

            # Emit run_completed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(
                    summary,
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

    def benchmark(self) -> Dict[str, Any]:
        """Execute benchmark mode (training with periodic evaluation).

        This method uses XuanCe's benchmark() function which trains
        for eval_interval steps, evaluates, and saves the best model.

        Returns:
            Dictionary with execution results (compatible with standardized interface).

        Raises:
            RuntimeError: If XuanCe is not installed.
            Exception: Propagated from XuanCe runner on failure.
        """
        # Emit run_started lifecycle event
        if self._lifecycle_emitter:
            self._lifecycle_emitter.run_started(
                {
                    "mode": "benchmark",
                    "method": self._config.method,
                    "env": self._config.env,
                    "env_id": self._config.env_id,
                    "dl_toolbox": self._config.dl_toolbox,
                    "running_steps": self._config.running_steps,
                },
                constant=LOG_WORKER_XUANCE_RUNTIME_STARTED,
            )

        if self._dry_run:
            LOGGER.info(
                "Dry-run benchmark mode | method=%s env=%s env_id=%s",
                self._config.method,
                self._config.env,
                self._config.env_id,
            )
            summary = {
                "status": "dry-run",
                "mode": "benchmark",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "runner_type": "unknown",
                "config": self._config.to_dict(),
            }
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(summary)
            return summary

        try:
            try:
                from xuance import get_runner
            except ImportError as e:
                raise RuntimeError(
                    "XuanCe is not installed. Install with: pip install -e 3rd_party/xuance_worker"
                ) from e

            # Normalize method name for XuanCe config lookup
            normalized_method = _normalize_method_name(self._config.method)

            LOGGER.info(
                "Starting XuanCe benchmark | method=%s (normalized=%s) env=%s env_id=%s backend=%s steps=%d",
                self._config.method,
                normalized_method,
                self._config.env,
                self._config.env_id,
                self._config.dl_toolbox,
                self._config.running_steps,
            )

            parser_args = self._build_parser_args()

            runner = get_runner(
                method=normalized_method,
                env=self._config.env,
                env_id=self._config.env_id,
                config_path=self._config.config_path,
                parser_args=parser_args,
                is_test=self._config.test_mode,
            )

            runner_type = type(runner).__name__
            LOGGER.info("Created XuanCe runner for benchmark: %s", runner_type)

            # Log agent/runner created
            if _HAS_GYM_GUI and log_constant:
                log_constant(
                    LOGGER,
                    LOG_WORKER_XUANCE_AGENT_CREATED,
                    extra={
                        "run_id": self._config.run_id,
                        "runner_type": runner_type,
                        "method": self._config.method,
                        "mode": "benchmark",
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
                        "mode": "benchmark",
                    },
                )

            # Execute benchmark
            runner.benchmark()

            LOGGER.info(
                "XuanCe benchmark completed | method=%s runner=%s",
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
                        "mode": "benchmark",
                    },
                )

            # Generate analytics manifest
            manifest_path = None
            if _HAS_ANALYTICS:
                try:
                    manifest_path = write_analytics_manifest(
                        self._config,
                        notes=f"XuanCe {self._config.method} benchmark on {self._config.env_id}",
                    )
                    LOGGER.info("Analytics manifest written to: %s", manifest_path)
                except Exception as e:
                    LOGGER.warning("Failed to write analytics manifest: %s", e)

            summary = {
                "status": "completed",
                "mode": "benchmark",
                "method": self._config.method,
                "env_id": self._config.env_id,
                "runner_type": runner_type,
                "config": self._config.to_dict(),
                "analytics_manifest": str(manifest_path) if manifest_path else None,
            }

            # Emit run_completed lifecycle event
            if self._lifecycle_emitter:
                self._lifecycle_emitter.run_completed(
                    summary,
                    constant=LOG_WORKER_XUANCE_RUNTIME_STOPPED,
                )

            return summary

        except Exception as e:
            LOGGER.error("XuanCe benchmark failed: %s", e, exc_info=True)

            error_summary = {
                "status": "failed",
                "mode": "benchmark",
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

        LOGGER.info(
            "InteractiveRuntime initialized | env=%s method=%s policy=%s",
            self._env_id,
            self._method,
            self._policy_path,
        )

    def _load_policy(self) -> None:
        """Load trained XuanCe policy from checkpoint."""
        import gymnasium as gym

        if not self._policy_path:
            raise ValueError("policy_path is required for interactive mode")

        policy_file = Path(self._policy_path).expanduser()
        if not policy_file.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_file}")

        LOGGER.info("Loading XuanCe policy from %s", policy_file)

        # Auto-import environment packages to register their environments
        if self._env_id.startswith("MiniGrid") or self._env_id.startswith("BabyAI"):
            try:
                import minigrid  # noqa: F401 - registers MiniGrid/BabyAI envs
                LOGGER.debug("Imported minigrid to register environments")
            except ImportError:
                LOGGER.warning("minigrid package not installed")

        # Create vectorized environment with render mode
        env_id = self._env_id
        is_minigrid = env_id.startswith("MiniGrid") or env_id.startswith("BabyAI")

        def make_env():
            env = gym.make(env_id, render_mode="rgb_array")
            # Apply MiniGrid wrappers if needed
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

        # Determine environment family from env_id
        env_family = self._config.env
        if is_minigrid:
            env_family = "minigrid"
        elif "CartPole" in env_id or "MountainCar" in env_id or "Pendulum" in env_id:
            env_family = "classic_control"
        elif "Pong" in env_id or "Breakout" in env_id:
            env_family = "atari"

        # Build parser args for XuanCe
        parser_args = SimpleNamespace()
        parser_args.dl_toolbox = self._dl_toolbox
        parser_args.device = self._device
        parser_args.parallels = 1
        parser_args.running_steps = 1  # Not training, just loading

        # Create runner to get agent
        try:
            runner = get_runner(
                method=self._method,
                env=env_family,
                env_id=self._env_id,
                parser_args=parser_args,
                is_test=True,  # Test mode
            )

            # Load the model weights
            if hasattr(runner, 'agent'):
                self._agent = runner.agent
                if hasattr(self._agent, 'load_model'):
                    self._agent.load_model(str(policy_file))
                    LOGGER.info("XuanCe agent model loaded")
                else:
                    # Try loading directly via torch
                    try:
                        import torch
                        state_dict = torch.load(str(policy_file), map_location=self._device)
                        if hasattr(self._agent, 'policy'):
                            self._agent.policy.load_state_dict(state_dict)
                        LOGGER.info("Loaded model via torch.load")
                    except Exception as e:
                        LOGGER.warning("Could not load model weights: %s", e)
            else:
                LOGGER.warning("Runner has no agent attribute")

        except Exception as e:
            LOGGER.warning("XuanCe runner creation failed: %s, using fallback", e)
            # Fallback: create a simple random agent for testing
            self._agent = None

        LOGGER.info("Policy loading completed")

    def _get_action(self, obs) -> int:
        """Get action from loaded policy."""
        import numpy as np

        if self._agent is None:
            # Fallback: random action
            return self._envs.single_action_space.sample()

        try:
            # XuanCe agents typically have an action() method
            if hasattr(self._agent, 'action'):
                action = self._agent.action(obs)
            elif hasattr(self._agent, 'act'):
                action = self._agent.act(obs)
            elif hasattr(self._agent, 'policy'):
                # Direct policy access
                import torch
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(self._device)
                    action = self._agent.policy(obs_tensor)
                    if hasattr(action, 'cpu'):
                        action = action.cpu().numpy()
            else:
                # Fallback
                action = self._envs.single_action_space.sample()

            # Extract scalar if needed
            if hasattr(action, '__len__'):
                if len(action) == 1:
                    action = action[0]
                elif len(action) > 1 and hasattr(action[0], '__len__'):
                    action = action[0][0]

            return int(action)

        except Exception as e:
            LOGGER.warning("Action selection failed: %s, using random", e)
            return self._envs.single_action_space.sample()

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
        print(json.dumps(data), flush=True)

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
