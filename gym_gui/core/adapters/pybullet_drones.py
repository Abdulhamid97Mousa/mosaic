"""PyBullet Drones environment adapters for quadcopter control.

gym-pybullet-drones is a PyBullet-based Gymnasium environment for single and
multi-agent reinforcement learning of quadcopter (Crazyflie 2.x) control.
It provides realistic physics simulation including aerodynamic effects.

Paper: Panerati, J., et al. (2021). Learning to Fly - a Gym Environment with
       PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control.
Repository: https://github.com/utiasDSL/gym-pybullet-drones
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode

_LOGGER = logging.getLogger(__name__)

# Try to import gym-pybullet-drones
try:
    import gymnasium as gym
    from gym_pybullet_drones.envs import HoverAviary, MultiHoverAviary
    from gym_pybullet_drones.utils.enums import (
        ActionType,
        DroneModel,
        ObservationType,
        Physics,
    )
    _PYBULLET_DRONES_AVAILABLE = True
except ImportError:
    _PYBULLET_DRONES_AVAILABLE = False
    gym = None  # type: ignore[assignment]
    HoverAviary = None  # type: ignore[assignment, misc]
    MultiHoverAviary = None  # type: ignore[assignment, misc]
    ActionType = None  # type: ignore[assignment, misc]
    DroneModel = None  # type: ignore[assignment, misc]
    ObservationType = None  # type: ignore[assignment, misc]
    Physics = None  # type: ignore[assignment, misc]


_PYBULLET_DRONES_STEP_LOG_FREQUENCY = 50


@dataclass
class PyBulletDronesConfig:
    """Configuration for PyBullet Drones environments.

    Attributes:
        env_id: The environment identifier (e.g., 'hover-aviary-v0')
        drone_model: Drone model to use ('cf2x', 'cf2p', 'race')
        num_drones: Number of drones (for multi-agent environments)
        physics: Physics simulation type
        pyb_freq: PyBullet simulation frequency in Hz
        ctrl_freq: Control/action frequency in Hz
        gui: Whether to enable PyBullet GUI visualization
        obs_type: Observation type ('kin' for kinematic, 'rgb' for vision)
        act_type: Action type ('rpm', 'pid', 'vel', 'one_d_rpm', 'one_d_pid')
        record: Whether to record video frames
    """
    env_id: str = "hover-aviary-v0"
    drone_model: str = "cf2x"
    num_drones: int = 1
    physics: str = "pyb"
    pyb_freq: int = 240
    ctrl_freq: int = 30
    gui: bool = False
    obs_type: str = "kin"
    act_type: str = "rpm"
    record: bool = False

    def to_gym_kwargs(self) -> dict[str, Any]:
        """Convert config to gymnasium.make() kwargs."""
        kwargs: dict[str, Any] = {
            "gui": self.gui,
            "record": self.record,
            "pyb_freq": self.pyb_freq,
            "ctrl_freq": self.ctrl_freq,
        }

        # Only add drone model and physics if the package is available
        if _PYBULLET_DRONES_AVAILABLE and DroneModel is not None:
            drone_model_map = {
                "cf2x": DroneModel.CF2X,
                "cf2p": DroneModel.CF2P,
                "race": DroneModel.RACE,
            }
            kwargs["drone_model"] = drone_model_map.get(
                self.drone_model.lower(), DroneModel.CF2X
            )

        if _PYBULLET_DRONES_AVAILABLE and Physics is not None:
            physics_map = {
                "pyb": Physics.PYB,
                "dyn": Physics.DYN,
                "pyb_gnd": Physics.PYB_GND,
                "pyb_drag": Physics.PYB_DRAG,
                "pyb_dw": Physics.PYB_DW,
                "pyb_gnd_drag_dw": Physics.PYB_GND_DRAG_DW,
            }
            kwargs["physics"] = physics_map.get(
                self.physics.lower(), Physics.PYB
            )

        if _PYBULLET_DRONES_AVAILABLE and ObservationType is not None:
            obs_type_map = {
                "kin": ObservationType.KIN,
                "rgb": ObservationType.RGB,
            }
            kwargs["obs"] = obs_type_map.get(
                self.obs_type.lower(), ObservationType.KIN
            )

        if _PYBULLET_DRONES_AVAILABLE and ActionType is not None:
            act_type_map = {
                "rpm": ActionType.RPM,
                "pid": ActionType.PID,
                "vel": ActionType.VEL,
                "one_d_rpm": ActionType.ONE_D_RPM,
                "one_d_pid": ActionType.ONE_D_PID,
            }
            kwargs["act"] = act_type_map.get(
                self.act_type.lower(), ActionType.RPM
            )

        return kwargs


@dataclass(slots=True)
class _PyBulletDronesMetrics:
    """Container for PyBullet Drones telemetry metrics."""

    position: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float] | None = None  # roll, pitch, yaw
    angular_velocity: tuple[float, float, float] | None = None
    motor_rpms: tuple[float, ...] | None = None
    num_drones: int = 1


def _ensure_pybullet_drones() -> None:
    """Ensure gym-pybullet-drones is available, raise helpful error if not."""
    if not _PYBULLET_DRONES_AVAILABLE:
        raise ImportError(
            "gym-pybullet-drones is not available. Install via:\n"
            "  pip install gym-pybullet-drones pybullet\n"
            "or:\n"
            "  pip install -e '.[pybullet-drones]'"
        )


class PyBulletDronesAdapter(EnvironmentAdapter[np.ndarray, np.ndarray]):
    """Base adapter for gym-pybullet-drones quadcopter environments.

    These environments provide kinematic or vision-based observations and
    have continuous action spaces for motor control.
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (ControlMode.AGENT_ONLY,)

    # Subclasses override with their specific environment ID
    DEFAULT_ENV_ID = "hover-aviary-v0"

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: PyBulletDronesConfig | None = None,
    ) -> None:
        super().__init__(context)
        if config is None:
            config = PyBulletDronesConfig(env_id=self.DEFAULT_ENV_ID)
        self._config = config
        self._env_id = config.env_id or self.DEFAULT_ENV_ID
        self._step_counter = 0
        self._render_warning_emitted = False

    @property
    def id(self) -> str:  # type: ignore[override]
        return self._env_id

    def gym_kwargs(self) -> dict[str, Any]:
        kwargs = super().gym_kwargs()
        kwargs.update(self._config.to_gym_kwargs())
        return kwargs

    def load(self) -> None:
        """Load the PyBullet Drones environment."""
        _ensure_pybullet_drones()

        try:
            import gymnasium as gym

            # Get kwargs from config
            env_kwargs = self._config.to_gym_kwargs()

            # Create environment using gymnasium.make()
            self._env = gym.make(self._env_id, **env_kwargs)

            _LOGGER.info(
                "PyBullet Drones environment loaded: %s (drones=%d, obs=%s, act=%s)",
                self._env_id,
                self._config.num_drones,
                self._config.obs_type,
                self._config.act_type,
            )
        except Exception as exc:
            _LOGGER.error(
                "Failed to load PyBullet Drones environment %s: %s",
                self._env_id,
                exc,
            )
            raise

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:
        """Reset the environment."""
        env = self._require_env()

        reset_kwargs: dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options is not None:
            reset_kwargs["options"] = options

        observation, info = env.reset(**reset_kwargs)
        processed_obs = self._process_observation(observation)

        self._step_counter = 0

        _LOGGER.debug(
            "PyBullet Drones reset: %s (seed=%s)",
            self._env_id,
            seed,
        )

        return self._package_step(processed_obs, 0.0, False, False, info)

    def step(self, action: np.ndarray) -> AdapterStep[np.ndarray]:
        """Execute one step in the environment."""
        env = self._require_env()

        observation, reward, terminated, truncated, info = env.step(action)
        processed_obs = self._process_observation(observation)

        self._step_counter += 1

        if self._step_counter % _PYBULLET_DRONES_STEP_LOG_FREQUENCY == 1:
            _LOGGER.debug(
                "PyBullet Drones step %d: reward=%.4f, terminated=%s, truncated=%s",
                self._step_counter,
                float(reward),
                terminated,
                truncated,
            )

        return self._package_step(
            processed_obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> dict[str, Any]:
        """Render the environment."""
        env = self._require_env()

        # PyBullet Drones uses gui=True for rendering or record=True for frames
        # When gui=False and record=False, we may not get visual output
        try:
            frame = env.render()
            if frame is not None:
                array = np.asarray(frame)
                return {
                    "mode": RenderMode.RGB_ARRAY.value,
                    "rgb": array,
                    "game_id": self._env_id,
                }
        except Exception as exc:
            if not self._render_warning_emitted:
                self._render_warning_emitted = True
                _LOGGER.warning(
                    "PyBullet Drones render failed for %s: %s. "
                    "Set gui=True or record=True in config for visual output.",
                    self._env_id,
                    exc,
                )

        # Return empty frame placeholder if rendering not available
        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "game_id": self._env_id,
        }

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as exc:
                _LOGGER.warning("Error closing PyBullet Drones environment: %s", exc)
            self._env = None
            _LOGGER.info("PyBullet Drones environment closed: %s", self._env_id)

    def build_step_state(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> StepState:
        """Build step state from observation and info."""
        metrics = _PyBulletDronesMetrics()

        # Extract drone state from kinematic observation
        # KIN observation: [x, y, z, r, p, y, vx, vy, vz, wx, wy, wz]
        if observation.ndim == 1 and len(observation) >= 12:
            metrics.position = tuple(observation[0:3].tolist())  # type: ignore[assignment]
            metrics.orientation = tuple(observation[3:6].tolist())  # type: ignore[assignment]
            metrics.velocity = tuple(observation[6:9].tolist())  # type: ignore[assignment]
            metrics.angular_velocity = tuple(observation[9:12].tolist())  # type: ignore[assignment]

        metrics_map: dict[str, Any] = {}
        if metrics.position:
            metrics_map["position"] = metrics.position
            metrics_map["altitude"] = metrics.position[2]
        if metrics.velocity:
            metrics_map["velocity"] = metrics.velocity
        if metrics.orientation:
            metrics_map["orientation"] = metrics.orientation
        if metrics.angular_velocity:
            metrics_map["angular_velocity"] = metrics.angular_velocity

        environment_meta: dict[str, Any] = {
            "env_id": self._env_id,
            "step": self._step_counter,
        }

        return StepState(
            metrics=metrics_map,
            environment=environment_meta,
            raw={"observation_shape": observation.shape},
        )

    def build_render_hint(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
        state: StepState,
    ) -> Mapping[str, Any] | None:
        """Build render hint for visualization."""
        base_hint = super().build_render_hint(observation, info, state) or {}
        hint: dict[str, Any] = dict(base_hint)
        hint["observation_shape"] = observation.shape if hasattr(observation, "shape") else None
        hint["env_type"] = "pybullet_drones"
        return hint or None

    def _process_observation(self, observation: Any) -> np.ndarray:
        """Process observation to ensure correct format."""
        return np.asarray(observation, dtype=np.float32)


class HoverAviaryAdapter(PyBulletDronesAdapter):
    """Adapter for hover-aviary-v0 single-agent hover task.

    The goal is to reach a target altitude (z=1.0m) and stabilize.
    """

    DEFAULT_ENV_ID = GameId.PYBULLET_HOVER_AVIARY.value

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: PyBulletDronesConfig | None = None,
    ) -> None:
        if config is None:
            config = PyBulletDronesConfig(env_id=self.DEFAULT_ENV_ID)
        super().__init__(context, config=config)


class MultiHoverAviaryAdapter(PyBulletDronesAdapter):
    """Adapter for multihover-aviary-v0 multi-agent hover task.

    Multiple drones must learn to hover at different target altitudes
    while accounting for aerodynamic interactions (downwash).
    """

    DEFAULT_ENV_ID = GameId.PYBULLET_MULTIHOVER_AVIARY.value
    supported_control_modes = (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: PyBulletDronesConfig | None = None,
    ) -> None:
        if config is None:
            config = PyBulletDronesConfig(
                env_id=self.DEFAULT_ENV_ID,
                num_drones=2,
            )
        super().__init__(context, config=config)


class CtrlAviaryAdapter(PyBulletDronesAdapter):
    """Adapter for ctrl-aviary-v0 low-level RPM control.

    Direct motor RPM control environment for trajectory tracking
    and low-level control research.
    """

    DEFAULT_ENV_ID = GameId.PYBULLET_CTRL_AVIARY.value

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: PyBulletDronesConfig | None = None,
    ) -> None:
        if config is None:
            config = PyBulletDronesConfig(env_id=self.DEFAULT_ENV_ID)
        super().__init__(context, config=config)


class VelocityAviaryAdapter(PyBulletDronesAdapter):
    """Adapter for velocity-aviary-v0 high-level velocity control.

    Velocity-based control environment with internal PID controller
    for high-level planning and navigation research.
    """

    DEFAULT_ENV_ID = GameId.PYBULLET_VELOCITY_AVIARY.value

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: PyBulletDronesConfig | None = None,
    ) -> None:
        if config is None:
            config = PyBulletDronesConfig(env_id=self.DEFAULT_ENV_ID)
        super().__init__(context, config=config)


# Adapter registry for this family
PYBULLET_DRONES_ADAPTERS: dict[GameId, type[PyBulletDronesAdapter]] = {
    GameId.PYBULLET_HOVER_AVIARY: HoverAviaryAdapter,
    GameId.PYBULLET_MULTIHOVER_AVIARY: MultiHoverAviaryAdapter,
    GameId.PYBULLET_CTRL_AVIARY: CtrlAviaryAdapter,
    GameId.PYBULLET_VELOCITY_AVIARY: VelocityAviaryAdapter,
}


__all__ = [
    "PyBulletDronesAdapter",
    "PyBulletDronesConfig",
    "HoverAviaryAdapter",
    "MultiHoverAviaryAdapter",
    "CtrlAviaryAdapter",
    "VelocityAviaryAdapter",
    "PYBULLET_DRONES_ADAPTERS",
]
