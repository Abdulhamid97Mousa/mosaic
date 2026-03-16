"""MalmoEnv adapters — Microsoft Malmo (Java-based Minecraft) via the MalmoEnv Python package.

MalmoEnv connects directly to a running Minecraft instance (launched via launchClient.sh)
over a TCP socket on port 9000.  Each adapter wraps a specific mission XML file bundled
with the MalmoEnv package.

Before using any of these adapters, start Minecraft:
    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
    cd 3rd_party/environments/malmo/Minecraft
    bash launchClient.sh -port 9000 -env

Repository: 3rd_party/environments/malmo/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterStep,
    EnvironmentAdapter,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CLOSED,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
)

try:
    import malmoenv
    _MALMOENV_AVAILABLE = True
except ImportError:
    malmoenv = None  # type: ignore[assignment]
    _MALMOENV_AVAILABLE = False

# Path to bundled MalmoEnv mission XMLs
_MISSIONS_DIR = (
    Path(__file__).parents[3]
    / "3rd_party"
    / "environments"
    / "malmo"
    / "MalmoEnv"
    / "missions"
)

# Malmo action names matching default mission XML command handlers
MALMOENV_ACTIONS = [
    "move 1",       # 0: move forward
    "move -1",      # 1: move backward
    "strafe -1",    # 2: strafe left
    "strafe 1",     # 3: strafe right
    "turn -1",      # 4: turn left
    "turn 1",       # 5: turn right
    "jump 1",       # 6: jump
    "attack 1",     # 7: attack / break block
]

_DEFAULT_PORT = 9000
_DEFAULT_SERVER = "localhost"


class MalmoEnvAdapter(EnvironmentAdapter[np.ndarray, int]):
    """Base adapter for MalmoEnv (Microsoft Malmo) missions.

    Overrides the full lifecycle because MalmoEnv does not use gym.make() —
    it requires ``env.init(xml, port)`` and connects to a running Minecraft
    server.  The observation is an RGB numpy array; ``step()`` returns the
    old 4-tuple gym format which we translate to a standard AdapterStep.
    """

    mission_xml: str = "mobchase_single_agent.xml"  # override in subclasses
    malmo_port: int = _DEFAULT_PORT
    malmo_server: str = _DEFAULT_SERVER

    default_render_mode = RenderMode.MOSAIC_MALMO
    supported_render_modes = (RenderMode.MOSAIC_MALMO, RenderMode.RGB_ARRAY)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    _malmo_env: Any = None
    _last_obs: np.ndarray | None = None
    _last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if not _MALMOENV_AVAILABLE:
            raise RuntimeError(
                "malmoenv package is not installed. "
                "Run: pip install -e 3rd_party/environments/malmo/MalmoEnv/"
            )
        xml_path = _MISSIONS_DIR / self.mission_xml
        if not xml_path.exists():
            raise FileNotFoundError(f"Mission XML not found: {xml_path}")
        xml_content = xml_path.read_text()
        env = malmoenv.Env()
        env.init(xml_content, self.malmo_port, server=self.malmo_server, reshape=True)
        self._malmo_env = env
        self.log_constant(
            LOG_ADAPTER_ENV_CREATED,
            extra={
                "env_id": self.id,
                "render_mode": RenderMode.MOSAIC_MALMO.value,
                "gym_kwargs": f"port={self.malmo_port},server={self.malmo_server}",
                "wrapped_class": "malmoenv.Env",
            },
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> AdapterStep[np.ndarray]:
        if self._malmo_env is None:
            raise RuntimeError(f"Adapter '{self.id}' has not been loaded.")
        self._episode_step = 0
        self._episode_return = 0.0
        obs = self._malmo_env.reset()
        if obs is None or len(obs) == 0:
            h = getattr(self._malmo_env, "height", 84)
            w = getattr(self._malmo_env, "width", 84)
            obs = np.zeros((h, w, 3), dtype=np.uint8)
        obs = np.flipud(np.asarray(obs, dtype=np.uint8))
        self._last_obs = obs
        self._last_info = {"episode_step": 0}
        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self.id,
                "seed": seed if seed is not None else "None",
                "has_options": bool(options),
            },
        )
        return self._package_step(obs, 0.0, False, False, self._last_info)

    def step(self, action: int) -> AdapterStep[np.ndarray]:
        if self._malmo_env is None:
            raise RuntimeError(f"Adapter '{self.id}' has not been loaded.")
        # MalmoEnv returns old gym 4-tuple: (obs, reward, done, info)
        obs, reward, done, info = self._malmo_env.step(action)
        if obs is None or len(obs) == 0:
            h = getattr(self._malmo_env, "height", 84)
            w = getattr(self._malmo_env, "width", 84)
            obs = (
                self._last_obs
                if self._last_obs is not None
                else np.zeros((h, w, 3), dtype=np.uint8)
            )
        obs = np.flipud(np.asarray(obs, dtype=np.uint8))
        r = float(reward) if reward is not None else 0.0
        self._episode_step += 1
        self._episode_return += r
        info_dict: dict[str, Any] = dict(info) if isinstance(info, Mapping) else {}
        info_dict["episode_step"] = self._episode_step
        info_dict["episode_score"] = self._episode_return
        self._last_obs = obs
        self._last_info = info_dict
        self.log_constant(
            LOG_ADAPTER_STEP_SUMMARY,
            extra={
                "env_id": self.id,
                "action": repr(action),
                "reward": r,
                "terminated": done,
                "truncated": False,
            },
        )
        return self._package_step(obs, r, bool(done), False, info_dict)

    def close(self) -> None:
        if self._malmo_env is not None:
            self.log_constant(LOG_ADAPTER_ENV_CLOSED, extra={"env_id": self.id})
            try:
                self._malmo_env.close()
            except Exception:
                pass
            self._malmo_env = None

    @property
    def action_space(self):  # type: ignore[override]
        if self._malmo_env is not None:
            return self._malmo_env.action_space
        return None

    @property
    def observation_space(self):  # type: ignore[override]
        if self._malmo_env is not None:
            return self._malmo_env.observation_space
        return None

    def render(self) -> dict[str, Any]:
        """Return an ``{"rgb": ndarray}`` payload for MosaicMalmoRendererStrategy."""
        frame = self._last_obs
        if frame is None:
            h = getattr(self._malmo_env, "height", 84) if self._malmo_env else 84
            w = getattr(self._malmo_env, "width", 84) if self._malmo_env else 84
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        payload: dict[str, Any] = {
            "mode": RenderMode.MOSAIC_MALMO.value,
            "rgb": np.asarray(frame),
            "mission": self.mission_xml.replace(".xml", ""),
        }
        for key in ("ms_per_tick", "agent_pos", "agent_yaw", "agent_pitch", "tick", "reward"):
            if key in self._last_info:
                payload[key] = self._last_info[key]
        return payload


# ---------------------------------------------------------------------------
# Mission-specific subclasses
# ---------------------------------------------------------------------------

class MalmoEnvMobChaseAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_MOBCHASE.value
    mission_xml = "mobchase_single_agent.xml"


class MalmoEnvMazeRunnerAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_MAZERUNNER.value
    mission_xml = "mazerunner.xml"


class MalmoEnvVerticalAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_VERTICAL.value
    mission_xml = "vertical.xml"


class MalmoEnvCliffWalkingAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_CLIFFWALKING.value
    mission_xml = "cliffwalking.xml"


class MalmoEnvCatchTheMobAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_CATCHTHEMOB.value
    mission_xml = "catchthemob.xml"


class MalmoEnvFindTheGoalAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_FINDTHEGOAL.value
    mission_xml = "findthegoal.xml"


class MalmoEnvAtticAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_ATTIC.value
    mission_xml = "attic.xml"


class MalmoEnvDefaultFlatWorldAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_DEFAULTFLATWORLD.value
    mission_xml = "defaultflatworld.xml"


class MalmoEnvDefaultWorldAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_DEFAULTWORLD.value
    mission_xml = "defaultworld.xml"


class MalmoEnvEatingAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_EATING.value
    mission_xml = "eating.xml"


class MalmoEnvObstaclesAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_OBSTACLES.value
    mission_xml = "obstacles.xml"


class MalmoEnvTrickyArenaAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_TRICKYARENA.value
    mission_xml = "trickyarena.xml"


class MalmoEnvTreasureHuntAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_TREASUREHUNT.value
    mission_xml = "treasurehunt.xml"


# Adapter registry — imported by adapters/__init__.py and factories/adapters.py
MALMOENV_ADAPTERS = {
    GameId.MALMOENV_MOBCHASE: MalmoEnvMobChaseAdapter,
    GameId.MALMOENV_MAZERUNNER: MalmoEnvMazeRunnerAdapter,
    GameId.MALMOENV_VERTICAL: MalmoEnvVerticalAdapter,
    GameId.MALMOENV_CLIFFWALKING: MalmoEnvCliffWalkingAdapter,
    GameId.MALMOENV_CATCHTHEMOB: MalmoEnvCatchTheMobAdapter,
    GameId.MALMOENV_FINDTHEGOAL: MalmoEnvFindTheGoalAdapter,
    GameId.MALMOENV_ATTIC: MalmoEnvAtticAdapter,
    GameId.MALMOENV_DEFAULTFLATWORLD: MalmoEnvDefaultFlatWorldAdapter,
    GameId.MALMOENV_DEFAULTWORLD: MalmoEnvDefaultWorldAdapter,
    GameId.MALMOENV_EATING: MalmoEnvEatingAdapter,
    GameId.MALMOENV_OBSTACLES: MalmoEnvObstaclesAdapter,
    GameId.MALMOENV_TRICKYARENA: MalmoEnvTrickyArenaAdapter,
    GameId.MALMOENV_TREASUREHUNT: MalmoEnvTreasureHuntAdapter,
}

# Legacy aliases so any existing imports of the old names still resolve
MOSAIC_MALMO_ADAPTERS = MALMOENV_ADAPTERS
MOSAIC_MALMO_ACTIONS = MALMOENV_ACTIONS
