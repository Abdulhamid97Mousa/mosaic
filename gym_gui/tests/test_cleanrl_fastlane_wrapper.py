import importlib
import os
import sys
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces

from gym_gui.fastlane import FastLaneReader


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "3rd_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))


class _DummyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step = 0
        obs = np.zeros(4, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = np.full(4, self._step, dtype=np.float32)
        reward = float(self._step)
        terminated = self._step >= 2
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        value = np.uint8(min(self._step * 10, 255))
        return np.full((8, 8, 3), value, dtype=np.uint8)


@pytest.mark.parametrize("fastlane_slot", [0])
def test_cleanrl_fastlane_wrapper_publishes_frames(monkeypatch, fastlane_slot):
    run_id = "test-fastlane-run"
    monkeypatch.setenv("GYM_GUI_FASTLANE_ONLY", "1")
    monkeypatch.setenv("GYM_GUI_FASTLANE_SLOT", str(fastlane_slot))
    monkeypatch.setenv("CLEANRL_NUM_ENVS", "1")
    monkeypatch.setenv("CLEANRL_RUN_ID", run_id)
    monkeypatch.setenv("CLEANRL_AGENT_ID", "agent-99")

    module = importlib.import_module("cleanrl_worker.MOSAIC_CLEANRL_WORKER.fastlane")
    module = importlib.reload(module)

    env = _DummyEnv()
    wrapper = module.FastLaneTelemetryWrapper(env, module._CONFIG, fastlane_slot)
    wrapper.reset()
    try:
        for _ in range(3):
            wrapper.step(0)
    except PermissionError as exc:
        pytest.skip(f"Shared memory creation not permitted in this environment: {exc}")

    reader = FastLaneReader.attach(run_id)
    frame = reader.latest_frame()
    assert frame is not None, "FastLane reader should return a frame"
    assert frame.width == 8 and frame.height == 8
    assert frame.channels == 3
    assert frame.metrics.last_reward > 0

    wrapper.close()
    reader.close()
    with suppress(FileNotFoundError):
        reader.unlink()
