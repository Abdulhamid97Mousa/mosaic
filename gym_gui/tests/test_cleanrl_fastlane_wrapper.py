import importlib
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any
from multiprocessing import shared_memory

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces

from gym_gui.fastlane import FastLaneReader
from gym_gui.fastlane import buffer as fastlane_buffer
from gym_gui.telemetry.semconv import TelemetryEnv, VideoModes


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
    monkeypatch.setenv(TelemetryEnv.FASTLANE_ONLY, "1")
    monkeypatch.setenv(TelemetryEnv.FASTLANE_SLOT, str(fastlane_slot))
    monkeypatch.setenv("CLEANRL_NUM_ENVS", "1")
    monkeypatch.setenv("CLEANRL_RUN_ID", run_id)
    monkeypatch.setenv("CLEANRL_AGENT_ID", "agent-99")

    module = importlib.import_module("cleanrl_worker.fastlane")
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


def test_fastlane_grid_mode_tiles_frames(monkeypatch):
    run_id = "grid-mode-run"
    monkeypatch.setenv(TelemetryEnv.FASTLANE_ONLY, "1")
    monkeypatch.setenv(TelemetryEnv.FASTLANE_SLOT, "0")
    monkeypatch.setenv(TelemetryEnv.FASTLANE_VIDEO_MODE, VideoModes.GRID)
    monkeypatch.setenv(TelemetryEnv.FASTLANE_GRID_LIMIT, "4")
    monkeypatch.setenv("CLEANRL_NUM_ENVS", "4")
    monkeypatch.setenv("CLEANRL_RUN_ID", run_id)
    monkeypatch.setenv("CLEANRL_AGENT_ID", "agent-grid")

    module = importlib.import_module("cleanrl_worker.fastlane")
    module = importlib.reload(module)

    module._GRID_COORDINATORS.clear()

    published: list[tuple[Any, bytes]] = []

    class _StubWriter:
        def __init__(self, config):
            self.config = config

        @classmethod
        def create(cls, run_id, config):
            return cls(config)

        def publish(self, payload, metrics=None):
            published.append((self.config, payload))

        def close(self):
            pass

    original_writer = getattr(module, "FastLaneWriter", None)
    module.FastLaneWriter = _StubWriter  # type: ignore

    try:
        envs = [module.maybe_wrap_env(_DummyEnv()) for _ in range(4)]
        for env in envs:
            env.reset()

        for _ in range(2):
            for env in envs:
                env.step(0)

        assert published, "Grid mode did not publish any frames"
        config, payload = published[-1]
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(config.height, config.width, config.channels)
        assert frame.shape[0] == 16 and frame.shape[1] == 16
    finally:
        module._GRID_COORDINATORS.clear()
        if original_writer is not None:
            module.FastLaneWriter = original_writer  # type: ignore


def test_fastlane_reader_handles_zero_capacity_header():
    size = fastlane_buffer._HEADER_STRUCT.size  # type: ignore[attr-defined]
    try:
        shm = shared_memory.SharedMemory(create=True, size=size)
    except PermissionError as exc:
        pytest.skip(f"Shared memory creation not permitted in this environment: {exc}")
    try:
        reader = fastlane_buffer.FastLaneReader(shm)
        assert reader.latest_frame() is None
        reader.close()
    finally:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
