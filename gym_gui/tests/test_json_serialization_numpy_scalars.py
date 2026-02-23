import json
import os
from pathlib import Path
import tempfile

import numpy as np

from gym_gui.utils import json_serialization as js
from gym_gui.storage.session import EpisodeRecord, SessionRecorder


def test_numpy_scalar_tuple_roundtrip() -> None:
    payload = {"seeds": (np.uint32(1), np.uint32(2))}
    dumped = js.dumps(payload)
    assert isinstance(dumped, (bytes, bytearray))
    loaded = js.loads(dumped)
    assert isinstance(loaded, dict)
    assert loaded["seeds"] == (1, 2)
    assert all(isinstance(x, int) for x in loaded["seeds"])  # coerced to Python ints


def test_session_recorder_summarizes_ndarray() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        recorder = SessionRecorder(
            base_dir=Path(tmp),
            ring_buffer_limit=16,
            retain_episodes=2,
            compress_frames=False,
            telemetry_only=False,
        )
        # Large RGB observation should be summarized, not expanded into lists
        obs = np.zeros((96, 96, 3), dtype=np.uint8)
        rec = EpisodeRecord(
            episode_id="ep-test",
            step_index=0,
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"note": "test"},
        )
        recorder.write_step(rec)
        recorder.finalize_episode()

        ep_dir = Path(tmp) / "ep-test"
        assert ep_dir.exists()
        jsonl = ep_dir / "episode.jsonl"
        assert jsonl.exists()
        with jsonl.open("r", encoding="utf-8") as f:
            line = f.readline()
        data = json.loads(line)
        assert "observation" in data
        obs_payload = data["observation"]
        assert isinstance(obs_payload, dict)
        assert obs_payload.get("__type__") == "ndarray_summary"
        assert obs_payload.get("shape") == [96, 96, 3]
        assert obs_payload.get("dtype") == "uint8"


def test_session_recorder_summarizes_lazyframes_like() -> None:
    class LazyFrames:
        def __init__(self):
            self.frame_shape = (96, 96, 3)
            self.shape = (4, 96, 96, 3)
            self.dtype = np.uint8
            self._frames = [np.zeros(self.frame_shape, dtype=self.dtype) for _ in range(4)]

    with tempfile.TemporaryDirectory() as tmp:
        recorder = SessionRecorder(
            base_dir=Path(tmp),
            ring_buffer_limit=16,
            retain_episodes=2,
            compress_frames=False,
            telemetry_only=False,
        )
        obs = LazyFrames()
        rec = EpisodeRecord(
            episode_id="ep-test2",
            step_index=0,
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
        )
        recorder.write_step(rec)
        recorder.finalize_episode()

        ep_dir = Path(tmp) / "ep-test2"
        with (ep_dir / "episode.jsonl").open("r", encoding="utf-8") as f:
            line = f.readline()
        data = json.loads(line)
        payload = data.get("observation")
        assert isinstance(payload, dict)
        assert payload.get("__type__") == "lazyframes_summary"
        # Either shape or frame_shape should be present
        assert (
            payload.get("shape") == [4, 96, 96, 3]
            or payload.get("frame_shape") == [96, 96, 3]
        )
