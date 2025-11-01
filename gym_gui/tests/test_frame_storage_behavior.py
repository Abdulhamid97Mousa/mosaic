import numpy as np
from PIL import Image

from gym_gui.services.frame_storage import FrameStorageService
from gym_gui.services.storage import StorageRecorderService
from gym_gui.services.telemetry import TelemetryService
from gym_gui.core.data_model import StepRecord


class _DummyStore:
    def __init__(self) -> None:
        self.records: list[StepRecord] = []

    def record_step(self, record: StepRecord) -> None:  # pragma: no cover - simple container
        self.records.append(record)

    # TelemetryService may call these in other flows; no-ops keep interface compatible
    def record_episode(self, *_, **__):
        pass

    def delete_episode(self, *_, **__):
        pass

    def delete_all_episodes(self, *_, **__):
        pass

    def delete_run(self, *_, **__):
        pass

    def archive_run(self, *_, **__):
        pass

    def flush(self):
        pass


def test_box2d_storage_profile_disables_frame_capture() -> None:
    """Box2D profile now disables frame capture to avoid per-step PNG churn."""

    service = StorageRecorderService()
    service.ensure_profiles_loaded()
    service.set_active_profile("box2d")

    profile = service.get_active_profile()

    assert profile.capture_frames is False
    assert profile.compress_frames is False
    assert profile.drop_render_payload is True
    assert profile.drop_observation is True
    assert profile.strategy == "ndjson_png"


def test_frame_storage_writes_png_files(tmp_path) -> None:
    """Saving a single frame produces a PNG on disk."""

    storage = FrameStorageService(base_dir=tmp_path)
    frame = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)

    saved = storage.save_frame(frame, "frames/test.png", run_id="run123")

    assert saved is not None
    assert saved.exists()

    loaded = np.array(Image.open(saved))
    assert loaded.shape == frame.shape


def test_frame_storage_per_step_creates_many_pngs(tmp_path) -> None:
    """Multiple save calls create many PNG files, mirroring per-step overhead."""

    storage = FrameStorageService(base_dir=tmp_path)
    base_shape = (16, 16, 3)

    for index in range(5):
        frame = np.full(base_shape, index, dtype=np.uint8)
        ref = f"frames/{index:06d}.png"
        storage.save_frame(frame, ref, run_id="runABC")

    frame_dir = tmp_path / "runABC" / "frames"
    assert frame_dir.exists()
    saved_files = sorted(frame_dir.glob("*.png"))
    assert len(saved_files) == 5


def test_telemetry_service_drops_heavy_fields_when_configured() -> None:
    storage = StorageRecorderService()
    storage.ensure_profiles_loaded()
    storage.set_active_profile("box2d")

    dummy_store = _DummyStore()
    telemetry = TelemetryService()
    telemetry.attach_storage(storage)
    telemetry.attach_store(dummy_store)

    record = StepRecord(
        episode_id="ep-1",
        step_index=0,
        action=None,
        observation=np.zeros((4, 4, 3), dtype=np.uint8),
        reward=0.0,
        terminated=False,
        truncated=False,
        render_payload={"mode": "rgb_array", "rgb": np.zeros((4, 4, 3), dtype=np.uint8)},
        frame_ref="frames/test.png",
    )

    telemetry.record_step(record)

    assert dummy_store.records, "Telemetry store should receive at least one record"
    stored = dummy_store.records[0]
    assert stored.observation is None
    assert stored.render_payload is None
    assert stored.frame_ref is None
