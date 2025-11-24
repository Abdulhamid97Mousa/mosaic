"""Shared-memory ring buffer for fast lane frame delivery.

The fast lane mirrors the architecture described in
``docs/1.0_DAY_24/TASK_3/proposing_improvement.md`` and
``docs/1.0_DAY_25/TASK_1/Rendering_Telemetry_pipeline.md``:
trainers/workers publish the freshest RGB frame + HUD scalars into shared memory
so the GUI can render immediately, while the existing telemetry pipeline
continues to provide durable storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import struct
from typing import Iterable, Optional

try:  # NumPy is optional here; we tolerate absence for headless tests
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore

MAGIC = b"FLAN"
VERSION = 1

# Header layout (little endian):
# magic(4s) | version(u32) | flags(u32) | capacity(u32) |
# slot_size(u32) | payload_size(u32) | width(u32) | height(u32) |
# channels(u32) | fmt_hash(u32) | head(u64) | tail(u64) |
# last_reward(f64) | rolling_return(f64) | step_rate(f64)
_HEADER_STRUCT = struct.Struct("<4sIIIIIIIIIQQddd")
_HEADER_SIZE = _HEADER_STRUCT.size

_IDX_HEAD = 10
_IDX_TAIL = 11
_IDX_LAST_REWARD = 12
_IDX_ROLLING = 13
_IDX_STEP_RATE = 14

# Slot meta layout: seq(u64) | payload_len(u32) | reserved(u32)
_SLOT_META_STRUCT = struct.Struct("<QII")
_SLOT_META_SIZE = _SLOT_META_STRUCT.size

_DEFAULT_CAPACITY = 128


@dataclass(frozen=True)
class FastLaneConfig:
    width: int
    height: int
    channels: int = 3
    pixel_format: str = "RGB"
    capacity: int = _DEFAULT_CAPACITY

    @property
    def payload_bytes(self) -> int:
        return max(1, self.width * self.height * self.channels)


@dataclass(frozen=True)
class FastLaneMetrics:
    last_reward: float = 0.0
    rolling_return: float = 0.0
    step_rate_hz: float = 0.0


@dataclass(frozen=True)
class FastLaneFrame:
    data: bytes
    width: int
    height: int
    channels: int
    metrics: FastLaneMetrics


def create_fastlane_name(run_id: str) -> str:
    return f"mosaic.fastlane.{run_id}".replace("/", "_")


class FastLaneBase:
    def __init__(self, shm: shared_memory.SharedMemory) -> None:
        self._shm = shm
        self._buffer = shm.buf
        self._mv = memoryview(self._buffer)

    # ------------------------------------------------------------------
    # Header helpers
    # ------------------------------------------------------------------
    def _read_header(self) -> tuple:
        return _HEADER_STRUCT.unpack_from(self._mv, 0)

    def _write_header(self, values: Iterable[int | float | bytes]) -> None:
        _HEADER_STRUCT.pack_into(self._mv, 0, *values)

    @property
    def capacity(self) -> int:
        return _HEADER_STRUCT.unpack_from(self._mv, 0)[3]

    @property
    def payload_size(self) -> int:
        return _HEADER_STRUCT.unpack_from(self._mv, 0)[5]

    @property
    def slot_size(self) -> int:
        return _HEADER_STRUCT.unpack_from(self._mv, 0)[4]

    @property
    def width(self) -> int:
        return _HEADER_STRUCT.unpack_from(self._mv, 0)[6]

    @property
    def height(self) -> int:
        return _HEADER_STRUCT.unpack_from(self._mv, 0)[7]

    @property
    def channels(self) -> int:
        return _HEADER_STRUCT.unpack_from(self._mv, 0)[8]

    @property
    def head(self) -> int:
        return _HEADER_STRUCT.unpack_from(self._mv, 0)[10]

    def _set_head(self, value: int) -> None:
        header = list(_HEADER_STRUCT.unpack_from(self._mv, 0))
        header[_IDX_HEAD] = value
        _HEADER_STRUCT.pack_into(self._mv, 0, *header)

    def _set_metrics(self, metrics: FastLaneMetrics) -> None:
        header = list(_HEADER_STRUCT.unpack_from(self._mv, 0))
        header[_IDX_LAST_REWARD] = metrics.last_reward
        header[_IDX_ROLLING] = metrics.rolling_return
        header[_IDX_STEP_RATE] = metrics.step_rate_hz
        _HEADER_STRUCT.pack_into(self._mv, 0, *header)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._mv.release()
        self._buffer.release()
        self._shm.close()

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except FileNotFoundError:  # already gone
            pass


class FastLaneWriter(FastLaneBase):
    def __init__(self, shm: shared_memory.SharedMemory, config: FastLaneConfig) -> None:
        super().__init__(shm)
        self._config = config
        self._slot_offset = _HEADER_SIZE
        self._slot_payload_bytes = config.payload_bytes

    @classmethod
    def create(
        cls,
        run_id: str,
        config: FastLaneConfig,
        *,
        capacity: int | None = None,
    ) -> "FastLaneWriter":
        cap = max(1, capacity or config.capacity)
        payload_size = config.payload_bytes
        slot_size = _SLOT_META_SIZE + payload_size
        total_size = _HEADER_SIZE + cap * slot_size
        name = create_fastlane_name(run_id)
        shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)
        writer = cls(shm, FastLaneConfig(
            width=config.width,
            height=config.height,
            channels=config.channels,
            pixel_format=config.pixel_format,
            capacity=cap,
        ))
        fmt_hash = _hash_format(config.pixel_format)
        writer._write_header(
            (
                MAGIC,
                VERSION,
                0,
                cap,
                slot_size,
                payload_size,
                config.width,
                config.height,
                config.channels,
                fmt_hash,
                0,
                0,
                0.0,
                0.0,
                0.0,
            )
        )
        return writer

    def publish(
        self,
        frame: bytes | memoryview | "np.ndarray",
        *,
        metrics: FastLaneMetrics | None = None,
    ) -> int:
        payload = _coerce_bytes(frame)
        if len(payload) > self._slot_payload_bytes:
            raise ValueError("Frame payload exceeds slot capacity")

        if metrics is not None:
            self._set_metrics(metrics)

        head = self.head + 1
        slot_index = (head - 1) % self.capacity
        slot_offset = self._slot_offset + slot_index * self.slot_size
        seq_addr = slot_offset
        payload_addr = slot_offset + _SLOT_META_SIZE

        seq = head * 2
        _SLOT_META_STRUCT.pack_into(self._mv, seq_addr, seq + 1, len(payload), 0)
        payload_view = self._mv[payload_addr:payload_addr + self._slot_payload_bytes]
        payload_view[:len(payload)] = payload
        if len(payload) < self._slot_payload_bytes:
            payload_view[len(payload):] = b"\x00" * (self._slot_payload_bytes - len(payload))
        _SLOT_META_STRUCT.pack_into(self._mv, seq_addr, seq + 2, len(payload), 0)
        self._set_head(head)
        return head


class FastLaneReader(FastLaneBase):
    def __init__(self, shm: shared_memory.SharedMemory) -> None:
        super().__init__(shm)
        self._slot_offset = _HEADER_SIZE
        self._slot_payload_bytes = self.payload_size
        self._last_seq = 0

    @classmethod
    def attach(cls, run_id: str) -> "FastLaneReader":
        name = create_fastlane_name(run_id)
        shm = shared_memory.SharedMemory(name=name, create=False)
        return cls(shm)

    def latest_frame(self) -> Optional[FastLaneFrame]:
        capacity = self.capacity
        if capacity <= 0:
            # Writer has not finished initializing the header yet.
            return None
        slot_size = self.slot_size
        if slot_size <= 0:
            return None
        head = self.head
        if head == 0:
            return None
        attempts = min(capacity, head)
        for idx in range(attempts):
            seq_head = head - idx
            slot_index = (seq_head - 1) % capacity
            slot_offset = self._slot_offset + slot_index * slot_size
            seq_addr = slot_offset
            payload_addr = slot_offset + _SLOT_META_SIZE
            seq1, payload_len, _reserved = _SLOT_META_STRUCT.unpack_from(self._mv, seq_addr)
            if seq1 % 2 == 1:
                continue  # writer mid-flight
            payload_bytes = bytes(self._mv[payload_addr:payload_addr + payload_len])
            seq2, _, _ = _SLOT_META_STRUCT.unpack_from(self._mv, seq_addr)
            if seq1 != seq2:
                continue
            self._last_seq = seq2
            metrics = self.metrics()
            return FastLaneFrame(
                data=payload_bytes,
                width=self.width,
                height=self.height,
                channels=self.channels,
                metrics=metrics,
            )
        return None

    def metrics(self) -> FastLaneMetrics:
        header = _HEADER_STRUCT.unpack_from(self._mv, 0)
        return FastLaneMetrics(
            last_reward=header[_IDX_LAST_REWARD],
            rolling_return=header[_IDX_ROLLING],
            step_rate_hz=header[_IDX_STEP_RATE],
        )


def _coerce_bytes(frame: bytes | memoryview | "np.ndarray") -> bytes:
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    if isinstance(frame, memoryview):
        return frame.tobytes()
    if np is not None and hasattr(frame, "tobytes"):
        return frame.tobytes()
    raise TypeError("Frame must be bytes-like or numpy array")


def _hash_format(fmt: str) -> int:
    import zlib
    return zlib.crc32(fmt.encode("utf-8")) & 0xFFFFFFFF
