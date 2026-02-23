"""Shared-memory ring buffer for fast lane frame delivery.

Trainers/workers publish the freshest RGB frame + HUD scalars into shared memory
so the GUI can render immediately, while the existing telemetry pipeline
continues to provide durable storage.

See Also:
    - :doc:`/documents/architecture/fastlane` for the Fast Lane architecture overview
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

_IDX_FLAGS = 2
_IDX_HEAD = 10
_IDX_TAIL = 11
_IDX_LAST_REWARD = 12
_IDX_ROLLING = 13
_IDX_STEP_RATE = 14

# Flag bits stored in the header flags field.
FLAG_INVALIDATED = 0x1  # Buffer is stale; readers must detach and reattach.

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
    metadata_size: int = 0  # Max size for JSON metadata (0 = disabled)

    @property
    def payload_bytes(self) -> int:
        return max(1, self.width * self.height * self.channels)

    @property
    def slot_payload_bytes(self) -> int:
        """Total slot payload: frame + metadata."""
        return self.payload_bytes + self.metadata_size


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
    metadata: Optional[bytes] = None  # JSON metadata bytes (if available)


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
    def flags(self) -> int:
        return _HEADER_STRUCT.unpack_from(self._mv, 0)[_IDX_FLAGS]

    @property
    def is_invalidated(self) -> bool:
        """True when the writer has marked this buffer as stale."""
        return bool(self.flags & FLAG_INVALIDATED)

    def invalidate(self) -> None:
        """Mark this buffer as invalidated so readers detach and reattach."""
        header = list(_HEADER_STRUCT.unpack_from(self._mv, 0))
        header[_IDX_FLAGS] = header[_IDX_FLAGS] | FLAG_INVALIDATED
        _HEADER_STRUCT.pack_into(self._mv, 0, *header)

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
        self._frame_payload_bytes = config.payload_bytes
        self._metadata_size = config.metadata_size
        self._slot_payload_bytes = config.slot_payload_bytes

    @classmethod
    def create(
        cls,
        run_id: str,
        config: FastLaneConfig,
        *,
        capacity: int | None = None,
    ) -> "FastLaneWriter":
        cap = max(1, capacity or config.capacity)
        # Total slot payload includes frame + metadata
        slot_payload_size = config.slot_payload_bytes
        slot_size = _SLOT_META_SIZE + slot_payload_size
        total_size = _HEADER_SIZE + cap * slot_size
        name = create_fastlane_name(run_id)
        shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)
        writer = cls(shm, FastLaneConfig(
            width=config.width,
            height=config.height,
            channels=config.channels,
            pixel_format=config.pixel_format,
            capacity=cap,
            metadata_size=config.metadata_size,
        ))
        fmt_hash = _hash_format(config.pixel_format)
        writer._write_header(
            (
                MAGIC,
                VERSION,
                0,
                cap,
                slot_size,
                slot_payload_size,  # payload_size now includes metadata space
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
        metadata: bytes | str | None = None,
    ) -> int:
        """Publish a frame with optional metrics and metadata.

        Args:
            frame: RGB/RGBA frame bytes or numpy array.
            metrics: Optional metrics to update in header.
            metadata: Optional JSON metadata (str or bytes). Will be truncated
                     if larger than metadata_size configured at creation.

        Returns:
            New head sequence number.
        """
        frame_bytes = _coerce_bytes(frame)
        if len(frame_bytes) > self._frame_payload_bytes:
            raise ValueError("Frame payload exceeds slot capacity")

        # Handle metadata
        metadata_bytes = b""
        if metadata is not None:
            if isinstance(metadata, str):
                metadata_bytes = metadata.encode("utf-8")
            else:
                metadata_bytes = bytes(metadata)
            # Truncate if too large
            if len(metadata_bytes) > self._metadata_size:
                metadata_bytes = metadata_bytes[:self._metadata_size]

        if metrics is not None:
            self._set_metrics(metrics)

        head = self.head + 1
        slot_index = (head - 1) % self.capacity
        slot_offset = self._slot_offset + slot_index * self.slot_size
        seq_addr = slot_offset
        payload_addr = slot_offset + _SLOT_META_SIZE

        seq = head * 2
        # Use reserved field (index 2) for metadata_len
        _SLOT_META_STRUCT.pack_into(self._mv, seq_addr, seq + 1, len(frame_bytes), len(metadata_bytes))

        # Write frame data
        payload_view = self._mv[payload_addr:payload_addr + self._slot_payload_bytes]
        payload_view[:len(frame_bytes)] = frame_bytes

        # Write metadata after frame
        if metadata_bytes:
            metadata_start = len(frame_bytes)
            metadata_end = metadata_start + len(metadata_bytes)
            payload_view[metadata_start:metadata_end] = metadata_bytes

        # Zero-fill remaining space
        used = len(frame_bytes) + len(metadata_bytes)
        if used < self._slot_payload_bytes:
            payload_view[used:] = b"\x00" * (self._slot_payload_bytes - used)

        _SLOT_META_STRUCT.pack_into(self._mv, seq_addr, seq + 2, len(frame_bytes), len(metadata_bytes))
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
            seq1, frame_len, metadata_len = _SLOT_META_STRUCT.unpack_from(self._mv, seq_addr)
            if seq1 % 2 == 1:
                continue  # writer mid-flight
            # Read frame data
            frame_bytes = bytes(self._mv[payload_addr:payload_addr + frame_len])
            # Read metadata if present (metadata_len > 0)
            metadata_bytes: Optional[bytes] = None
            if metadata_len > 0:
                metadata_start = payload_addr + frame_len
                metadata_bytes = bytes(self._mv[metadata_start:metadata_start + metadata_len])
            seq2, _, _ = _SLOT_META_STRUCT.unpack_from(self._mv, seq_addr)
            if seq1 != seq2:
                continue
            self._last_seq = seq2
            metrics = self.metrics()
            return FastLaneFrame(
                data=frame_bytes,
                width=self.width,
                height=self.height,
                channels=self.channels,
                metrics=metrics,
                metadata=metadata_bytes,
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
