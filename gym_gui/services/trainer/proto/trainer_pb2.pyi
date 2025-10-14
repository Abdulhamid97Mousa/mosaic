from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RunStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUN_STATUS_UNSPECIFIED: _ClassVar[RunStatus]
    RUN_STATUS_PENDING: _ClassVar[RunStatus]
    RUN_STATUS_DISPATCHING: _ClassVar[RunStatus]
    RUN_STATUS_RUNNING: _ClassVar[RunStatus]
    RUN_STATUS_COMPLETED: _ClassVar[RunStatus]
    RUN_STATUS_FAILED: _ClassVar[RunStatus]
    RUN_STATUS_CANCELLED: _ClassVar[RunStatus]
RUN_STATUS_UNSPECIFIED: RunStatus
RUN_STATUS_PENDING: RunStatus
RUN_STATUS_DISPATCHING: RunStatus
RUN_STATUS_RUNNING: RunStatus
RUN_STATUS_COMPLETED: RunStatus
RUN_STATUS_FAILED: RunStatus
RUN_STATUS_CANCELLED: RunStatus

class SubmitRunRequest(_message.Message):
    __slots__ = ("run_id", "config_json")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_JSON_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    config_json: str
    def __init__(self, run_id: _Optional[str] = ..., config_json: _Optional[str] = ...) -> None: ...

class SubmitRunResponse(_message.Message):
    __slots__ = ("run_id", "digest")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    digest: str
    def __init__(self, run_id: _Optional[str] = ..., digest: _Optional[str] = ...) -> None: ...

class CancelRunRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class CancelRunResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RunRecord(_message.Message):
    __slots__ = ("run_id", "status", "digest", "created_at", "updated_at", "last_heartbeat", "gpu_slot", "failure_reason", "gpu_slots")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    GPU_SLOT_FIELD_NUMBER: _ClassVar[int]
    FAILURE_REASON_FIELD_NUMBER: _ClassVar[int]
    GPU_SLOTS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    status: RunStatus
    digest: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    last_heartbeat: _timestamp_pb2.Timestamp
    gpu_slot: int
    failure_reason: str
    gpu_slots: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, run_id: _Optional[str] = ..., status: _Optional[_Union[RunStatus, str]] = ..., digest: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., last_heartbeat: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., gpu_slot: _Optional[int] = ..., failure_reason: _Optional[str] = ..., gpu_slots: _Optional[_Iterable[int]] = ...) -> None: ...

class ListRunsRequest(_message.Message):
    __slots__ = ("status_filter",)
    STATUS_FILTER_FIELD_NUMBER: _ClassVar[int]
    status_filter: _containers.RepeatedScalarFieldContainer[RunStatus]
    def __init__(self, status_filter: _Optional[_Iterable[_Union[RunStatus, str]]] = ...) -> None: ...

class ListRunsResponse(_message.Message):
    __slots__ = ("runs",)
    RUNS_FIELD_NUMBER: _ClassVar[int]
    runs: _containers.RepeatedCompositeFieldContainer[RunRecord]
    def __init__(self, runs: _Optional[_Iterable[_Union[RunRecord, _Mapping]]] = ...) -> None: ...

class WatchRunsRequest(_message.Message):
    __slots__ = ("status_filter",)
    STATUS_FILTER_FIELD_NUMBER: _ClassVar[int]
    status_filter: _containers.RepeatedScalarFieldContainer[RunStatus]
    def __init__(self, status_filter: _Optional[_Iterable[_Union[RunStatus, str]]] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
