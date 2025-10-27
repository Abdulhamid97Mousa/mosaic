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
    __slots__ = ("run_id", "status", "digest", "created_at", "updated_at", "last_heartbeat", "gpu_slot", "failure_reason", "gpu_slots", "seq_id")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    GPU_SLOT_FIELD_NUMBER: _ClassVar[int]
    FAILURE_REASON_FIELD_NUMBER: _ClassVar[int]
    GPU_SLOTS_FIELD_NUMBER: _ClassVar[int]
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    status: RunStatus
    digest: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    last_heartbeat: _timestamp_pb2.Timestamp
    gpu_slot: int
    failure_reason: str
    gpu_slots: _containers.RepeatedScalarFieldContainer[int]
    seq_id: int
    def __init__(self, run_id: _Optional[str] = ..., status: _Optional[_Union[RunStatus, str]] = ..., digest: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., last_heartbeat: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., gpu_slot: _Optional[int] = ..., failure_reason: _Optional[str] = ..., gpu_slots: _Optional[_Iterable[int]] = ..., seq_id: _Optional[int] = ...) -> None: ...

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
    __slots__ = ("status_filter", "since_seq")
    STATUS_FILTER_FIELD_NUMBER: _ClassVar[int]
    SINCE_SEQ_FIELD_NUMBER: _ClassVar[int]
    status_filter: _containers.RepeatedScalarFieldContainer[RunStatus]
    since_seq: int
    def __init__(self, status_filter: _Optional[_Iterable[_Union[RunStatus, str]]] = ..., since_seq: _Optional[int] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("pid", "started_at", "listen_address", "healthy")
    PID_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    LISTEN_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    pid: int
    started_at: _timestamp_pb2.Timestamp
    listen_address: str
    healthy: bool
    def __init__(self, pid: _Optional[int] = ..., started_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., listen_address: _Optional[str] = ..., healthy: bool = ...) -> None: ...

class StreamStepsRequest(_message.Message):
    __slots__ = ("run_id", "since_seq")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SINCE_SEQ_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    since_seq: int
    def __init__(self, run_id: _Optional[str] = ..., since_seq: _Optional[int] = ...) -> None: ...

class StreamEpisodesRequest(_message.Message):
    __slots__ = ("run_id", "since_seq")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SINCE_SEQ_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    since_seq: int
    def __init__(self, run_id: _Optional[str] = ..., since_seq: _Optional[int] = ...) -> None: ...

class RunStep(_message.Message):
    __slots__ = ("run_id", "episode_index", "step_index", "action_json", "observation_json", "reward", "terminated", "truncated", "timestamp", "policy_label", "backend", "seq_id", "agent_id", "render_hint_json", "frame_ref", "payload_version", "render_payload_json", "episode_seed")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_INDEX_FIELD_NUMBER: _ClassVar[int]
    STEP_INDEX_FIELD_NUMBER: _ClassVar[int]
    ACTION_JSON_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_JSON_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    POLICY_LABEL_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    RENDER_HINT_JSON_FIELD_NUMBER: _ClassVar[int]
    FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_VERSION_FIELD_NUMBER: _ClassVar[int]
    RENDER_PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    EPISODE_SEED_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    episode_index: int
    step_index: int
    action_json: str
    observation_json: str
    reward: float
    terminated: bool
    truncated: bool
    timestamp: _timestamp_pb2.Timestamp
    policy_label: str
    backend: str
    seq_id: int
    agent_id: str
    render_hint_json: str
    frame_ref: str
    payload_version: int
    render_payload_json: str
    episode_seed: int
    def __init__(self, run_id: _Optional[str] = ..., episode_index: _Optional[int] = ..., step_index: _Optional[int] = ..., action_json: _Optional[str] = ..., observation_json: _Optional[str] = ..., reward: _Optional[float] = ..., terminated: bool = ..., truncated: bool = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., policy_label: _Optional[str] = ..., backend: _Optional[str] = ..., seq_id: _Optional[int] = ..., agent_id: _Optional[str] = ..., render_hint_json: _Optional[str] = ..., frame_ref: _Optional[str] = ..., payload_version: _Optional[int] = ..., render_payload_json: _Optional[str] = ..., episode_seed: _Optional[int] = ...) -> None: ...

class RunEpisode(_message.Message):
    __slots__ = ("run_id", "episode_index", "total_reward", "steps", "terminated", "truncated", "metadata_json", "timestamp", "seq_id", "agent_id")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_REWARD_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    METADATA_JSON_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    episode_index: int
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    metadata_json: str
    timestamp: _timestamp_pb2.Timestamp
    seq_id: int
    agent_id: str
    def __init__(self, run_id: _Optional[str] = ..., episode_index: _Optional[int] = ..., total_reward: _Optional[float] = ..., steps: _Optional[int] = ..., terminated: bool = ..., truncated: bool = ..., metadata_json: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., seq_id: _Optional[int] = ..., agent_id: _Optional[str] = ...) -> None: ...

class PublishTelemetryResponse(_message.Message):
    __slots__ = ("accepted", "dropped")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DROPPED_FIELD_NUMBER: _ClassVar[int]
    accepted: int
    dropped: int
    def __init__(self, accepted: _Optional[int] = ..., dropped: _Optional[int] = ...) -> None: ...
