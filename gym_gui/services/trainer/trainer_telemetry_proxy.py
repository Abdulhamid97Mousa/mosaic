# gym_gui/services/trainer/trainer_telemetry_proxy.py
"""
Telemetry JSONL → gRPC proxy sidecar.

Runs a worker subprocess that emits JSONL telemetry to stdout,
translates each line into RunStep/RunEpisode protos, and streams
them to the daemon via PublishRunSteps/PublishRunEpisodes.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import logging
from typing import Any, AsyncIterator, Dict, Optional

import grpc

from gym_gui.services.trainer.proto import trainer_pb2, trainer_pb2_grpc
from gym_gui.core.subprocess_validation import validated_create_subprocess_exec

_LOGGER = logging.getLogger("gym_gui.trainer.telemetry_proxy")


class JsonlTailer:
    """Async wrapper that yields parsed JSON objects from a process' stdout."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    async def __aiter__(self):
        if not self._proc.stdout:
            return
        async for raw in self._proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # best-effort: skip malformed line
                continue
            yield obj


def _ts_from_unix_ns(ns: int) -> Any:
    """Convert nanoseconds to protobuf Timestamp."""
    from google.protobuf.timestamp_pb2 import Timestamp
    ts = Timestamp()
    seconds, nanos = divmod(int(ns), 1_000_000_000)
    ts.seconds = seconds
    ts.nanos = nanos
    return ts


def _coerce_str(x: Any) -> str:
    """Coerce any value to string for JSON fields."""
    if x is None:
        return ""
    if isinstance(x, (dict, list, bool, int, float)):
        return json.dumps(x, separators=(",", ":"))
    return str(x)


def _mk_runstep(
    ev: Dict[str, Any],
    run_id: str,
    default_agent: str,
    worker_id: str | None,
) -> trainer_pb2.RunStep:
    """Build RunStep proto from JSONL event dict."""
    # CRITICAL: Extract episode_index from metadata dict
    # The trainer emits: episode (display value = seed + episode_index)
    # The metadata dict contains: episode_index (0-based counter)
    # We MUST use episode_index from metadata for proper synchronization
    episode_index_val = None

    # Extract metadata and get episode_index from it
    metadata_json = ev.get("metadata_json", ev.get("metadata", "{}"))
    if isinstance(metadata_json, str):
        try:
            metadata = json.loads(metadata_json)
            episode_index_val = metadata.get("episode_index")
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(metadata_json, dict):
        episode_index_val = metadata_json.get("episode_index")

    # Fallback to episode field if episode_index not found in metadata
    if episode_index_val is None:
        episode_index_val = ev.get("episode", 0)

    episode_val = episode_index_val
    step_val = ev.get("step_index", ev.get("step", 0))

    # DEBUG: Log render_payload presence
    render_payload = ev.get("render_payload")
    if render_payload:
        _LOGGER.debug(f"[PROXY] render_payload found: {type(render_payload)}, keys: {list(render_payload.keys()) if isinstance(render_payload, dict) else 'not_dict'}")
    else:
        _LOGGER.debug(f"[PROXY] render_payload NOT found in event. Available keys: {list(ev.keys())}")

    msg = trainer_pb2.RunStep(
        run_id=run_id,
        episode_index=int(episode_val),
        step_index=int(step_val),
        action_json=_coerce_str(ev.get("action_json", ev.get("action"))),
        observation_json=_coerce_str(ev.get("observation_json", ev.get("observation"))),
        reward=float(ev.get("reward", 0.0)),
        terminated=bool(ev.get("terminated", False)),
        truncated=bool(ev.get("truncated", False)),
        policy_label=str(ev.get("policy_label", "")),
        backend=str(ev.get("backend", "")),
        agent_id=str(ev.get("agent_id") or default_agent),
        render_hint_json=_coerce_str(ev.get("render_hint_json", ev.get("render"))),
        render_payload_json=_coerce_str(ev.get("render_payload")),
        frame_ref=str(ev.get("frame_ref", "")),
        payload_version=int(ev.get("payload_version", 0)),
        episode_seed=int(ev.get("episode_seed", 0)),  # NEW: Unique seed per episode for environment variation
        worker_id=str(worker_id or ev.get("worker_id", "")) or "",
    )
    ts_ns = ev.get("ts_unix_ns")
    if isinstance(ts_ns, (int, float, str)) and str(ts_ns).replace('.', '').replace('-', '').isdigit():
        msg.timestamp.CopyFrom(_ts_from_unix_ns(int(float(ts_ns))))
    return msg


def _mk_runepisode(
    ev: Dict[str, Any],
    run_id: str,
    default_agent: str,
    worker_id: str | None,
) -> trainer_pb2.RunEpisode:
    """Build RunEpisode proto from JSONL event dict."""
    # CRITICAL: Extract episode_index from metadata dict
    # The trainer emits: episode (display value = seed + episode_index)
    # The metadata dict contains: episode_index (0-based counter)
    # We MUST use episode_index from metadata for proper synchronization
    episode_index_val = None

    # Extract metadata and get episode_index from it
    metadata_json = ev.get("metadata_json", ev.get("metadata", "{}"))
    if isinstance(metadata_json, str):
        try:
            metadata = json.loads(metadata_json)
            episode_index_val = metadata.get("episode_index")
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(metadata_json, dict):
        episode_index_val = metadata_json.get("episode_index")

    # Fallback to episode field if episode_index not found in metadata
    if episode_index_val is None:
        episode_index_val = ev.get("episode", 0)

    msg = trainer_pb2.RunEpisode(
        run_id=run_id,
        episode_index=int(episode_index_val),
        total_reward=float(ev.get("total_reward", ev.get("reward", 0.0))),
        steps=int(ev.get("steps", 0)),
        terminated=bool(ev.get("terminated", False)),
        truncated=bool(ev.get("truncated", False)),
        metadata_json=_coerce_str(ev.get("metadata_json", ev.get("metadata", {}))),
        agent_id=str(ev.get("agent_id") or default_agent),
        worker_id=str(worker_id or ev.get("worker_id", "")) or "",
    )
    ts_ns = ev.get("ts_unix_ns")
    if isinstance(ts_ns, (int, float, str)) and str(ts_ns).replace('.', '').replace('-', '').isdigit():
        msg.timestamp.CopyFrom(_ts_from_unix_ns(int(float(ts_ns))))
    return msg


async def _publish_steps(
    stub: trainer_pb2_grpc.TrainerServiceStub,
    queue: "asyncio.Queue[Optional[trainer_pb2.RunStep]]"
) -> trainer_pb2.PublishTelemetryResponse:
    """Publish step stream from queue."""
    async def gen() -> AsyncIterator[trainer_pb2.RunStep]:
        while True:
            item = await queue.get()
            if item is None:  # sentinel
                break
            yield item
    return await stub.PublishRunSteps(gen())


async def _publish_episodes(
    stub: trainer_pb2_grpc.TrainerServiceStub,
    queue: "asyncio.Queue[Optional[trainer_pb2.RunEpisode]]"
) -> trainer_pb2.PublishTelemetryResponse:
    """Publish episode stream from queue."""
    async def gen() -> AsyncIterator[trainer_pb2.RunEpisode]:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    return await stub.PublishRunEpisodes(gen())


async def run_proxy(
    target: str,
    run_id: str,
    agent_id: str,
    worker_id: str | None,
    worker_argv: list[str],
    max_queue: int = 2048
) -> int:
    """
    Run worker subprocess, tail JSONL stdout, and proxy to gRPC streams.
    
    Returns worker exit code.
    """
    _LOGGER.info(
        "Proxy launching worker",
        extra={
            "run_id": run_id,
            "agent_id": agent_id,
            "worker_id": worker_id,
            "worker_cmd": worker_argv,
        },
    )

    # Start worker subprocess whose stdout is JSONL
    proc = await validated_create_subprocess_exec(
        *worker_argv,
        run_id=run_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )

    # Create gRPC channel & stub
    options = (
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
    )
    channel = grpc.aio.insecure_channel(target, options=options)
    await channel.channel_ready()

    stub = trainer_pb2_grpc.TrainerServiceStub(channel)

    # Perform capability handshake before streaming telemetry
    resolved_worker_id = worker_id or f"proxy-{os.getpid()}"
    try:
        response = await stub.RegisterWorker(
            trainer_pb2.RegisterWorkerRequest(
                run_id=run_id,
                worker_id=resolved_worker_id,
                worker_kind="telemetry_proxy",
                proto_version="MOSAIC/1.0",
                schema_id="jsonl.telemetry",
                schema_version=1,
                supports_pause=False,
                supports_checkpoint=False,
            )
        )
        session_token = response.session_token
        _LOGGER.info(
            "RegisterWorker accepted",
            extra={
                "run_id": run_id,
                "worker_id": resolved_worker_id,
                "session_token": session_token,
                "accepted_version": response.accepted_version,
            },
        )
    except grpc.aio.AioRpcError as exc:
        _LOGGER.error(
            "RegisterWorker failed",
            extra={
                "run_id": run_id,
                "worker_id": resolved_worker_id,
                "code": exc.code().name,
                "details": exc.details(),
            },
        )
        await channel.close()
        # Ensure worker process is terminated since handshake failed
        with contextlib.suppress(ProcessLookupError):
            if proc.returncode is None:
                proc.terminate()
        await proc.wait()
        return int(proc.returncode or 1)

    # Queues feeding client-streaming RPCs
    step_q: asyncio.Queue[Optional[trainer_pb2.RunStep]] = asyncio.Queue(maxsize=max_queue)
    ep_q: asyncio.Queue[Optional[trainer_pb2.RunEpisode]] = asyncio.Queue(maxsize=max_queue)
    step_count = 0
    episode_count = 0

    # Tasks to publish streams
    steps_task = asyncio.create_task(_publish_steps(stub, step_q), name="publish-steps")
    eps_task = asyncio.create_task(_publish_episodes(stub, ep_q), name="publish-episodes")

    async def read_worker() -> None:
        """Tail worker stdout and enqueue parsed events."""
        nonlocal step_count, episode_count
        tail = JsonlTailer(proc)
        async for ev in tail:
            typ = str(ev.get("type", "")).lower()
            if typ == "step":
                try:
                    step_msg = _mk_runstep(ev, run_id, agent_id, worker_id)
                    step_q.put_nowait(step_msg)
                    step_count += 1
                    if step_count <= 5:
                        _LOGGER.debug(
                            "Proxy enqueued step",
                            extra={
                                "run_id": run_id,
                                "agent_id": step_msg.agent_id,
                                "episode_index": step_msg.episode_index,
                                "step_index": step_msg.step_index,
                            },
                        )
                except asyncio.QueueFull:
                    # Drop oldest by pulling one, then push new
                    _ = step_q.get_nowait()
                    step_q.put_nowait(step_msg)
                    _LOGGER.warning(
                        "Proxy step queue full; dropping oldest",
                        extra={"run_id": run_id},
                    )
            elif typ == "episode":
                try:
                    ep_msg = _mk_runepisode(ev, run_id, agent_id, worker_id)
                    ep_q.put_nowait(ep_msg)
                    episode_count += 1
                    if episode_count <= 3:
                        _LOGGER.debug(
                            "Proxy enqueued episode",
                            extra={
                                "run_id": run_id,
                                "agent_id": ep_msg.agent_id,
                                "episode_index": ep_msg.episode_index,
                            },
                        )
                except asyncio.QueueFull:
                    _ = ep_q.get_nowait()
                    ep_q.put_nowait(ep_msg)
                    _LOGGER.warning(
                        "Proxy episode queue full; dropping oldest",
                        extra={"run_id": run_id},
                    )
            # else: ignore other event types

    async def pump_stderr() -> None:
        """Forward worker stderr to our stderr."""
        if not proc.stderr:
            return
        async for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            # Surface stderr to our own stderr so daemon logs show issues
            sys.stderr.write(f"[worker stderr] {line}\n")

    # Drive everything until worker exits
    reader = asyncio.create_task(read_worker(), name="read-worker")
    errpump = asyncio.create_task(pump_stderr(), name="read-stderr")

    # Wait for worker to finish
    rc = await proc.wait()
    _LOGGER.info(
        "Worker process exited",
        extra={
            "run_id": run_id,
            "exit_code": rc,
            "steps_forwarded": step_count,
            "episodes_forwarded": episode_count,
        },
    )

    # Stop publishers
    await step_q.put(None)
    await ep_q.put(None)
    # Await publishing completion (ignore errors so we can report worker rc)
    try:
        await asyncio.gather(steps_task, eps_task, return_exceptions=True)
    finally:
        await channel.close()

    # Ensure readers are done
    for t in (reader, errpump):
        if not t.done():
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

    return int(rc)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for sidecar proxy."""
    # Manual handling of '--' separator for worker command
    if argv is None:
        argv = sys.argv[1:]
    
    # Split on '--' separator
    try:
        sep_idx = argv.index("--")
        proxy_args = argv[:sep_idx]
        worker_cmd = argv[sep_idx + 1:]
    except ValueError:
        print("Proxy requires a worker command after `--`.", file=sys.stderr)
        return 2
    
    if not worker_cmd:
        print("Proxy requires a worker command after `--`.", file=sys.stderr)
        return 2
    
    # Parse proxy-specific arguments
    p = argparse.ArgumentParser(description="Telemetry JSONL → gRPC proxy sidecar")
    p.add_argument("--target", required=True, help="daemon address, e.g. 127.0.0.1:50055")
    p.add_argument("--run-id", required=True)
    p.add_argument("--agent-id", default="agent_1")
    p.add_argument("--worker-id", default="", help="Worker identifier for distributed runs")
    args = p.parse_args(proxy_args)

    return asyncio.run(
        run_proxy(
            args.target,
            args.run_id,
            args.agent_id,
            args.worker_id or None,
            worker_cmd,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
