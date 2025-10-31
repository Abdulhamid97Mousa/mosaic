from __future__ import annotations

"""Async dispatcher that orchestrates trainer worker lifecycle."""

import sys
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import os
import json
import contextlib
from pathlib import Path
import signal
import re
from typing import Any, Callable, Optional

from gym_gui.services.trainer import RunRecord, RunRegistry, RunStatus
from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.services.trainer.gpu import GPUAllocator
from gym_gui.core.subprocess_validation import validated_create_subprocess_exec
from gym_gui.core.agent_config import get_agent_config
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import get_constant_by_code

_LOGGER = logging.getLogger("gym_gui.trainer.dispatcher")

# Import signals for lifecycle events (lazy initialization)
_signals = None

def _get_signals():
    """Get or initialize TrainerSignals lazily."""
    global _signals
    if _signals is None:
        try:
            from gym_gui.services.trainer.signals import get_trainer_signals
            _signals = get_trainer_signals()
        except Exception as e:
            _LOGGER.warning(f"Failed to initialize TrainerSignals: {e}")
            _signals = None
    return _signals


# Pattern to recognize LOG_CODE from worker output: "LOG_XXX | message | extra={...}"
_LOG_CODE_PATTERN = re.compile(
    r'^(?P<code>LOG\d+)\s+\|\s+(?P<message>.+?)\s+\|\s+extra=(?P<extra>.*)$'
)


def _parse_structured_log_line(line: str) -> tuple[str | None, dict[str, Any] | None]:
    """Parse a structured log line from worker output.
    
    Expected format: "LOG_CODE | message | extra={...}"
    
    Returns:
        Tuple of (log_code, extra_dict) if parseable, else (None, None).
    """
    match = _LOG_CODE_PATTERN.match(line.strip())
    if not match:
        return None, None
    
    code = match.group('code')
    message = match.group('message')
    extra_str = match.group('extra')
    
    try:
        extra = json.loads(extra_str)
    except json.JSONDecodeError:
        return None, None
    
    return code, extra


def _re_emit_worker_log(run_id: str, code: str, extra: dict[str, Any]) -> None:
    """Re-emit a worker log as a structured log constant.
    
    Looks up the LOG_CODE, extracts component/subcomponent, and calls log_constant.
    Falls back gracefully if the code is not recognized.
    """
    constant = get_constant_by_code(code)
    if not constant:
        _LOGGER.debug(
            f"Unknown LOG_CODE from worker: {code}",
            extra={"run_id": run_id, "code": code, "worker_extra": extra}
        )
        return
    
    # Ensure component and subcomponent are present in extra dict
    worker_extra = dict(extra)
    worker_extra.setdefault('component', constant.component)
    worker_extra.setdefault('subcomponent', constant.subcomponent)
    
    try:
        log_constant(_LOGGER, constant, extra=worker_extra)
    except Exception as e:
        _LOGGER.debug(
            f"Failed to re-emit worker log: {e}",
            extra={"run_id": run_id, "code": code, "error": str(e)}
        )


class WorkerHandle:
    """Tracks a subprocess worker and its metadata."""

    def __init__(
        self,
        run_id: str,
        process: asyncio.subprocess.Process,
        gpu_slots: list[int],
        started_at: datetime,
    ) -> None:
        self.run_id = run_id
        self.process = process
        self.gpu_slots = gpu_slots
        self.started_at = started_at
        self.cancelled = False


class TrainerDispatcher:
    """Manages worker subprocess lifecycle and state transitions."""

    def __init__(
        self,
        registry: RunRegistry,
        gpu_allocator: GPUAllocator,
        *,
        broadcaster: Optional[Callable[[str], Any]] = None,
        heartbeat_timeout: int = 300,
        dispatch_interval: float = 2.0,
    ) -> None:
        self._registry = registry
        self._gpu_allocator = gpu_allocator
        self._broadcaster = broadcaster
        self._heartbeat_timeout = heartbeat_timeout
        self._dispatch_interval = dispatch_interval
        self._workers: dict[str, WorkerHandle] = {}
        self._stop_event = asyncio.Event()
        self._dispatch_task: Optional[asyncio.Task[None]] = None
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._worker_config_paths: dict[str, Path] = {}

    async def start(self) -> None:
        """Start dispatcher, monitor, and heartbeat checker tasks."""
        self._stop_event.clear()
        self._dispatch_task = asyncio.create_task(self._dispatch_loop(), name="trainer-dispatch")
        self._monitor_task = asyncio.create_task(self._monitor_loop(), name="trainer-monitor")
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="trainer-heartbeat")
        _LOGGER.info("Trainer dispatcher started")

    async def stop(self) -> None:
        """Stop all dispatcher tasks and terminate active workers."""
        self._stop_event.set()
        tasks = [
            task
            for task in (self._dispatch_task, self._monitor_task, self._heartbeat_task)
            if task
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._dispatch_task = None
        self._monitor_task = None
        self._heartbeat_task = None
        await self._terminate_all_workers()
        _LOGGER.info("Trainer dispatcher stopped")

    async def cancel_run(self, run_id: str) -> bool:
        """Request cancellation of a running worker."""
        handle = self._workers.get(run_id)
        if not handle:
            return False
        handle.cancelled = True
        await self._terminate_worker(handle, reason="user_cancel")
        return True

    # ------------------------------------------------------------------
    async def _dispatch_loop(self) -> None:
        """Poll for PENDING runs and dispatch them."""
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._dispatch_interval
                    )
                    break
                except asyncio.TimeoutError:
                    _LOGGER.debug("Dispatch loop tick")
                    await self._dispatch_pending_runs()
        except asyncio.CancelledError:
            _LOGGER.debug("Dispatch loop cancelled")
            raise

    async def _dispatch_pending_runs(self) -> None:
        """Dispatch all PENDING runs."""
        pending = self._registry.load_runs([RunStatus.PENDING])
        _LOGGER.info("Pending runs polled", extra={"count": len(pending)})
        for run in pending:
            if run.run_id in self._workers:
                _LOGGER.debug("Run already has worker", extra={"run_id": run.run_id})
                continue
            _LOGGER.info("Dispatching run", extra={"run_id": run.run_id})
            try:
                await self._dispatch_run(run)
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.exception("Failed to dispatch run", extra={"run_id": run.run_id, "error": str(exc)})
                self._registry.update_status(
                    run.run_id, RunStatus.FAILED, failure_reason=f"dispatch_error: {exc}"
                )
                self._gpu_allocator.release_many([run.run_id])
                self._registry.update_gpu_slots(run.run_id, [])
                await self._broadcast_update(run.run_id)

    async def _dispatch_run(self, run: RunRecord) -> None:
        """Transition run to DISPATCHING, spawn worker subprocess, transition to RUNNING."""
        self._registry.update_status(run.run_id, RunStatus.DISPATCHING)
        await self._broadcast_update(run.run_id)

        # Build worker command
        cmd = self._build_worker_command(run)
        env = self._build_worker_env(run)

        _LOGGER.info("Spawning worker", extra={"run_id": run.run_id, "cmd": cmd})

        # Spawn worker with validated command arguments
        process = await validated_create_subprocess_exec(
            *cmd,
            run_id=run.run_id,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        handle = WorkerHandle(
            run_id=run.run_id,
            process=process,
            gpu_slots=run.gpu_slots,
            started_at=datetime.now(timezone.utc),
        )
        self._workers[run.run_id] = handle

        self._registry.update_status(run.run_id, RunStatus.RUNNING)
        await self._broadcast_update(run.run_id)

        # Emit training_started signal
        signals = _get_signals()
        if signals:
            try:
                config_json = json.loads(self._registry.get_run_config_json(run.run_id) or "{}")
                metadata = config_json.get("metadata", {})
                signals.emit_training_started(run.run_id, metadata)
                _LOGGER.debug("Emitted training_started signal", extra={"run_id": run.run_id})
            except Exception as e:
                _LOGGER.warning(f"Failed to emit training_started signal: {e}", extra={"run_id": run.run_id})

        # Start log streaming tasks
        asyncio.create_task(self._stream_stdout(handle), name=f"stdout-{run.run_id}")
        asyncio.create_task(self._stream_stderr(handle), name=f"stderr-{run.run_id}")

    def _build_worker_command(self, run: RunRecord) -> list[str]:
        """Build the subprocess command for the trainer worker."""

        
        config_payload = self._registry.get_run_config_json(run.run_id)
        worker_cmd: list[str] | None = None
        config_path: Optional[Path] = None

        if config_payload:
            try:
                config_json = json.loads(config_payload)
                _LOGGER.debug("Loaded run configuration JSON", extra={"run_id": run.run_id})
            except json.JSONDecodeError:
                _LOGGER.warning("Failed to parse run config JSON", extra={"run_id": run.run_id})
                config_json = None
        else:
            config_json = None

        worker_meta = config_json.get("metadata", {}).get("worker", {}) if config_json else {}
        worker_id = ""
        if isinstance(worker_meta, dict):
            raw_worker_id = worker_meta.get("worker_id")
            if raw_worker_id is not None:
                worker_id = str(raw_worker_id).strip()

        if config_json and worker_meta:
            module = worker_meta.get("module")
            script = worker_meta.get("script")
            args = list(worker_meta.get("arguments", []))
            grpc_target = worker_meta.get("grpc_target", "127.0.0.1:50055")
            use_grpc = worker_meta.get("use_grpc", True)
            worker_config = worker_meta.get("config")

            if worker_config:
                config_dir = VAR_TRAINER_DIR / "configs"
                config_dir.mkdir(parents=True, exist_ok=True)
                suffix = f"-{worker_id}" if worker_id else ""
                config_path = config_dir / f"worker-{run.run_id}{suffix}.json"
                worker_payload = dict(worker_config)
                worker_payload.setdefault("run_id", run.run_id)
                if worker_id:
                    worker_payload.setdefault("worker_id", worker_id)
                schema_id = worker_payload.get("schema_id")
                schema_definition = worker_payload.get("schema_definition")
                if schema_definition:
                    _LOGGER.debug(
                        "Worker schema supplied",
                        extra={"run_id": run.run_id, "schema_id": schema_id},
                    )
                else:
                    _LOGGER.warning(
                        "Worker schema definition missing",
                        extra={"run_id": run.run_id, "schema_id": schema_id},
                    )
                config_path.write_text(json.dumps(worker_payload, indent=2), encoding="utf-8")
                self._worker_config_paths[run.run_id] = config_path
                _LOGGER.debug(
                    "Persisted worker config",
                    extra={"run_id": run.run_id, "path": str(config_path)},
                )

            if module:
                worker_cmd = [sys.executable, "-m", module]
            elif script:
                worker_cmd = [script]

            if worker_cmd is not None:
                if config_path is not None and "--config" not in args:
                    args.extend(["--config", str(config_path)])
                if use_grpc:
                    if "--grpc" not in args:
                        args.append("--grpc")
                    if "--grpc-target" not in args:
                        args.extend(["--grpc-target", grpc_target])
                if worker_id and "--worker-id" not in args:
                    args.extend(["--worker-id", worker_id])
                
                # Add BDI flags if BDI mode is enabled
                agent_type = worker_meta.get("agent_type", "Headless")
                if worker_meta.get("bdi_enabled"):
                    if "--bdi" not in args:
                        args.append("--bdi")
                    bdi_config = worker_meta.get("bdi_config", {})
                    bdi_jid = bdi_config.get("jid", "agent@localhost")
                    bdi_password = bdi_config.get("password", "secret")
                    
                    if "--bdi-jid" not in args:
                        args.extend(["--bdi-jid", bdi_jid])
                    if "--bdi-password" not in args:
                        args.extend(["--bdi-password", bdi_password])
                    
                    if "asl_file" in bdi_config:
                        asl_file = bdi_config.get("asl_file")
                        if asl_file and "--asl-file" not in args:
                            args.extend(["--asl-file", asl_file])
                    
                    # Log BDI configuration details
                    agent_config = get_agent_config("BDI")
                    schema = agent_config.get_telemetry_schema()
                    required_fields = agent_config.get_required_fields()
                    optional_fields = agent_config.get_optional_fields()
                    
                    _LOGGER.info(
                        "BDI Agent worker command prepared",
                        extra={
                            "run_id": run.run_id,
                            "agent_type": agent_type,
                            "bdi_jid": bdi_jid,
                            "bdi_password": "[***]",
                            "bdi_asl_file": bdi_config.get("asl_file", "(not provided)"),
                            "bdi_config_keys": list(bdi_config.keys()),
                            "telemetry_schema_categories": list(schema.keys()),
                            "required_telemetry_fields": sorted(required_fields),
                            "optional_telemetry_fields": sorted(optional_fields),
                        },
                    )
                else:
                    # Log Headless agent configuration
                    agent_config = get_agent_config(agent_type)
                    schema = agent_config.get_telemetry_schema()
                    required_fields = agent_config.get_required_fields()
                    
                    _LOGGER.info(
                        "Headless Agent worker command prepared",
                        extra={
                            "run_id": run.run_id,
                            "agent_type": agent_type,
                            "algorithm": worker_meta.get("algorithm", "unknown"),
                            "telemetry_schema_categories": list(schema.keys()),
                            "required_telemetry_fields": sorted(required_fields),
                        },
                    )
                
                worker_cmd.extend(args)
                _LOGGER.info(
                    "Prepared worker command from metadata",
                    extra={
                        "run_id": run.run_id,
                        "cmd": worker_cmd,
                        "agent_id": worker_meta.get("agent_id"),
                    },
                )

        if worker_cmd is None:
            if run.run_id in self._worker_config_paths:
                path = self._worker_config_paths.pop(run.run_id)
                with contextlib.suppress(Exception):
                    path.unlink()
            # Fallback legacy behaviour (demo worker)
            agent_id = f"agent_{run.run_id[:8]}"
            grpc_target = "127.0.0.1:50055"
            worker_entry = os.environ.get("GYM_GUI_WORKER_CMD")
            if not worker_entry:
                worker_cmd = [
                    sys.executable, "-m", "gym_gui.workers.demo_worker",
                    "--run-id", run.run_id,
                    "--agent-id", agent_id,
                    "--episodes", "3",
                    "--steps", "15",
                    "--delay", "0.03",
                ]
            else:
                worker_cmd = worker_entry.split()

            proxy_cmd = [
                sys.executable, "-m", "gym_gui.services.trainer.trainer_telemetry_proxy",
                "--target", grpc_target,
                "--run-id", run.run_id,
                "--agent-id", agent_id,
            ]
            if worker_id:
                proxy_cmd.extend(["--worker-id", worker_id])
            proxy_cmd.append("--")
            return proxy_cmd + worker_cmd

        proxy_cmd = [
            sys.executable, "-m", "gym_gui.services.trainer.trainer_telemetry_proxy",
            "--target", worker_meta.get("grpc_target", "127.0.0.1:50055"),
            "--run-id", run.run_id,
            "--agent-id", worker_meta.get("agent_id", f"agent_{run.run_id[:8]}"),
        ]
        if worker_id:
            proxy_cmd.extend(["--worker-id", worker_id])
        proxy_cmd.append("--")

        _LOGGER.debug(
            "Assembled proxy command",
            extra={"run_id": run.run_id, "proxy_cmd": proxy_cmd, "worker_cmd": worker_cmd},
        )
        return proxy_cmd + worker_cmd

    def _build_worker_env(self, run: RunRecord) -> dict[str, str]:
        """Build environment variables for the worker subprocess."""
        env = os.environ.copy()
        if run.gpu_slots:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(slot) for slot in run.gpu_slots)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        config_path = self._worker_config_paths.get(run.run_id)
        if config_path is not None:
            env["TRAINER_WORKER_CONFIG_PATH"] = str(config_path)

        # Merge environment overrides from run configuration.
        config_payload = self._registry.get_run_config_json(run.run_id)
        config_json: dict[str, Any] = {}
        if config_payload:
            try:
                config_json = json.loads(config_payload)
            except json.JSONDecodeError:
                _LOGGER.warning("Failed to parse run config JSON when building env", extra={"run_id": run.run_id})

        env_overrides = config_json.get("environment") if isinstance(config_json, dict) else None
        if isinstance(env_overrides, dict):
            for key, value in env_overrides.items():
                if key and value is not None:
                    env[str(key)] = str(value)

        worker_meta = config_json.get("metadata", {}).get("worker", {}) if isinstance(config_json, dict) else {}
        worker_id = ""
        if isinstance(worker_meta, dict):
            raw_worker_id = worker_meta.get("worker_id")
            if raw_worker_id is not None:
                worker_id = str(raw_worker_id).strip()
        if worker_id and not env.get("WORKER_ID"):
            env["WORKER_ID"] = worker_id
        return env

    async def _stream_stdout(self, handle: WorkerHandle) -> None:
        """Stream stdout from worker subprocess.
        
        Attempts to parse structured LOG_CODE lines and re-emit as log constants.
        Falls back to plain DEBUG logging for unstructured output.
        """
        if not handle.process.stdout:
            return
        _LOGGER.debug("Starting worker stdout stream", extra={"run_id": handle.run_id})
        try:
            async for line in handle.process.stdout:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                
                # Try to parse as structured log
                code, extra = _parse_structured_log_line(decoded)
                if code is not None and extra is not None:
                    _re_emit_worker_log(handle.run_id, code, extra)
                else:
                    # Fallback: log as plain DEBUG message
                    _LOGGER.debug(
                        "Worker stdout",
                        extra={"run_id": handle.run_id, "line": decoded}
                    )
        except asyncio.CancelledError:
            pass

    async def _stream_stderr(self, handle: WorkerHandle) -> None:
        """Stream stderr from worker subprocess."""
        if not handle.process.stderr:
            return
        _LOGGER.debug("Starting worker stderr stream", extra={"run_id": handle.run_id})
        try:
            async for line in handle.process.stderr:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                _LOGGER.warning("Worker stderr", extra={"run_id": handle.run_id, "line": decoded})
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    async def _monitor_loop(self) -> None:
        """Monitor active workers for completion."""
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                    break
                except asyncio.TimeoutError:
                    _LOGGER.debug("Monitor loop tick", extra={"active_workers": len(self._workers)})
                    await self._check_workers()
        except asyncio.CancelledError:
            _LOGGER.debug("Monitor loop cancelled")
            raise

    async def _check_workers(self) -> None:
        """Check for completed workers and reconcile their state."""
        finished: list[str] = []
        for run_id, handle in self._workers.items():
            if handle.process.returncode is not None:
                finished.append(run_id)
                await self._reconcile_worker(handle)
        for run_id in finished:
            self._workers.pop(run_id, None)

    async def _reconcile_worker(self, handle: WorkerHandle) -> None:
        """Reconcile terminal worker state and release resources."""
        exit_code = handle.process.returncode
        if handle.cancelled:
            status = RunStatus.CANCELLED
            reason = "user_cancel"
        elif exit_code == 0:
            status = RunStatus.COMPLETED
            reason = None
        else:
            status = RunStatus.FAILED
            reason = f"exit_code_{exit_code}"

        _LOGGER.info(
            "Worker finished",
            extra={"run_id": handle.run_id, "exit_code": exit_code, "status": status.value},
        )

        self._registry.update_status(handle.run_id, status, failure_reason=reason)
        self._gpu_allocator.release_many([handle.run_id])
        self._registry.update_gpu_slots(handle.run_id, [])

        # Emit training_finished signal
        signals = _get_signals()
        if signals:
            try:
                outcome = "succeeded" if status == RunStatus.COMPLETED else "failed" if status == RunStatus.FAILED else "canceled"
                signals.emit_training_finished(handle.run_id, outcome, reason)
                _LOGGER.debug("Emitted training_finished signal", extra={"run_id": handle.run_id, "outcome": outcome})
            except Exception as e:
                _LOGGER.warning(f"Failed to emit training_finished signal: {e}", extra={"run_id": handle.run_id})

        await self._broadcast_update(handle.run_id)
        config_path = self._worker_config_paths.pop(handle.run_id, None)
        if config_path and config_path.exists():
            with contextlib.suppress(Exception):
                config_path.unlink()

    async def _terminate_worker(self, handle: WorkerHandle, *, reason: str) -> None:
        """Terminate a worker with SIGTERM → SIGKILL escalation."""
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(handle.process.pid), signal.SIGTERM)
            else:
                handle.process.terminate()
            try:
                await asyncio.wait_for(handle.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                _LOGGER.warning(
                    "Worker did not respond to SIGTERM; escalating to SIGKILL",
                    extra={"run_id": handle.run_id},
                )
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(handle.process.pid), signal.SIGKILL)
                else:
                    handle.process.kill()
                await handle.process.wait()
        except ProcessLookupError:
            pass

    async def _terminate_all_workers(self) -> None:
        """Terminate all active workers during shutdown."""
        for handle in list(self._workers.values()):
            await self._terminate_worker(handle, reason="daemon_shutdown")
        self._workers.clear()

    # ------------------------------------------------------------------
    async def _heartbeat_loop(self) -> None:
        """Check for stale workers and mark them FAILED if heartbeat timeout exceeded."""
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=30.0)
                    break
                except asyncio.TimeoutError:
                    _LOGGER.debug("Heartbeat loop tick")
                    await self._check_heartbeats()
        except asyncio.CancelledError:
            _LOGGER.debug("Heartbeat loop cancelled")
            raise

    async def _check_heartbeats(self) -> None:
        """Mark runs as FAILED if last_heartbeat is stale."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._heartbeat_timeout)
        running = self._registry.load_runs([RunStatus.RUNNING])
        for run in running:
            if run.last_heartbeat and run.last_heartbeat < cutoff:
                _LOGGER.warning(
                    "Worker heartbeat timeout",
                    extra={"run_id": run.run_id, "last_heartbeat": run.last_heartbeat.isoformat()},
                )
                self._registry.update_status(
                    run.run_id, RunStatus.FAILED, failure_reason="worker_timeout"
                )
                self._gpu_allocator.release_many([run.run_id])
                self._registry.update_gpu_slots(run.run_id, [])
                await self._broadcast_update(run.run_id)
                # Terminate the worker if still tracked
                handle = self._workers.pop(run.run_id, None)
                if handle:
                    await self._terminate_worker(handle, reason="heartbeat_timeout")

    async def _broadcast_update(self, run_id: str) -> None:
        """Publish run update if broadcaster is configured."""
        if self._broadcaster:
            try:
                await self._broadcaster(run_id)
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.exception("Broadcast failed", extra={"run_id": run_id, "error": str(exc)})


__all__ = ["TrainerDispatcher", "WorkerHandle"]
