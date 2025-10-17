from __future__ import annotations

"""Async dispatcher that orchestrates trainer worker lifecycle."""

import sys
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import signal
import subprocess
from typing import Any, Callable, Optional

from gym_gui.services.trainer import RunRecord, RunRegistry, RunStatus
from gym_gui.services.trainer.gpu import GPUAllocator

_LOGGER = logging.getLogger("gym_gui.trainer.dispatcher")


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
                    await self._dispatch_pending_runs()
        except asyncio.CancelledError:
            _LOGGER.debug("Dispatch loop cancelled")
            raise

    async def _dispatch_pending_runs(self) -> None:
        """Dispatch all PENDING runs."""
        pending = self._registry.load_runs([RunStatus.PENDING])
        for run in pending:
            if run.run_id in self._workers:
                continue
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

        process = await asyncio.create_subprocess_exec(
            *cmd,
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

        # Start log streaming tasks
        asyncio.create_task(self._stream_stdout(handle), name=f"stdout-{run.run_id}")
        asyncio.create_task(self._stream_stderr(handle), name=f"stderr-{run.run_id}")

    def _build_worker_command(self, run: RunRecord) -> list[str]:
        """Build the subprocess command for the trainer worker."""

        
        # Derive agent_id from run_id for telemetry correlation
        agent_id = f"agent_{run.run_id[:8]}"
        
        # gRPC daemon target (hardcoded for now, could be config)
        grpc_target = "127.0.0.1:50055"
        
        # Check for custom worker command from environment or config
        worker_entry = os.environ.get("GYM_GUI_WORKER_CMD")
        if not worker_entry:
            # Default to demo worker for testing
            worker_cmd = [
                sys.executable, "-m", "gym_gui.workers.demo_worker",
                "--run-id", run.run_id,
                "--agent-id", agent_id,
                "--episodes", "3",
                "--steps", "15",
                "--delay", "0.03",
            ]
        else:
            # Parse custom worker command (TODO: handle quoted args properly)
            worker_cmd = worker_entry.split()
        
        # Wrap worker with telemetry proxy
        proxy_cmd = [
            sys.executable, "-m", "gym_gui.services.trainer.trainer_telemetry_proxy",
            "--target", grpc_target,
            "--run-id", run.run_id,
            "--agent-id", agent_id,
            "--",  # Separator for proxy args vs worker args
        ]
        
        return proxy_cmd + worker_cmd

    def _build_worker_env(self, run: RunRecord) -> dict[str, str]:
        """Build environment variables for the worker subprocess."""
        env = os.environ.copy()
        if run.gpu_slots:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(slot) for slot in run.gpu_slots)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    async def _stream_stdout(self, handle: WorkerHandle) -> None:
        """Stream stdout from worker subprocess."""
        if not handle.process.stdout:
            return
        try:
            async for line in handle.process.stdout:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                _LOGGER.debug("Worker stdout", extra={"run_id": handle.run_id, "line": decoded})
        except asyncio.CancelledError:
            pass

    async def _stream_stderr(self, handle: WorkerHandle) -> None:
        """Stream stderr from worker subprocess."""
        if not handle.process.stderr:
            return
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
        await self._broadcast_update(handle.run_id)

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
