"""Profile CleanRL training runs with resource sampling."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Dict, Iterable, Optional, Tuple


def _probe_gpu() -> bool:
    try:
        from shutil import which

        if which("nvidia-smi") is None:
            return False
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(result.strip())
    except Exception:
        return False


def _query_gpu_metrics() -> Optional[Dict[str, float]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            return None
        util, mem = map(float, parts)
        return {"gpu_util_percent": util, "gpu_mem_mb": mem}
    except Exception:
        return None


def _read_proc_cpu(pid: int) -> float:
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "%cpu", "--no-headers"],
            text=True,
        ).strip()
        return float(output) if output else 0.0
    except Exception:
        return 0.0


def _read_proc_mem(pid: int) -> float:
    status_path = Path("/proc") / str(pid) / "status"
    try:
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    mem_kb = float(parts[1])
                    return mem_kb / 1024.0
    except FileNotFoundError:
        pass
    return 0.0


@dataclass
class Sample:
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_util_percent: Optional[float]
    gpu_mem_mb: Optional[float]


class ResourceMonitor:
    def __init__(self, pid: int, interval: float = 0.5, enable_gpu: bool = False) -> None:
        self.pid = pid
        self.interval = interval
        self.enable_gpu = enable_gpu
        self._samples: list[Sample] = []
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def samples(self) -> Iterable[Sample]:
        return list(self._samples)

    def summary(self) -> Dict[str, float]:
        if not self._samples:
            return {}
        cpu = [s.cpu_percent for s in self._samples]
        mem = [s.memory_mb for s in self._samples]
        gpu_util = [s.gpu_util_percent for s in self._samples if s.gpu_util_percent is not None]
        gpu_mem = [s.gpu_mem_mb for s in self._samples if s.gpu_mem_mb is not None]
        return {
            "cpu_avg_percent": sum(cpu) / len(cpu),
            "cpu_peak_percent": max(cpu),
            "mem_avg_mb": sum(mem) / len(mem),
            "mem_peak_mb": max(mem),
            "gpu_avg_percent": (sum(gpu_util) / len(gpu_util)) if gpu_util else 0.0,
            "gpu_peak_percent": max(gpu_util) if gpu_util else 0.0,
            "gpu_avg_mem_mb": (sum(gpu_mem) / len(gpu_mem)) if gpu_mem else 0.0,
            "gpu_peak_mem_mb": max(gpu_mem) if gpu_mem else 0.0,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            if not Path(f"/proc/{self.pid}").exists():
                break
            cpu = _read_proc_cpu(self.pid)
            mem = _read_proc_mem(self.pid)
            gpu_metrics = _query_gpu_metrics() if self.enable_gpu else None
            self._samples.append(
                Sample(
                    timestamp=time.perf_counter(),
                    cpu_percent=cpu,
                    memory_mb=mem,
                    gpu_util_percent=gpu_metrics.get("gpu_util_percent") if gpu_metrics else None,
                    gpu_mem_mb=gpu_metrics.get("gpu_mem_mb") if gpu_metrics else None,
                )
            )
            time.sleep(self.interval)


def _latest_run_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CleanRL training")
    parser.add_argument("--algo", default="cleanrl/dqn.py", help="Relative path to CleanRL algorithm script")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--output", default="docs/data/cleanrl_profile.json")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    script_path = Path(args.algo)
    if not script_path.exists():
        raise SystemExit(f"Algorithm script not found: {script_path}")

    run_dir_root = Path("runs")
    before_run = _latest_run_dir(run_dir_root)

    cmd = [
        sys.executable,
        str(script_path),
        "--env-id",
        args.env_id,
        "--total-timesteps",
        str(args.total_timesteps),
    ]

    gpu_available = _probe_gpu()

    start = time.perf_counter()
    env = os.environ.copy()
    if args.cuda_visible_devices != "__keep__":
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    monitor = ResourceMonitor(proc.pid, interval=args.interval, enable_gpu=gpu_available)
    monitor.start()
    stdout, stderr = proc.communicate()
    monitor.stop()
    duration = time.perf_counter() - start

    run_dir_after = _latest_run_dir(run_dir_root)

    def _tail(lines: str) -> Iterable[str]:
        data = lines.splitlines()
        return data[-5:] if data else []

    payload = {
        "engine": "cleanrl",
        "command": cmd,
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "exit_code": proc.returncode,
        "duration_seconds": duration,
        "resource_summary": monitor.summary(),
        "stdout_tail": list(_tail(stdout)),
        "stderr_tail": list(_tail(stderr)),
        "gpu_available": gpu_available,
    }

    if run_dir_after and run_dir_after != before_run:
        payload["run_dir"] = str(run_dir_after)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
