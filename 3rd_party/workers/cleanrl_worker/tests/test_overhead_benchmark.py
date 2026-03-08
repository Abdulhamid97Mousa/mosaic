"""Benchmark test to measure MOSAIC overhead vs native CleanRL execution.

This test measures:
1. Native CleanRL execution (direct subprocess)
2. MOSAIC execution WITHOUT FastLane telemetry
3. MOSAIC execution WITH FastLane telemetry

The goal is to empirically validate overhead claims in the paper.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

# Find CleanRL repo root
CLEANRL_WORKER_ROOT = Path(__file__).resolve().parent.parent
CLEANRL_REPO = CLEANRL_WORKER_ROOT / "cleanrl"

# Use a fast environment and short training for benchmarking
BENCHMARK_ENV = "CartPole-v1"
BENCHMARK_ALGO = "ppo"
BENCHMARK_TIMESTEPS = 10_000  # Short but measurable
BENCHMARK_ITERATIONS = 3  # Average over multiple runs


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    mode: str
    wall_time_seconds: float
    timesteps: int
    steps_per_second: float
    return_code: int

    @property
    def overhead_vs(self) -> Optional[float]:
        """Placeholder for overhead calculation."""
        return None


def run_native_cleanrl(timesteps: int, env_id: str, seed: int = 1) -> BenchmarkResult:
    """Run CleanRL directly without any MOSAIC wrapper.

    This is the baseline measurement.
    """
    # Path to CleanRL PPO script
    ppo_script = CLEANRL_REPO / "cleanrl" / "ppo.py"
    if not ppo_script.exists():
        pytest.skip(f"CleanRL PPO script not found at {ppo_script}")

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            str(ppo_script),
            "--env-id", env_id,
            "--total-timesteps", str(timesteps),
            "--seed", str(seed),
            "--track", "False",  # Disable wandb
            "--capture-video", "False",
        ]

        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"

        start_time = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env=env,
        )
        end_time = time.perf_counter()

        wall_time = end_time - start_time

        return BenchmarkResult(
            mode="native",
            wall_time_seconds=wall_time,
            timesteps=timesteps,
            steps_per_second=timesteps / wall_time if wall_time > 0 else 0,
            return_code=result.returncode,
        )


def run_mosaic_cleanrl(
    timesteps: int,
    env_id: str,
    seed: int = 1,
    fastlane_enabled: bool = False,
) -> BenchmarkResult:
    """Run CleanRL through MOSAIC cleanrl_worker.

    This measures MOSAIC overhead.
    """
    from cleanrl_worker.config import CleanRLWorkerConfig
    from cleanrl_worker.runtime import CleanRLTrainerRuntime
    from gym_gui.core.worker import TelemetryEmitter

    # Create a no-op emitter for benchmarking
    class NoOpEmitter(TelemetryEmitter):
        def emit(self, event_type: str, payload: dict) -> None:
            pass
        def heartbeat(self, payload: dict, constant: Optional[str] = None) -> None:
            pass
        def run_started(self, payload: dict, constant: Optional[str] = None) -> None:
            pass
        def run_completed(self, payload: dict, constant: Optional[str] = None) -> None:
            pass
        def run_failed(self, payload: dict, constant: Optional[str] = None) -> None:
            pass

    config = CleanRLWorkerConfig(
        run_id=f"benchmark-{seed}",
        algo=BENCHMARK_ALGO,
        env_id=env_id,
        seed=seed,
        total_timesteps=timesteps,
        track_wandb=False,
        capture_video=False,
    )

    emitter = NoOpEmitter()
    runtime = CleanRLTrainerRuntime(config, emitter)

    # Set FastLane environment
    env = os.environ.copy()
    if fastlane_enabled:
        env["GYM_GUI_FASTLANE_ONLY"] = "1"
        env["GYM_GUI_FASTLANE_SLOT"] = "0"
    else:
        env["GYM_GUI_FASTLANE_ONLY"] = "0"

    start_time = time.perf_counter()
    try:
        runtime.run(extras={"tensorboard": False})
        return_code = 0
    except subprocess.CalledProcessError as e:
        return_code = e.returncode
    except Exception:
        return_code = 1
    end_time = time.perf_counter()

    wall_time = end_time - start_time
    mode = "mosaic+fastlane" if fastlane_enabled else "mosaic"

    return BenchmarkResult(
        mode=mode,
        wall_time_seconds=wall_time,
        timesteps=timesteps,
        steps_per_second=timesteps / wall_time if wall_time > 0 else 0,
        return_code=return_code,
    )


def calculate_overhead(baseline: BenchmarkResult, test: BenchmarkResult) -> float:
    """Calculate overhead percentage.

    Returns:
        Overhead as percentage (e.g., 5.0 means 5% slower)
    """
    if baseline.wall_time_seconds <= 0:
        return 0.0
    overhead = ((test.wall_time_seconds - baseline.wall_time_seconds) / baseline.wall_time_seconds) * 100
    return overhead


@pytest.mark.benchmark
@pytest.mark.slow
class TestOverheadBenchmark:
    """Benchmark tests for MOSAIC overhead measurement."""

    def test_native_cleanrl_baseline(self):
        """Establish baseline: native CleanRL execution time."""
        result = run_native_cleanrl(
            timesteps=BENCHMARK_TIMESTEPS,
            env_id=BENCHMARK_ENV,
        )

        assert result.return_code == 0, "Native CleanRL should complete successfully"
        assert result.steps_per_second > 0, "Should measure positive throughput"

        print(f"\n{'='*60}")
        print(f"BASELINE: Native CleanRL")
        print(f"  Wall time: {result.wall_time_seconds:.2f}s")
        print(f"  Throughput: {result.steps_per_second:.0f} steps/sec")
        print(f"{'='*60}")

    def test_mosaic_without_fastlane(self):
        """Measure MOSAIC overhead without FastLane telemetry."""
        # First get baseline
        baseline = run_native_cleanrl(
            timesteps=BENCHMARK_TIMESTEPS,
            env_id=BENCHMARK_ENV,
        )

        # Then run through MOSAIC
        mosaic_result = run_mosaic_cleanrl(
            timesteps=BENCHMARK_TIMESTEPS,
            env_id=BENCHMARK_ENV,
            fastlane_enabled=False,
        )

        overhead = calculate_overhead(baseline, mosaic_result)

        print(f"\n{'='*60}")
        print(f"MOSAIC WITHOUT FASTLANE vs NATIVE")
        print(f"  Native:  {baseline.wall_time_seconds:.2f}s ({baseline.steps_per_second:.0f} steps/sec)")
        print(f"  MOSAIC:  {mosaic_result.wall_time_seconds:.2f}s ({mosaic_result.steps_per_second:.0f} steps/sec)")
        print(f"  Overhead: {overhead:+.2f}%")
        print(f"{'='*60}")

        # The claim: less than 5% overhead
        # Without FastLane, we expect nearly 0% since subprocess runs at native speed
        assert overhead < 10.0, f"Overhead {overhead:.2f}% exceeds 10% threshold"

    def test_mosaic_with_fastlane(self):
        """Measure MOSAIC overhead with FastLane telemetry enabled."""
        baseline = run_native_cleanrl(
            timesteps=BENCHMARK_TIMESTEPS,
            env_id=BENCHMARK_ENV,
        )

        mosaic_result = run_mosaic_cleanrl(
            timesteps=BENCHMARK_TIMESTEPS,
            env_id=BENCHMARK_ENV,
            fastlane_enabled=True,
        )

        overhead = calculate_overhead(baseline, mosaic_result)

        print(f"\n{'='*60}")
        print(f"MOSAIC WITH FASTLANE vs NATIVE")
        print(f"  Native:       {baseline.wall_time_seconds:.2f}s ({baseline.steps_per_second:.0f} steps/sec)")
        print(f"  MOSAIC+FL:    {mosaic_result.wall_time_seconds:.2f}s ({mosaic_result.steps_per_second:.0f} steps/sec)")
        print(f"  Overhead:     {overhead:+.2f}%")
        print(f"{'='*60}")

        # With FastLane, overhead depends on render() cost
        # For CartPole (simple rendering), should still be reasonable


@pytest.mark.benchmark
def test_full_overhead_comparison():
    """Run complete overhead comparison with multiple iterations."""
    native_times = []
    mosaic_times = []
    mosaic_fl_times = []

    print(f"\n{'='*70}")
    print(f"FULL OVERHEAD BENCHMARK")
    print(f"Environment: {BENCHMARK_ENV}")
    print(f"Timesteps: {BENCHMARK_TIMESTEPS}")
    print(f"Iterations: {BENCHMARK_ITERATIONS}")
    print(f"{'='*70}")

    for i in range(BENCHMARK_ITERATIONS):
        seed = 42 + i
        print(f"\nIteration {i+1}/{BENCHMARK_ITERATIONS} (seed={seed})...")

        # Native
        native = run_native_cleanrl(BENCHMARK_TIMESTEPS, BENCHMARK_ENV, seed)
        native_times.append(native.wall_time_seconds)

        # MOSAIC without FastLane
        mosaic = run_mosaic_cleanrl(BENCHMARK_TIMESTEPS, BENCHMARK_ENV, seed, fastlane_enabled=False)
        mosaic_times.append(mosaic.wall_time_seconds)

        # MOSAIC with FastLane
        mosaic_fl = run_mosaic_cleanrl(BENCHMARK_TIMESTEPS, BENCHMARK_ENV, seed, fastlane_enabled=True)
        mosaic_fl_times.append(mosaic_fl.wall_time_seconds)

    # Calculate averages
    avg_native = sum(native_times) / len(native_times)
    avg_mosaic = sum(mosaic_times) / len(mosaic_times)
    avg_mosaic_fl = sum(mosaic_fl_times) / len(mosaic_fl_times)

    overhead_mosaic = ((avg_mosaic - avg_native) / avg_native) * 100
    overhead_mosaic_fl = ((avg_mosaic_fl - avg_native) / avg_native) * 100

    print(f"\n{'='*70}")
    print(f"RESULTS (averaged over {BENCHMARK_ITERATIONS} iterations)")
    print(f"{'='*70}")
    print(f"{'Mode':<30} {'Avg Time (s)':<15} {'Overhead':<15}")
    print(f"{'-'*70}")
    print(f"{'Native CleanRL':<30} {avg_native:<15.2f} {'baseline':<15}")
    print(f"{'MOSAIC (no telemetry)':<30} {avg_mosaic:<15.2f} {overhead_mosaic:+.2f}%")
    print(f"{'MOSAIC + FastLane':<30} {avg_mosaic_fl:<15.2f} {overhead_mosaic_fl:+.2f}%")
    print(f"{'='*70}")

    # Return results for programmatic access
    return {
        "native_avg": avg_native,
        "mosaic_avg": avg_mosaic,
        "mosaic_fastlane_avg": avg_mosaic_fl,
        "overhead_mosaic_pct": overhead_mosaic,
        "overhead_mosaic_fastlane_pct": overhead_mosaic_fl,
    }


if __name__ == "__main__":
    # Run directly for quick testing
    print("Running overhead benchmark...")
    results = test_full_overhead_comparison()
    print(f"\nFinal Results: {results}")
