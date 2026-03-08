"""Evaluation helpers for CleanRL algorithms under MOSAIC."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median, pstdev
from typing import Callable, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tolerate missing tensorboard
    SummaryWriter = None  # type: ignore

__all__ = [
    "EvalBatchSummary",
    "EvalRunResult",
    "run_batched_evaluation",
]


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalBatchSummary:
    """Aggregate statistics for a single evaluation batch."""

    index: int
    episodes: int
    returns: List[float]
    avg_return: float
    min_return: float
    max_return: float
    median_return: float
    std_return: float
    duration_sec: float
    updated_at: float


@dataclass(frozen=True)
class EvalRunResult:
    """Collection of batch summaries produced during evaluation."""

    batches: Sequence[EvalBatchSummary]

    @property
    def returns(self) -> List[float]:
        aggregated: List[float] = []
        for batch in self.batches:
            aggregated.extend(batch.returns)
        return aggregated


def run_batched_evaluation(
    evaluate_fn: Callable[..., Sequence[float]],
    *,
    policy_path: str,
    make_env: Callable,
    env_id: str,
    agent_cls,
    device,
    capture_video: bool,
    gamma: float,
    episodes_per_batch: int,
    repeat: bool,
    log_dir: Path | str,
    summary_path: Path | str,
    run_name_prefix: str,
    on_batch: Optional[Callable[[EvalBatchSummary], None]] = None,
    on_batch_start: Optional[Callable[[int], None]] = None,
) -> EvalRunResult:
    """Run evaluation in batches, writing TensorBoard + summary artifacts.

    Args:
        evaluate_fn: CleanRL evaluate helper.
        policy_path: Path to the serialized `.cleanrl_model` checkpoint.
        make_env: Callable returning a Gym env compatible with the agent.
        env_id: Environment identifier (e.g., ``Walker2d-v5``).
        agent_cls: CleanRL agent class used to load the policy.
        device: Torch device or device string.
        capture_video: Whether to emit mp4s via CleanRL's RecordVideo wrapper.
        gamma: Discount factor forwarded to the env factory/evaluator.
        episodes_per_batch: How many episodes to roll per evaluation batch.
        repeat: When True, keep running batches until cancelled.
        log_dir: Directory for TensorBoard event files.
        summary_path: JSON file updated with the latest batch statistics.
        run_name_prefix: Prefix forwarded to CleanRL's ``run_name`` argument so
            each batch stores artifacts in its own subfolder.
        on_batch: Optional callback invoked after each batch with the summary.
        on_batch_start: Optional callback invoked before each batch begins.
    """

    episodes_per_batch = max(1, int(episodes_per_batch))
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    summary_path = Path(summary_path)
    writer: Optional[SummaryWriter]
    if SummaryWriter is not None:  # pragma: no cover - tensorboard optional
        writer = SummaryWriter(log_dir=str(log_dir_path))
    else:  # pragma: no cover - best-effort fallback
        writer = None

    aggregated: List[EvalBatchSummary] = []
    batch_index = 0
    global_episode_offset = 0

    try:
        while True:
            if on_batch_start is not None:
                try:
                    on_batch_start(batch_index)
                except Exception:  # pragma: no cover - defensive callback isolation
                    LOGGER.exception(
                        "Policy eval batch start callback failed | batch=%s", batch_index
                    )
            LOGGER.info(
                "Policy eval invoking helper | batch=%s episodes=%s repeat=%s run=%s",
                batch_index,
                episodes_per_batch,
                repeat,
                run_name_prefix,
            )
            start = time.perf_counter()
            raw_returns = evaluate_fn(
                policy_path,
                make_env,
                env_id,
                eval_episodes=episodes_per_batch,
                run_name=f"{run_name_prefix}-eval-b{batch_index}",
                Model=agent_cls,
                device=device,
                capture_video=capture_video,
                gamma=gamma,
            )
            returns = [float(value) for value in raw_returns or []]
            duration = time.perf_counter() - start
            LOGGER.info(
                "Policy eval helper completed | batch=%s episodes=%s duration=%.2fs",
                batch_index,
                len(returns),
                duration,
            )
            if not returns:
                LOGGER.warning(
                    "Policy eval helper returned no episodes | batch=%s run=%s",
                    batch_index,
                    run_name_prefix,
                )
                break
            summary = _summarize_batch(batch_index, returns, duration)
            aggregated.append(summary)
            _write_summary_file(summary_path, summary)

            if writer is not None:
                _write_tensorboard_batch(writer, summary, global_episode_offset)

            if on_batch is not None:
                on_batch(summary)

            global_episode_offset += summary.episodes
            batch_index += 1

            if not repeat:
                break
    finally:
        if writer is not None:  # pragma: no cover - tensorboard optional
            writer.flush()
            writer.close()

    return EvalRunResult(tuple(aggregated))


def _summarize_batch(index: int, returns: Sequence[float], duration: float) -> EvalBatchSummary:
    avg_value = float(sum(returns) / len(returns))
    min_value = float(min(returns))
    max_value = float(max(returns))
    median_value = float(median(returns))
    std_value = float(pstdev(returns)) if len(returns) > 1 else 0.0
    return EvalBatchSummary(
        index=index,
        episodes=len(returns),
        returns=list(returns),
        avg_return=avg_value,
        min_return=min_value,
        max_return=max_value,
        median_return=median_value,
        std_return=std_value,
        duration_sec=float(duration),
        updated_at=time.time(),
    )


def _write_summary_file(path: Path, summary: EvalBatchSummary) -> None:
    payload = {
        "batch_index": summary.index,
        "episodes": summary.episodes,
        "avg_return": summary.avg_return,
        "min_return": summary.min_return,
        "max_return": summary.max_return,
        "median_return": summary.median_return,
        "std_return": summary.std_return,
        "duration_sec": summary.duration_sec,
        "updated_at": summary.updated_at,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:  # pragma: no cover - avoid crashing evaluation on I/O
        return


def _write_tensorboard_batch(
    writer: SummaryWriter,
    summary: EvalBatchSummary,
    global_episode_offset: int,
) -> None:
    for offset, value in enumerate(summary.returns):
        writer.add_scalar(
            "eval/episode_return",
            value,
            global_episode_offset + offset,
        )
    writer.add_scalar("eval/avg_return", summary.avg_return, summary.index)
    writer.add_scalar("eval/min_return", summary.min_return, summary.index)
    writer.add_scalar("eval/max_return", summary.max_return, summary.index)
    writer.add_scalar("eval/median_return", summary.median_return, summary.index)
    writer.add_scalar("eval/std_return", summary.std_return, summary.index)
    writer.add_scalar("eval/episodes", summary.episodes, summary.index)
    writer.add_scalar("eval/duration_sec", summary.duration_sec, summary.index)
    writer.flush()
