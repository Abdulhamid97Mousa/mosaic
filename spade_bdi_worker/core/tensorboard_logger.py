"""TensorBoard integration helpers for the SPADE-BDI worker runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Optional

from gym_gui.config.paths import VAR_TENSORBOARD_DIR, ensure_var_directories


def _resolve_summary_writer() -> Optional[Callable[..., Any]]:
    """Best-effort import for a TensorBoard ``SummaryWriter`` implementation."""
    try:
        from torch.utils.tensorboard import SummaryWriter as torch_summary_writer  # type: ignore

        return torch_summary_writer
    except Exception:  # noqa: BLE001
        pass

    try:
        from tensorboardX import SummaryWriter as tensorboardx_summary_writer  # type: ignore

        return tensorboardx_summary_writer
    except Exception:  # noqa: BLE001
        return None


@dataclass(slots=True)
class TensorboardLogger:
    """Simple wrapper around a TensorBoard ``SummaryWriter``."""

    _writer: Any
    log_dir: Path

    @classmethod
    def from_run_config(cls, config: "RunConfig") -> Optional["TensorboardLogger"]:
        """Instantiate a logger for the provided run configuration.

        Returns ``None`` when tensorboard logging is explicitly disabled or when
        no compatible ``SummaryWriter`` implementation can be imported.
        """

        extra = config.extra
        tensorboard_settings = extra.get("tensorboard")
        if isinstance(tensorboard_settings, Mapping):
            disabled = bool(tensorboard_settings.get("disabled"))
            requested_dir = tensorboard_settings.get("log_dir")
        else:
            tensorboard_settings = {}
            disabled = False
            requested_dir = None

        if bool(extra.get("disable_tensorboard")) or disabled:
            return None

        summary_writer_factory = _resolve_summary_writer()
        if summary_writer_factory is None:
            raise RuntimeError(
                "TensorBoard integration requested but no SummaryWriter implementation is available. "
                "Install torch>=1.4 or tensorboardX."
            )

        ensure_var_directories()
        run_dir = VAR_TENSORBOARD_DIR / config.run_id
        if requested_dir:
            log_dir = Path(str(requested_dir)).expanduser().resolve()
        else:
            log_dir = (run_dir / "tensorboard").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        writer = summary_writer_factory(log_dir=str(log_dir))

        settings_dict = dict(tensorboard_settings)
        settings_dict.update(
            {
                "enabled": True,
                "log_dir": str(log_dir),
                "relative_path": f"var/trainer/runs/{config.run_id}/tensorboard",
            }
        )
        extra["tensorboard"] = settings_dict

        return cls(writer, log_dir)

    def on_run_started(self, config_payload: Mapping[str, Any]) -> None:
        """Log initial run metadata to TensorBoard."""
        try:
            payload_json = json.dumps(config_payload, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            payload_json = json.dumps({"error": "config_serialization_failed"})
        self._writer.add_text("run/config", payload_json)
        schema_id = config_payload.get("schema_id")
        if schema_id is not None:
            self._writer.add_text("run/schema_id", str(schema_id))
        schema_version = config_payload.get("schema_version")
        if schema_version is not None:
            try:
                version_value = int(schema_version)
                self._writer.add_scalar("run/schema_version", version_value, 0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                self._writer.add_text("run/schema_version", str(schema_version))
        self._writer.flush()

    def log_episode(
        self,
        *,
        episode_number: int,
        reward: float,
        steps: int,
        epsilon: float,
        success: bool,
    ) -> None:
        """Record episode-level metrics."""
        self._writer.add_scalar("episode/return", reward, episode_number)
        self._writer.add_scalar("episode/length", steps, episode_number)
        self._writer.add_scalar("exploration/epsilon", epsilon, episode_number)
        self._writer.add_scalar("episode/success", 1.0 if success else 0.0, episode_number)
        self._writer.flush()

    def log_run_summary(self, summaries: Iterable[Any]) -> None:
        """Aggregate episode metrics for run-level dashboards."""
        materialized = list(summaries)
        if not materialized:
            return

        returns = [float(getattr(item, "total_reward", 0.0)) for item in materialized]
        lengths = [int(getattr(item, "steps", 0)) for item in materialized]
        successes = [1.0 if getattr(item, "success", False) else 0.0 for item in materialized]

        self._writer.add_scalar("run/avg_return", mean(returns), len(materialized))
        self._writer.add_scalar("run/max_return", max(returns), len(materialized))
        self._writer.add_scalar("run/avg_length", mean(lengths), len(materialized))
        self._writer.add_scalar("run/success_rate", mean(successes), len(materialized))
        self._writer.flush()

    def on_run_completed(self, *, status: str, error: Optional[str] = None) -> None:
        """Persist terminal status markers."""
        self._writer.add_text("run/status", status)
        if error:
            self._writer.add_text("run/error", error)
        self._writer.flush()

    def close(self) -> None:
        """Release the underlying writer resources."""
        try:
            self._writer.flush()
        finally:
            self._writer.close()


# Avoid circular import for type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import RunConfig
