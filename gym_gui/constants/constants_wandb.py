"""Weights & Biases (W&B) defaults and helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WandbDefaults:
    """Runtime defaults for presenting W&B analytics inside the GUI."""

    app_base_url: str = "https://wandb.ai"
    api_base_url: str = "https://api.wandb.ai"
    status_probe_timeout_s: float = 3.0
    status_probe_interval_ms: int = 5000


DEFAULT_WANDB = WandbDefaults()


def build_wandb_run_url(run_path: str, *, base_url: str | None = None) -> str:
    """Construct a canonical W&B web URL from a run path.

    Args:
        run_path: W&B run path in the form ``entity/project/runs/<run_id>``.
        base_url: Optional base URL override (defaults to DEFAULT_WANDB.app_base_url).

    Returns:
        Fully-qualified URL string that can be opened in a browser.
    """

    base = (base_url or DEFAULT_WANDB.app_base_url).rstrip("/")
    cleaned = run_path.strip().strip("/")
    return f"{base}/{cleaned}"


__all__ = [
    "DEFAULT_WANDB",
    "WandbDefaults",
    "build_wandb_run_url",
]
