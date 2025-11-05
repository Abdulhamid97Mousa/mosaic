"""Presenter for CleanRL worker analytics lane."""

from __future__ import annotations

from typing import Any, List, Optional


class CleanRlWorkerPresenter:
    """Placeholder presenter for the CleanRL analytics worker.

    The CleanRL worker does not yet support policy evaluation or live telemetry tabs.
    Training runs originate from the CleanRlTrainForm, and analytics tabs will be
    implemented once manifest ingestion lands.
    """

    @property
    def id(self) -> str:
        return "cleanrl_worker"

    def build_train_request(self, policy_path: Any, current_game: Optional[Any]) -> dict:
        """Policy evaluation is not yet supported for CleanRL workers."""
        raise NotImplementedError("CleanRL worker does not support policy evaluation yet.")

    def create_tabs(self, run_id: str, agent_id: str, first_payload: dict, parent: Any) -> List[Any]:
        """Analytics tabs will be attached post-run once manifest ingestion lands."""
        return []


__all__ = ["CleanRlWorkerPresenter"]
