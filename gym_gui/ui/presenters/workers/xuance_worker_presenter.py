"""Presenter for XuanCe worker analytics lane."""

from __future__ import annotations

from typing import Any, List, Optional


class XuanCeWorkerPresenter:
    """Placeholder presenter for the XuanCe analytics worker.

    The XuanCe worker provides 46+ RL algorithms for single-agent and multi-agent
    training. This presenter is a placeholder until full analytics integration lands.

    Training runs originate from the XuanCe training form, and analytics tabs will be
    implemented once manifest ingestion is complete.

    Supported algorithms:
    - Single-agent: DQN, PPO, SAC, TD3, DDPG, DreamerV3
    - Multi-agent: MAPPO, QMIX, MADDPG, VDN, COMA, IAC

    Supported environments:
    - Gymnasium (classic_control, atari, mujoco, box2d)
    - PettingZoo (MPE, SISL)
    - SMAC (StarCraft Multi-Agent Challenge)
    - Google Football
    """

    @property
    def id(self) -> str:
        return "xuance_worker"

    def build_train_request(self, policy_path: Any, current_game: Optional[Any]) -> dict:
        """Policy evaluation is not yet supported for XuanCe workers.

        Policy loading and evaluation will be implemented in Phase 6.
        """
        raise NotImplementedError("XuanCe worker does not support policy evaluation yet.")

    def create_tabs(self, run_id: str, agent_id: str, first_payload: dict, parent: Any) -> List[Any]:
        """Analytics tabs will be attached post-run once manifest ingestion lands.

        TensorBoard and WandB integration will be available after training completes.
        """
        return []


__all__ = ["XuanCeWorkerPresenter"]
