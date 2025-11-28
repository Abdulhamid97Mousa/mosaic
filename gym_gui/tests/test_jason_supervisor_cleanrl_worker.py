from __future__ import annotations

from gym_gui.services.actor import CleanRLWorkerActor, StepSnapshot, EpisodeSummary
from gym_gui.workers.jason_supervisor_cleanrl_worker import (
    JasonSupervisorCleanRLWorkerActor,
)


def test_actor_protocol_minimal():
    actor = JasonSupervisorCleanRLWorkerActor()
    assert isinstance(actor, CleanRLWorkerActor)
    assert actor.id == "cleanrl_worker"

    # select_action abstains (CleanRL decides externally)
    step = StepSnapshot(
        step_index=0,
        observation=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        seed=None,
        info={},
    )
    assert actor.select_action(step) is None

    # on_step and on_episode_end shouldn't raise
    actor.on_step(step)

    ep = EpisodeSummary(
        episode_index=0,
        total_reward=0.0,
        steps=1,
        metadata={},
    )
    actor.on_episode_end(ep)
