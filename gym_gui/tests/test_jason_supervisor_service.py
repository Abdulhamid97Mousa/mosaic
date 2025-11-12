from __future__ import annotations

from gym_gui.constants.constants_jason_supervisor import DEFAULT_SUPERVISOR
from gym_gui.services.jason_supervisor import JasonSupervisorService
from gym_gui.validations.validations_telemetry import ValidationService


def _make_service() -> JasonSupervisorService:
    validator = ValidationService(strict_mode=False)
    svc = JasonSupervisorService(validator=validator)
    # ensure baseline
    assert svc.snapshot()["actions_emitted"] == 0
    return svc


def test_apply_control_update_accepts_and_updates_state():
    svc = _make_service()
    ok = svc.apply_control_update(
        {
            "run_id": "run_ok",
            "reason": "plateau_reanneal",
            "source": "jason_supervisor",
            "params": {"epsilon": 0.05},
            "available_credits": DEFAULT_SUPERVISOR.min_available_credits,
        }
    )
    assert ok is True
    snap = svc.snapshot()
    assert snap["last_action"] == "plateau_reanneal"
    assert snap["actions_emitted"] == 1


def test_apply_control_update_rejects_low_credits():
    svc = _make_service()
    # provide insufficient credits
    ok = svc.apply_control_update(
        {
            "run_id": "run_low_credit",
            "reason": "plateau_reanneal",
            "source": "jason_supervisor",
            "params": {"epsilon": 0.05},
            "available_credits": max(0, DEFAULT_SUPERVISOR.min_available_credits - 1),
        }
    )
    assert ok is False
    snap = svc.snapshot()
    # last_action remains default; no actions emitted
    assert snap["actions_emitted"] == 0


def test_apply_control_update_rejects_invalid_params():
    svc = _make_service()
    # epsilon outside [0,1]
    ok = svc.apply_control_update(
        {
            "run_id": "run_bad",
            "reason": "invalid_eps",
            "source": "jason_supervisor",
            "params": {"epsilon": 1.5},
            "available_credits": DEFAULT_SUPERVISOR.min_available_credits,
        }
    )
    assert ok is False
    snap = svc.snapshot()
    # last_error is recorded by the service on validation failure
    assert snap["last_error"] == "validation_failed"


def test_record_rollback_updates_last_action():
    svc = _make_service()
    svc.record_rollback(reason="negative_reward_streak")
    snap = svc.snapshot()
    assert "rollback:" in snap["last_action"]
