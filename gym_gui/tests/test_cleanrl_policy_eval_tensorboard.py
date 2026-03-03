import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "3rd_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from cleanrl_worker.eval import run_batched_evaluation

MODEL_PATH = (
    ROOT / "var/trainer/runs/01KAW8JYRDJZDG1G9G29PYGZ91/"
    "runs/Walker2d-v5__ppo_continuous_action__1__1764032618/ppo_continuous_action.cleanrl_model"
).resolve()

LOG_DIR = (
    ROOT / "var/trainer/evals/01KAWCW4PZP8R9BBQD8F595WST/tensorboard"
).resolve()

RUN_DIR = LOG_DIR.parent


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="reference policy checkpoint not found")
def test_cleanrl_policy_eval_populates_tensorboard(tmp_path):
    """Run a lightweight eval pass and verify TensorBoard + summary artifacts."""

    existing_event_files = set(LOG_DIR.glob("events.out.tfevents.*"))
    summary_path = RUN_DIR / "eval_summary_pytest.json"
    if summary_path.exists():
        summary_path.unlink()

    def fake_evaluate(
        model_path,
        make_env,
        env_id,
        eval_episodes,
        run_name,
        Model,
        device,
        capture_video,
        gamma,
    ):
        # Load the real checkpoint to ensure it is readable.
        state_dict = torch.load(model_path, map_location=device)
        assert isinstance(state_dict, dict)
        # return deterministic pseudo evaluations
        return [float(index + 1) for index in range(eval_episodes)]

    class _DummyAgent:
        pass

    result = run_batched_evaluation(
        fake_evaluate,
        policy_path=str(MODEL_PATH),
        make_env=lambda *args, **kwargs: None,
        env_id="Walker2d-v5",
        agent_cls=_DummyAgent,
        device=torch.device("cpu"),
        capture_video=False,
        gamma=0.99,
        episodes_per_batch=2,
        repeat=False,
        log_dir=LOG_DIR,
        summary_path=summary_path,
        run_name_prefix="pytest-cleanrl-eval",
    )

    assert result.returns, "run_batched_evaluation should report aggregated batches"

    current_event_files = set(LOG_DIR.glob("events.out.tfevents.*"))
    new_event_files = current_event_files - existing_event_files
    assert new_event_files, "expected a new TensorBoard event file to be created"
    for event_file in new_event_files:
        assert event_file.stat().st_size > 256, "event file should contain scalar data"

    assert summary_path.exists(), "eval_summary file was not written"
    payload = json.loads(summary_path.read_text())
    assert payload["episodes"] == 2
    assert pytest.approx(payload["avg_return"], rel=1e-5) == 1.5
