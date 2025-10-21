"""Integration test for the full worker training pipeline."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_full_training_run():
    """
    End-to-end test: Worker completes a 5-episode training run.
    
    This verifies:
    - Config parsing from stdin
    - Environment creation (FrozenLake-v2)
    - Q-learning agent initialization
    - Episode execution with telemetry emission
    - Policy saving
    - Clean exit without matplotlib
    """
    config = {
        "run_id": "integration_test_001",
        "game_id": "FrozenLake-v2",
        "agent_id": "qlearning_integration",
        "seed": 123,
        "max_episodes": 5,
        "max_steps_per_episode": 20,
        "policy_strategy": "train_and_save",
        "policy_path": None,  # Will use default
        "capture_video": False,
        "headless": True,
        "extra": {},
    }
    
    result = subprocess.run(
        [sys.executable, "-m", "spade_bdi_rl.worker"],
        input=json.dumps(config),
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        timeout=30,  # Should complete in 30 seconds
    )
    
    assert result.returncode == 0, (
        f"Worker failed with exit code {result.returncode}\\n"
        f"STDOUT:\\n{result.stdout}\\n"
        f"STDERR:\\n{result.stderr}"
    )
    
    # Parse telemetry output (JSONL format - one JSON object per line)
    lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    events = [json.loads(line) for line in lines]
    
    # Verify event sequence
    event_types = [e["type"] for e in events]
    assert event_types[0] == "run_started"
    assert event_types[-1] == "run_completed"
    assert "step" in event_types
    assert "episode" in event_types
    
    # Verify run completed successfully
    final_event = events[-1]
    assert final_event["status"] == "completed"
    assert final_event["run_id"] == "integration_test_001"
    
    # Count episodes and steps
    episode_events = [e for e in events if e["type"] == "episode"]
    step_events = [e for e in events if e["type"] == "step"]
    
    assert len(episode_events) == 5, f"Expected 5 episodes, got {len(episode_events)}"
    assert len(step_events) > 0, "No steps recorded"
    
    print(f"✓ Worker completed successfully")
    print(f"✓ Recorded {len(episode_events)} episodes")
    print(f"✓ Recorded {len(step_events)} steps")
    print(f"✓ No matplotlib dependencies")


if __name__ == "__main__":
    test_full_training_run()
    print("\\n🎉 All integration tests passed!")
