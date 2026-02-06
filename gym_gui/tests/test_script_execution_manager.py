"""Test Script Execution Manager - decoupled from Manual Mode.

Uses pytest-qt for idiomatic Qt signal testing (qtbot.waitSignal).
"""

import pytest

from gym_gui.services.operator_script_execution_manager import OperatorScriptExecutionManager
from gym_gui.services.operator import OperatorConfig


@pytest.fixture
def execution_manager(qapp):
    """Create execution manager instance.

    Uses pytest-qt's auto-provided qapp fixture (session-scoped QApplication).
    """
    manager = OperatorScriptExecutionManager()
    yield manager


def test_execution_manager_initialization(execution_manager):
    """Test that execution manager initializes correctly."""
    assert not execution_manager.is_running
    assert execution_manager.current_episode == 0
    assert execution_manager.total_episodes == 0


def test_start_experiment_new_operators(execution_manager):
    """Test starting experiment with new operators."""
    # Create operator configs
    configs = [
        OperatorConfig.single_agent(
            operator_id="random_minigrid",
            display_name="Random MiniGrid",
            worker_id="operators_worker",
            worker_type="baseline",
            env_name="minigrid",
            task="MiniGrid-Empty-8x8-v0",
            settings={"behavior": "random"},
            max_steps=100,
        )
    ]

    # Execution config
    execution_config = {
        "num_episodes": 3,
        "seeds": [1000, 1001, 1002]
    }

    # Track signal emissions
    launch_signals = []
    execution_manager.launch_operator.connect(
        lambda op_id, cfg, seed: launch_signals.append((op_id, seed))
    )

    # Start experiment
    execution_manager.start_experiment(configs, execution_config)

    # Should be running and have launched operators
    assert execution_manager.is_running
    assert execution_manager.total_episodes == 3
    assert execution_manager.current_episode == 1
    assert len(launch_signals) == 1
    assert launch_signals[0] == ("random_minigrid", 1000)


def test_on_ready_triggers_stepping(execution_manager):
    """Test that ready response triggers automatic stepping."""
    # Start experiment
    configs = [
        OperatorConfig.single_agent(
            operator_id="op1",
            display_name="Op 1",
            worker_id="operators_worker",
            worker_type="baseline",
            env_name="minigrid",
            task="MiniGrid-Empty-8x8-v0",
            settings={"behavior": "random"},
            max_steps=100,
        )
    ]
    execution_config = {"num_episodes": 2, "seeds": [1000, 1001]}
    execution_manager.start_experiment(configs, execution_config)

    # Track step signals
    step_signals = []
    execution_manager.step_operator.connect(
        lambda op_id: step_signals.append(op_id)
    )

    # Simulate ready response
    execution_manager.on_ready_received("op1")

    # Should have sent first step
    assert len(step_signals) == 1
    assert step_signals[0] == "op1"


def test_on_step_triggers_next_step(execution_manager, qtbot):
    """Test that step response triggers next step after pacing delay.

    on_step_received() uses QTimer.singleShot for pacing, so we use
    qtbot.waitSignal to let the Qt event loop process the deferred emit.
    """
    # Start experiment with a short pacing delay for fast testing
    configs = [
        OperatorConfig.single_agent(
            operator_id="op1",
            display_name="Op 1",
            worker_id="operators_worker",
            worker_type="baseline",
            env_name="minigrid",
            task="MiniGrid-Empty-8x8-v0",
            settings={"behavior": "random"},
            max_steps=100,
        )
    ]
    execution_config = {"num_episodes": 2, "seeds": [1000, 1001], "step_delay_ms": 10}
    execution_manager.start_experiment(configs, execution_config)

    # Use qtbot.waitSignal to block until the deferred step_operator fires
    with qtbot.waitSignal(execution_manager.step_operator, timeout=1000) as blocker:
        execution_manager.on_step_received("op1")

    # Verify the signal was emitted (not timed out) with correct args
    assert blocker.signal_triggered
    assert blocker.args == ["op1"]


def test_on_episode_ended_advances_to_next(execution_manager):
    """Test that episode end advances to next episode."""
    # Start experiment
    configs = [
        OperatorConfig.single_agent(
            operator_id="op1",
            display_name="Op 1",
            worker_id="operators_worker",
            worker_type="baseline",
            env_name="minigrid",
            task="MiniGrid-Empty-8x8-v0",
            settings={"behavior": "random"},
            max_steps=100,
        )
    ]
    execution_config = {"num_episodes": 3, "seeds": [1000, 1001, 1002]}
    execution_manager.start_experiment(configs, execution_config)

    # Track reset signals
    reset_signals = []
    execution_manager.reset_operator.connect(
        lambda op_id, seed: reset_signals.append((op_id, seed))
    )

    # Track progress signals
    progress_signals = []
    execution_manager.progress_updated.connect(
        lambda ep, total, seed: progress_signals.append((ep, seed))
    )

    # Simulate episode end
    execution_manager.on_episode_ended("op1", True, False)

    # Should advance to episode 2
    assert execution_manager.current_episode == 2
    assert len(reset_signals) == 1
    assert reset_signals[0] == ("op1", 1001)
    assert len(progress_signals) == 1
    assert progress_signals[0] == (2, 1001)


def test_experiment_completes_after_all_episodes(execution_manager):
    """Test that experiment completes after all episodes."""
    # Start experiment
    configs = [
        OperatorConfig.single_agent(
            operator_id="op1",
            display_name="Op 1",
            worker_id="operators_worker",
            worker_type="baseline",
            env_name="minigrid",
            task="MiniGrid-Empty-8x8-v0",
            settings={"behavior": "random"},
            max_steps=100,
        )
    ]
    execution_config = {"num_episodes": 2, "seeds": [1000, 1001]}
    execution_manager.start_experiment(configs, execution_config)

    # Track completion signal
    completion_signals = []
    execution_manager.experiment_completed.connect(
        lambda num: completion_signals.append(num)
    )

    # Complete first episode
    execution_manager.on_episode_ended("op1", True, False)
    assert execution_manager.is_running
    assert len(completion_signals) == 0

    # Complete second episode
    execution_manager.on_episode_ended("op1", True, False)
    assert not execution_manager.is_running
    assert len(completion_signals) == 1
    assert completion_signals[0] == 2


def test_stop_experiment(execution_manager):
    """Test stopping experiment."""
    # Start experiment
    configs = [
        OperatorConfig.single_agent(
            operator_id="op1",
            display_name="Op 1",
            worker_id="operators_worker",
            worker_type="baseline",
            env_name="minigrid",
            task="MiniGrid-Empty-8x8-v0",
            settings={"behavior": "random"},
            max_steps=100,
        )
    ]
    execution_config = {"num_episodes": 2, "seeds": [1000, 1001]}
    execution_manager.start_experiment(configs, execution_config)

    # Simulate operator becoming ready (marks it as running)
    execution_manager.on_ready_received("op1")

    # Track stop signals
    stop_signals = []
    execution_manager.stop_operator.connect(
        lambda op_id: stop_signals.append(op_id)
    )

    # Stop experiment
    execution_manager.stop_experiment()

    # Should stop running and emit stop signal
    assert not execution_manager.is_running
    assert len(stop_signals) == 1
    assert stop_signals[0] == "op1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
