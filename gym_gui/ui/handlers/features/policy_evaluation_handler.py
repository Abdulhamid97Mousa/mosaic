"""Policy evaluation handler for MainWindow.

Extracts policy evaluation logic from MainWindow into a dedicated handler.
Manages the EvaluationWorker QThread and result dialogs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QStatusBar, QWidget

if TYPE_CHECKING:
    from gym_gui.ui.widgets.render_tabs import RenderTabs

_LOGGER = logging.getLogger(__name__)


class EvaluationWorker(QThread):
    """Background worker for policy evaluation."""

    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        eval_run_id: str,
        env_id: str,
        env_family: str,
        checkpoint_path: str,
    ):
        super().__init__()
        self.eval_run_id = eval_run_id
        self.env_id = env_id
        self.env_family = env_family
        self.checkpoint_path = checkpoint_path

    def run(self):
        try:
            # Try to import the policy evaluator - it may not be available
            try:
                from ray_worker.policy_evaluator import (
                    PolicyEvaluator,
                    EvaluationConfig,
                )
            except ImportError:
                _LOGGER.warning(
                    "ray_worker.policy_evaluator not available - "
                    "policy evaluation requires the ray_worker package"
                )
                self.error_signal.emit(
                    "Policy evaluation not available: ray_worker.policy_evaluator module not found. "
                    "Please ensure the ray_worker package is properly installed."
                )
                return

            config = EvaluationConfig(
                env_id=self.env_id,
                env_family=self.env_family,
                checkpoint_path=self.checkpoint_path,
                run_id=self.eval_run_id,
                policy_id="shared",
                num_episodes=5,
                deterministic=True,
                fastlane_enabled=True,
                seed=42,
            )

            evaluator = PolicyEvaluator(config)
            evaluator.setup()
            evaluator.run()
            summary = evaluator.get_summary()
            evaluator.cleanup()

            self.finished_signal.emit(summary)

        except Exception as e:
            _LOGGER.exception("Evaluation failed: %s", e)
            self.error_signal.emit(str(e))


class PolicyEvaluationHandler:
    """Handler for policy evaluation requests.

    Manages:
    - Validation of evaluation configuration
    - Creation of FastLane tabs for visualization
    - Background evaluation worker lifecycle
    - Result dialogs and status messages
    """

    def __init__(
        self,
        parent: QWidget,
        status_bar: QStatusBar,
        open_ray_fastlane_tabs: Callable[..., None],
    ):
        """Initialize the handler.

        Args:
            parent: Parent widget for dialogs.
            status_bar: Status bar for messages.
            open_ray_fastlane_tabs: Callback to open FastLane tabs.
        """
        self._parent = parent
        self._status_bar = status_bar
        self._open_ray_fastlane_tabs = open_ray_fastlane_tabs
        self._evaluation_workers: List[EvaluationWorker] = []

    def handle_evaluate_request(self, config: dict) -> None:
        """Handle policy evaluation request from PolicyAssignmentPanel.

        Args:
            config: Evaluation configuration containing:
                - mode: "evaluate"
                - agent_policies: {agent_id: checkpoint_path}
                - policy_types: {agent_id: "ray" | "cleanrl" | "random"}
                - agents: list of agent IDs
                - env_id: environment ID
                - env_family: environment family
        """
        from ulid import ULID

        env_id = config.get("env_id", "unknown")
        env_family = config.get("env_family", "sisl")
        agent_policies = config.get("agent_policies", {})
        policy_types = config.get("policy_types", {})
        agents = config.get("agents", [])

        _LOGGER.info(
            "Policy evaluation requested: env=%s, agents=%s",
            env_id,
            agents,
        )

        # Find the first Ray checkpoint path to use
        checkpoint_path = None
        for agent_id, path in agent_policies.items():
            if path and policy_types.get(agent_id) == "ray":
                checkpoint_path = path
                break

        if not checkpoint_path:
            QMessageBox.warning(
                self._parent,
                "No Policy Selected",
                "Please select at least one Ray RLlib checkpoint for evaluation.\n"
                "Random policies cannot be visualized in evaluation mode.",
            )
            return

        # Generate evaluation run ID
        eval_run_id = str(ULID())

        # Log the policy assignments
        for agent_id, policy_path in agent_policies.items():
            policy_type = policy_types.get(agent_id, "unknown")
            if policy_path:
                _LOGGER.info(
                    "  %s: [%s] %s",
                    agent_id,
                    policy_type,
                    policy_path,
                )
            else:
                _LOGGER.info("  %s: Random (no policy)", agent_id)

        # Show status message
        num_policies = sum(1 for p in agent_policies.values() if p)
        self._status_bar.showMessage(
            f"Starting evaluation: {env_id} with {num_policies}/{len(agents)} policies",
            5000,
        )

        # Create FastLane tab for evaluation
        self._open_ray_fastlane_tabs(
            run_id=eval_run_id,
            agent_id="eval",
            run_mode="policy_eval",
            env_id=env_id,
            num_workers=0,
        )

        # Launch evaluation in background thread
        self._launch_evaluation(
            eval_run_id=eval_run_id,
            env_id=env_id,
            env_family=env_family,
            checkpoint_path=checkpoint_path,
        )

    def _launch_evaluation(
        self,
        eval_run_id: str,
        env_id: str,
        env_family: str,
        checkpoint_path: str,
    ) -> None:
        """Launch policy evaluation in a background thread."""
        worker = EvaluationWorker(
            eval_run_id=eval_run_id,
            env_id=env_id,
            env_family=env_family,
            checkpoint_path=checkpoint_path,
        )

        def on_finished(summary: dict):
            _LOGGER.info("Evaluation completed: %s", summary)
            results_dir = summary.get("results_dir", "")
            msg = (
                f"Evaluation complete: {summary.get('num_episodes', 0)} episodes, "
                f"mean reward: {summary.get('mean_reward', 0):.2f}"
            )
            if results_dir:
                msg += f" | Results: {results_dir}"
            self._status_bar.showMessage(msg, 15000)

            # Show results dialog with path to files
            if results_dir:
                QMessageBox.information(
                    self._parent,
                    "Evaluation Complete",
                    f"Evaluation finished successfully!\n\n"
                    f"Episodes: {summary.get('num_episodes', 0)}\n"
                    f"Mean Reward: {summary.get('mean_reward', 0):.2f}\n"
                    f"Std Reward: {summary.get('std_reward', 0):.2f}\n"
                    f"Min/Max: {summary.get('min_reward', 0):.2f} / {summary.get('max_reward', 0):.2f}\n"
                    f"Mean Length: {summary.get('mean_length', 0):.0f} steps\n"
                    f"Total Time: {summary.get('total_duration', 0):.1f}s\n\n"
                    f"Results saved to:\n{results_dir}\n\n"
                    f"Files:\n"
                    f"  - config.json (evaluation config)\n"
                    f"  - summary.json (summary statistics)\n"
                    f"  - episodes.csv (per-episode metrics)\n"
                    f"  - agents.csv (per-agent breakdown)",
                )

        def on_error(error: str):
            _LOGGER.error("Evaluation error: %s", error)
            self._status_bar.showMessage(f"Evaluation failed: {error}", 10000)
            QMessageBox.critical(
                self._parent,
                "Evaluation Error",
                f"Policy evaluation failed:\n\n{error}",
            )

        worker.finished_signal.connect(on_finished)
        worker.error_signal.connect(on_error)

        # Store reference to prevent garbage collection
        self._evaluation_workers.append(worker)

        worker.start()
        _LOGGER.info("Evaluation worker started: run_id=%s", eval_run_id)
