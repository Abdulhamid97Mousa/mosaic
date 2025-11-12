"""Default configuration constants for the Jason-based supervisor.

These values govern plateau detection, safety thresholds, and control
backoff intervals used when the Jason Supervisor proposes training
parameter adjustments (e.g. epsilon re-anneal, PER alpha/beta tweaks).

The constants are intentionally conservative; higher-risk tuning (like
aggressive learning-rate decay) should remain opt-in until sufficient
telemetry confidence accrues.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SupervisorDefaults:
	# Plateau detection
	reward_plateau_window: int = 25  # episodes to average for plateau detection
	reward_plateau_delta: float = 0.01  # min improvement threshold to exit plateau
	max_plateau_actions: int = 3  # max consecutive control actions during a plateau

	# Safety thresholds
	max_negative_reward_streak: int = 5  # consecutive episodes with net negative reward before rollback
	variance_spike_multiplier: float = 3.0  # stddev multiplier considered a volatility spike
	max_rollback_depth: int = 2  # number of parameter layers to rollback

	# Control cadence
	min_control_interval_episodes: int = 10  # minimum episodes between control updates
	cooldown_after_rollback_episodes: int = 15  # enforced cooldown after a rollback

	# Exploration (epsilon) tuning bounds
	epsilon_min: float = 0.01
	epsilon_max: float = 0.9
	epsilon_anneal_factor: float = 0.85  # multiplicative decay when tightening exploration
	epsilon_recover_factor: float = 1.10  # multiplicative increase when exploration deemed too low

	# PER (Prioritized Experience Replay) tuning bounds
	per_alpha_min: float = 0.4
	per_alpha_max: float = 0.9
	per_beta_min: float = 0.4
	per_beta_max: float = 1.0
	per_alpha_step: float = 0.05  # step size when adjusting alpha
	per_beta_step: float = 0.05  # step size when adjusting beta

	# Learning rate tuning (conservative bounds)
	lr_min_multiplier: float = 0.2  # relative to initial LR
	lr_max_multiplier: float = 1.0
	lr_decay_factor: float = 0.5  # applied when decaying
	lr_recover_factor: float = 1.25  # applied when recovering (capped by max_multiplier)

	# Target network update (tau) tuning
	tau_min: float = 0.001
	tau_max: float = 0.02
	tau_step: float = 0.001

	# Credit / backpressure awareness (integration with TelemetryAsyncHub)
	min_available_credits: int = 4  # do not emit control updates if credits below this

	# Logging sampling limits
	max_log_control_events: int = 100  # bound number of emitted control events


DEFAULT_SUPERVISOR = SupervisorDefaults()

__all__ = ["SupervisorDefaults", "DEFAULT_SUPERVISOR"]
