"""Validation helpers for the agent training form UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from gym_gui.validations.validations_ui import IntRangeValidator
from gym_gui.constants import WORKER_ID_WIDTH


@dataclass(frozen=True)
class AgentTrainFormInputs:
    """Collected values from the training form used for validation."""

    episodes: int
    max_steps_per_episode: int
    seed: int
    learning_rate: str
    discount: str
    epsilon_decay: str
    agent_id: str
    worker_id: str


def validate_agent_train_form(inputs: AgentTrainFormInputs) -> List[str]:
    """Return a list of validation error messages for the training form."""

    errors: list[str] = []

    episodes_validator = IntRangeValidator(1, 1_000_000, "Episodes")
    episodes_result = episodes_validator.validate(str(inputs.episodes))
    if not episodes_result.is_valid:
        errors.append(episodes_result.message)

    max_steps_validator = IntRangeValidator(1, 100_000, "Max Steps per Episode")
    max_steps_result = max_steps_validator.validate(str(inputs.max_steps_per_episode))
    if not max_steps_result.is_valid:
        errors.append(max_steps_result.message)

    # Learning rate in (0, 1]
    lr_text = inputs.learning_rate.strip()
    if not lr_text:
        errors.append("Learning Rate cannot be empty")
    else:
        try:
            lr_val = float(lr_text)
            if lr_val <= 0 or lr_val > 1:
                errors.append(f"Learning Rate must be in (0, 1], got {lr_val}")
        except ValueError:
            errors.append(f"Learning Rate must be a number, got '{lr_text}'")

    # Discount in [0, 1)
    discount_text = inputs.discount.strip()
    if not discount_text:
        errors.append("Discount (γ) cannot be empty")
    else:
        try:
            discount_val = float(discount_text)
            if discount_val < 0 or discount_val >= 1:
                errors.append(f"Discount (γ) must be in [0, 1), got {discount_val}")
        except ValueError:
            errors.append(f"Discount (γ) must be a number, got '{discount_text}'")

    # Epsilon decay in (0, 1]
    epsilon_text = inputs.epsilon_decay.strip()
    if not epsilon_text:
        errors.append("Epsilon Decay cannot be empty")
    else:
        try:
            epsilon_val = float(epsilon_text)
            if epsilon_val <= 0 or epsilon_val > 1:
                errors.append(f"Epsilon Decay must be in (0, 1], got {epsilon_val}")
        except ValueError:
            errors.append(f"Epsilon Decay must be a number, got '{epsilon_text}'")

    seed_validator = IntRangeValidator(0, 999_999, "Random Seed")
    seed_result = seed_validator.validate(str(inputs.seed))
    if not seed_result.is_valid:
        errors.append(seed_result.message)

    agent_id = inputs.agent_id.strip()
    if not agent_id:
        errors.append("Agent ID cannot be empty")
    elif len(agent_id) > 256:
        errors.append("Agent ID must be at most 256 characters")

    worker_id = inputs.worker_id.strip()
    if not worker_id:
        errors.append("Worker ID cannot be empty")
    elif not worker_id.replace("_", "").replace("-", "").isalnum():
        errors.append("Worker ID must contain only letters, digits, hyphen, or underscore")
    elif len(worker_id) > WORKER_ID_WIDTH:
        errors.append(
            f"Worker ID must be at most {WORKER_ID_WIDTH} characters"
        )

    return errors


__all__ = ["AgentTrainFormInputs", "validate_agent_train_form"]
