"""Shared validation utilities across MOSAIC.

This package centralizes validation logic that was previously scattered across
core, services, and UI modules. Submodules are organized by domain:

- validations_pydantic: Pydantic models for telemetry events and configs
- validations_telemetry: Runtime validation service for telemetry payloads
- validations_ui: Qt-oriented validators and widgets
- validations_agent_train_form: High-level helpers for the training form

Additional validation helpers can be added here to keep the project
maintainable and consistent.
"""

from gym_gui.validations.validations_pydantic import *  # noqa: F401,F403
from gym_gui.validations.validations_telemetry import *  # noqa: F401,F403
from gym_gui.validations.validations_ui import *  # noqa: F401,F403
from gym_gui.validations.validations_agent_train_form import *  # noqa: F401,F403

__all__ = []  # populated by star imports above
