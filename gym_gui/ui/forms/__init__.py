"""Worker form registration utilities."""

from .factory import WorkerFormFactory, get_worker_form_factory

# Ensure default worker forms register themselves with the factory.
# Import side-effects are intentional.
from gym_gui.ui.widgets import spade_bdi_worker_tabs  # noqa: F401
from gym_gui.ui.widgets.cleanrl_train_form import CleanRlTrainForm  # noqa: F401

__all__ = ["WorkerFormFactory", "get_worker_form_factory", "CleanRlTrainForm"]
