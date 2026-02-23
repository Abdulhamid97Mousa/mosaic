"""Factory for creating environment-specific grid configuration dialogs.

This module provides the GridConfigDialogFactory which creates the
appropriate configuration dialog for each supported grid environment.

New environments can be registered dynamically using the register() method.

Example:
    from gym_gui.ui.widgets.operators_grid_config_form import GridConfigDialogFactory

    # Check if environment is supported
    if GridConfigDialogFactory.supports("MiniGrid-Empty-5x5-v0"):
        dialog = GridConfigDialogFactory.create("MiniGrid-Empty-5x5-v0", initial_state, parent)
        if dialog.exec() == QDialog.Accepted:
            custom_state = dialog.get_state()

    # Register a new environment family
    GridConfigDialogFactory.register_family("Crafter", CrafterConfigDialog)
"""

import logging
import re
from functools import partial
from typing import Dict, Type, Optional, List, Tuple, Callable

from PyQt6 import QtWidgets

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_OP_GRID_CONFIG_FACTORY_CREATE,
    LOG_OP_GRID_CONFIG_UNSUPPORTED_ENV,
)
from .base import GridConfigDialog

_LOGGER = logging.getLogger(__name__)
_log = partial(log_constant, _LOGGER)


class GridConfigDialogFactory:
    """Factory to create environment-specific grid configuration dialogs.

    This factory maintains a registry of environment patterns to dialog classes,
    supporting both exact matches and pattern-based matching for environment families.

    Supported environments:
    - MiniGrid-*: All MiniGrid environments
    - BabyAI-*: All BabyAI environments
    - MultiGrid-*: Multi-agent grid environments
    - MeltingPot-*: Multi-agent social scenarios
    """

    # Registry: family_name -> (dialog_class, pattern_matcher)
    _registry: Dict[str, Tuple[Type[GridConfigDialog], Callable[[str], bool]]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Lazy initialization of the registry with built-in environments."""
        if cls._initialized:
            return

        # IMPORTANT: Set flag FIRST to prevent recursion
        cls._initialized = True

        # Register MiniGrid family
        cls._register_minigrid()

        # Register BabyAI family (language-grounded)
        cls._register_babyai()

        # Register MultiGrid family
        cls._register_multigrid()

        # Register MeltingPot family
        cls._register_meltingpot()

    @classmethod
    def _register_minigrid(cls) -> None:
        """Register MiniGrid environment family."""
        try:
            from .minigrid_editor import MiniGridConfigDialog

            pattern = r"^MiniGrid-.*$"
            compiled = re.compile(pattern)
            cls._registry["MiniGrid"] = (
                MiniGridConfigDialog,
                lambda env_id: bool(compiled.match(env_id))
            )
            _LOGGER.debug("Registered MiniGridConfigDialog for MiniGrid")
        except ImportError as e:
            _LOGGER.warning(f"Failed to register MiniGrid editor: {e}")

    @classmethod
    def _register_babyai(cls) -> None:
        """Register BabyAI language-grounded environment family."""
        try:
            from .babyai_editor import BabyAIConfigDialog

            pattern = r"^BabyAI-.*$"
            compiled = re.compile(pattern)
            cls._registry["BabyAI"] = (
                BabyAIConfigDialog,
                lambda env_id: bool(compiled.match(env_id))
            )
            _LOGGER.debug("Registered BabyAIConfigDialog for BabyAI")
        except ImportError as e:
            _LOGGER.debug(f"BabyAI editor not available: {e}")

    @classmethod
    def _register_multigrid(cls) -> None:
        """Register MultiGrid environment family."""
        try:
            from .multigrid_editor import MultiGridConfigDialog

            pattern = r"^(multigrid|MultiGrid|SoccerGame|CollectGame).*$"
            compiled = re.compile(pattern)
            cls._registry["MultiGrid"] = (
                MultiGridConfigDialog,
                lambda env_id: bool(compiled.match(env_id))
            )
            _LOGGER.debug("Registered MultiGridConfigDialog for MultiGrid")
        except ImportError as e:
            _LOGGER.debug(f"MultiGrid editor not available: {e}")

    @classmethod
    def _register_meltingpot(cls) -> None:
        """Register MeltingPot environment family."""
        try:
            from .meltingpot_editor import MeltingPotConfigDialog

            pattern = r"^(meltingpot|collaborative_cooking|clean_up|commons_harvest|territory|prisoners_dilemma|stag_hunt|allelopathic_harvest|king_of_the_hill).*$"
            compiled = re.compile(pattern)
            cls._registry["MeltingPot"] = (
                MeltingPotConfigDialog,
                lambda env_id: bool(compiled.match(env_id))
            )
            _LOGGER.debug("Registered MeltingPotConfigDialog for MeltingPot")
        except ImportError as e:
            _LOGGER.debug(f"MeltingPot editor not available: {e}")

    @classmethod
    def create(
        cls,
        env_id: str,
        initial_state: Optional[Dict] = None,
        parent: Optional[QtWidgets.QWidget] = None
    ) -> GridConfigDialog:
        """Create the appropriate dialog for the given environment.

        Args:
            env_id: Environment identifier (e.g., "MiniGrid-Empty-5x5-v0")
            initial_state: Optional initial state dictionary
            parent: Parent widget

        Returns:
            Environment-specific configuration dialog instance

        Raises:
            ValueError: If env_id is not supported
        """
        cls._ensure_initialized()

        dialog_class = cls._find_dialog_class(env_id)
        if dialog_class is None:
            supported = cls.get_supported_families()
            _log(
                LOG_OP_GRID_CONFIG_UNSUPPORTED_ENV,
                extra={"env_id": env_id, "supported_families": supported},
            )
            raise ValueError(
                f"No configuration dialog for environment '{env_id}'. "
                f"Supported families: {', '.join(supported) or 'none'}"
            )

        _log(
            LOG_OP_GRID_CONFIG_FACTORY_CREATE,
            extra={"env_id": env_id, "dialog_class": dialog_class.__name__},
        )
        return dialog_class(initial_state, parent, env_id=env_id)

    @classmethod
    def _find_dialog_class(cls, env_id: str) -> Optional[Type[GridConfigDialog]]:
        """Find the dialog class for an environment ID."""
        for family_name, (dialog_class, matcher) in cls._registry.items():
            if matcher(env_id):
                return dialog_class
        return None

    @classmethod
    def supports(cls, env_id: str) -> bool:
        """Check if an environment has a configuration dialog."""
        cls._ensure_initialized()
        return cls._find_dialog_class(env_id) is not None

    @classmethod
    def register_family(
        cls,
        family_name: str,
        dialog_class: Type[GridConfigDialog],
        pattern: str
    ) -> None:
        """Register a new environment family.

        This allows external modules to add support for new environments
        without modifying the factory code.

        Args:
            family_name: Family identifier (e.g., "Crafter")
            dialog_class: Dialog class implementing GridConfigDialog
            pattern: Regex pattern to match environment IDs
        """
        cls._ensure_initialized()

        compiled_pattern = re.compile(pattern)
        matcher = lambda env_id: bool(compiled_pattern.match(env_id))

        if family_name in cls._registry:
            _LOGGER.warning(
                f"Overwriting existing dialog for {family_name}: "
                f"{cls._registry[family_name][0].__name__} -> {dialog_class.__name__}"
            )

        cls._registry[family_name] = (dialog_class, matcher)
        _LOGGER.info(f"Registered {dialog_class.__name__} for {family_name}")

    @classmethod
    def unregister_family(cls, family_name: str) -> bool:
        """Unregister an environment family."""
        cls._ensure_initialized()

        if family_name in cls._registry:
            del cls._registry[family_name]
            _LOGGER.info(f"Unregistered dialog for {family_name}")
            return True
        return False

    @classmethod
    def get_supported_families(cls) -> List[str]:
        """Get list of supported environment families."""
        cls._ensure_initialized()
        return list(cls._registry.keys())

    @classmethod
    def get_dialog_class(cls, family_name: str) -> Optional[Type[GridConfigDialog]]:
        """Get the dialog class for a family without creating an instance."""
        cls._ensure_initialized()
        entry = cls._registry.get(family_name)
        return entry[0] if entry else None

    @classmethod
    def get_family_for_env(cls, env_id: str) -> Optional[str]:
        """Get the family name for an environment ID."""
        cls._ensure_initialized()
        for family_name, (_, matcher) in cls._registry.items():
            if matcher(env_id):
                return family_name
        return None
