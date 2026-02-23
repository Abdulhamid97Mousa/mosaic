"""Optional dependency loaders for worker integrations.

This module centralizes all try/except import patterns for optional dependencies.
Each worker has a lazy loader that raises a helpful error if the dependency is
not installed.

Usage:
    from gym_gui.constants.optional_deps import get_mjpc_launcher, get_vizdoom_env

    # These will raise ImportError with installation instructions if not available
    launcher = get_mjpc_launcher()
    env_class = get_vizdoom_env()

Design:
    - Each optional dependency has a getter function that returns the actual import
    - If the import fails, a descriptive error is raised with installation instructions
    - TYPE_CHECKING blocks provide type hints without runtime imports
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    # These imports are only for type hints, not runtime
    pass

T = TypeVar("T")


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is not installed."""

    def __init__(self, package: str, install_cmd: str, extra_info: str = "") -> None:
        self.package = package
        self.install_cmd = install_cmd
        message = (
            f"{package} is not installed.\n"
            f"Install with: {install_cmd}"
        )
        if extra_info:
            message += f"\n{extra_info}"
        super().__init__(message)


# =============================================================================
# MuJoCo MPC Worker
# =============================================================================

_mjpc_launcher: Any = None
_mjpc_launcher_loaded: bool = False


def get_mjpc_launcher() -> Any:
    """Get the MuJoCo MPC launcher.

    Returns:
        MJPCLauncher instance from mujoco_mpc_worker

    Raises:
        OptionalDependencyError: If mujoco_mpc_worker is not installed
    """
    global _mjpc_launcher, _mjpc_launcher_loaded

    if _mjpc_launcher_loaded:
        if _mjpc_launcher is None:
            raise OptionalDependencyError(
                package="MuJoCo MPC worker",
                install_cmd="pip install -e 3rd_party/mujoco_mpc_worker",
                extra_info="Also requires building the MJPC agent_server binary. "
                "See: 3rd_party/mujoco_mpc_worker/mujoco_mpc/README.md"
            )
        return _mjpc_launcher()

    try:
        from mujoco_mpc_worker import get_launcher
        _mjpc_launcher = get_launcher
        _mjpc_launcher_loaded = True
        return _mjpc_launcher()
    except ImportError:
        _mjpc_launcher = None
        _mjpc_launcher_loaded = True
        raise OptionalDependencyError(
            package="MuJoCo MPC worker",
            install_cmd="pip install -e 3rd_party/mujoco_mpc_worker",
            extra_info="Also requires building the MJPC agent_server binary. "
            "See: 3rd_party/mujoco_mpc_worker/mujoco_mpc/README.md"
        )


def is_mjpc_available() -> bool:
    """Check if MuJoCo MPC worker is available without raising an error."""
    global _mjpc_launcher, _mjpc_launcher_loaded

    if _mjpc_launcher_loaded:
        return _mjpc_launcher is not None

    try:
        from mujoco_mpc_worker import get_launcher
        _mjpc_launcher = get_launcher
        _mjpc_launcher_loaded = True
        return True
    except ImportError:
        _mjpc_launcher = None
        _mjpc_launcher_loaded = True
        return False


# =============================================================================
# Godot Game Engine Worker
# =============================================================================

_godot_launcher: Any = None
_godot_launcher_loaded: bool = False


def get_godot_launcher() -> Any:
    """Get the Godot game engine launcher.

    Returns:
        GodotLauncher instance from godot_worker

    Raises:
        OptionalDependencyError: If godot_worker is not installed
    """
    global _godot_launcher, _godot_launcher_loaded

    if _godot_launcher_loaded:
        if _godot_launcher is None:
            raise OptionalDependencyError(
                package="Godot worker",
                install_cmd="pip install -e 3rd_party/godot_worker",
                extra_info="Also requires the Godot binary in 3rd_party/godot_worker/bin/"
            )
        return _godot_launcher()

    try:
        from godot_worker import get_launcher  # type: ignore[import-not-found]
        _godot_launcher = get_launcher
        _godot_launcher_loaded = True
        return _godot_launcher()
    except ImportError:
        _godot_launcher = None
        _godot_launcher_loaded = True
        raise OptionalDependencyError(
            package="Godot worker",
            install_cmd="pip install -e 3rd_party/godot_worker",
            extra_info="Also requires the Godot binary in 3rd_party/godot_worker/bin/"
        )


def is_godot_available() -> bool:
    """Check if Godot worker is available without raising an error."""
    global _godot_launcher, _godot_launcher_loaded

    if _godot_launcher_loaded:
        return _godot_launcher is not None

    try:
        from godot_worker import get_launcher  # type: ignore[import-not-found]
        _godot_launcher = get_launcher
        _godot_launcher_loaded = True
        return True
    except ImportError:
        _godot_launcher = None
        _godot_launcher_loaded = True
        return False


# =============================================================================
# ViZDoom Environment
# =============================================================================

_vizdoom_available: bool | None = None


def is_vizdoom_available() -> bool:
    """Check if ViZDoom is available without raising an error."""
    global _vizdoom_available

    if _vizdoom_available is not None:
        return _vizdoom_available

    try:
        import vizdoom  # noqa: F401
        _vizdoom_available = True
        return True
    except ImportError:
        _vizdoom_available = False
        return False


def require_vizdoom() -> None:
    """Require ViZDoom to be installed.

    Raises:
        OptionalDependencyError: If vizdoom is not installed
    """
    if not is_vizdoom_available():
        raise OptionalDependencyError(
            package="ViZDoom",
            install_cmd="pip install -r requirements/workers/vizdoom.txt",
            extra_info="ViZDoom requires additional system libraries. "
            "See: https://vizdoom.farama.org/introduction/installation/"
        )


# =============================================================================
# PettingZoo Multi-Agent Environments
# =============================================================================

_pettingzoo_available: bool | None = None


def is_pettingzoo_available() -> bool:
    """Check if PettingZoo is available without raising an error."""
    global _pettingzoo_available

    if _pettingzoo_available is not None:
        return _pettingzoo_available

    try:
        import pettingzoo  # noqa: F401
        _pettingzoo_available = True
        return True
    except ImportError:
        _pettingzoo_available = False
        return False


def require_pettingzoo() -> None:
    """Require PettingZoo to be installed.

    Raises:
        OptionalDependencyError: If pettingzoo is not installed
    """
    if not is_pettingzoo_available():
        raise OptionalDependencyError(
            package="PettingZoo",
            install_cmd="pip install -r requirements/workers/pettingzoo.txt",
        )


# =============================================================================
# Stockfish Chess Engine
# =============================================================================

_stockfish_available: bool | None = None


def is_stockfish_available() -> bool:
    """Check if Stockfish Python bindings are available without raising an error."""
    global _stockfish_available

    if _stockfish_available is not None:
        return _stockfish_available

    try:
        from stockfish import Stockfish  # noqa: F401
        _stockfish_available = True
        return True
    except ImportError:
        _stockfish_available = False
        return False


def require_stockfish() -> None:
    """Require Stockfish to be installed.

    Raises:
        OptionalDependencyError: If stockfish is not installed
    """
    if not is_stockfish_available():
        raise OptionalDependencyError(
            package="Stockfish",
            install_cmd="pip install stockfish && sudo apt install stockfish",
            extra_info="Stockfish requires both the Python bindings and the system binary."
        )


# =============================================================================
# CleanRL Worker
# =============================================================================

_cleanrl_available: bool | None = None


def is_cleanrl_available() -> bool:
    """Check if CleanRL worker is available without raising an error."""
    global _cleanrl_available

    if _cleanrl_available is not None:
        return _cleanrl_available

    try:
        import cleanrl_worker  # type: ignore[import-not-found]  # noqa: F401
        _cleanrl_available = True
        return True
    except ImportError:
        _cleanrl_available = False
        return False


def require_cleanrl() -> None:
    """Require CleanRL worker to be installed.

    Raises:
        OptionalDependencyError: If cleanrl_worker is not installed
    """
    if not is_cleanrl_available():
        raise OptionalDependencyError(
            package="CleanRL worker",
            install_cmd="pip install -e 3rd_party/cleanrl_worker && "
            "pip install -r requirements/workers/cleanrl.txt",
        )


# =============================================================================
# Torch (for training)
# =============================================================================

_torch_available: bool | None = None


def is_torch_available() -> bool:
    """Check if PyTorch is available without raising an error."""
    global _torch_available

    if _torch_available is not None:
        return _torch_available

    try:
        import torch  # noqa: F401
        _torch_available = True
        return True
    except ImportError:
        _torch_available = False
        return False


def require_torch() -> None:
    """Require PyTorch to be installed.

    Raises:
        OptionalDependencyError: If torch is not installed
    """
    if not is_torch_available():
        raise OptionalDependencyError(
            package="PyTorch",
            install_cmd="pip install torch",
            extra_info="For GPU support, see: https://pytorch.org/get-started/locally/"
        )


__all__ = [
    # Error class
    "OptionalDependencyError",
    # MuJoCo MPC
    "get_mjpc_launcher",
    "is_mjpc_available",
    # Godot
    "get_godot_launcher",
    "is_godot_available",
    # ViZDoom
    "is_vizdoom_available",
    "require_vizdoom",
    # PettingZoo
    "is_pettingzoo_available",
    "require_pettingzoo",
    # Stockfish
    "is_stockfish_available",
    "require_stockfish",
    # CleanRL
    "is_cleanrl_available",
    "require_cleanrl",
    # Torch
    "is_torch_available",
    "require_torch",
]
