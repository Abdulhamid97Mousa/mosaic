"""MOSAIC Ray worker site customizations.

This module is auto-imported by Python when present on PYTHONPATH. It patches
PettingZoo environment factories to automatically wrap with FastLane when
enabled, similar to how CleanRL's sitecustomize patches gymnasium.make().

Environment Variables:
    RAY_FASTLANE_ENABLED: Set to "1" to enable FastLane wrapping
    RAY_FASTLANE_RUN_ID: Unique identifier for the training run
    RAY_FASTLANE_ENV_NAME: Environment name for tab title
    RAY_FASTLANE_THROTTLE_MS: Minimum ms between frame publishes

Usage:
    # Set env vars before importing pettingzoo
    os.environ["RAY_FASTLANE_ENABLED"] = "1"
    os.environ["RAY_FASTLANE_RUN_ID"] = "waterworld_ppo_001"
    os.environ["RAY_FASTLANE_ENV_NAME"] = "Waterworld"

    # Now any PettingZoo env will be wrapped
    from pettingzoo.sisl import waterworld_v4
    env = waterworld_v4.env()  # Automatically wrapped with FastLane
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Import FastLane helpers
try:
    from .fastlane import is_fastlane_enabled, maybe_wrap_env
except ImportError:
    try:
        from ray_worker.fastlane import is_fastlane_enabled, maybe_wrap_env
    except ImportError:
        def is_fastlane_enabled() -> bool:
            return False

        def maybe_wrap_env(env: Any) -> Any:
            return env


# --- WandB defaults (same pattern as CleanRL) ---
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB__SERVICE", "disabled")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))


# --- Ray defaults ---
# Suppress some Ray warnings
os.environ.setdefault("RAY_DEDUP_LOGS", "0")


# --- PettingZoo environment patching ---

# Registry of patched modules to avoid double-patching
_PATCHED_MODULES: Dict[str, bool] = {}


def _create_wrapper(original_fn: Callable, module_name: str) -> Callable:
    """Create a wrapper function that applies FastLane wrapping.

    Args:
        original_fn: Original environment factory function
        module_name: Name of the module being patched

    Returns:
        Wrapped function that applies FastLane if enabled
    """
    def wrapped_env_fn(*args, **kwargs) -> Any:
        env = original_fn(*args, **kwargs)
        if is_fastlane_enabled():
            return maybe_wrap_env(env)
        return env

    # Preserve function metadata
    wrapped_env_fn.__name__ = original_fn.__name__
    wrapped_env_fn.__doc__ = original_fn.__doc__
    wrapped_env_fn._mosaic_wrapped = True
    wrapped_env_fn._mosaic_original = original_fn

    return wrapped_env_fn


def _patch_pettingzoo_module(module: Any, module_name: str) -> None:
    """Patch a PettingZoo environment module to wrap with FastLane.

    Args:
        module: The PettingZoo module (e.g., pettingzoo.sisl.waterworld_v4)
        module_name: Full module name for tracking
    """
    if module_name in _PATCHED_MODULES:
        return

    # Patch the .env() function
    if hasattr(module, 'env') and callable(module.env):
        if not getattr(module.env, '_mosaic_wrapped', False):
            original_env = module.env
            module.env = _create_wrapper(original_env, module_name)
            _PATCHED_MODULES[module_name] = True
            _log_debug(f"Patched {module_name}.env()")

    # Also patch .parallel_env() if it exists
    if hasattr(module, 'parallel_env') and callable(module.parallel_env):
        if not getattr(module.parallel_env, '_mosaic_wrapped', False):
            original_parallel = module.parallel_env
            module.parallel_env = _create_wrapper(original_parallel, module_name)
            _log_debug(f"Patched {module_name}.parallel_env()")


def _log_debug(message: str) -> None:
    """Log debug message if verbose mode enabled."""
    if os.getenv("RAY_FASTLANE_VERBOSE", "").lower() in {"1", "true", "yes"}:
        print(f"[RAY-SITECUSTOMIZE] {message}", file=sys.stderr, flush=True)


# --- Patch SISL environments ---
try:
    from pettingzoo.sisl import waterworld_v4
    _patch_pettingzoo_module(waterworld_v4, "pettingzoo.sisl.waterworld_v4")
except ImportError:
    pass

try:
    from pettingzoo.sisl import multiwalker_v9
    _patch_pettingzoo_module(multiwalker_v9, "pettingzoo.sisl.multiwalker_v9")
except ImportError:
    pass

try:
    from pettingzoo.sisl import pursuit_v4
    _patch_pettingzoo_module(pursuit_v4, "pettingzoo.sisl.pursuit_v4")
except ImportError:
    pass


# --- Patch Classic environments ---
try:
    from pettingzoo.classic import chess_v6
    _patch_pettingzoo_module(chess_v6, "pettingzoo.classic.chess_v6")
except ImportError:
    pass

try:
    from pettingzoo.classic import go_v5
    _patch_pettingzoo_module(go_v5, "pettingzoo.classic.go_v5")
except ImportError:
    pass

try:
    from pettingzoo.classic import connect_four_v3
    _patch_pettingzoo_module(connect_four_v3, "pettingzoo.classic.connect_four_v3")
except ImportError:
    pass

try:
    from pettingzoo.classic import tictactoe_v3
    _patch_pettingzoo_module(tictactoe_v3, "pettingzoo.classic.tictactoe_v3")
except ImportError:
    pass


# --- Patch Butterfly environments ---
try:
    from pettingzoo.butterfly import knights_archers_zombies_v10
    _patch_pettingzoo_module(knights_archers_zombies_v10, "pettingzoo.butterfly.knights_archers_zombies_v10")
except ImportError:
    pass

try:
    from pettingzoo.butterfly import cooperative_pong_v5
    _patch_pettingzoo_module(cooperative_pong_v5, "pettingzoo.butterfly.cooperative_pong_v5")
except ImportError:
    pass

try:
    from pettingzoo.butterfly import pistonball_v6
    _patch_pettingzoo_module(pistonball_v6, "pettingzoo.butterfly.pistonball_v6")
except ImportError:
    pass


# --- Patch MPE environments ---
try:
    from pettingzoo.mpe import simple_spread_v3
    _patch_pettingzoo_module(simple_spread_v3, "pettingzoo.mpe.simple_spread_v3")
except ImportError:
    pass

try:
    from pettingzoo.mpe import simple_adversary_v3
    _patch_pettingzoo_module(simple_adversary_v3, "pettingzoo.mpe.simple_adversary_v3")
except ImportError:
    pass

try:
    from pettingzoo.mpe import simple_tag_v3
    _patch_pettingzoo_module(simple_tag_v3, "pettingzoo.mpe.simple_tag_v3")
except ImportError:
    pass


# --- TensorBoard log redirection ---
try:
    from torch.utils import tensorboard as _tb_pkg
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
    from torch.utils.tensorboard import writer as _tb_writer_mod

    def _resolve_tensorboard_logdir(root: str, log_dir: str | os.PathLike | None) -> str:
        base = Path(root)
        if not log_dir:
            return str(base)
        leaf = Path(log_dir).name or "events"
        return str(base / leaf)

    class _MosaicSummaryWriter(_TorchSummaryWriter):
        def __init__(self, log_dir=None, *args, **kwargs):
            override_root = os.getenv("RAY_TENSORBOARD_DIR")
            if override_root:
                log_dir = _resolve_tensorboard_logdir(override_root, log_dir)
            super().__init__(log_dir=log_dir, *args, **kwargs)

    _tb_writer_mod.SummaryWriter = _MosaicSummaryWriter
    _tb_pkg.SummaryWriter = _MosaicSummaryWriter
except Exception:
    pass


# --- Torch save directory auto-creation ---
try:
    import torch

    _ORIG_TORCH_SAVE = torch.save

    def _mosaic_torch_save(obj, f, *args, **kwargs):
        if isinstance(f, (str, Path)):
            Path(f).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return _ORIG_TORCH_SAVE(obj, f, *args, **kwargs)

    torch.save = _mosaic_torch_save
except Exception:
    pass


# --- Public API ---

def patch_pettingzoo_env(module: Any, module_name: Optional[str] = None) -> None:
    """Manually patch a PettingZoo environment module.

    Use this to patch additional environments not covered by auto-patching.

    Args:
        module: The PettingZoo module to patch
        module_name: Optional name for logging (defaults to module.__name__)

    Example:
        from pettingzoo.atari import pong_v3
        from ray_worker.sitecustomize import patch_pettingzoo_env
        patch_pettingzoo_env(pong_v3)
    """
    name = module_name or getattr(module, '__name__', str(module))
    _patch_pettingzoo_module(module, name)


def get_patched_modules() -> Dict[str, bool]:
    """Get list of modules that have been patched.

    Returns:
        Dictionary of module names to patch status
    """
    return dict(_PATCHED_MODULES)


__all__ = [
    "patch_pettingzoo_env",
    "get_patched_modules",
]
