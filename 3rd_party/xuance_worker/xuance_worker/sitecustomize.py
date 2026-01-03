"""MOSAIC XuanCe site customizations.

This module is auto-imported by Python when present on PYTHONPATH. We use it to
set sane defaults and provide optional hooks without patching the vendored
XuanCe sources.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- WANDB defaults -------------------------------------------------------
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB__SERVICE", "disabled")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))

try:  # pragma: no cover - wandb optional
    import wandb

    _ORIG_INIT = wandb.init

    def _patched_init(*args, **kwargs):
        kwargs.setdefault("reinit", True)
        if "project" not in kwargs:
            default_project = os.getenv("WANDB_PROJECT") or os.getenv("WANDB_PROJECT_NAME")
            if default_project:
                kwargs["project"] = default_project
        if "entity" not in kwargs:
            default_entity = os.getenv("WANDB_ENTITY") or os.getenv("WANDB_ENTITY_NAME")
            if default_entity:
                kwargs["entity"] = default_entity
        if "name" not in kwargs:
            default_name = os.getenv("WANDB_NAME")
            if default_name:
                kwargs["name"] = default_name
        return _ORIG_INIT(*args, **kwargs)

    wandb.init = _patched_init
except Exception:  # pragma: no cover - wandb optional
    pass

# --- Gymnasium environment wrapper ----------------------------------------
try:  # pragma: no cover - gym optional
    import gymnasium as gym

    if hasattr(gym.make, "_mosaic_wrapped"):
        _ORIG_MAKE = getattr(gym.make, "_mosaic_orig_make")
    else:
        _ORIG_MAKE = gym.make

    def _wrapped_make(env_id, *args, **kwargs):
        """Wrap gym.make to support MOSAIC environment variables."""
        render_kwargs = dict(kwargs)

        # Auto-enable rgb_array for Fast Lane telemetry
        if os.getenv("MOSAIC_FASTLANE_ENABLED") == "1" and "render_mode" not in render_kwargs:
            try:
                env = _ORIG_MAKE(env_id, *args, render_mode="rgb_array", **render_kwargs)
            except TypeError:
                env = _ORIG_MAKE(env_id, *args, **render_kwargs)
        else:
            env = _ORIG_MAKE(env_id, *args, **render_kwargs)

        return env

    _wrapped_make._mosaic_wrapped = True  # type: ignore[attr-defined]
    _wrapped_make._mosaic_orig_make = _ORIG_MAKE  # type: ignore[attr-defined]
    gym.make = _wrapped_make
except Exception:  # pragma: no cover - gym optional
    pass

# --- XuanCe FastLane environment wrapping ---------------------------------
try:  # pragma: no cover - xuance optional
    import logging as _logging
    from xuance.environment.utils import XuanCeEnvWrapper, XuanCeAtariEnvWrapper, XuanCeMultiAgentEnvWrapper
    from xuance_worker.fastlane import maybe_wrap_env, is_fastlane_enabled

    _logger = _logging.getLogger("xuance_worker.sitecustomize")

    # Only patch if FastLane is enabled
    if is_fastlane_enabled():
        # Patch XuanCeEnvWrapper
        _ORIG_XUANCE_ENV_INIT = XuanCeEnvWrapper.__init__

        def _patched_xuance_env_init(self, env):
            """Wrap XuanCe environments with FastLane telemetry."""
            # Call original __init__
            _ORIG_XUANCE_ENV_INIT(self, env)
            # Wrap the internal env with FastLane
            self.env = maybe_wrap_env(self.env)

        XuanCeEnvWrapper.__init__ = _patched_xuance_env_init

        # Patch XuanCeAtariEnvWrapper
        _ORIG_ATARI_ENV_INIT = XuanCeAtariEnvWrapper.__init__

        def _patched_atari_env_init(self, env):
            """Wrap XuanCe Atari environments with FastLane telemetry."""
            _ORIG_ATARI_ENV_INIT(self, env)
            self.env = maybe_wrap_env(self.env)

        XuanCeAtariEnvWrapper.__init__ = _patched_atari_env_init

        # Patch XuanCeMultiAgentEnvWrapper
        _ORIG_MAENV_INIT = XuanCeMultiAgentEnvWrapper.__init__

        def _patched_maenv_init(self, env):
            """Wrap XuanCe multi-agent environments with FastLane telemetry."""
            _ORIG_MAENV_INIT(self, env)
            self.env = maybe_wrap_env(self.env)

        XuanCeMultiAgentEnvWrapper.__init__ = _patched_maenv_init

        _logger.info("XuanCe FastLane environment wrapping enabled")
except Exception as e:  # pragma: no cover - xuance optional
    _logging.getLogger("xuance_worker.sitecustomize").warning(
        "XuanCe FastLane wrapping failed: %s", e, exc_info=True
    )
    pass

# --- PyTorch save helper ----------------------------------------
try:  # pragma: no cover - torch optional
    import torch

    _ORIG_TORCH_SAVE = torch.save

    def _mosaic_torch_save(obj, f, *args, **kwargs):
        """Ensure parent directories exist before saving."""
        if isinstance(f, (str, Path)):
            Path(f).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return _ORIG_TORCH_SAVE(obj, f, *args, **kwargs)

    torch.save = _mosaic_torch_save
except Exception:  # pragma: no cover - torch optional
    pass

# --- TensorBoard log redirection ----------------------------------------
try:  # pragma: no cover - tensorboard optional
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
            override_root = os.getenv("XUANCE_TENSORBOARD_DIR")
            if override_root:
                log_dir = _resolve_tensorboard_logdir(override_root, log_dir)
            super().__init__(log_dir=log_dir, *args, **kwargs)

    _tb_writer_mod.SummaryWriter = _MosaicSummaryWriter
    _tb_pkg.SummaryWriter = _MosaicSummaryWriter
except Exception:  # pragma: no cover - tensorboard optional
    pass

# --- XuanCe config override support ----------------------------------------
try:  # pragma: no cover - xuance optional
    import xuance

    # Store original get_config if available
    if hasattr(xuance, "get_config"):
        _ORIG_GET_CONFIG = xuance.get_config

        def _patched_get_config(*args, **kwargs):
            """Patch XuanCe config loading to respect MOSAIC overrides."""
            config = _ORIG_GET_CONFIG(*args, **kwargs)

            # Apply MOSAIC environment variable overrides
            if os.getenv("XUANCE_DEVICE"):
                config.device = os.getenv("XUANCE_DEVICE")
            if os.getenv("XUANCE_SEED"):
                try:
                    config.seed = int(os.getenv("XUANCE_SEED"))
                except ValueError:
                    pass
            if os.getenv("XUANCE_PARALLELS"):
                try:
                    config.parallels = int(os.getenv("XUANCE_PARALLELS"))
                except ValueError:
                    pass
            if os.getenv("XUANCE_RUNNING_STEPS"):
                try:
                    config.running_steps = int(os.getenv("XUANCE_RUNNING_STEPS"))
                except ValueError:
                    pass
            if os.getenv("XUANCE_LOG_DIR"):
                config.log_dir = os.getenv("XUANCE_LOG_DIR")
            if os.getenv("XUANCE_MODEL_DIR"):
                config.model_dir = os.getenv("XUANCE_MODEL_DIR")

            return config

        xuance.get_config = _patched_get_config
except Exception:  # pragma: no cover - xuance optional
    pass

# --- Checkpoint resume support ----------------------------------------
try:  # pragma: no cover - torch optional
    import torch.nn as nn

    _MOSAIC_MODULE_TO_PATCHED = getattr(nn.Module.to, "_mosaic_patched", False)

    if not _MOSAIC_MODULE_TO_PATCHED:
        _ORIG_MODULE_TO = nn.Module.to
    else:
        _ORIG_MODULE_TO = getattr(nn.Module, "_orig_to", nn.Module.to)

    _RESUME_CHECKPOINT_LOADED = False

    def _mosaic_module_to(self, *args, **kwargs):
        global _RESUME_CHECKPOINT_LOADED
        result = _ORIG_MODULE_TO(self, *args, **kwargs)

        resume_path = os.getenv("XUANCE_RESUME_PATH")
        if not resume_path or _RESUME_CHECKPOINT_LOADED:
            return result

        checkpoint_file = Path(resume_path).expanduser().resolve()
        if not checkpoint_file.exists():
            return result

        # Determine device from args
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]

        try:
            checkpoint = torch.load(
                checkpoint_file,
                map_location=device,
                weights_only=True,
            )

            model_keys = set(self.state_dict().keys())

            # Handle dict format with model_state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                checkpoint_keys = set(checkpoint["model_state_dict"].keys())
                if model_keys == checkpoint_keys:
                    self.load_state_dict(checkpoint["model_state_dict"])
                    _RESUME_CHECKPOINT_LOADED = True
                    if "global_step" in checkpoint:
                        os.environ["XUANCE_RESUME_GLOBAL_STEP"] = str(checkpoint["global_step"])
                    print(f"[MOSAIC] Resumed from checkpoint: {checkpoint_file}")
            else:
                # Direct state_dict format
                checkpoint_keys = set(checkpoint.keys())
                if model_keys == checkpoint_keys:
                    self.load_state_dict(checkpoint)
                    _RESUME_CHECKPOINT_LOADED = True
                    print(f"[MOSAIC] Resumed from checkpoint: {checkpoint_file}")
        except Exception:
            pass

        return result

    _mosaic_module_to._mosaic_patched = True
    nn.Module._orig_to = _ORIG_MODULE_TO
    nn.Module.to = _mosaic_module_to
except Exception:  # pragma: no cover - torch optional
    pass
