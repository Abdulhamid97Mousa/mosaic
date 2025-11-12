"""MOSAIC CleanRL site customizations.

This module is auto-imported by Python when present on PYTHONPATH. We use it to
set sane WANDB defaults and provide optional video hooks without patching the
vendored CleanRL sources.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- WANDB defaults -------------------------------------------------------
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB__SERVICE", "disabled")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_DISABLE_GYM", "true")
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

# --- Gymnasium RecordVideo helper ----------------------------------------
try:  # pragma: no cover - gym optional
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    _ORIG_MAKE = gym.make
    _RGB_MODES = {"rgb_array", "rgb_array_list"}

    if not hasattr(RecordVideo, "enabled"):

        def _get_enabled(self):
            return getattr(self, "_mosaic_recording_enabled", True)

        def _set_enabled(self, value):
            self._mosaic_recording_enabled = bool(value)

        RecordVideo.enabled = property(_get_enabled, _set_enabled)  # type: ignore[attr-defined]

    if not hasattr(RecordVideo, "path"):
        _ORIG_STOP_RECORDING = RecordVideo.stop_recording

        def _wrapped_stop_recording(self):
            video_folder = getattr(self, "video_folder", None)
            video_name = getattr(self, "_video_name", None)
            result = _ORIG_STOP_RECORDING(self)
            if video_folder and video_name:
                candidate = os.path.join(video_folder, f"{video_name}.mp4")
                self._mosaic_video_path = candidate
            return result

        def _get_path(self):
            return getattr(self, "_mosaic_video_path", None)

        def _set_path(self, value):
            self._mosaic_video_path = value

        RecordVideo.stop_recording = _wrapped_stop_recording
        RecordVideo.path = property(_get_path, _set_path)  # type: ignore[attr-defined]

    def _wrapped_make(env_id, *args, **kwargs):
        env = _ORIG_MAKE(env_id, *args, **kwargs)
        if os.getenv("MOSAIC_CAPTURE_VIDEO") != "1":
            return env

        if isinstance(env, RecordVideo):
            return env

        render_mode = getattr(env, "render_mode", None)
        if render_mode not in _RGB_MODES:
            return env

        video_dir = os.getenv("MOSAIC_VIDEOS_DIR", "videos")
        try:
            env = RecordVideo(env, video_folder=video_dir)
        except Exception:
            return env
        return env

    gym.make = _wrapped_make
except Exception:  # pragma: no cover - gym optional
    pass

try:  # pragma: no cover - gym optional
    import gymnasium as gym
    from gymnasium.wrappers import TransformObservation as _OrigTransformObservation

    class _MosaicTransformObservation(_OrigTransformObservation):
        def __init__(self, env, func, *, observation_space=None):
            obs_space = observation_space or getattr(env, "observation_space", None)
            if obs_space is None:
                raise ValueError("TransformObservation requires an observation_space")
            super().__init__(env, func, observation_space=obs_space)

    gym.wrappers.TransformObservation = _MosaicTransformObservation
except Exception:  # pragma: no cover - gym optional
    pass

try:  # pragma: no cover - torch optional
    import torch

    _ORIG_TORCH_SAVE = torch.save

    def _mosaic_torch_save(obj, f, *args, **kwargs):
        if isinstance(f, (str, Path)):
            Path(f).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return _ORIG_TORCH_SAVE(obj, f, *args, **kwargs)

    torch.save = _mosaic_torch_save
except Exception:  # pragma: no cover - torch optional
    pass

try:  # pragma: no cover - cleanrl optional
    import os
    from cleanrl_utils.evals import ppo_eval as _ppo_eval

    _ORIG_PPO_EVALUATE = _ppo_eval.evaluate

    def _mosaic_ppo_evaluate(model_path, make_env, env_id, eval_episodes, run_name, Model, *, device="cpu", capture_video=True, gamma=0.99):
        override = os.getenv("MOSAIC_CLEANRL_EVAL_EPISODES")
        if override is not None:
            try:
                eval_episodes = int(override)
            except ValueError:
                pass
        return _ORIG_PPO_EVALUATE(
            model_path,
            make_env,
            env_id,
            eval_episodes,
            run_name,
            Model,
            device=device,
            capture_video=capture_video,
            gamma=gamma,
        )

    _ppo_eval.evaluate = _mosaic_ppo_evaluate
except Exception:  # pragma: no cover - cleanrl optional
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
            override_root = os.getenv("CLEANRL_TENSORBOARD_DIR")
            if override_root:
                log_dir = _resolve_tensorboard_logdir(override_root, log_dir)
            super().__init__(log_dir=log_dir, *args, **kwargs)

    _tb_writer_mod.SummaryWriter = _MosaicSummaryWriter
    _tb_pkg.SummaryWriter = _MosaicSummaryWriter
except Exception:  # pragma: no cover - tensorboard optional
    pass
