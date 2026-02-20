"""XuanCe v1.4.0 compatibility shims for MOSAIC.

We do NOT modify 3rd-party code in the xuance submodule.  Instead, all
upstream compatibility work-arounds live here as runtime shims applied
before any XuanCe runners or learners are created.

Both ``runtime.py`` (standard training) and
``multi_agent_curriculum_training.py`` (curriculum training) call
``apply_shims()`` before creating a runner.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shim 1: LearnerMAS missing ``use_cnn`` attribute
# ---------------------------------------------------------------------------

_LEARNER_PATCHED = False


def _shim_learner_use_cnn() -> None:
    """Add the missing ``use_cnn`` attribute to LearnerMAS.__init__.

    XuanCe v1.4.0 references ``self.use_cnn`` in IAC_Learner.build_training_data()
    and MAPPO_Learner.update(), but LearnerMAS.__init__ never sets it.  The
    attribute IS set in the agent base class (MARLAgents), but not in the
    learner hierarchy, causing:

        AttributeError: 'MAPPO_Learner' object has no attribute 'use_cnn'

    This wraps LearnerMAS.__init__ so every learner subclass inherits
    ``use_cnn`` from config (defaulting to False).
    """
    global _LEARNER_PATCHED
    if _LEARNER_PATCHED:
        return

    try:
        from xuance.torch.learners.learner import LearnerMAS
    except ImportError:
        return  # xuance not installed

    _original_init = LearnerMAS.__init__

    def _patched_init(self, config, model_keys, agent_keys, policy, callback):
        _original_init(self, config, model_keys, agent_keys, policy, callback)
        if not hasattr(self, "use_cnn"):
            self.use_cnn = getattr(config, "use_cnn", False)

    LearnerMAS.__init__ = _patched_init
    _LEARNER_PATCHED = True
    LOGGER.debug("Shim applied: LearnerMAS.__init__ now sets use_cnn")


# ---------------------------------------------------------------------------
# Shim 2: tqdm suppression (prevents pipe buffer deadlock)
# ---------------------------------------------------------------------------

def _shim_tqdm_disabled() -> None:
    """Ensure tqdm is disabled to prevent pipe buffer deadlock.

    XuanCe's on_policy_marl.py uses tqdm(range(n_steps)) which writes to
    stderr on every iteration.  When stderr is piped to the telemetry proxy,
    the 1MB pipe buffer fills up and the training process deadlocks.

    The module-level ``TQDM_DISABLE`` env var (set in
    ``multi_agent_curriculum_training.py``) handles the common case.  This
    function wraps the XuanCe module directly if tqdm was imported before
    the env var took effect.
    """
    tqdm_mod = sys.modules.get("tqdm")
    if tqdm_mod is None:
        return  # Not imported yet; module-level env var will handle it

    # Wrap the tqdm class in XuanCe's on-policy MARL module if already loaded
    marl_mod = sys.modules.get("xuance.torch.agents.core.on_policy_marl")
    if marl_mod is not None and hasattr(marl_mod, "tqdm"):
        original_tqdm = marl_mod.tqdm

        def _silent_tqdm(iterable=None, *args, **kwargs):
            kwargs["disable"] = True
            return original_tqdm(iterable, *args, **kwargs)

        marl_mod.tqdm = _silent_tqdm
        LOGGER.debug("Shim applied: tqdm disabled in on_policy_marl")


# ---------------------------------------------------------------------------
# Shim 3: Redirect log/model/result dirs under var/
# ---------------------------------------------------------------------------
# XuanCe's get_runner() (common_tools.py) defaults log_dir to "logs/{algo}"
# and result_dir to "results/{algo}" -- both RELATIVE to CWD.  When callers
# (e.g. InteractiveRuntime) don't explicitly set log_dir/model_dir on
# parser_args, XuanCe creates directories like logs/ppo/CartPole-v1 in the
# project root.
#
# This shim wraps get_runner() so parser_args always carries a fallback
# directory under var/trainer/fallback/.  Callers that already set log_dir
# (e.g. runtime.py for normal training) are unaffected.
# ---------------------------------------------------------------------------

_RUNNER_DIR_PATCHED = False


def _shim_get_runner_directories() -> None:
    """Wrap ``get_runner`` to keep log/model/result dirs under ``var/``."""
    global _RUNNER_DIR_PATCHED
    if _RUNNER_DIR_PATCHED:
        return

    # We need the MOSAIC paths infrastructure.  If it is not available
    # (e.g. running xuance_worker standalone without gym_gui), skip.
    try:
        from gym_gui.config.paths import VAR_TRAINER_DIR
    except ImportError:
        LOGGER.debug(
            "gym_gui.config.paths not available; skipping get_runner dir shim"
        )
        return

    try:
        import xuance.common.common_tools as _ct
    except ImportError:
        return  # XuanCe not installed

    _fallback_dir = str(VAR_TRAINER_DIR / "fallback")
    _original_get_runner = _ct.get_runner

    def _wrapped_get_runner(
        algo, env=None, env_id=None, config_path=None, parser_args=None
    ):
        # Guarantee parser_args exists and carries dir defaults so XuanCe
        # never falls back to relative "logs/{algo}" in CWD.
        if parser_args is None:
            parser_args = SimpleNamespace()
        if not hasattr(parser_args, "log_dir"):
            parser_args.log_dir = _fallback_dir
        if not hasattr(parser_args, "model_dir"):
            parser_args.model_dir = _fallback_dir

        runner = _original_get_runner(
            algo, env, env_id, config_path, parser_args
        )

        # Fix result_dir which is always hardcoded as "results/{algo}/{env_id}"
        # with no getattr fallback -- parser_args cannot override it.
        def _abs_result_dir(args_obj):
            rd = getattr(args_obj, "result_dir", None)
            if rd and not os.path.isabs(rd):
                args_obj.result_dir = os.path.join(_fallback_dir, rd)

        if hasattr(runner, "args"):
            if isinstance(runner.args, list):
                for a in runner.args:
                    _abs_result_dir(a)
            else:
                _abs_result_dir(runner.args)

        return runner

    # Wrap both the module-level function and the top-level shortcut
    _ct.get_runner = _wrapped_get_runner
    try:
        import xuance
        xuance.get_runner = _wrapped_get_runner
    except (ImportError, AttributeError):
        pass

    _RUNNER_DIR_PATCHED = True
    LOGGER.debug(
        "Shim applied: get_runner dirs redirected under %s", _fallback_dir
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_shims() -> None:
    """Apply all XuanCe v1.4.0 compatibility shims.

    Safe to call multiple times -- each shim is guarded and only applied once.
    Call this before creating any XuanCe runner or learner.
    """
    _shim_learner_use_cnn()
    _shim_tqdm_disabled()
    _shim_get_runner_directories()
