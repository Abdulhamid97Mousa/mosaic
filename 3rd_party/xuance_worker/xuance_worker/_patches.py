"""XuanCe v1.4.0 shim-layer patches.

We do NOT modify 3rd-party code in the xuance submodule.  Instead, all
upstream bug work-arounds live here as monkey-patches applied at runtime
before any XuanCe runners or learners are created.

Both ``runtime.py`` (standard training) and
``multi_agent_curriculum_training.py`` (curriculum training) call
``apply_xuance_patches()`` before creating a runner.
"""

from __future__ import annotations

import logging
import os
import sys

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patch 1: LearnerMAS missing ``use_cnn`` attribute
# ---------------------------------------------------------------------------

_LEARNER_PATCHED = False


def _patch_learner_use_cnn() -> None:
    """Patch LearnerMAS.__init__ to add the missing ``use_cnn`` attribute.

    XuanCe v1.4.0 references ``self.use_cnn`` in IAC_Learner.build_training_data()
    and MAPPO_Learner.update(), but LearnerMAS.__init__ never sets it.  The
    attribute IS set in the agent base class (MARLAgents), but not in the
    learner hierarchy, causing:

        AttributeError: 'MAPPO_Learner' object has no attribute 'use_cnn'

    This monkey-patches LearnerMAS.__init__ so every learner subclass
    inherits ``use_cnn`` from config (defaulting to False).
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
    LOGGER.debug("Patched LearnerMAS.__init__ to add use_cnn attribute")


# ---------------------------------------------------------------------------
# Patch 2: tqdm suppression (prevents pipe buffer deadlock)
# ---------------------------------------------------------------------------

def _ensure_tqdm_disabled() -> None:
    """Ensure tqdm is disabled to prevent pipe buffer deadlock.

    XuanCe's on_policy_marl.py uses tqdm(range(n_steps)) which writes to
    stderr on every iteration.  When stderr is piped to the telemetry proxy,
    the 1MB pipe buffer fills up and the training process deadlocks.

    The module-level ``TQDM_DISABLE`` env var (set in
    ``multi_agent_curriculum_training.py``) handles the common case.  This
    function patches the XuanCe module directly if tqdm was imported before
    the env var took effect.
    """
    tqdm_mod = sys.modules.get("tqdm")
    if tqdm_mod is None:
        return  # Not imported yet; module-level env var will handle it

    # Patch the tqdm class in XuanCe's on-policy MARL module if already loaded
    marl_mod = sys.modules.get("xuance.torch.agents.core.on_policy_marl")
    if marl_mod is not None and hasattr(marl_mod, "tqdm"):
        original_tqdm = marl_mod.tqdm

        def _silent_tqdm(iterable=None, *args, **kwargs):
            kwargs["disable"] = True
            return original_tqdm(iterable, *args, **kwargs)

        marl_mod.tqdm = _silent_tqdm
        LOGGER.debug("Patched tqdm in on_policy_marl module")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_xuance_patches() -> None:
    """Apply all XuanCe v1.4.0 shim-layer patches.

    Safe to call multiple times -- each patch is guarded and only applied once.
    Call this before creating any XuanCe runner or learner.
    """
    _patch_learner_use_cnn()
    _ensure_tqdm_disabled()
