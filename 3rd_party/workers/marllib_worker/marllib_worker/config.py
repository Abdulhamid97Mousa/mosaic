"""Configuration dataclass for the MARLlib worker.

Implements the MOSAIC ``WorkerConfig`` protocol so that the GUI and
trainer daemon can create, serialise and validate run configurations
uniformly across all workers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol

    _HAS_PROTOCOL = True
except ImportError:
    WorkerConfigProtocol = object  # type: ignore[assignment,misc]
    _HAS_PROTOCOL = False


@dataclass(frozen=True)
class MARLlibWorkerConfig:
    """Structured configuration for a MARLlib training run.

    Implements the MOSAIC ``WorkerConfig`` protocol (``run_id``, ``seed``,
    ``to_dict()``, ``from_dict()``).
    """

    # --- MOSAIC protocol required ---
    run_id: str
    seed: Optional[int] = None

    # --- MARLlib core ---
    algo: str = ""
    environment_name: str = ""
    map_name: str = ""
    force_coop: bool = False
    hyperparam_source: str = "common"
    share_policy: str = "all"
    core_arch: str = "mlp"
    encode_layer: str = "128-256"

    # --- Ray / training ---
    num_gpus: int = 1
    num_workers: int = 2
    local_mode: bool = False
    framework: str = "torch"
    checkpoint_freq: int = 100
    checkpoint_end: bool = True

    # --- Stop conditions ---
    stop_timesteps: int = 1_000_000
    stop_reward: float = 999_999.0
    stop_iters: int = 9_999_999

    # --- Checkpoint restore ---
    restore_model_path: str = ""
    restore_params_path: str = ""

    # --- Override bags ---
    algo_params: Dict[str, Any] = field(default_factory=dict)
    env_params: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    # --- MOSAIC bookkeeping ---
    worker_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("marllib_worker config missing required field 'run_id'")
        if not self.algo:
            raise ValueError("marllib_worker config missing required field 'algo'")
        if not self.environment_name:
            raise ValueError(
                "marllib_worker config missing required field 'environment_name'"
            )
        if not self.map_name:
            raise ValueError("marllib_worker config missing required field 'map_name'")

        from .registries import validate_algo, SHARE_POLICY_OPTIONS, CORE_ARCH_OPTIONS

        validate_algo(self.algo)

        if self.share_policy not in SHARE_POLICY_OPTIONS:
            raise ValueError(
                f"share_policy must be one of {SHARE_POLICY_OPTIONS}, "
                f"got '{self.share_policy}'"
            )
        if self.core_arch not in CORE_ARCH_OPTIONS:
            raise ValueError(
                f"core_arch must be one of {CORE_ARCH_OPTIONS}, "
                f"got '{self.core_arch}'"
            )

        if _HAS_PROTOCOL:
            assert isinstance(self, WorkerConfigProtocol), (
                "MARLlibWorkerConfig must implement WorkerConfig protocol"
            )

    # --- WorkerConfig protocol methods ---

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MARLlibWorkerConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        ctor: Dict[str, Any] = {}
        extras: Dict[str, Any] = dict(data.get("extras", {}) or {})

        for key, value in data.items():
            if key in known:
                ctor[key] = value
            elif key != "extras":
                extras[key] = value

        ctor["extras"] = extras
        ctor.setdefault("raw", dict(data))
        return cls(**ctor)

    # --- Convenience ---

    def with_overrides(self, **overrides: Any) -> "MARLlibWorkerConfig":
        """Return a new config with *overrides* applied (``None`` values skipped)."""
        filtered = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **filtered)


# ------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------


def load_worker_config(path: Path) -> MARLlibWorkerConfig:
    """Load a worker config from a JSON file.

    Handles both the direct format and the nested
    ``metadata.worker.config`` format emitted by the MOSAIC GUI.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, Mapping) and "metadata" in payload:
        worker_section = payload["metadata"].get("worker", {})
        config_data = worker_section.get("config", payload)
    else:
        config_data = payload

    return MARLlibWorkerConfig.from_dict(config_data)
