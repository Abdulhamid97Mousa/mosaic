"""Algorithm and environment registries for MARLlib worker.

Static data extracted from MARLlib source:
- Algorithm paradigms: marllib/marl/common.py (algo_type_dict)
- Environment list: marllib/envs/base_env/config/*.yaml
- Policy registry: marllib/marl/algos/scripts/__init__.py (POlICY_REGISTRY)
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Algorithm paradigm classification
# Source: marllib/marl/common.py  check_algo_type()
# ---------------------------------------------------------------------------

ALGO_TYPE_DICT: Dict[str, List[str]] = {
    "IL": ["ia2c", "iddpg", "itrpo", "ippo"],
    "VD": ["vda2c", "vdppo", "facmac", "iql", "vdn", "qmix"],
    "CC": ["maa2c", "maddpg", "mappo", "matrpo", "happo", "hatrpo", "coma"],
}

ALL_ALGORITHMS: Set[str] = set()
for _algos in ALGO_TYPE_DICT.values():
    ALL_ALGORITHMS.update(_algos)

# ---------------------------------------------------------------------------
# Environment list (from marllib/envs/base_env/config/*.yaml)
# ---------------------------------------------------------------------------

ALL_ENVIRONMENTS: Tuple[str, ...] = (
    "aircombat",
    "football",
    "gobigger",
    "gymnasium_mamujoco",
    "gymnasium_mpe",
    "hanabi",
    "hns",
    "lbf",
    "magent",
    "mamujoco",
    "mate",
    "metadrive",
    "mpe",
    "overcooked",
    "pommerman",
    "rware",
    "sisl",
    "smac",
    "voltage",
)

# ---------------------------------------------------------------------------
# Constraint sets
# ---------------------------------------------------------------------------

DISCRETE_ONLY_ALGOS: Set[str] = {"iql", "coma", "vdn", "qmix"}
CONTINUOUS_ONLY_ALGOS: Set[str] = {"iddpg", "maddpg", "facmac"}

SHARE_POLICY_OPTIONS: Tuple[str, ...] = ("all", "group", "individual")
CORE_ARCH_OPTIONS: Tuple[str, ...] = ("mlp", "gru", "lstm")
HYPERPARAM_SPECIAL_SOURCES: Tuple[str, ...] = ("common", "test")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_algo_type(algo_name: str) -> str:
    """Return 'IL', 'VD', or 'CC' for a given algorithm name.

    Raises:
        ValueError: If *algo_name* is not a recognised MARLlib algorithm.
    """
    for algo_type, algos in ALGO_TYPE_DICT.items():
        if algo_name in algos:
            return algo_type
    raise ValueError(
        f"Unknown algorithm '{algo_name}'. "
        f"Available: {sorted(ALL_ALGORITHMS)}"
    )


def validate_algo(algo_name: str) -> None:
    """Raise ``ValueError`` if *algo_name* is not recognised."""
    if algo_name not in ALL_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Available: {sorted(ALL_ALGORITHMS)}"
        )


def validate_environment(env_name: str) -> None:
    """Raise ``ValueError`` if *env_name* is not recognised."""
    if env_name not in ALL_ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Available: {sorted(ALL_ENVIRONMENTS)}"
        )
