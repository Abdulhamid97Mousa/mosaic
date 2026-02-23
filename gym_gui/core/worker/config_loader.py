"""Standard config loading utilities for MOSAIC workers.

Handles both direct config format and nested GUI format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Type, TypeVar

T = TypeVar("T")


def extract_worker_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract worker config from nested or flat format.

    Supports two formats:
    1. Nested (from GUI):
       {"metadata": {"worker": {"config": {...actual config...}}}}

    2. Direct (standalone):
       {...actual config...}

    Args:
        payload: Raw JSON payload

    Returns:
        Extracted worker configuration dictionary

    Example:
        # Nested format
        payload = {
            "metadata": {
                "worker": {
                    "config": {"run_id": "abc", "algo": "ppo"}
                }
            }
        }
        config = extract_worker_config(payload)
        # Returns: {"run_id": "abc", "algo": "ppo"}

        # Direct format
        payload = {"run_id": "abc", "algo": "ppo"}
        config = extract_worker_config(payload)
        # Returns: {"run_id": "abc", "algo": "ppo"}
    """
    # Check for nested GUI format
    if "metadata" in payload and "worker" in payload["metadata"]:
        worker_section = payload["metadata"]["worker"]
        if "config" in worker_section:
            return worker_section["config"]

    # Direct format
    return payload


def load_worker_config_from_file(
    path: Path,
    config_class: Type[T],
) -> T:
    """Load worker configuration from JSON file.

    Automatically handles nested and flat formats, then deserializes
    using the provided config class.

    Args:
        path: Path to JSON configuration file
        config_class: Configuration class with from_dict() class method

    Returns:
        Deserialized configuration instance

    Raises:
        FileNotFoundError: If path doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        TypeError: If config_class doesn't have from_dict method
        ValueError: If config validation fails

    Example:
        @dataclass
        class MyConfig:
            run_id: str
            algo: str

            @classmethod
            def from_dict(cls, data):
                return cls(**data)

        config = load_worker_config_from_file(
            Path("config.json"),
            MyConfig
        )
    """
    # Read JSON file
    payload = json.loads(path.read_text())

    # Extract worker config (handles nested/flat format)
    config_dict = extract_worker_config(payload)

    # Deserialize using config class
    if not hasattr(config_class, "from_dict"):
        raise TypeError(
            f"{config_class.__name__} must implement from_dict() class method"
        )

    return config_class.from_dict(config_dict)
