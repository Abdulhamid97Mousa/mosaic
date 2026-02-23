"""MOSAIC LLM Worker Observations - Theory of Mind Study.

This module provides observation text generation for multi-agent LLMs.
Two observation modes are implemented for research on how social
information affects coordination.
"""

from .theory_of_mind import (
    OBJECT_TYPES,
    COLORS,
    DIRECTIONS,
    describe_observation_egocentric,
    describe_observation_with_teammates,
    extract_visible_teammates,
)

__all__ = [
    "OBJECT_TYPES",
    "COLORS",
    "DIRECTIONS",
    "describe_observation_egocentric",
    "describe_observation_with_teammates",
    "extract_visible_teammates",
]
