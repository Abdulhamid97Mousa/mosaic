"""BARLOG Worker - LLM-based agent for BALROG environments.

This worker provides a subprocess interface for running LLM agents
on BALROG benchmark environments (BabyAI, MiniHack, Crafter).
"""

__version__ = "0.1.0"
__author__ = "MOSAIC Team"

from barlog_worker.config import BarlogWorkerConfig

__all__ = ["BarlogWorkerConfig", "__version__"]
