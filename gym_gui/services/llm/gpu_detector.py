"""GPU detection and monitoring for local LLM inference.

Detects NVIDIA GPUs using nvidia-smi and provides VRAM information
for model compatibility checking.
"""

from __future__ import annotations

import logging
import subprocess
import re
from dataclasses import dataclass
from typing import List, Optional

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_UI_CHAT_GPU_DETECTION_COMPLETED,
    LOG_UI_CHAT_GPU_DETECTION_ERROR,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    temperature_c: Optional[int] = None
    utilization_percent: Optional[int] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None

    @property
    def total_memory_gb(self) -> float:
        """Total VRAM in GB."""
        return self.total_memory_mb / 1024

    @property
    def free_memory_gb(self) -> float:
        """Free VRAM in GB."""
        return self.free_memory_mb / 1024

    @property
    def used_memory_gb(self) -> float:
        """Used VRAM in GB."""
        return self.used_memory_mb / 1024

    @property
    def memory_usage_percent(self) -> float:
        """Memory usage percentage."""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100

    def can_fit_model(self, model_size_gb: float, buffer_gb: float = 2.0) -> bool:
        """Check if this GPU can fit a model of given size.

        Args:
            model_size_gb: Model size in GB
            buffer_gb: Extra buffer for runtime overhead (default 2GB)

        Returns:
            True if model can fit in free VRAM
        """
        return self.free_memory_gb >= (model_size_gb + buffer_gb)

    def __str__(self) -> str:
        return (
            f"GPU {self.index}: {self.name} | "
            f"{self.free_memory_gb:.1f}GB free / {self.total_memory_gb:.1f}GB total "
            f"({self.memory_usage_percent:.0f}% used)"
        )


@dataclass
class GPUDetectionResult:
    """Result of GPU detection."""

    gpus: List[GPUInfo]
    nvidia_smi_available: bool
    cuda_available: bool
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def has_gpus(self) -> bool:
        """Check if any GPUs are available."""
        return len(self.gpus) > 0

    @property
    def total_vram_gb(self) -> float:
        """Total VRAM across all GPUs in GB."""
        return sum(gpu.total_memory_gb for gpu in self.gpus)

    @property
    def total_free_vram_gb(self) -> float:
        """Total free VRAM across all GPUs in GB."""
        return sum(gpu.free_memory_gb for gpu in self.gpus)

    def get_best_gpu_for_model(self, model_size_gb: float) -> Optional[GPUInfo]:
        """Get the best GPU for running a model of given size.

        Returns the GPU with most free VRAM that can fit the model.
        """
        suitable_gpus = [
            gpu for gpu in self.gpus
            if gpu.can_fit_model(model_size_gb)
        ]
        if not suitable_gpus:
            return None
        return max(suitable_gpus, key=lambda g: g.free_memory_gb)


class GPUDetector:
    """Detect and monitor NVIDIA GPUs."""

    # Approximate model sizes in GB (for quantized versions)
    MODEL_SIZES = {
        "Llama-3.2-1B": 2,
        "Llama-3.2-3B": 4,
        "Mistral-7B": 8,
        "Llama-3.1-8B": 10,
        "Llama-3.1-70B": 45,
    }

    @staticmethod
    def detect() -> GPUDetectionResult:
        """Detect all NVIDIA GPUs and their VRAM status.

        Returns:
            GPUDetectionResult with all detected GPUs
        """
        result = GPUDetectionResult(
            gpus=[],
            nvidia_smi_available=False,
            cuda_available=False,
        )

        # Try nvidia-smi first
        try:
            nvidia_smi_output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,driver_version",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if nvidia_smi_output.returncode == 0:
                result.nvidia_smi_available = True
                result = GPUDetector._parse_nvidia_smi(
                    nvidia_smi_output.stdout, result
                )

                # Get CUDA version
                cuda_version = GPUDetector._get_cuda_version()
                result.cuda_version = cuda_version
                result.cuda_available = cuda_version is not None

            else:
                result.error_message = "nvidia-smi command failed"
                log_constant(
                    _LOGGER,
                    LOG_UI_CHAT_GPU_DETECTION_ERROR,
                    message=f"nvidia-smi failed: {nvidia_smi_output.stderr}",
                )

        except FileNotFoundError:
            result.error_message = "nvidia-smi not found. NVIDIA drivers may not be installed."
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_GPU_DETECTION_COMPLETED,
                message="nvidia-smi not found - no NVIDIA GPU detected",
            )
        except subprocess.TimeoutExpired:
            result.error_message = "nvidia-smi timed out"
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_GPU_DETECTION_ERROR,
                message="nvidia-smi timed out",
            )
        except Exception as e:
            result.error_message = f"GPU detection failed: {e}"
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_GPU_DETECTION_ERROR,
                message="GPU detection failed",
                exc_info=e,
            )

        # Try PyTorch CUDA as fallback/supplement
        try:
            import torch
            if torch.cuda.is_available():
                result.cuda_available = True
                if not result.gpus:
                    # Populate from PyTorch if nvidia-smi failed
                    result = GPUDetector._detect_via_torch(result)
        except ImportError:
            pass  # PyTorch not installed

        return result

    @staticmethod
    def _parse_nvidia_smi(output: str, result: GPUDetectionResult) -> GPUDetectionResult:
        """Parse nvidia-smi CSV output."""
        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue

            try:
                gpu = GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_memory_mb=int(parts[2]),
                    used_memory_mb=int(parts[3]),
                    free_memory_mb=int(parts[4]),
                    temperature_c=int(parts[5]) if len(parts) > 5 and parts[5] else None,
                    utilization_percent=int(parts[6]) if len(parts) > 6 and parts[6] else None,
                    driver_version=parts[7] if len(parts) > 7 else None,
                )
                result.gpus.append(gpu)

                # Set driver version from first GPU
                if result.driver_version is None and gpu.driver_version:
                    result.driver_version = gpu.driver_version

            except (ValueError, IndexError) as e:
                log_constant(
                    _LOGGER,
                    LOG_UI_CHAT_GPU_DETECTION_ERROR,
                    message=f"Failed to parse GPU info: {line}",
                    exc_info=e,
                )

        return result

    @staticmethod
    def _get_cuda_version() -> Optional[str]:
        """Get CUDA version from nvidia-smi."""
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if output.returncode == 0:
                # Get CUDA version from nvcc or nvidia-smi
                nvcc_output = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if nvcc_output.returncode == 0:
                    # Parse "Cuda compilation tools, release X.Y"
                    match = re.search(r"release (\d+\.\d+)", nvcc_output.stdout)
                    if match:
                        return match.group(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: try PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda
        except ImportError:
            pass

        return None

    @staticmethod
    def _detect_via_torch(result: GPUDetectionResult) -> GPUDetectionResult:
        """Detect GPUs via PyTorch (fallback)."""
        try:
            import torch

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory // (1024 * 1024)

                # Get current memory usage
                torch.cuda.set_device(i)
                used_mb = torch.cuda.memory_allocated(i) // (1024 * 1024)
                free_mb = total_mb - used_mb

                gpu = GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_mb=total_mb,
                    used_memory_mb=used_mb,
                    free_memory_mb=free_mb,
                )
                result.gpus.append(gpu)

            result.cuda_version = torch.version.cuda

        except Exception as e:
            log_constant(
                _LOGGER,
                LOG_UI_CHAT_GPU_DETECTION_ERROR,
                message="PyTorch GPU detection failed",
                exc_info=e,
            )

        return result

    @staticmethod
    def get_recommended_models(detection: GPUDetectionResult) -> List[str]:
        """Get list of models that can run on detected GPUs.

        Args:
            detection: GPU detection result

        Returns:
            List of model names that can fit in available VRAM
        """
        if not detection.has_gpus:
            return []

        max_vram = max(gpu.free_memory_gb for gpu in detection.gpus)
        recommended = []

        for model_name, size_gb in sorted(
            GPUDetector.MODEL_SIZES.items(),
            key=lambda x: x[1]
        ):
            # Add 2GB buffer for runtime overhead
            if size_gb + 2 <= max_vram:
                recommended.append(model_name)

        return recommended

    @staticmethod
    def refresh() -> GPUDetectionResult:
        """Refresh GPU detection (alias for detect)."""
        return GPUDetector.detect()


__all__ = ["GPUInfo", "GPUDetectionResult", "GPUDetector"]
