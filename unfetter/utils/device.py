"""
Hardware detection utilities.

Provides detailed hardware information for optimal backend selection.
"""

import logging
import platform
from typing import Dict, Any

import torch

logger = logging.getLogger(__name__)


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive hardware information.

    Returns:
        Dict with CPU, RAM, GPU, and system details.
    """
    info = {
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "count_logical": 1,
            "count_physical": 1,
        },
        "memory": {
            "total_gb": 0,
            "available_gb": 0,
            "percent_used": 0,
        },
        "gpu": {
            "available": False,
            "count": 0,
            "devices": [],
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda or "N/A",
            "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
        },
    }

    # CPU info
    try:
        import psutil
        info["cpu"]["count_logical"] = psutil.cpu_count(logical=True)
        info["cpu"]["count_physical"] = psutil.cpu_count(logical=False)

        mem = psutil.virtual_memory()
        info["memory"]["total_gb"] = round(mem.total / (1024 ** 3), 2)
        info["memory"]["available_gb"] = round(mem.available / (1024 ** 3), 2)
        info["memory"]["percent_used"] = mem.percent
    except ImportError:
        import os
        info["cpu"]["count_logical"] = os.cpu_count() or 1

    # GPU info
    if torch.cuda.is_available():
        info["gpu"]["available"] = True
        info["gpu"]["count"] = torch.cuda.device_count()

        for i in range(info["gpu"]["count"]):
            props = torch.cuda.get_device_properties(i)
            info["gpu"]["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_mem / (1024 ** 3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })

    # MPS (Apple Silicon)
    info["mps"] = {
        "available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    return info


def get_optimal_device() -> str:
    """
    Get the optimal device string for computation.

    Returns:
        Device string ("cuda", "mps", or "cpu").
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def estimate_model_memory(
    num_params: int,
    dtype_bytes: int = 2,
    overhead: float = 1.2,
) -> float:
    """
    Estimate memory required to load a model.

    Args:
        num_params: Number of model parameters.
        dtype_bytes: Bytes per parameter (2 for fp16, 4 for fp32, 0.5 for 4-bit).
        overhead: Overhead multiplier (1.2 = 20% overhead).

    Returns:
        Estimated memory in GB.
    """
    base_memory = (num_params * dtype_bytes) / (1024 ** 3)
    return round(base_memory * overhead, 2)
