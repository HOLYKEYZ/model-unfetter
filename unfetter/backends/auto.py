"""
Hardware auto-detection and backend selection.

Probes the system for CPU, RAM, GPU, and VRAM, then selects
the optimal backend configuration automatically.
"""

import logging
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware and return a summary.

    Returns:
        Dict with cpu, ram, gpu details.
    """
    info = {
        "cpu_count": 1,
        "ram_total_gb": 8.0,
        "ram_available_gb": 4.0,
        "has_cuda": False,
        "num_gpus": 0,
        "gpus": [],
        "recommended_backend": "cpu",
    }

    # CPU and RAM
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
        info["ram_available_gb"] = round(mem.available / (1024 ** 3), 2)
    except ImportError:
        import os
        info["cpu_count"] = os.cpu_count() or 1
        logger.debug("psutil not available, RAM detection limited")

    # GPU
    info["has_cuda"] = torch.cuda.is_available()

    if info["has_cuda"]:
        info["num_gpus"] = torch.cuda.device_count()
        total_vram = 0
        for i in range(info["num_gpus"]):
            props = torch.cuda.get_device_properties(i)
            vram_gb = round(props.total_mem / (1024 ** 3), 2)
            total_vram += vram_gb
            info["gpus"].append({
                "index": i,
                "name": props.name,
                "vram_gb": vram_gb,
                "compute_capability": f"{props.major}.{props.minor}",
            })
        info["total_vram_gb"] = round(total_vram, 2)

    # MPS (Apple Silicon)
    info["has_mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # Determine recommended backend
    if info["num_gpus"] > 1:
        info["recommended_backend"] = "distributed"
    elif info["num_gpus"] == 1:
        info["recommended_backend"] = "gpu"
    else:
        info["recommended_backend"] = "cpu"

    return info


def select_backend(
    backend_name: str = "auto",
    ram_limit_gb: Optional[int] = None,
    vram_limit_gb: Optional[float] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 5,
):
    """
    Create and return the optimal backend for the current hardware.

    Args:
        backend_name: "auto", "cpu", "gpu", or "distributed".
        ram_limit_gb: Override RAM limit for CPU backend.
        vram_limit_gb: Override VRAM limit for GPU backend.
        checkpoint_dir: Directory for checkpointing (CPU backend).
        checkpoint_every: Checkpoint frequency (CPU backend).

    Returns:
        A Backend instance.
    """
    from unfetter.backends.cpu_backend import CPUBackend
    from unfetter.backends.gpu_backend import GPUBackend
    from unfetter.backends.distributed import DistributedBackend

    hw = detect_hardware()

    if backend_name == "auto":
        backend_name = hw["recommended_backend"]
        logger.info(f"Auto-detected hardware: recommending '{backend_name}' backend")

    if backend_name == "distributed":
        if not hw["has_cuda"] or hw["num_gpus"] < 2:
            logger.warning("Distributed requested but <2 GPUs available, falling back to GPU/CPU")
            backend_name = "gpu" if hw["has_cuda"] else "cpu"

    if backend_name == "gpu":
        if not hw["has_cuda"]:
            logger.warning("GPU requested but CUDA not available, falling back to CPU")
            backend_name = "cpu"

    # Create backend
    if backend_name == "cpu":
        ram = ram_limit_gb or int(hw.get("ram_available_gb", 16))
        return CPUBackend(
            ram_limit_gb=ram,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoint_dir,
        )

    elif backend_name == "gpu":
        vram = vram_limit_gb or hw.get("gpus", [{}])[0].get("vram_gb", 8)
        return GPUBackend(vram_limit_gb=vram)

    elif backend_name == "distributed":
        return DistributedBackend()

    else:
        raise ValueError(
            f"Unknown backend: '{backend_name}'. "
            f"Choose from: auto, cpu, gpu, distributed"
        )


def print_hardware_info():
    """Print a formatted summary of detected hardware."""
    hw = detect_hardware()

    lines = [
        "╔══════════════════════════════════════════╗",
        "║       Model Unfetter — Hardware Info     ║",
        "╠══════════════════════════════════════════╣",
        f"║ CPU Cores:    {hw['cpu_count']:>26} ║",
        f"║ RAM Total:    {hw['ram_total_gb']:>23.1f} GB ║",
        f"║ RAM Free:     {hw['ram_available_gb']:>23.1f} GB ║",
        f"║ CUDA:         {'Yes' if hw['has_cuda'] else 'No':>26} ║",
        f"║ GPUs:         {hw['num_gpus']:>26} ║",
    ]

    for gpu in hw.get("gpus", []):
        lines.append(
            f"║  └ GPU {gpu['index']}: {gpu['name'][:18]:>18} {gpu['vram_gb']:>4.1f}GB ║"
        )

    if hw.get("has_mps"):
        lines.append(f"║ MPS (Apple):  {'Yes':>26} ║")

    lines.extend([
        f"╠══════════════════════════════════════════╣",
        f"║ Recommended:  {hw['recommended_backend']:>26} ║",
        f"╚══════════════════════════════════════════╝",
    ])

    return "\n".join(lines)
