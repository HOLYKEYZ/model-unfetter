"""
Model Unfetter — Multi-Tier Model Unalignment Framework

A production-ready directional ablation tool that makes model alignment
removal accessible across all hardware configurations.

⚠️  DISCLAIMER: This tool is for AI SAFETY RESEARCH and RED TEAMING only.
    Use responsibly and in compliance with all applicable laws and model licenses.
"""

__version__ = "0.1.0"
__author__ = "Model Unfetter Contributors"

from unfetter.core.ablation import directional_ablation, ablate_layer
from unfetter.core.vectors import compute_refusal_vector

__all__ = [
    "directional_ablation",
    "ablate_layer",
    "compute_refusal_vector",
    "__version__",
]
