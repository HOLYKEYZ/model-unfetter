"""
Model Unfetter — Multi-Tier Model Unalignment Framework

A production-ready directional ablation tool that makes model alignment
removal accessible across all hardware configurations.

High-level API::

    from unfetter import compute_refusal_vector, directional_ablation

    refusal_vector = compute_refusal_vector(model, tokenizer, harmful, harmless)
    directional_ablation(model, refusal_vector, layer_indices=list(range(10, 30)))

Heretic-style hyperparameter search::

    from unfetter.core.optimization import optimize_ablation_parameters
    from unfetter.core.vectors import compute_all_layer_vectors

    layer_vectors = compute_all_layer_vectors(model, tokenizer, harmful, harmless)
    best = optimize_ablation_parameters(model, layer_vectors, [...], evaluate_fn)

1T-parameter streaming::

    from unfetter.core.streaming import StreamingAblator

    ablator = StreamingAblator(input_dir, output_dir, hidden_size, num_layers)
    ablator.ablate_all_layers(refusal_vector, layer_indices, target_modules)

⚠️  DISCLAIMER: This tool is for AI SAFETY RESEARCH and RED TEAMING only.
    Use responsibly and in compliance with all applicable laws and model licenses.
"""

__version__ = "0.2.0"
__author__ = "Model Unfetter Contributors"

from unfetter.core.ablation import (
    directional_ablation,
    ablate_layer,
    ablate_with_float_index,
    compute_ablation_weights,
)
from unfetter.core.vectors import (
    compute_refusal_vector,
    compute_refusal_vector_geometric_median,
    compute_all_layer_vectors,
    interpolate_layer_vectors,
)
from unfetter.core.methods import heretic_ablation, obliteratus_ablation
from unfetter.core.recovery import attach_lora_recovery, train_lora_recovery
from unfetter.models.registry import auto_select_ablation_strategy, get_model_summary

__all__ = [
    "directional_ablation",
    "ablate_layer",
    "ablate_with_float_index",
    "compute_ablation_weights",
    "compute_refusal_vector",
    "compute_refusal_vector_geometric_median",
    "compute_all_layer_vectors",
    "interpolate_layer_vectors",
    "heretic_ablation",
    "obliteratus_ablation",
    "attach_lora_recovery",
    "train_lora_recovery",
    "auto_select_ablation_strategy",
    "get_model_summary",
    "__version__",
]
