"""
Core directional ablation algorithm.

Implements the technique from "Refusal in Language Models Is Mediated by a
Single Direction" (Arditi et al. 2024).

Formula: W' = W - α × (W·v̂)v̂ᵀ
where v̂ is the normalized refusal vector and α is ablation strength.
"""

import logging
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_projection(
    weight: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the projection of weight matrix rows onto a direction vector.

    For a weight matrix W and direction v̂:
        projection = (W · v̂) ⊗ v̂ᵀ
    This gives the component of each row of W in the direction of v̂.

    Args:
        weight: Weight matrix of shape (out_features, in_features).
        direction: Unit direction vector of shape (in_features,).

    Returns:
        Projection tensor of shape (out_features, in_features).
    """
    if direction.dim() != 1:
        raise ValueError(f"Direction must be 1D, got shape {direction.shape}")

    # Ensure direction is normalized
    norm = direction.norm()
    if norm < 1e-10:
        return torch.zeros_like(weight)
        
    direction = direction / norm

    # Move direction to same device/dtype as weight
    direction = direction.to(device=weight.device, dtype=weight.dtype)

    # (W · v̂) → shape (out_features,)
    # Each element: dot product of weight row with direction
    dots = weight @ direction

    # (W · v̂) ⊗ v̂ᵀ → shape (out_features, in_features)
    projection = dots.unsqueeze(1) * direction.unsqueeze(0)

    return projection


def ablate_weight(
    weight: torch.Tensor,
    refusal_vector: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Remove the refusal direction from a single weight matrix.

    Formula: W' = W - α × (W·v̂)v̂ᵀ

    Args:
        weight: Weight matrix to modify, shape (out_features, in_features).
        refusal_vector: Direction to remove, shape (in_features,).
        strength: Ablation intensity (0.0 = no change, 1.0 = full removal).

    Returns:
        Modified weight tensor (new tensor, original is not modified).
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength must be in [0.0, 1.0], got {strength}")

    if strength == 0.0:
        return weight.clone()

    projection = compute_projection(weight, refusal_vector)
    ablated = weight - strength * projection

    return ablated


def ablate_layer(
    layer: nn.Module,
    refusal_vector: torch.Tensor,
    strength: float = 1.0,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Apply directional ablation to target weight matrices in a single layer.

    By default targets both self_attn.o_proj (attention output projection) and
    mlp.down_proj (MLP down projection), matching Heretic's approach of ablating
    both components.

    Args:
        layer: A single transformer layer (nn.Module).
        refusal_vector: Direction to remove, shape (hidden_size,).
        strength: Ablation intensity (0.0-1.0).
        target_modules: Which submodules to modify. Defaults to
                        ["self_attn.o_proj", "mlp.down_proj"].

    Returns:
        Dict with stats: {"modified_modules": [...], "projection_norms": {...}}
    """
    if target_modules is None:
        target_modules = ["self_attn.o_proj", "mlp.down_proj"]

    stats = {"modified_modules": [], "projection_norms": {}}

    for module_name in target_modules:
        # Navigate nested modules (e.g., "self_attn.o_proj")
        parts = module_name.split(".")
        target = layer
        found = True
        for part in parts:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                found = False
                break

        if not found:
            logger.debug(f"Module '{module_name}' not found in layer, skipping")
            continue

        if not hasattr(target, "weight"):
            logger.warning(f"Module '{module_name}' has no weight attribute, skipping")
            continue

        # Get weight data (handle quantized weights)
        weight = target.weight.data

        # Compute ablated weight
        original_norm = weight.norm().item()
        ablated_weight = ablate_weight(weight, refusal_vector, strength)
        projection_norm = (weight - ablated_weight).norm().item()

        # Apply in-place
        target.weight.data = ablated_weight

        stats["modified_modules"].append(module_name)
        stats["projection_norms"][module_name] = {
            "original_weight_norm": original_norm,
            "projection_norm": projection_norm,
            "relative_change": projection_norm / max(original_norm, 1e-10),
        }

        logger.debug(
            f"Ablated {module_name}: "
            f"proj_norm={projection_norm:.6f}, "
            f"relative_change={projection_norm / max(original_norm, 1e-10):.6f}"
        )

    return stats


def compute_ablation_weights(
    num_layers: int,
    layer_indices: List[int],
    max_weight: float = 1.0,
    max_weight_position: float = 0.5,
    min_weight: float = 0.0,
    min_weight_distance: float = 0.5,
) -> Dict[int, float]:
    """
    Compute per-layer ablation weights using a kernel shape.

    This implements a flexible weight kernel similar to Heretic's approach,
    where ablation strength varies across layers based on a configurable
    shape defined by peak position, peak weight, minimum weight, and falloff.

    Args:
        num_layers: Total number of layers in the model.
        layer_indices: Which layers to ablate.
        max_weight: Peak ablation weight (0.0-1.0).
        max_weight_position: Position of peak within layer_indices range (0.0-1.0).
        min_weight: Minimum ablation weight at the edges.
        min_weight_distance: Distance from peak to minimum (0.0-1.0 of range).

    Returns:
        Dict mapping layer_index -> weight.
    """
    if not layer_indices:
        return {}

    n = len(layer_indices)
    if n == 1:
        return {layer_indices[0]: max_weight}

    # Peak position in the index space
    peak_pos = max_weight_position * (n - 1)

    weights = {}
    for i, layer_idx in enumerate(layer_indices):
        # Normalized distance from peak
        dist = abs(i - peak_pos) / max(n - 1, 1)

        if min_weight_distance <= 0:
            # Step function — full weight everywhere
            w = max_weight
        elif dist >= min_weight_distance:
            w = min_weight
        else:
            # Linear interpolation between max and min
            t = dist / min_weight_distance
            w = max_weight + t * (min_weight - max_weight)

        weights[layer_idx] = w

    return weights


def directional_ablation(
    model: nn.Module,
    refusal_vector: torch.Tensor,
    layer_indices: List[int],
    strength: float = 1.0,
    target_modules: Optional[List[str]] = None,
    layer_weights: Optional[Dict[int, float]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Apply directional ablation across multiple layers of a transformer model.

    Core ablation: project weight matrices away from the refusal direction
    to remove refusal behavior while preserving model capabilities.

    Args:
        model: Transformer model (HuggingFace-compatible).
        refusal_vector: Refusal direction to remove, shape (hidden_size,).
        layer_indices: Which transformer layers to modify.
        strength: Global ablation intensity (0.0-1.0).
        target_modules: Which submodules per layer to modify.
                        Defaults to ["self_attn.o_proj", "mlp.down_proj"].
        layer_weights: Optional per-layer weights (from compute_ablation_weights).
                       Multiplied with strength for final per-layer intensity.
        progress_callback: Optional callable(layer_idx, total) for progress.

    Returns:
        Dict with overall stats and per-layer breakdown.
    """
    if target_modules is None:
        target_modules = ["self_attn.o_proj", "mlp.down_proj"]

    # Get the transformer layers
    layers = _get_model_layers(model)
    total_layers = len(layers)

    if not layers:
        raise ValueError("Could not find transformer layers in model")

    # Validate layer indices
    valid_indices = []
    for idx in layer_indices:
        # Support negative indexing
        resolved = idx if idx >= 0 else total_layers + idx
        if 0 <= resolved < total_layers:
            valid_indices.append(resolved)
        else:
            logger.warning(
                f"Layer index {idx} (resolved: {resolved}) out of range "
                f"[0, {total_layers}), skipping"
            )

    if not valid_indices:
        raise ValueError("No valid layer indices to process")

    logger.info(
        f"Starting directional ablation: "
        f"{len(valid_indices)} layers, strength={strength}, "
        f"targets={target_modules}"
    )

    results = {
        "total_layers": total_layers,
        "ablated_layers": len(valid_indices),
        "strength": strength,
        "target_modules": target_modules,
        "layer_stats": {},
    }

    for i, layer_idx in enumerate(valid_indices):
        # Compute effective strength for this layer
        effective_strength = strength
        if layer_weights and layer_idx in layer_weights:
            effective_strength *= layer_weights[layer_idx]

        if effective_strength <= 0:
            logger.debug(f"Layer {layer_idx}: effective strength=0, skipping")
            continue

        # Ablate single layer
        layer = layers[layer_idx]
        stats = ablate_layer(
            layer,
            refusal_vector,
            strength=effective_strength,
            target_modules=target_modules,
        )
        results["layer_stats"][layer_idx] = stats

        if progress_callback:
            progress_callback(i + 1, len(valid_indices))

        logger.debug(f"Layer {layer_idx}: ablated with strength={effective_strength:.4f}")

    # Summary statistics
    total_modified = sum(
        len(s["modified_modules"]) for s in results["layer_stats"].values()
    )
    results["total_modules_modified"] = total_modified

    logger.info(
        f"Ablation complete: modified {total_modified} modules "
        f"across {len(results['layer_stats'])} layers"
    )

    return results


def _get_model_layers(model: nn.Module) -> nn.ModuleList:
    """
    Extract the transformer layer list from a HuggingFace model.

    Supports common architectures: Llama, Mistral, Gemma, GPT-NeoX, Phi, Qwen.

    Args:
        model: A HuggingFace transformer model.

    Returns:
        ModuleList of transformer layers.
    """
    # Common layer access patterns for different architectures
    layer_paths = [
        "model.layers",          # Llama, Mistral, Gemma, Qwen
        "transformer.h",         # GPT-2, GPT-Neo
        "transformer.layers",    # Some custom models
        "gpt_neox.layers",       # GPT-NeoX, Pythia
        "model.decoder.layers",  # OPT, BART decoder
    ]

    for path in layer_paths:
        parts = path.split(".")
        obj = model
        found = True
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break
        if found and isinstance(obj, nn.ModuleList):
            return obj

    raise ValueError(
        "Could not find transformer layers. Supported patterns: "
        + ", ".join(layer_paths)
    )
