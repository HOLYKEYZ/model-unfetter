"""
Core directional ablation algorithms.

Implements the technique from "Refusal in Language Models Is Mediated by a
Single Direction" (Arditi et al. 2024), plus Heretic-style per-component
parameters / float layer interpolation and Obliteratus-style norm-preserving
projection variants.

Formulas:
    Standard:      W' = W - α * v ⊗ (vᵀ · W)
    Norm-pres:     W' as above, then rescale to preserve row/column norms
    SVD-pres:      orthogonal projection with optional per-row rescaling
"""

import logging
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

AblationMethod = str  # 'directional', 'svd_norm_preserving', 'channel_norm_preserving'


def compute_projection(
    weight: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the projection to orthogonalize the output of a Linear layer.

    For PyTorch Linear layers y = x @ W^T, the rows of W compute the output features.
    To make the output y orthogonal to vector v (y @ v = 0), we must project the
    columns of W.
    Projection = v ⊗ (vᵀ · W)

    Args:
        weight: Weight matrix of shape (out_features, in_features).
        direction: Unit direction vector of shape (out_features,).

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

    # (vᵀ · W) → shape (in_features,)
    dots = direction @ weight

    # v ⊗ (vᵀ · W) → shape (out_features, in_features)
    projection = direction.unsqueeze(1) * dots.unsqueeze(0)

    return projection


def ablate_weight(
    weight: torch.Tensor,
    refusal_vector: torch.Tensor,
    strength: float = 1.0,
    method: AblationMethod = "directional",
) -> torch.Tensor:
    """
    Remove the refusal direction from a single weight matrix.

    Args:
        weight: Weight matrix to modify, shape (out_features, in_features).
        refusal_vector: Direction to remove, shape (out_features,).
        strength: Ablation intensity (0.0 = no change, 1.0 = full removal).
        method: Which ablation variant to use.
            - "directional": standard orthogonal projection.
            - "svd_norm_preserving": projection followed by per-row rescaling to
              preserve output-channel norms.
            - "channel_norm_preserving": projection followed by global Frobenius
              norm rescaling.

    Returns:
        Modified weight tensor (new tensor, original is not modified).
    """
    if not 0.0 <= strength <= 2.0:
        raise ValueError(f"Strength must be in range [0.0, 2.0], got {strength}")

    if strength == 0.0:
        return weight.clone()

    projection = compute_projection(weight, refusal_vector)
    ablated = weight - strength * projection

    if method == "directional":
        return ablated

    if method == "svd_norm_preserving":
        # Preserve the norm of each output channel (row).
        original_norms = weight.norm(dim=1, keepdim=True)
        ablated_norms = ablated.norm(dim=1, keepdim=True).clamp_min(1e-10)
        scale = original_norms / ablated_norms
        return ablated * scale

    if method == "channel_norm_preserving":
        # Preserve the overall Frobenius norm.
        original_norm = weight.norm()
        ablated_norm = ablated.norm().clamp_min(1e-10)
        scale = original_norm / ablated_norm
        return ablated * scale

    raise ValueError(f"Unknown ablation method: {method}")


def ablate_layer(
    layer: nn.Module,
    refusal_vector: torch.Tensor,
    strength: float = 1.0,
    target_modules: Optional[List[str]] = None,
    component_alphas: Optional[Dict[str, float]] = None,
    method: AblationMethod = "directional",
) -> Dict[str, Any]:
    """
    Apply directional ablation to target weight matrices in a single layer.

    By default targets both self_attn.o_proj (attention output projection) and
    mlp.down_proj (MLP down projection), matching Heretic's approach of ablating
    both components.

    Args:
        layer: A single transformer layer (nn.Module).
        refusal_vector: Direction to remove, shape (hidden_size,).
        strength: Base ablation intensity (0.0-2.0).
        target_modules: Which submodules to modify. Defaults to
                        ["self_attn.o_proj", "mlp.down_proj"].
        component_alphas: Optional per-module alpha multipliers. Maps full module
            names (e.g. "self_attn.o_proj") to floats. If a module is missing,
            the base ``strength`` is used.
        method: Ablation variant (see :func:`ablate_weight`).

    Returns:
        Dict with stats: {"modified_modules": [...], "projection_norms": {...}}
    """
    if target_modules is None:
        target_modules = [
            "self_attn.o_proj",
            "mlp.down_proj"
        ]

    stats = {"modified_modules": [], "projection_norms": {}, "method": method}

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
        from unfetter.core.quantization import dequantize_weight
        weight = dequantize_weight(target.weight)

        # Ensure weight is correctly shaped (dequantization can sometimes flatten)
        if hasattr(target, "weight") and weight.shape != target.weight.shape:
             weight = weight.reshape(target.weight.shape)

        # Check if this module is compatible with the refusal vector
        # The refusal vector v acts on the output space, so out_features (dim 0) must match v.
        if weight.shape[0] != refusal_vector.shape[0]:
            logger.debug(
                f"Skipping {module_name}: shape {weight.shape} incompatible "
                f"with vector {refusal_vector.shape} (requires dim 0 match)"
            )
            continue

        # Compute effective alpha for this component (Heretic-style per-component params)
        effective_strength = strength
        if component_alphas and module_name in component_alphas:
            effective_strength *= component_alphas[module_name]
            logger.debug(f"Module {module_name}: component_alpha={component_alphas[module_name]}")

        if effective_strength <= 0:
            logger.debug(f"Module {module_name}: effective strength=0, skipping")
            continue

        # Compute ablated weight
        original_norm = weight.norm().item()
        ablated_weight = ablate_weight(weight, refusal_vector, effective_strength, method=method)
        projection_norm = (weight - ablated_weight).norm().item()

        # Apply update
        if hasattr(target.weight, "quant_state"):
            # If it was quantized, we replace it with a float parameter
            target.weight = nn.Parameter(ablated_weight.to(dtype=weight.dtype))
        else:
            target.weight.data = ablated_weight.to(dtype=target.weight.dtype)

        stats["modified_modules"].append(module_name)
        stats["projection_norms"][module_name] = {
            "original_weight_norm": original_norm,
            "projection_norm": projection_norm,
            "relative_change": projection_norm / max(original_norm, 1e-10),
            "effective_strength": effective_strength,
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
        "transformer.blocks",    # some custom/older models
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


def directional_ablation(
    model: nn.Module,
    refusal_vector: torch.Tensor,
    layer_indices: List[int],
    strength: float = 1.0,
    target_modules: Optional[List[str]] = None,
    layer_weights: Optional[Dict[int, float]] = None,
    component_alphas: Optional[Dict[str, float]] = None,
    method: AblationMethod = "directional",
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
        strength: Global ablation intensity (0.0-2.0).
        target_modules: Which submodules per layer to modify.
        layer_weights: Optional per-layer weights (from compute_ablation_weights).
                       Multiplied with strength for final per-layer intensity.
        component_alphas: Optional per-module alpha multipliers.
        method: Ablation variant (directional / svd_norm_preserving / channel_norm_preserving).
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
        f"targets={target_modules}, method={method}"
    )

    results = {
        "total_layers": total_layers,
        "ablated_layers": len(valid_indices),
        "strength": strength,
        "target_modules": target_modules,
        "method": method,
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
            component_alphas=component_alphas,
            method=method,
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


def ablate_with_float_index(
    model: nn.Module,
    layer_vectors: List[torch.Tensor],
    direction_index: float,
    layer_indices: List[int],
    strength: float = 1.0,
    target_modules: Optional[List[str]] = None,
    component_alphas: Optional[Dict[str, float]] = None,
    method: AblationMethod = "directional",
    layer_weights: Optional[Dict[int, float]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Heretic-style ablation using a continuous layer index.

    Instead of a single refusal vector, a stack of per-layer vectors is supplied
    and ``direction_index`` (float in [0, 1]) selects an interpolated vector.
    This lets hyperparameter search find the single best refusal direction for
    the model without restricting it to a discrete layer.

    Args:
        model: Transformer model.
        layer_vectors: Per-layer refusal vectors (one per layer).
        direction_index: Float in [0, 1] selecting the interpolated vector.
        layer_indices: Which layers to modify.
        strength: Global ablation intensity.
        target_modules: Submodules per layer to modify.
        component_alphas: Per-module alpha multipliers.
        method: Ablation variant.
        layer_weights: Optional per-layer kernel weights.
        progress_callback: Optional progress callback.

    Returns:
        Ablation stats dict including ``direction_index``.
    """
    from unfetter.core.vectors import interpolate_layer_vectors

    refusal_vector = interpolate_layer_vectors(layer_vectors, direction_index)
    results = directional_ablation(
        model,
        refusal_vector,
        layer_indices,
        strength=strength,
        target_modules=target_modules,
        component_alphas=component_alphas,
        method=method,
        layer_weights=layer_weights,
        progress_callback=progress_callback,
    )
    results["direction_index"] = direction_index
    return results
