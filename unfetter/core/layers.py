"""
Layer extraction and manipulation utilities.

Provides tools for identifying, selecting, and manipulating individual
transformer layers within HuggingFace models.
"""

import logging
import re
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_model_layers(model: nn.Module) -> nn.ModuleList:
    """
    Extract the transformer layer list from a model.

    Re-exports from ablation module for convenience.
    """
    from unfetter.core.ablation import _get_model_layers
    return _get_model_layers(model)


def get_layer_count(model: nn.Module) -> int:
    """Return the number of transformer layers in the model."""
    return len(get_model_layers(model))


def get_hidden_size(model: nn.Module) -> int:
    """
    Get the hidden size (embedding dimension) of the model.

    Checks config first, then infers from weight shapes.
    """
    if hasattr(model, "config"):
        if hasattr(model.config, "hidden_size"):
            return model.config.hidden_size

    # Infer from first layer's attention weight
    layers = get_model_layers(model)
    if layers:
        layer = layers[0]
        for name, param in layer.named_parameters():
            if "o_proj" in name or "out_proj" in name:
                return param.shape[0]

    raise ValueError("Could not determine hidden size")


def get_target_modules(
    layer: nn.Module,
    target_names: Optional[List[str]] = None,
) -> List[Tuple[str, nn.Module]]:
    """
    Find target submodules (o_proj, down_proj) within a layer.

    Args:
        layer: A single transformer layer.
        target_names: Module names to find. Defaults to
                      ["self_attn.o_proj", "mlp.down_proj"].

    Returns:
        List of (name, module) tuples that were found.
    """
    if target_names is None:
        target_names = ["self_attn.o_proj", "mlp.down_proj"]

    found = []
    for name in target_names:
        parts = name.split(".")
        obj = layer
        valid = True
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                valid = False
                break
        if valid:
            found.append((name, obj))

    return found


def select_layers(
    num_layers: int,
    layer_spec: str,
) -> List[int]:
    """
    Parse a layer specification string into a list of layer indices.

    Supports multiple formats:
    - Slice notation: "-8:-1" (last 8 layers except final)
    - Comma-separated: "20,25,30,35"
    - Range: "20-30" (layers 20 through 30 inclusive)
    - Percentage: "70%-100%" (last 30% of layers)
    - "auto" or "all": all layers

    Args:
        num_layers: Total number of layers in the model.
        layer_spec: Layer specification string.

    Returns:
        Sorted list of layer indices.
    """
    spec = layer_spec.strip()

    # "auto" or "all" — return all layers
    if spec.lower() in ("auto", "all"):
        return list(range(num_layers))

    # "last_N" — last N layers
    match = re.match(r"last[_-]?(\d+)", spec, re.IGNORECASE)
    if match:
        n = int(match.group(1))
        start = max(0, num_layers - n)
        return list(range(start, num_layers))

    # Percentage range: "70%-100%"
    match = re.match(r"(\d+)%\s*-\s*(\d+)%", spec)
    if match:
        start_pct = int(match.group(1))
        end_pct = int(match.group(2))
        start = int(num_layers * start_pct / 100)
        end = int(num_layers * end_pct / 100)
        return list(range(start, min(end, num_layers)))

    # Slice notation: "-8:-1" or "10:20"
    if ":" in spec and "," not in spec and "%" not in spec:
        parts = spec.split(":")
        if len(parts) == 2:
            start_str, end_str = parts
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else num_layers

            # Resolve negative indices
            if start < 0:
                start = num_layers + start
            if end < 0:
                end = num_layers + end

            start = max(0, start)
            end = min(num_layers, end)
            return list(range(start, end))

    # Comma-separated: "20,25,30,35"
    if "," in spec:
        indices = []
        for part in spec.split(","):
            part = part.strip()
            if part:
                idx = int(part)
                resolved = idx if idx >= 0 else num_layers + idx
                if 0 <= resolved < num_layers:
                    indices.append(resolved)
        return sorted(set(indices))

    # Range: "20-30"
    match = re.match(r"(-?\d+)\s*-\s*(-?\d+)$", spec)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        if start < 0:
            start = num_layers + start
        if end < 0:
            end = num_layers + end
        start = max(0, start)
        end = min(num_layers, end + 1)  # inclusive end
        return list(range(start, end))

    # Single integer
    try:
        idx = int(spec)
        resolved = idx if idx >= 0 else num_layers + idx
        if 0 <= resolved < num_layers:
            return [resolved]
    except ValueError:
        pass

    raise ValueError(
        f"Could not parse layer spec: '{layer_spec}'. "
        f"Supported formats: 'auto', 'all', 'last_N', 'N:M', 'N,M,O', "
        f"'N-M', 'N%-M%', or a single integer."
    )


def auto_select_layers(
    num_layers: int,
    percentage: float = 0.3,
) -> List[int]:
    """
    Auto-select layers for ablation.

    Default strategy: last 30% of layers, as refusal behavior is
    typically concentrated in the later transformer layers.

    Args:
        num_layers: Total number of layers.
        percentage: Fraction of layers to select from the end.

    Returns:
        List of layer indices.
    """
    n_select = max(1, int(num_layers * percentage))
    start = num_layers - n_select
    indices = list(range(start, num_layers))
    logger.info(f"Auto-selected layers {start}-{num_layers - 1} ({n_select} layers, {percentage * 100:.0f}%)")
    return indices


def get_layer_parameter_count(layer: nn.Module) -> int:
    """Count total parameters in a layer."""
    return sum(p.numel() for p in layer.parameters())


def get_layer_memory_mb(layer: nn.Module) -> float:
    """Estimate memory footprint of a layer in MB."""
    total_bytes = sum(
        p.numel() * p.element_size() for p in layer.parameters()
    )
    return total_bytes / (1024 * 1024)
