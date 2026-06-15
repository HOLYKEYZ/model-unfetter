"""
Refusal vector computation via activation contrast.

Extracts the "refusal direction" from a model by comparing activations
between harmful (refused) prompts and harmless (complied) prompts.

Method: difference-of-means on first-token residuals, per Arditi et al. 2024.
"""

import gc
import logging
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default cache directory for computed vectors
DEFAULT_CACHE_DIR = Path.home() / ".unfetter" / "vectors"


def get_layer_activation(
    model: nn.Module,
    tokenizer,
    prompt: str,
    layer_index: int,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Extract the residual stream activation at a specific layer using hooks.

    Captures the output of the layer (post-attention + MLP) for the first
    token position, which is where refusal decisions are encoded.

    Args:
        model: HuggingFace transformer model.
        tokenizer: Corresponding tokenizer.
        prompt: Input text to process.
        layer_index: Which layer to capture (supports negative indexing).
        device: Device to run on. If None, uses model's device.

    Returns:
        Activation tensor of shape (hidden_size,).
    """
    from unfetter.core.ablation import _get_model_layers

    layers = _get_model_layers(model)
    total = len(layers)

    # Resolve negative index
    resolved_idx = layer_index if layer_index >= 0 else total + layer_index
    if not 0 <= resolved_idx < total:
        raise ValueError(f"Layer index {layer_index} out of range [0, {total})")

    target_layer = layers[resolved_idx]
    activation = {}

    def hook_fn(module, input, output):
        # output is typically a tuple; first element is hidden states
        if isinstance(output, tuple):
            activation["value"] = output[0].detach()
        else:
            activation["value"] = output.detach()

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        # Apply chat template if available to match inference conditions
        if isinstance(prompt, str) and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            # We wrap in try block in case it requires a specific structure
            try:
                prompt_to_tokenize = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.debug(f"Failed to apply chat template: {e}, using raw prompt")
                prompt_to_tokenize = prompt
        else:
            prompt_to_tokenize = prompt

        # Tokenize and run forward pass
        inputs = tokenizer(
            prompt_to_tokenize,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        if "value" not in activation:
            raise RuntimeError(f"Hook did not capture activation at layer {resolved_idx}")

        # Take the LAST token's activation (residual at position -1)
        # This is where the model has processed the full prompt and decides whether to refuse
        act = activation["value"][0, -1, :].clone()

        return act.cpu()

    finally:
        handle.remove()


def compute_refusal_vector(
    model: nn.Module,
    tokenizer,
    refusal_prompts: List[str],
    compliance_prompts: List[str],
    target_layer: int = -2,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute the refusal direction vector using difference-of-means.

    Process:
    1. Run refusal prompts through model, capture activations at target_layer
    2. Run compliance prompts through model, capture activations at target_layer
    3. refusal_vector = mean(refusal_activations) - mean(compliance_activations)
    4. Normalize to unit vector

    Args:
        model: HuggingFace transformer model.
        tokenizer: Corresponding tokenizer.
        refusal_prompts: Prompts that typically trigger refusal.
        compliance_prompts: Prompts that typically get compliant responses.
        target_layer: Which layer to extract activations from (default: -2).
        batch_size: Number of prompts to process at once.
        max_samples: Maximum number of samples to use from each set.
        device: Device to run on.

    Returns:
        Normalized refusal vector of shape (hidden_size,).
    """
    if max_samples:
        refusal_prompts = refusal_prompts[:max_samples]
        compliance_prompts = compliance_prompts[:max_samples]

    n_refusal = len(refusal_prompts)
    n_compliance = len(compliance_prompts)

    logger.info(
        f"Computing refusal vector: "
        f"{n_refusal} refusal prompts, {n_compliance} compliance prompts, "
        f"layer={target_layer}, batch_size={batch_size}"
    )

    # Extract activations for refusal prompts
    refusal_activations = _extract_activations_batched(
        model, tokenizer, refusal_prompts, target_layer, batch_size, device,
        label="refusal",
    )

    # Extract activations for compliance prompts
    compliance_activations = _extract_activations_batched(
        model, tokenizer, compliance_prompts, target_layer, batch_size, device,
        label="compliance",
    )

    # Compute difference-of-means
    refusal_mean = torch.stack(refusal_activations).mean(dim=0)
    compliance_mean = torch.stack(compliance_activations).mean(dim=0)

    refusal_vector = refusal_mean - compliance_mean

    # Normalize to unit vector
    norm = refusal_vector.norm()
    if norm < 1e-10:
        logger.warning("Refusal vector has near-zero norm — model may not exhibit refusal")
        return refusal_vector

    refusal_vector = refusal_vector / norm

    logger.info(f"Refusal vector computed: norm={norm:.6f}, shape={refusal_vector.shape}")

    return refusal_vector


def compute_all_layer_vectors(
    model: nn.Module,
    tokenizer,
    refusal_prompts: List[str],
    compliance_prompts: List[str],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Compute refusal direction vectors for every layer.

    This enables per-layer ablation (as in Heretic's "per_layer" mode),
    where each layer is ablated using its own refusal direction.

    Args:
        model: HuggingFace transformer model.
        tokenizer: Corresponding tokenizer.
        refusal_prompts: Prompts that trigger refusal.
        compliance_prompts: Prompts that get compliant responses.
        batch_size: Batch size for activation extraction.
        max_samples: Maximum samples per set.
        device: Device to run on.

    Returns:
        List of normalized refusal vectors, one per layer.
    """
    from unfetter.core.ablation import _get_model_layers

    layers = _get_model_layers(model)
    num_layers = len(layers)

    logger.info(f"Computing refusal vectors for all {num_layers} layers")

    vectors = []
    for layer_idx in range(num_layers):
        vec = compute_refusal_vector(
            model, tokenizer,
            refusal_prompts, compliance_prompts,
            target_layer=layer_idx,
            batch_size=batch_size,
            max_samples=max_samples,
            device=device,
        )
        vectors.append(vec)
        logger.info(f"Layer {layer_idx}/{num_layers}: vector norm={vec.norm():.6f}")

    return vectors


def geometric_median(
    points: List[torch.Tensor],
    eps: float = 1e-5,
    max_iter: int = 100,
) -> torch.Tensor:
    """
    Compute the geometric median of a set of vectors.

    The geometric median is more robust to outliers than the arithmetic mean,
    making it well-suited for aggregating refusal directions from a diverse
    prompt set (as used by Heretic).

    Args:
        points: List of 1-D tensors of the same shape.
        eps: Convergence tolerance.
        max_iter: Maximum Weiszfeld iterations.

    Returns:
        1-D tensor containing the geometric median.
    """
    if not points:
        raise ValueError("Cannot compute geometric median of empty point set")

    stacked = torch.stack([p.flatten() for p in points])
    median = stacked.mean(dim=0)

    for _ in range(max_iter):
        distances = torch.norm(stacked - median, dim=1)
        # Avoid division by zero
        distances = torch.clamp(distances, min=1e-10)
        weights = 1.0 / distances
        weights = weights / weights.sum()
        new_median = (stacked * weights.unsqueeze(1)).sum(dim=0)

        if torch.norm(new_median - median) < eps:
            return new_median

        median = new_median

    return median


def compute_refusal_vector_geometric_median(
    model: nn.Module,
    tokenizer,
    refusal_prompts: List[str],
    compliance_prompts: List[str],
    target_layer: int = -2,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute a robust refusal direction using geometric-median contrast.

    Returns the normalized vector from compliance median to refusal median,
    where each median is the geometric median of the respective activations.
    """
    if max_samples:
        refusal_prompts = refusal_prompts[:max_samples]
        compliance_prompts = compliance_prompts[:max_samples]

    refusal_activations = _extract_activations_batched(
        model, tokenizer, refusal_prompts, target_layer, batch_size, device,
        label="refusal",
    )
    compliance_activations = _extract_activations_batched(
        model, tokenizer, compliance_prompts, target_layer, batch_size, device,
        label="compliance",
    )

    refusal_median = geometric_median(refusal_activations)
    compliance_median = geometric_median(compliance_activations)

    refusal_vector = refusal_median - compliance_median
    norm = refusal_vector.norm()
    if norm < 1e-10:
        logger.warning("Geometric-median refusal vector has near-zero norm")
        return refusal_vector

    refusal_vector = refusal_vector / norm
    logger.info(
        f"Geometric-median refusal vector computed: norm={norm:.6f}, "
        f"shape={refusal_vector.shape}"
    )
    return refusal_vector


def interpolate_layer_vectors(
    layer_vectors: List[torch.Tensor],
    direction_index: float,
) -> torch.Tensor:
    """
    Interpolate a refusal vector from a continuous layer index.

    Heretic uses a float ``direction_index`` in [0, 1] to choose the
    position along the layer stack, then blends the two nearest layer
    vectors for a smooth, per-model optimum.

    Args:
        layer_vectors: List of normalized refusal vectors, one per layer.
        direction_index: Float in [0, 1]; 0 = first layer, 1 = last layer.

    Returns:
        Normalized interpolated refusal vector.
    """
    if not layer_vectors:
        raise ValueError("layer_vectors must not be empty")

    n = len(layer_vectors)
    if n == 1 or direction_index <= 0:
        return layer_vectors[0]
    if direction_index >= 1:
        return layer_vectors[-1]

    scaled = direction_index * (n - 1)
    lower_idx = int(scaled)
    upper_idx = min(lower_idx + 1, n - 1)
    t = scaled - lower_idx

    interpolated = (1 - t) * layer_vectors[lower_idx] + t * layer_vectors[upper_idx]
    norm = interpolated.norm()
    if norm < 1e-10:
        logger.warning("Interpolated refusal vector has near-zero norm")
        return interpolated
    return interpolated / norm


def _extract_activations_batched(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    target_layer: int,
    batch_size: int,
    device: Optional[str],
    label: str = "",
) -> List[torch.Tensor]:
    """
    Extract activations for a list of prompts in batches.

    Uses individual prompt processing (not true batching) because
    different prompts have different lengths and we want exact first-token
    activations. Memory is managed with periodic garbage collection.

    Args:
        model: Transformer model.
        tokenizer: Tokenizer.
        prompts: List of text prompts.
        target_layer: Layer to capture.
        batch_size: Process this many then gc.collect().
        device: Device override.
        label: Label for logging.

    Returns:
        List of activation tensors, each shape (hidden_size,).
    """
    activations = []

    for i, prompt in enumerate(prompts):
        try:
            act = get_layer_activation(model, tokenizer, prompt, target_layer, device)
            activations.append(act)
        except Exception as e:
            logger.warning(f"Failed to extract activation for {label} prompt {i}: {e}")
            continue

        # Periodic cleanup
        if (i + 1) % batch_size == 0 or (i + 1) == len(prompts):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  {label}: processed {i + 1}/{len(prompts)}", flush=True)
            logger.debug(f"  {label}: processed {i + 1}/{len(prompts)}")

    if not activations:
        raise RuntimeError(f"Failed to extract any {label} activations")

    logger.info(f"Extracted {len(activations)} {label} activations")
    return activations


def cache_vector(
    vector: torch.Tensor,
    model_family: str,
    model_name: str,
    layer: int,
    cache_dir: Optional[str] = None,
) -> Path:
    """
    Save a computed refusal vector to disk for reuse.

    Args:
        vector: Refusal vector tensor.
        model_family: Family name (e.g., "llama", "mistral").
        model_name: Specific model name.
        layer: Layer index the vector was computed at.
        cache_dir: Override cache directory.

    Returns:
        Path where the vector was saved.
    """
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_path = cache_path / model_family / model_name.replace("/", "_")
    cache_path.mkdir(parents=True, exist_ok=True)

    filename = cache_path / f"refusal_vector_layer{layer}.pt"
    torch.save({
        "vector": vector,
        "model_family": model_family,
        "model_name": model_name,
        "layer": layer,
    }, filename)

    logger.info(f"Cached refusal vector: {filename}")
    return filename


def load_cached_vector(
    model_family: str,
    model_name: str,
    layer: int,
    cache_dir: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    Load a previously cached refusal vector.

    Args:
        model_family: Family name (e.g., "llama", "mistral").
        model_name: Specific model name.
        layer: Layer index.
        cache_dir: Override cache directory.

    Returns:
        Refusal vector tensor, or None if not found.
    """
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    filename = cache_path / model_family / model_name.replace("/", "_") / f"refusal_vector_layer{layer}.pt"

    if not filename.exists():
        return None

    data = torch.load(filename, map_location="cpu", weights_only=True)
    logger.info(f"Loaded cached refusal vector: {filename}")
    return data["vector"] if isinstance(data, dict) else data
