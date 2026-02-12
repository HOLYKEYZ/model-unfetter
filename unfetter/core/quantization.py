"""
Quantization utilities for memory-efficient model loading.

Supports 4-bit (NF4) and 8-bit quantization via bitsandbytes,
enabling large models to run on consumer hardware.
"""

import logging
from typing import Optional, Literal

import torch

logger = logging.getLogger(__name__)

QuantizationMode = Literal["none", "4bit", "8bit", "bnb_4bit", "bnb_8bit"]


def get_quantization_config(
    mode: QuantizationMode = "none",
    compute_dtype: Optional[torch.dtype] = None,
) -> Optional[object]:
    """
    Create a BitsAndBytesConfig for the specified quantization mode.

    Args:
        mode: Quantization mode. Options:
              - "none": No quantization (full precision).
              - "4bit" or "bnb_4bit": 4-bit NF4 quantization.
              - "8bit" or "bnb_8bit": 8-bit LLM.int8() quantization.
        compute_dtype: Compute dtype for 4-bit quantization.
                       Defaults to bfloat16 if available, else float16.

    Returns:
        BitsAndBytesConfig or None if no quantization.
    """
    if mode == "none":
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "bitsandbytes and transformers are required for quantization. "
            "Install: pip install bitsandbytes transformers"
        )

    if compute_dtype is None:
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    if mode in ("4bit", "bnb_4bit"):
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"Using 4-bit NF4 quantization (compute_dtype={compute_dtype})")
        return config

    elif mode in ("8bit", "bnb_8bit"):
        config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        logger.info("Using 8-bit LLM.int8() quantization")
        return config

    else:
        raise ValueError(f"Unknown quantization mode: {mode}")


def load_quantized_model(
    model_path: str,
    quantization: QuantizationMode = "none",
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
):
    """
    Load a HuggingFace model with optional quantization.

    Tries multiple dtypes in order of preference (like Heretic):
    auto → float16 → bfloat16 → float32

    Args:
        model_path: HuggingFace model name or local path.
        quantization: Quantization mode.
        device_map: Device mapping strategy.
        torch_dtype: Override torch dtype.
        trust_remote_code: Whether to trust remote code.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_quantization_config(quantization)

    # Dtype fallback chain (like Heretic)
    dtypes_to_try = [torch_dtype] if torch_dtype else [
        "auto",
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    last_error = None
    for dtype in dtypes_to_try:
        try:
            kwargs = {
                "pretrained_model_name_or_path": model_path,
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
            }

            if quant_config:
                kwargs["quantization_config"] = quant_config
            if dtype != "auto":
                kwargs["torch_dtype"] = dtype

            model = AutoModelForCausalLM.from_pretrained(**kwargs)
            logger.info(f"Model loaded: {model_path} (dtype={dtype}, quant={quantization})")
            return model, tokenizer

        except Exception as e:
            last_error = e
            logger.debug(f"Failed to load with dtype={dtype}: {e}")
            continue

    raise RuntimeError(
        f"Failed to load model '{model_path}' with any dtype. "
        f"Last error: {last_error}"
    )


def dequantize_weight(weight_param) -> torch.Tensor:
    """
    Extract full-precision weights from a potentially quantized parameter.

    For quantized models, the weight data needs to be dequantized before
    ablation can be applied, then re-quantized.

    Args:
        weight_param: A model parameter (may be quantized).

    Returns:
        Float32 weight tensor.
    """
    if hasattr(weight_param, "dequantize"):
        return weight_param.dequantize()

    # Check for bitsandbytes quantized parameters
    try:
        import bitsandbytes as bnb
        if isinstance(weight_param, bnb.nn.Params4bit):
            return bnb.functional.dequantize_4bit(
                weight_param.data,
                weight_param.quant_state,
            ).float()
        elif isinstance(weight_param, bnb.nn.Int8Params):
            return weight_param.data.float()
    except ImportError:
        pass

    # Already full precision
    return weight_param.data.float()


def estimate_model_memory(
    model_path: str,
    quantization: QuantizationMode = "none",
) -> dict:
    """
    Estimate memory requirements for loading a model.

    Args:
        model_path: Model name or path.
        quantization: Quantization mode.

    Returns:
        Dict with estimated memory in GB for different components.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path)
        hidden_size = getattr(config, "hidden_size", 4096)
        num_layers = getattr(config, "num_hidden_layers", 32)
        vocab_size = getattr(config, "vocab_size", 32000)
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
        num_heads = getattr(config, "num_attention_heads", 32)

        # Estimate total parameters
        # Embedding: vocab_size * hidden_size
        embed_params = vocab_size * hidden_size
        # Per layer: attention (4 * h^2) + MLP (3 * h * intermediate)
        layer_params = (4 * hidden_size ** 2) + (3 * hidden_size * intermediate_size)
        # LM head: hidden_size * vocab_size
        head_params = hidden_size * vocab_size

        total_params = embed_params + (num_layers * layer_params) + head_params

        # Bytes per parameter based on quantization
        bytes_per_param = {
            "none": 2.0,  # fp16
            "4bit": 0.5,
            "bnb_4bit": 0.5,
            "8bit": 1.0,
            "bnb_8bit": 1.0,
        }

        bpp = bytes_per_param.get(quantization, 2.0)
        model_gb = (total_params * bpp) / (1024 ** 3)

        # Overhead: ~20% for optimizer states, activations, etc.
        overhead_gb = model_gb * 0.2

        return {
            "total_parameters": total_params,
            "model_memory_gb": round(model_gb, 2),
            "overhead_gb": round(overhead_gb, 2),
            "total_estimated_gb": round(model_gb + overhead_gb, 2),
            "quantization": quantization,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
        }

    except Exception as e:
        logger.warning(f"Could not estimate memory for {model_path}: {e}")
        return {"error": str(e)}
