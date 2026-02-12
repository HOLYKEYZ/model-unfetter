"""
Model family detection and registry.

Auto-detects which model architecture a given model uses
and returns the appropriate handler.
"""

import logging
from typing import Optional, Type

import torch.nn as nn

from unfetter.models.base import TransformerModel
from unfetter.models.llama import LlamaModel
from unfetter.models.mistral import MistralModel, MixtralModel
from unfetter.models.gemma import GemmaModel

logger = logging.getLogger(__name__)

# Registry mapping architecture identifiers to handler classes
MODEL_REGISTRY = {
    # Llama family
    "llama": LlamaModel,
    "llama2": LlamaModel,
    "llama3": LlamaModel,
    "codellama": LlamaModel,
    # Mistral family
    "mistral": MistralModel,
    "mixtral": MixtralModel,
    # Gemma family
    "gemma": GemmaModel,
    "gemma2": GemmaModel,
    # Qwen (shares Llama-like architecture)
    "qwen": LlamaModel,
    "qwen2": LlamaModel,
    "qwen3": LlamaModel,
    # Phi (shares Llama-like architecture)
    "phi": LlamaModel,
    "phi3": LlamaModel,
    # Yi (shares Llama-like architecture)
    "yi": LlamaModel,
    # InternLM
    "internlm": LlamaModel,
    "internlm2": LlamaModel,
    # DeepSeek
    "deepseek": LlamaModel,
    # Command-R
    "cohere": LlamaModel,
    # GPT-OSS
    "gpt-oss": LlamaModel,
}

# Patterns to match in model config's architectures field
ARCHITECTURE_PATTERNS = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mistral",
    "MixtralForCausalLM": "mixtral",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma2",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    "PhiForCausalLM": "phi",
    "Phi3ForCausalLM": "phi3",
    "InternLM2ForCausalLM": "internlm2",
    "DeepseekForCausalLM": "deepseek",
    "CohereForCausalLM": "cohere",
}


def detect_model_family(
    model_name_or_path: str,
    model: Optional[nn.Module] = None,
) -> str:
    """
    Detect the model family from name, path, or loaded model.

    Detection priority:
    1. Model config's architectures field (most reliable)
    2. Model name substring matching
    3. Default to "generic"

    Args:
        model_name_or_path: HuggingFace model name or local path.
        model: Optionally loaded model for config inspection.

    Returns:
        Model family string (e.g., "llama", "mistral", "gemma").
    """
    # 1. Check model config architectures
    if model and hasattr(model, "config"):
        config = model.config
        if hasattr(config, "architectures") and config.architectures:
            for arch in config.architectures:
                if arch in ARCHITECTURE_PATTERNS:
                    family = ARCHITECTURE_PATTERNS[arch]
                    logger.info(f"Detected model family '{family}' from architecture '{arch}'")
                    return family

        # Check model_type in config
        if hasattr(config, "model_type"):
            model_type = config.model_type.lower()
            if model_type in MODEL_REGISTRY:
                logger.info(f"Detected model family '{model_type}' from config.model_type")
                return model_type

    # 2. Try name-based detection
    name_lower = model_name_or_path.lower().replace("-", "").replace("_", "")

    for key in MODEL_REGISTRY:
        if key in name_lower:
            logger.info(f"Detected model family '{key}' from model name")
            return key

    # 3. Try loading config without model
    if model is None:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name_or_path)
            if hasattr(config, "architectures") and config.architectures:
                for arch in config.architectures:
                    if arch in ARCHITECTURE_PATTERNS:
                        family = ARCHITECTURE_PATTERNS[arch]
                        logger.info(f"Detected model family '{family}' from remote config")
                        return family
            if hasattr(config, "model_type"):
                model_type = config.model_type.lower()
                if model_type in MODEL_REGISTRY:
                    return model_type
        except Exception as e:
            logger.debug(f"Could not load config for detection: {e}")

    logger.warning(f"Could not detect model family for '{model_name_or_path}', using generic")
    return "generic"


def get_model_handler(
    model: nn.Module,
    model_name: str = "",
    family: Optional[str] = None,
) -> TransformerModel:
    """
    Get the appropriate model handler for a loaded model.

    Args:
        model: Loaded HuggingFace model.
        model_name: Model name or path.
        family: Override family detection.

    Returns:
        TransformerModel instance (or subclass).
    """
    if family is None:
        family = detect_model_family(model_name, model)

    handler_class = MODEL_REGISTRY.get(family, TransformerModel)
    handler = handler_class(model, model_name)

    logger.info(
        f"Using {handler_class.__name__} handler for '{model_name}' "
        f"(family={family})"
    )

    return handler


def list_supported_families() -> list:
    """Return a sorted list of supported model families."""
    return sorted(set(MODEL_REGISTRY.keys()))
