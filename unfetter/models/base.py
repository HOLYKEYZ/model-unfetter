"""
Base transformer model handler.

Provides a generic interface for accessing model architecture details
across different transformer families.
"""

import logging
from typing import List, Optional, Tuple, Dict

import torch.nn as nn

logger = logging.getLogger(__name__)


class TransformerModel:
    """
    Generic wrapper for HuggingFace transformer models.

    Provides architecture-agnostic access to layers, hidden sizes,
    and target modules for ablation.
    """

    # Subclasses override these
    FAMILY = "generic"
    LAYER_PATH = "model.layers"
    ATTENTION_OUT = "self_attn.o_proj"
    MLP_DOWN = "mlp.down_proj"

    def __init__(self, model: nn.Module, model_name: str = ""):
        self.model = model
        self.model_name = model_name
        self.config = getattr(model, "config", None)

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        if self.config and hasattr(self.config, "num_hidden_layers"):
            return self.config.num_hidden_layers
        return len(self.get_layers())

    @property
    def hidden_size(self) -> int:
        """Embedding / hidden dimension."""
        if self.config and hasattr(self.config, "hidden_size"):
            return self.config.hidden_size
        raise ValueError("Cannot determine hidden_size from config")

    @property
    def num_attention_heads(self) -> int:
        if self.config and hasattr(self.config, "num_attention_heads"):
            return self.config.num_attention_heads
        return 0

    @property
    def vocab_size(self) -> int:
        if self.config and hasattr(self.config, "vocab_size"):
            return self.config.vocab_size
        return 0

    def get_layers(self) -> nn.ModuleList:
        """Get the list of transformer layers."""
        parts = self.LAYER_PATH.split(".")
        obj = self.model
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise ValueError(
                    f"Cannot access '{self.LAYER_PATH}' on model. "
                    f"Failed at '{part}'."
                )
        return obj

    def get_target_module_names(self) -> List[str]:
        """Return the target module names for ablation."""
        return [self.ATTENTION_OUT, self.MLP_DOWN]

    def get_layer_info(self, layer_idx: int) -> Dict:
        """Get info about a specific layer."""
        layers = self.get_layers()
        if layer_idx < 0:
            layer_idx = len(layers) + layer_idx

        layer = layers[layer_idx]
        param_count = sum(p.numel() for p in layer.parameters())

        return {
            "index": layer_idx,
            "parameter_count": param_count,
            "modules": [name for name, _ in layer.named_modules() if name],
        }

    def get_summary(self) -> Dict:
        """Return a summary of the model architecture."""
        return {
            "family": self.FAMILY,
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "vocab_size": self.vocab_size,
            "target_modules": self.get_target_module_names(),
            "layer_path": self.LAYER_PATH,
        }
