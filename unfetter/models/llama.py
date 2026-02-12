"""
Llama family model handler.

Supports: Llama 2, Llama 3, Llama 3.1, Llama 3.2, Llama 3.3
Architecture: model.layers[i].self_attn.o_proj / model.layers[i].mlp.down_proj
"""

from unfetter.models.base import TransformerModel


class LlamaModel(TransformerModel):
    """Handler for Meta's Llama model family."""

    FAMILY = "llama"
    LAYER_PATH = "model.layers"
    ATTENTION_OUT = "self_attn.o_proj"
    MLP_DOWN = "mlp.down_proj"

    # Llama uses GQA (Grouped Query Attention) in newer versions
    @property
    def num_key_value_heads(self) -> int:
        if self.config and hasattr(self.config, "num_key_value_heads"):
            return self.config.num_key_value_heads
        return self.num_attention_heads

    @property
    def intermediate_size(self) -> int:
        if self.config and hasattr(self.config, "intermediate_size"):
            return self.config.intermediate_size
        return self.hidden_size * 4

    @property
    def rope_theta(self) -> float:
        """RoPE base frequency (higher in newer Llama versions)."""
        if self.config and hasattr(self.config, "rope_theta"):
            return self.config.rope_theta
        return 10000.0

    def get_summary(self):
        summary = super().get_summary()
        summary.update({
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "rope_theta": self.rope_theta,
        })
        return summary
