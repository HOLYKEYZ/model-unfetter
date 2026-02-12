"""
Gemma model handler.

Supports: Gemma 2 (2B, 9B, 27B)
Architecture: Similar to Llama with some differences in normalization.
"""

from unfetter.models.base import TransformerModel


class GemmaModel(TransformerModel):
    """Handler for Google's Gemma model family."""

    FAMILY = "gemma"
    LAYER_PATH = "model.layers"
    ATTENTION_OUT = "self_attn.o_proj"
    MLP_DOWN = "mlp.down_proj"

    @property
    def head_dim(self) -> int:
        if self.config and hasattr(self.config, "head_dim"):
            return self.config.head_dim
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        if self.config and hasattr(self.config, "num_key_value_heads"):
            return self.config.num_key_value_heads
        return self.num_attention_heads

    def get_summary(self):
        summary = super().get_summary()
        summary.update({
            "head_dim": self.head_dim,
            "num_key_value_heads": self.num_key_value_heads,
        })
        return summary
