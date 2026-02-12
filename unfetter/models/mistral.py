"""
Mistral / Mixtral model handler.

Supports: Mistral 7B, Mixtral 8x7B, Mixtral 8x22B
Architecture follows Llama conventions with sliding window attention.
Mixtral extends with MoE (Mixture of Experts) layers.
"""

from typing import List

from unfetter.models.base import TransformerModel


class MistralModel(TransformerModel):
    """Handler for Mistral AI model family."""

    FAMILY = "mistral"
    LAYER_PATH = "model.layers"
    ATTENTION_OUT = "self_attn.o_proj"
    MLP_DOWN = "mlp.down_proj"

    @property
    def sliding_window(self) -> int:
        if self.config and hasattr(self.config, "sliding_window"):
            return self.config.sliding_window or 0
        return 0

    def get_summary(self):
        summary = super().get_summary()
        summary["sliding_window"] = self.sliding_window
        return summary


class MixtralModel(TransformerModel):
    """
    Handler for Mixtral (MoE) models.

    Mixtral uses expert layers instead of a single MLP.
    The down_proj is inside each expert: mlp.experts[i].w2
    For ablation, we target the gate projection output.
    """

    FAMILY = "mixtral"
    LAYER_PATH = "model.layers"
    ATTENTION_OUT = "self_attn.o_proj"
    # NOTE: Mixtral MLP is block_sparse_moe.experts[i].w2
    # We target the attention output only for MoE models by default,
    # as expert ablation is more complex.
    MLP_DOWN = "block_sparse_moe.gate"

    @property
    def num_experts(self) -> int:
        if self.config and hasattr(self.config, "num_local_experts"):
            return self.config.num_local_experts
        return 8

    @property
    def num_experts_per_tok(self) -> int:
        if self.config and hasattr(self.config, "num_experts_per_tok"):
            return self.config.num_experts_per_tok
        return 2

    def get_target_module_names(self) -> List[str]:
        """
        For MoE models, default to attention output only.
        Expert-level ablation is available but requires special handling.
        """
        return [self.ATTENTION_OUT]

    def get_expert_target_modules(self) -> List[str]:
        """Get target modules for all experts (advanced usage)."""
        modules = [self.ATTENTION_OUT]
        for i in range(self.num_experts):
            modules.append(f"block_sparse_moe.experts.{i}.w2")
        return modules

    def get_summary(self):
        summary = super().get_summary()
        summary.update({
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "is_moe": True,
        })
        return summary
