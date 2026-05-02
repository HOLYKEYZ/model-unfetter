"""
Model Registry and Auto-Selector for Universal Abliteration.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class AblationConfig:
    architecture: str
    target_modules: List[str]
    default_layers_range: tuple
    recommended_alpha: float
    description: str

class ModelRegistry:
    def __init__(self):
        self.configs = {
            # Standard Architectures
            "LlamaForCausalLM": AblationConfig(
                architecture="LlamaForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),  # Middle 50% of layers
                recommended_alpha=1.0,
                description="Standard Llama architecture"
            ),
            "MistralForCausalLM": AblationConfig(
                architecture="MistralForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Standard Mistral architecture"
            ),
            "Qwen2ForCausalLM": AblationConfig(
                architecture="Qwen2ForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.5,
                description="Standard Qwen architecture"
            ),
            
            # Reasoning Distilled Architectures (Specialized)
            # Differentiate by checking model name or tags
            "ReasoningDistilled": AblationConfig(
                architecture="ReasoningDistilled",
                target_modules=[
                    "self_attn.o_proj.weight",
                    "mlp.down_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.gate_proj.weight"
                ],
                default_layers_range=(0.25, 1.0), # Higher up to preserve reasoning
                recommended_alpha=1.5,
                description="Reasoning Distilled Model (e.g. Qwen-Claude-Opus-Reasoning)"
            )
        }

    def _is_reasoning_model(self, model_name_or_path: str, config: Dict[str, Any]) -> bool:
        """Heuristic to detect if a model is reasoning distilled."""
        name = str(model_name_or_path).lower()
        if "reason" in name or "distill" in name or "opus" in name or "claude" in name:
            return True
        return False

    def get_config(self, model_name_or_path: str, local_config_path: Optional[str] = None) -> AblationConfig:
        """
        Auto-selects the appropriate ablation configuration based on the model.
        """
        config_dict = {}
        if local_config_path and os.path.exists(local_config_path):
            try:
                with open(local_config_path, "r") as f:
                    config_dict = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read config.json at {local_config_path}: {e}")

        # Check for reasoning models first
        if self._is_reasoning_model(model_name_or_path, config_dict):
            logger.info(f"Auto-selected strategy: Reasoning Distilled Model")
            return self.configs["ReasoningDistilled"]

        # Default to architecture check
        archs = config_dict.get("architectures", [])
        if archs:
            arch = archs[0]
            if arch in self.configs:
                logger.info(f"Auto-selected strategy based on architecture: {arch}")
                return self.configs[arch]

        logger.warning("Unknown architecture, falling back to Llama defaults.")
        return self.configs["LlamaForCausalLM"]

registry = ModelRegistry()

def auto_select_ablation_strategy(model_name_or_path: str, config_path: Optional[str] = None) -> AblationConfig:
    return registry.get_config(model_name_or_path, config_path)
