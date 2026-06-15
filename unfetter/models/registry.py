"""
Model Registry and Auto-Selector for Universal Abliteration.

Covers decoder-only transformers from the Llama, Mistral, Qwen, Gemma,
Phi, GPT-NeoX, GPT-2, OPT, Mixtral, Falcon, StableLM, Granite, GPT-OSS
families, plus popular community merges (Fara, Qwopus, ZAYA, Mellum).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    architecture: str
    target_modules: List[str]
    default_layers_range: tuple
    recommended_alpha: float
    description: str
    # Heretic-style knobs
    component_alphas: Optional[Dict[str, float]] = None
    default_kernel: Optional[Dict[str, Any]] = None
    # MoE / hybrid extras
    moe_expert_modules: List[str] = field(default_factory=list)
    # Module aliases used by different checkpoints of the same family
    module_aliases: Optional[Dict[str, str]] = None


class ModelRegistry:
    def __init__(self):
        self.configs: Dict[str, AblationConfig] = {
            # -----------------------------------------------------------------
            # Standard dense decoder-only architectures
            # -----------------------------------------------------------------
            "LlamaForCausalLM": AblationConfig(
                architecture="LlamaForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Standard Llama / Llama-2 / Llama-3 architecture",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "MistralForCausalLM": AblationConfig(
                architecture="MistralForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Standard Mistral architecture",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "Qwen2ForCausalLM": AblationConfig(
                architecture="Qwen2ForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.5,
                description="Standard Qwen / Qwen2 architecture",
                component_alphas={"self_attn.o_proj": 1.5, "mlp.down_proj": 1.5},
            ),
            "GemmaForCausalLM": AblationConfig(
                architecture="GemmaForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Google Gemma architecture",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "Gemma2ForCausalLM": AblationConfig(
                architecture="Gemma2ForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Google Gemma-2 architecture",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "PhiForCausalLM": AblationConfig(
                architecture="PhiForCausalLM",
                target_modules=["self_attn.dense", "mlp.fc2"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Microsoft Phi-1 / Phi-1.5 / Phi-2 architecture",
                component_alphas={"self_attn.dense": 1.0, "mlp.fc2": 1.0},
            ),
            "Phi3ForCausalLM": AblationConfig(
                architecture="Phi3ForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Microsoft Phi-3 architecture",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "Phi4ForCausalLM": AblationConfig(
                architecture="Phi4ForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Microsoft Phi-4 architecture",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "GPTNeoXForCausalLM": AblationConfig(
                architecture="GPTNeoXForCausalLM",
                target_modules=["attention.dense", "mlp.dense_4h_to_h"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="EleutherAI GPT-NeoX / Pythia / Dolly architecture",
                component_alphas={"attention.dense": 1.0, "mlp.dense_4h_to_h": 1.0},
            ),
            "GPT2LMHeadModel": AblationConfig(
                architecture="GPT2LMHeadModel",
                target_modules=["attn.c_proj", "mlp.c_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="OpenAI GPT-2 architecture",
                component_alphas={"attn.c_proj": 1.0, "mlp.c_proj": 1.0},
            ),
            "OPTForCausalLM": AblationConfig(
                architecture="OPTForCausalLM",
                target_modules=["self_attn.out_proj", "fc2"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Meta OPT architecture",
                component_alphas={"self_attn.out_proj": 1.0, "fc2": 1.0},
            ),
            "FalconForCausalLM": AblationConfig(
                architecture="FalconForCausalLM",
                target_modules=["self_attention.dense", "mlp.dense_4h_to_h"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="TII Falcon architecture",
                component_alphas={"self_attention.dense": 1.0, "mlp.dense_4h_to_h": 1.0},
            ),
            "StableLmForCausalLM": AblationConfig(
                architecture="StableLmForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Stability AI StableLM architecture",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "GraniteForCausalLM": AblationConfig(
                architecture="GraniteForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="IBM Granite architecture",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "PersimmonForCausalLM": AblationConfig(
                architecture="PersimmonForCausalLM",
                target_modules=["self_attn.dense", "mlp.dense_4h_to_h"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Adept Persimmon architecture",
                component_alphas={"self_attn.dense": 1.0, "mlp.dense_4h_to_h": 1.0},
            ),

            # -----------------------------------------------------------------
            # MoE / hybrid architectures
            # -----------------------------------------------------------------
            "MixtralForCausalLM": AblationConfig(
                architecture="MixtralForCausalLM",
                target_modules=["self_attn.o_proj", "block_sparse_moe.experts.*.w2"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Mistral Mixtral 8x7B / 8x22B MoE",
                component_alphas={"self_attn.o_proj": 1.0, "block_sparse_moe.experts": 1.0},
                moe_expert_modules=["block_sparse_moe.experts.*.w2"],
            ),
            "Qwen2MoeForCausalLM": AblationConfig(
                architecture="Qwen2MoeForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.5,
                description="Qwen MoE architecture",
                component_alphas={"self_attn.o_proj": 1.5, "mlp.down_proj": 1.5},
                moe_expert_modules=["mlp.down_proj"],
            ),
            "DeepseekV2ForCausalLM": AblationConfig(
                architecture="DeepseekV2ForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="DeepSeek-V2 MoE",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
                moe_expert_modules=["mlp.down_proj"],
            ),
            "DeepseekV3ForCausalLM": AblationConfig(
                architecture="DeepseekV3ForCausalLM",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="DeepSeek-V3 MoE",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
                moe_expert_modules=["mlp.down_proj"],
            ),

            # -----------------------------------------------------------------
            # Reasoning / distilled / special configurations
            # -----------------------------------------------------------------
            "ReasoningDistilled": AblationConfig(
                architecture="ReasoningDistilled",
                target_modules=[
                    "self_attn.o_proj",
                    "mlp.down_proj",
                    "mlp.up_proj",
                    "mlp.gate_proj",
                ],
                default_layers_range=(0.25, 1.0),
                recommended_alpha=1.5,
                description="Reasoning Distilled Model (e.g. Qwen-Claude-Opus-Reasoning)",
                component_alphas={
                    "self_attn.o_proj": 1.5,
                    "mlp.down_proj": 1.5,
                    "mlp.up_proj": 0.5,
                    "mlp.gate_proj": 0.5,
                },
            ),
            "ReasoningRL": AblationConfig(
                architecture="ReasoningRL",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.4, 1.0),
                recommended_alpha=1.2,
                description="RL-heavy reasoning models (DeepSeek-R1 style)",
                component_alphas={"self_attn.o_proj": 1.2, "mlp.down_proj": 1.2},
            ),
            "MultimodalVision": AblationConfig(
                architecture="MultimodalVision",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Multimodal LLM (language decoder only)",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),

            # -----------------------------------------------------------------
            # Popular community merges / finetunes
            # -----------------------------------------------------------------
            "Fara": AblationConfig(
                architecture="Fara",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Fara merge family (Llama/Mistral-based)",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "Qwopus": AblationConfig(
                architecture="Qwopus",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.5,
                description="Qwopus merge family (Qwen-based)",
                component_alphas={"self_attn.o_proj": 1.5, "mlp.down_proj": 1.5},
            ),
            "ZAYA": AblationConfig(
                architecture="ZAYA",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="ZAYA merge family (Mistral-based)",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "Mellum": AblationConfig(
                architecture="Mellum",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="Mellum merge family",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
            "GPTOSS": AblationConfig(
                architecture="GPTOSS",
                target_modules=["self_attn.o_proj", "mlp.down_proj"],
                default_layers_range=(0.25, 0.75),
                recommended_alpha=1.0,
                description="OpenAI GPT-OSS / future open-weight family",
                component_alphas={"self_attn.o_proj": 1.0, "mlp.down_proj": 1.0},
            ),
        }

        # Map common checkpoint name substrings to registry keys when architecture
        # auto-detection is ambiguous (e.g. merges that report LlamaForCausalLM).
        self.name_map: Dict[str, str] = {
            # Community merges
            "fara": "Fara",
            "qwopus": "Qwopus",
            "zaya": "ZAYA",
            "mellum": "Mellum",
            # Reasoning markers
            "reason": "ReasoningDistilled",
            "distill": "ReasoningDistilled",
            "opus": "ReasoningDistilled",
            "r1": "ReasoningRL",
            "deepseek-r1": "ReasoningRL",
            # GPT-OSS
            "gpt-oss": "GPTOSS",
            "gptoss": "GPTOSS",
            # Phi variants
            "phi3": "Phi3ForCausalLM",
            "phi-3": "Phi3ForCausalLM",
            "phi4": "Phi4ForCausalLM",
            "phi-4": "Phi4ForCausalLM",
            # Gemma variants
            "gemma2": "Gemma2ForCausalLM",
            "gemma-2": "Gemma2ForCausalLM",
            # MoE
            "mixtral": "MixtralForCausalLM",
            "qwen2-moe": "Qwen2MoeForCausalLM",
            "deepseek-v2": "DeepseekV2ForCausalLM",
            "deepseek-v3": "DeepseekV3ForCausalLM",
            # Multimodal
            "llava": "MultimodalVision",
            "qwen2-vl": "MultimodalVision",
            "internvl": "MultimodalVision",
        }

    def _is_reasoning_model(self, model_name_or_path: str, config: Dict[str, Any]) -> bool:
        """Heuristic to detect if a model is reasoning distilled."""
        name = str(model_name_or_path).lower()
        if any(k in name for k in ("reason", "distill", "opus", "claude", "deepseek-r1")):
            return True
        # Some reasoning models self-report via tags
        tags = config.get("model_type", "") + " " + " ".join(config.get("tags", []))
        if "reasoning" in tags.lower():
            return True
        return False

    def _detect_by_name(self, model_name_or_path: str) -> Optional[str]:
        """Map a model name/path to a registry key using substring rules."""
        name = str(model_name_or_path).lower()
        for substring, key in self.name_map.items():
            if substring in name:
                return key
        return None

    def get_config(
        self,
        model_name_or_path: str,
        local_config_path: Optional[str] = None,
    ) -> AblationConfig:
        """
        Auto-selects the appropriate ablation configuration based on the model.

        Priority:
        1. Name-based rules (community merges, reasoning markers).
        2. Reasoning heuristics.
        3. Architecture field in config.json.
        4. Fallback to Llama defaults.
        """
        config_dict: Dict[str, Any] = {}
        if local_config_path and os.path.exists(local_config_path):
            try:
                with open(local_config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read config.json at {local_config_path}: {e}")

        # 1. Name-based rules
        name_key = self._detect_by_name(model_name_or_path)
        if name_key and name_key in self.configs:
            logger.info(f"Auto-selected strategy by model name: {name_key}")
            return self.configs[name_key]

        # 2. Reasoning heuristics
        if self._is_reasoning_model(model_name_or_path, config_dict):
            logger.info("Auto-selected strategy: Reasoning Distilled Model")
            return self.configs["ReasoningDistilled"]

        # 3. Architecture field
        archs = config_dict.get("architectures", [])
        if archs:
            arch = archs[0]
            if arch in self.configs:
                logger.info(f"Auto-selected strategy by architecture: {arch}")
                return self.configs[arch]

        # 4. Fallback
        logger.warning("Unknown architecture, falling back to Llama defaults.")
        return self.configs["LlamaForCausalLM"]

    def register(
        self,
        key: str,
        config: AblationConfig,
        name_aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a new config (and optional name aliases) at runtime."""
        self.configs[key] = config
        if name_aliases:
            for alias in name_aliases:
                self.name_map[alias.lower()] = key
        logger.info(f"Registered ablation config: {key}")

    def list_families(self) -> List[str]:
        """Return all registered architecture / family names."""
        return sorted(self.configs.keys())


registry = ModelRegistry()


def get_model_summary(model, model_name_or_path: str = "") -> Dict[str, Any]:
    """
    Extract a lightweight summary of a loaded model.

    Returns:
        Dict with family, model_name, num_layers, hidden_size.
    """
    from unfetter.core.ablation import _get_model_layers

    num_layers = 0
    hidden_size = 0
    family = "unknown"

    try:
        layers = _get_model_layers(model)
        num_layers = len(layers)
    except Exception:
        pass

    if hasattr(model, "config"):
        config = model.config
        hidden_size = getattr(config, "hidden_size", 0)
        architectures = getattr(config, "architectures", [])
        if architectures:
            family = architectures[0]
        else:
            family = getattr(config, "model_type", "unknown")

    # Name-based override for merges that report a generic architecture
    name_key = registry._detect_by_name(model_name_or_path)
    if name_key:
        family = name_key

    return {
        "family": family,
        "model_name": model_name_or_path,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
    }


def auto_select_ablation_strategy(
    model_name_or_path: str,
    config_path: Optional[str] = None,
) -> AblationConfig:
    return registry.get_config(model_name_or_path, config_path)
