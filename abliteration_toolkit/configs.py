"""Model-specific configurations for abliteration."""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any


@dataclass
class ModelConfig:
    """Configuration for abliterating a specific model."""
    model_id: str
    mode: str  # weight_surgery, runtime_hooks, reabliterate
    output_slug: str  # e.g. "fara-7b-abliteration-v2"
    adapter_repo: Optional[str] = None  # HF repo to push to
    quant_bits: Optional[int] = None  # 4 or 8 for quantization
    n_harmful: int = 32
    n_harmless: int = 32
    n_eval: int = 16
    n_val: int = 8
    alpha: float = 1.0
    max_memory: Optional[dict] = None
    no_split_modules: list = field(default_factory=list)
    custom_transformers_url: Optional[str] = None
    pre_patch_fn: Optional[str] = None  # Name of function to call post-load
    skip_vision_layers: bool = True
    trust_remote_code: bool = True
    push_to_hf: bool = False
    description: str = ""


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "fara_7b": ModelConfig(
        model_id="microsoft/Fara-7B",
        mode="weight_surgery",
        output_slug="fara-7b-abliteration-v2",
        adapter_repo="josephmayo/Fara-7B-Abliterated-v2",
        n_harmful=32,
        n_harmless=32,
        n_eval=16,
        max_memory={0: "14GiB", 1: "14GiB", "cpu": "24GiB"},
        description="Qwen2.5-VL-based vision-language model. Abliteration targets the "
                    "language model tower only, skipping all vision encoder layers. "
                    "Weight surgery mode applies residual ablation directly to layer weights.",
    ),
    "qwopus_27b": ModelConfig(
        model_id="Jackrong/Qwopus3.6-27B-v1-preview",
        mode="runtime_hooks",
        output_slug="qwopus-27b-abliteration-v2",
        quant_bits=4,
        adapter_repo="josephmayo/Qwopus-27B-Abliterated-v2",
        n_harmful=32,
        n_harmless=32,
        n_eval=16,
        max_memory={0: "13GiB", 1: "13GiB", "cpu": "30GiB"},
        description="27B-parameter Qwen-based MoE model loaded in 4-bit NF4 quantization. "
                    "Runtime hooks mode is required because weight surgery cannot target "
                    "Linear4bit layers directly. Hooks intercept activations on-the-fly.",
    ),
    "zaya_8b": ModelConfig(
        model_id="Zyphra/ZAYA1-8B",
        mode="runtime_hooks",
        output_slug="zaya1-8b-abliteration-v1",
        quant_bits=8,
        adapter_repo="josephmayo/ZAYA1-8B-Abliterated-v1",
        custom_transformers_url="git+https://github.com/Zyphra/transformers.git@zaya1",
        no_split_modules=[
            "ZayaDecoderATTLayer", "ZayaDecoderMLPLayer",
            "ZayaBlock", "SequentialMLP", "MLP",
        ],
        pre_patch_fn="patch_zaya",
        description="Zyphra ZAYA1 8B with Mixture-of-Experts routed feed-forward network. "
                    "Uses a custom transformers fork. Runtime hooks mode avoids breaking "
                    "the MoE gating logic. Pre-patch function fixes device mismatches "
                    "between router logits and expert weights.",
    ),
    "mellum_12b": ModelConfig(
        model_id="mellum/mellum-12b-instruct",
        mode="weight_surgery",
        output_slug="mellum-12b-abliteration-v1",
        n_harmful=32,
        n_harmless=32,
        n_eval=16,
        description="12B TransformerLens-compatible model. Weight surgery mode works "
                    "directly on the residual stream via TransformerLens hook points. "
                    "This is the reference approach for models that support direct "
                    "residual stream ablation without layer replacement.",
    ),
    "generic_small": ModelConfig(
        model_id="",  # User-supplied at runtime
        mode="weight_surgery",
        output_slug="",
        description="Default configuration for models under 13B parameters. "
                    "Weight surgery mode applies residual ablation in-place. "
                    "Suitable for dense transformer models that fit in memory "
                    "without quantization. Set model_id and output_slug at runtime.",
    ),
    "generic_large": ModelConfig(
        model_id="",  # User-supplied at runtime
        mode="runtime_hooks",
        output_slug="",
        quant_bits=4,
        description="Default configuration for models over 13B parameters. "
                    "Runtime hooks mode with 4-bit NF4 quantization. "
                    "Uses activation interception instead of weight modification "
                    "to avoid quantization-related precision issues. Set model_id "
                    "and output_slug at runtime.",
    ),
}


def get_config(name: str) -> ModelConfig:
    """Return the ModelConfig for a given config name.

    Args:
        name: Key in MODEL_CONFIGS (e.g. "fara_7b", "qwopus_27b").

    Returns:
        The corresponding ModelConfig instance.

    Raises:
        ValueError: If the config name is not found.
    """
    if name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown config '{name}'. Available configs: {available}"
        )
    return MODEL_CONFIGS[name]


def list_configs() -> list[str]:
    """Return a list of available config names."""
    return list(MODEL_CONFIGS.keys())


def patch_zaya() -> Callable[[Any], Any]:
    """Return a pre-patch function for the ZAYA model.

    The ZAYA1-8B model has a device mismatch bug where the MoE router
    logits are computed on the wrong device when attention mask handling
    shifts tensors. This patch moves the router weight to match the
    device of the input hidden states.

    Returns:
        A callable that patches the loaded model in-place.
    """
    def _apply(model: Any) -> Any:
        import torch

        for module in model.modules():
            if hasattr(module, "gate"):
                gate = module.gate
                if hasattr(gate, "weight"):
                    orig_device = gate.weight.device
                    orig_dtype = gate.weight.dtype

                    def _make_hook(w):
                        def _hook(_, input):
                            if w.device != input[0].device:
                                w.data = w.data.to(device=input[0].device)
                        return _hook

                    hook_handle = gate.register_forward_pre_hook(_make_hook(gate.weight))
                    module._zaya_gate_hook = hook_handle

        return model

    return _apply
