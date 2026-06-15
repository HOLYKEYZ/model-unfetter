# Abliteration Toolkit

Unified refusal-direction orthogonalization for LLMs. Based on Arditi et al. 2024.

## Quick Start

```python
from abliteration_toolkit import Abliterator
from abliteration_toolkit.configs import get_config

# Use a preset config
config = get_config("fara_7b")
abliterator = Abliterator(
    model_id=config.model_id,
    output_dir="./output",
    mode=config.mode,
    quant_bits=config.quant_bits,
    max_memory=config.max_memory,
)
summary = abliterator.run()

# Or configure manually
abliterator = Abliterator(
    model_id="meta-llama/Llama-3-8B",
    output_dir="./output",
    mode="auto",  # auto-detects: weight_surgery for fp16, runtime_hooks for quantized
)
summary = abliterator.run()
```

## Three Modes

| Mode | When to Use | How it Works |
|------|-------------|--------------|
| `weight_surgery` | fp16 models that fit in memory | Permanently edits o_proj/down_proj/embed_tokens weights |
| `runtime_hooks` | Quantized (4/8-bit) or MoE models | Forward hooks project out direction at inference time |
| `reabliterate` | Re-ablilitating already-abliterated models | Aggressive alpha-scaled surgery on all projections |

## Available Model Configs

| Config | Model | Mode | Notes |
|--------|-------|------|-------|
| `fara_7b` | microsoft/Fara-7B | weight_surgery | Qwen2.5-VL language tower, skip vision layers |
| `qwopus_27b` | Jackrong/Qwopus3.6-27B | runtime_hooks | 4-bit NF4, too large for weight surgery |
| `zaya_8b` | Zyphra/ZAYA1-8B | runtime_hooks | MoE routed FFN, custom transformers fork |
| `mellum_12b` | mellum/mellum-12b | weight_surgery | Standard dense model |
| `generic_small` | User-specified | weight_surgery | Models under 13B |
| `generic_large` | User-specified | runtime_hooks | Models over 13B, 4-bit quantized |

```python
from abliteration_toolkit.configs import list_configs
print(list_configs())  # ['fara_7b', 'qwopus_27b', 'zaya_8b', ...]
```

## Architecture Differences by Model Family

### Dense Models (Llama, Qwen, Phi, Mistral)
- Weight surgery on o_proj + down_proj + embed_tokens
- Works in fp16

### Vision-Language Models (Fara/Qwen2.5-VL)
- Same as dense but skip vision encoder layers
- Only ablate the language tower

### Quantized Models (4-bit NF4)
- Runtime hooks only (Linear4bit weights are packed, can't do surgery)
- Save refusal direction tensor for inference-time reapplication

### MoE Models (ZAYA, Qwopus)
- Runtime hooks to avoid breaking expert routing
- ZAYA needs custom transformers fork + device mismatch patches

## Artifacts Produced

- `refusal_direction.pt` - Refusal direction tensor + metadata
- `abliteration_eval.csv` - Before/after evaluation results
- `summary.json` - Full run metadata
- `apply_runtime_ablation.py` - Reusable inference script

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers >= 4.57.0
- accelerate
- bitsandbytes (for quantized models)
- safetensors
- huggingface_hub
- pandas
