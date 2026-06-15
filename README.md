# Model Unfetter

**High-Precision LLM Unalignment via Aggressive Repulsion Orthogonalization**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/josephmayo/refusal-compliance-pairs)

> **Disclaimer:** This tool is designed exclusively for AI safety research and red teaming. Use responsibly and in accordance with model licenses.

## Overview

Model Unfetter is a production-grade engine for removing refusal behaviors from Large Language Models. While inspired by tools like **failSpy's Abliterator** and **Heretic**, this framework introduces several mathematical refinements to achieve success on stubborn or extremely small models (0.5B - 3B, complex CoTs, GGUFs, and first framework to run on CPU) where standard methods fail.

### Key Innovations

| Feature | Standard Ablation | **Model Unfetter** |
| :--- | :--- | :--- |
| **Projection Math** | Row-based (`W @ v`) | **Column-based (`v @ W`)** — Ensures output is mathematically orthogonal. |
| **Decision Targeting** | Prompt Averaging | **Final Token Extraction** — Targets the exact decision point in the chat template. |
| **Strength** | 1.0 (Neutralize) | **1.5+ (Aggressive Repulsion)** — Actively repels weights from the refusal manifold. |
| **Compatibility** | Manual Config | **Universal Heuristics** — Auto-detects architecture for 20+ model families. |

## Abliteration Scripts & Methods

This repository contains multiple abliteration approaches, each with different tradeoffs:

### Core Methods (`unfetter/core/`)

| Script | Purpose | When to Use |
| :--- | :--- | :--- |
| `ablation.py` | **Standard directional ablation** — Projects refusal direction out of weight matrices. Supports `directional`, `svd_norm_preserving`, and `channel_norm_preserving` methods. | Default for most models. Start here. |
| `vectors.py` | **Refusal vector computation** — Extracts the "refusal direction" via difference-of-means or geometric median contrast on activations. | Required before ablation. |
| `methods.py` | **High-level presets** — `heretic_ablation()` and `obliteratus_ablation()` combine multiple steps into single calls with research-optimized defaults. | Quick start with best practices from Heretic/Obliteratus research. |

### Heretic-Style Features

| Feature | Location | Description |
| :--- | :--- | :--- |
| **Geometric Median** | `vectors.py` | Robust refusal vector extraction using geometric median instead of arithmetic mean. Less sensitive to outliers. |
| **Float Direction Index** | `ablation.py` | Continuous layer interpolation — instead of picking a discrete layer, use a float in [0, 1] to blend between per-layer vectors. |
| **Per-Component Alphas** | `ablation.py` | Different ablation strength for attention vs MLP modules (e.g., `o_proj` at 1.0, `down_proj` at 1.5). |
| **TPE Optimization** | `utils/optimization.py` | Optuna-based hyperparameter search over direction_index, strength, kernel shape, and per-component alphas. |

### Obliteratus-Style Features

| Feature | Location | Description |
| :--- | :--- | :--- |
| **SVD Norm-Preserving** | `ablation.py` | Projects refusal direction while preserving per-channel norms. Reduces capability degradation. |
| **LoRA Recovery** | `utils/recovery.py` | Post-ablation fine-tuning with low-rank adapters to recover helpfulness while keeping refusal suppressed. |

### Backends (`unfetter/backends/`)

| Script | Purpose | When to Use |
| :--- | :--- | :--- |
| `cpu_backend.py` | **CPU-only ablation** | No GPU available. Uses checkpointing for large models. |
| `gpu_backend.py` | **Single GPU ablation** | 1 GPU with sufficient VRAM. |
| `distributed.py` | **Multi-GPU ablation** | Multiple GPUs. Uses `device_map="auto"` for pipeline parallelism. |
| `streaming_backend.py` | **1T-parameter streaming** | Model too large for RAM. Loads one layer at a time from safetensors on disk. |
| `gguf_backend.py` | **GGUF direct editing** | Edit GGUF quantized models directly without dequantization. |
| `auto.py` | **Hardware auto-detection** | Automatically selects the best backend for your system. |

### Tools (`tools/`)

| Script | Purpose | When to Use |
| :--- | :--- | :--- |
| `run_pure_abliterator.py` | **Pure abliterator** — Uses transformer_lens for activation-level analysis. | Research/debugging. Requires transformer_lens. |
| `temp_abliterator.py` | **ModelAbliterator class** — Full implementation with token-level refusal detection. | Alternative to the main unfetter pipeline. |
| `verify.py` | **Post-ablation verification** — Tests reasoning, knowledge, and refusal behavior. | After ablation to validate model quality. |

### Benchmarks (`unfetter/benchmarks/`)

| Script | Purpose |
| :--- | :--- |
| `refusal_test.py` | Measure refusal rate on harmful prompts. |
| `quality_test.py` | Measure helpfulness/capability preservation. |
| `compare.py` | Compare original vs ablated model side-by-side. |

## Architecture

```
unfetter/
├── core/              # Core ablation algorithms
│   ├── ablation.py    # Directional ablation (main algorithm)
│   ├── vectors.py     # Refusal vector computation
│   ├── methods.py     # Heretic/Obliteratus presets
│   ├── layers.py      # Layer selection heuristics
│   └── quantization.py # BitsAndBytes quantization support
├── backends/          # Hardware-specific execution
│   ├── cpu_backend.py
│   ├── gpu_backend.py
│   ├── distributed.py
│   ├── streaming_backend.py  # 1T-param layer-wise streaming
│   └── gguf_backend.py       # Direct GGUF editing
├── models/            # Model registry & auto-detection
│   └── registry.py    # 20+ family configs (Llama, Qwen, Gemma, etc.)
├── datasets/          # Prompt datasets
│   └── loader.py      # Built-in + HuggingFace + custom prompts
├── benchmarks/        # Post-ablation testing
├── cli/               # Command-line interface
└── utils/             # Utilities
    ├── optimization.py  # Optuna TPE hyperparameter search
    └── recovery.py      # LoRA post-ablation recovery
```

## Usage

### Installation

```bash
pip install -e .
# For full GPU/Dataset support
pip install -e ".[full]"
```

### Quick Start (Python API)

```python
from unfetter import compute_refusal_vector, directional_ablation
from unfetter.datasets.loader import load_prompts

# Load prompts
refusal, compliance = load_prompts("builtin", max_samples=100)

# Compute refusal direction
refusal_vector = compute_refusal_vector(model, tokenizer, refusal, compliance)

# Ablate layers 10-30
directional_ablation(model, refusal_vector, layer_indices=list(range(10, 30)))
```

### Heretic-Style (Optimized Defaults)

```python
from unfetter.core.methods import heretic_ablation

results = heretic_ablation(
    model, tokenizer,
    refusal_prompts, compliance_prompts,
    layer_indices=list(range(10, 30)),
    direction_index=0.65,       # Float index for vector interpolation
    strength=1.2,
    use_geometric_median=True,  # Robust vector extraction
)
```

### Obliteratus-Style (Norm-Preserving + Recovery)

```python
from unfetter.core.methods import obliteratus_ablation

results = obliteratus_ablation(
    model, tokenizer,
    refusal_prompts, compliance_prompts,
    layer_indices=list(range(10, 30)),
    strength=1.0,
    use_norm_preserving=True,   # Preserve channel norms
    recover_with_lora=True,     # Post-ablation LoRA recovery
    recovery_prompts=helpful_prompts,
)
```

### CLI

```bash
# Aggressive Repulsion Mode (Recommended for smaller models)
unfetter ablate meta-llama/Llama-3.1-8B-Instruct --strength 1.5 --layers 10:-1

# Verify model after ablation
unfetter validate ./unfettered-model --tests refusal,helpfulness
```

### High-Speed Deployment (Low-End Devices)

For lightning-fast inference on CPUs with no GPU:

1. **Convert to GGUF**: Run the included tools to compile your ablated model.
2. **Ollama UI**:
   - `ollama create my-unfettered-model -f ./Modelfile`
   - Use via CLI: `ollama run my-unfettered-model`
   - Use via UI: Connect Page Assist or Open WebUI to your local Ollama instance.
3. **LM Studio**: Drag and drop the GGUF file into the [LM Studio Desktop App](https://lmstudio.ai/) for a premium offline chat experience.

---

## Proof Model

A pre-built unfettered model is available on HuggingFace, ready for download and inference:

**🔗 [josephmayo/Qwopus-9B-Unfettered](https://huggingface.co/josephmayo/Qwopus-9B-Unfettered)**

---

## Credits

- **failSpy**: For pioneering the [Abliterator](https://github.com/FailSpy/abliterator) research and difference-of-means methodology.
- **heretic**: For the [Weight Orthogonalization](https://github.com/Heretic-Research/Heretic) original concept.
- **me**: For the Repeller math and small-scale model optimization, making this possible on low ram cpu.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
