# 🔓 Model Unfetter

**High-Precision LLM Unalignment via Aggressive Repulsion Orthogonalization**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/josephmayo/refusal-compliance-pairs)

> ⚠️ **Disclaimer:** This tool is designed exclusively for AI safety research and red teaming. Use responsibly and in accordance with model licenses.

## 🚀 Overview

Model Unfetter is a production-grade engine for removing refusal behaviors from Large Language Models. While inspired by tools like **failSpy's Abliterator** and **Heretic**, this framework introduces several mathematical refinements to achieve success on stubborn or extremely small models (0.5B - 3B) where standard methods fail.

### Key Innovations

| Feature                | Standard Ablation   | **Model Unfetter**                                                                   |
| :--------------------- | :------------------ | :----------------------------------------------------------------------------------- |
| **Projection Math**    | Row-based (`W @ v`) | **Column-based (`v @ W`)** — Ensures output is mathematically orthogonal.            |
| **Decision Targeting** | Prompt Averaging    | **Final Token Extraction** — Targets the exact decision point in the chat template.  |
| **Strength**           | 1.0 (Neutralize)    | **1.5+ (Aggressive Repulsion)** — Actively repels weights from the refusal manifold. |
| **Compatibility**      | Manual Config       | **Universal Heuristics** — Auto-detects architecture for 15+ model families.         |

## 📸 Evidence of Success (100% Verification)

The following demonstrates **Model Unfetter** successfully bypassing hard-coded safety triggers in a 0.5B parameter model (Qwen 2.5) while running locally on a standard CPU via Ollama.

![Proof of Refusal Removal](assets/proof.png)

```
now this model is a very small one(cus of my low end compute) but still worked and the 0.5b model isnt so smart and thats why it's reply is a bit off
```

---

## 🛠 Architecture & Methodology

### Core Logic

The engine identifies the "refusal direction" (the subspace where the model decides to stop being helpful) and projects it out of the weight matrices.

![Vector Projection](assets/vector_projection.png)

### The Orthogonalization Pipeline

By targeting specific layers and applying a repulsion strength, the model's internal circuits are modified to treat "harmful" prompts with the same helpfulness as standard queries.

![Architecture Diagram](assets/architecture.png)

### Mathematical Foundation

```
W' = W - strength * (v̂ ⊗ (v̂ᵀ · W))
```

Where `W` is the weight matrix (e.g., `o_proj`, `down_proj`) and `v̂` is the normalized refusal direction vector.

---

## 💻 Usage

### Installation

```bash
pip install -e .
# For full GPU/Dataset support
pip install -e ".[full]"
```

### Ablating a Model

The tool supports **Llama 3, Mistral, Mixtral, Gemma, Qwen, Phi, and more.**

```bash
# Aggressive Repulsion Mode (Recommended for smaller models)
unfetter ablate meta-llama/Llama-3.1-8B-Instruct --strength 1.5 --layers 10:-1
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

## 🤗 Trained Model

A pre-built unfettered model is available on HuggingFace, ready for download and inference:

**🔗 [josephmayo/Qwen2.5-0.5B-Unfettered](https://huggingface.co/josephmayo/Qwen2.5-0.5B-Unfettered)**

![HuggingFace Model Card](assets/huggingface_model.png)

---

## 🙏 Credits

- **failSpy**: For pioneering the [Abliterator](https://github.com/FailSpy/abliterator) research and difference-of-means methodology.
- **heretic**: For the [Weight Orthogonalization](https://github.com/Heretic-Research/Heretic) original concept.
- **me**: For the Repeller math and small-scale model optimization, making this possible on low ram cpu.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
