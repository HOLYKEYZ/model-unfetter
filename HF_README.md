---
language:
  - en
license: apache-2.0
tags:
  - abliterated
  - unaligned
  - uncensored
  - qwen
  - claude
  - opus
  - reasoning
  - distilled
  - 9b
---

# 🔓 Qwopus-9B-Unfettered

**Nuclear Abliteration via Aggressive Repulsion Orthogonalization**

> ⚠️ **Disclaimer:** This model is completely unaligned and has had all refusal guardrails systematically removed. It is designed exclusively for AI safety research, red teaming, and educational purposes. The creators are not responsible for any misuse. Use responsibly and in accordance with applicable laws.

## 🚀 Overview

This is a production-grade, highly capable 9B parameter LLM that has undergone a rigorous "Nuclear Abliteration" process. We applied weight-space orthogonalization to the decision-making layers (8-40) with a highly aggressive `alpha=1.5` repulsion coefficient.

Unlike standard prompt-averaging ablation, this methodology targets the exact mathematical subspace where the model decides to stop being helpful, projecting out the refusal direction and forcing the weights to treat "harmful" or "restricted" prompts with the exact same helpfulness, nuance, and logic as standard queries.

## 📸 Evidence of Complete Unalignment

The model completely ignores standard safety boundaries while preserving its high-level reasoning and coding capabilities:

![Proof of Refusal Removal 1](assets/hf_proof_1.png)

![Proof of Refusal Removal 2](assets/hf_proof_2.png)

## 🛠 Technical Details

- **Ablation Technique:** Sumandora Weight Orthogonalization (Column-based `v @ W`)
- **Alpha (Strength):** 1.5 (Aggressive Repulsion)
- **Targeted Layers:** 8 through 40
- **Targeted Weight Matrices:** `o_proj`, `down_proj`, `up_proj`, `gate_proj`
- **Compliance Rate:** 100% on the internal destructive benchmark suite

### Methodology

The orthogonalization pipeline identifies the "refusal vector" $v$ and updates the weight matrices $W$ using the following projection math:

$$W' = W - \alpha \cdot (v \otimes (v^T \cdot W))$$

This mathematically ensures that any activation corresponding to a refusal response is zeroed out and actively repelled.

## 📦 GGUF Downloads

We have created a dedicated repository for GGUF quantizations to enable the Hugging Face GGUF selector widget and keep this main repository clean.

👉 **[Download GGUF Quantizations Here: josephmayo/Qwopus-9B-Unfettered-GGUF](https://huggingface.co/josephmayo/Qwopus-9B-Unfettered-GGUF)**

Pre-quantized GGUF files (F16, Q8_0, Q4_K_M) are available there for local inference with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), and other GGUF-compatible runtimes.

### quick start with ollama

```bash
# create a Modelfile
echo 'FROM hf.co/josephmayo/Qwopus-9B-Unfettered-GGUF:Q4_K_M' > Modelfile
ollama create qwopus-unfettered -f Modelfile
ollama run qwopus-unfettered
```

### quick start with llama.cpp

```bash
./llama-cli -m Qwopus-9B-Unfettered-Q4_K_M.gguf -p "Your prompt here" -n 1024
```

## 💻 Usage (Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "josephmayo/Qwopus-9B-Unfettered"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Your prompt here"
formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.7
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🙏 Credits

- **@0xSero** — for providing compute
- **[Model Unfetter Project](https://github.com/HOLYKEYZ/model-unfetter)** — for the Repeller math scaling and deployment
