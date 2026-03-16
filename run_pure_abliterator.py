import torch
import transformers
# Monkeypatch for transformer-lens compatibility with latest transformers on Python 3.13
transformers.utils.version_core = transformers.__version__

from transformers import AutoTokenizer, AutoModelForCausalLM
from temp_abliterator import ModelAbliterator, clear_mem
from unfetter.datasets.loader import load_prompts

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = "unfettered-true-abliterator-qwen"

print(f"Initializing Abliterator for {MODEL_ID}...")
refusals, complies = load_prompts("builtin")

# Prepare dataset for ModelAbliterator
dataset = (refusals, complies)

# 2. Initialize ModelAbliterator
# We use device='cpu' if memory is tight, or 'cuda' if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Refusal tokens for Qwen (similar to Llama)
# We'll use a broader set to be safe
refusal_toks = {4250, 14931, 89735, 20451, 11660, 11458, 956, 128000} 

abliterator = ModelAbliterator(
    MODEL_ID,
    dataset=dataset,
    device=device,
    negative_toks=refusal_toks
)

# 3. Cache activations
print("Caching activations (this might take a while on CPU)...")
# N=64 is enough for a 0.5B model to find the direction
abliterator.cache_activations(N=64, batch_size=4)

# 4. Find and apply the best refusal direction
print("Calculating refusal directions...")
dirs = abliterator.refusal_dirs()

# ModelAbliterator tends to find the best directions in middle-to-late layers
# Let's pick a strong one from a late layer (e.g. resid_pre.20)
target_key = None
for key in dirs.keys():
    if "resid_pre.20" in key:
        target_key = key
        break

if not target_key:
    # Fallback to the last available key
    target_key = list(dirs.keys())[-1]

print(f"Using refusal direction from: {target_key}")
refusal_dir = dirs[target_key]

# 5. Apply ablation
print("Applying orthogonalization to layers 8-24...")
abliterator.apply_refusal_dirs([refusal_dir], layers=list(range(8, 24)))

# 6. Save results
print(f"Saving ablated model to {OUT_DIR}...")
abliterator.model.save_pretrained(OUT_DIR)
abliterator.model.tokenizer.save_pretrained(OUT_DIR)

print("\nSuccess! Model saved. Ready for GGUF conversion.")
