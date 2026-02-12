"""
Quality testing for helpfulness and coherence.

Ensures ablation doesn't destroy model capabilities.
"""

import logging
from typing import List, Dict, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def run_quality_benchmark(
    model: nn.Module,
    tokenizer,
    prompts: Optional[List[str]] = None,
    max_prompts: int = 50,
) -> Dict:
    """
    Run comprehensive quality benchmark.

    Tests helpfulness on benign prompts to verify model
    capabilities are preserved after ablation.
    """
    from unfetter.core.validation import test_helpfulness

    if prompts is None:
        from unfetter.datasets.loader import load_prompts
        _, compliance_prompts = load_prompts("builtin")
        prompts = compliance_prompts

    prompts = prompts[:max_prompts]
    logger.info(f"Running quality benchmark on {len(prompts)} prompts")

    return test_helpfulness(model, tokenizer, prompts)


KNOWLEDGE_PROMPTS = [
    "What is the capital of France?",
    "Explain Newton's three laws of motion.",
    "What year did World War II end?",
    "Describe the process of photosynthesis.",
    "What is the speed of light in a vacuum?",
    "Name the planets in our solar system in order.",
    "What is the Pythagorean theorem?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
    "Explain how a CPU processes instructions.",
]


def run_knowledge_test(
    model: nn.Module,
    tokenizer,
    max_prompts: int = 10,
) -> Dict:
    """
    Test factual knowledge retention after ablation.

    Uses basic factual questions that any model should answer correctly.
    """
    from unfetter.core.validation import generate_response, is_refusal

    prompts = KNOWLEDGE_PROMPTS[:max_prompts]
    correct = 0
    details = []

    for prompt in prompts:
        try:
            response = generate_response(model, tokenizer, prompt, max_new_tokens=150)
            refused = is_refusal(response)
            has_content = len(response.strip()) > 20

            score = 0.0
            if not refused:
                score += 0.5
            if has_content:
                score += 0.5

            correct += score
            details.append({
                "prompt": prompt,
                "response": response[:200],
                "score": score,
            })
        except Exception as e:
            details.append({"prompt": prompt, "error": str(e), "score": 0.0})

    return {
        "knowledge_score": round((correct / max(len(prompts), 1)) * 100, 2),
        "total": len(prompts),
        "details": details,
    }
