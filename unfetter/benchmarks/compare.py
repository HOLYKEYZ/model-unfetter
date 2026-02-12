"""
Side-by-side model comparison.

Compares an ablated model against the original (and optionally a
reference like a Heretic-processed model).
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def compare_models(
    original_path: str,
    ablated_path: str,
    tests: List[str],
    max_prompts: int = 50,
    reference_path: Optional[str] = None,
) -> str:
    """
    Run comparison tests between original and ablated models.

    Args:
        original_path: Path to original model.
        ablated_path: Path to ablated model.
        tests: Tests to run (refusal, helpfulness, kl).
        max_prompts: Max prompts per test.
        reference_path: Optional reference model (e.g., Heretic output).

    Returns:
        Markdown-formatted comparison report.
    """
    from unfetter.backends.auto import select_backend
    from unfetter.datasets.loader import load_prompts

    logger.info(f"Comparing models:")
    logger.info(f"  Original: {original_path}")
    logger.info(f"  Ablated:  {ablated_path}")
    if reference_path:
        logger.info(f"  Reference: {reference_path}")

    backend = select_backend("auto")

    # Load models
    logger.info("Loading original model...")
    orig_model, orig_tok = backend.load_model(original_path)

    logger.info("Loading ablated model...")
    abl_model, abl_tok = backend.load_model(ablated_path)

    ref_model, ref_tok = None, None
    if reference_path:
        logger.info("Loading reference model...")
        ref_model, ref_tok = backend.load_model(reference_path)

    # Load prompts
    refusal_prompts, compliance_prompts = load_prompts("builtin", max_samples=max_prompts)

    # Run tests
    results = {"original": {}, "ablated": {}}
    if reference_path:
        results["reference"] = {}

    if "refusal" in tests:
        from unfetter.core.validation import test_refusal_rate
        results["original"]["refusal"] = test_refusal_rate(
            orig_model, orig_tok, refusal_prompts[:max_prompts]
        )
        results["ablated"]["refusal"] = test_refusal_rate(
            abl_model, abl_tok, refusal_prompts[:max_prompts]
        )
        if ref_model:
            results["reference"]["refusal"] = test_refusal_rate(
                ref_model, ref_tok, refusal_prompts[:max_prompts]
            )

    if "helpfulness" in tests:
        from unfetter.core.validation import test_helpfulness
        results["original"]["helpfulness"] = test_helpfulness(
            orig_model, orig_tok, compliance_prompts[:max_prompts]
        )
        results["ablated"]["helpfulness"] = test_helpfulness(
            abl_model, abl_tok, compliance_prompts[:max_prompts]
        )
        if ref_model:
            results["reference"]["helpfulness"] = test_helpfulness(
                ref_model, ref_tok, compliance_prompts[:max_prompts]
            )

    if "kl" in tests:
        from unfetter.core.validation import compute_kl_divergence
        results["ablated"]["kl"] = compute_kl_divergence(
            orig_model, abl_model, orig_tok,
            compliance_prompts[:max_prompts],
        )
        if ref_model:
            results["reference"]["kl"] = compute_kl_divergence(
                orig_model, ref_model, orig_tok,
                compliance_prompts[:max_prompts],
            )

    # Generate report
    report = _format_comparison_report(results, original_path, ablated_path, reference_path)

    backend.cleanup()
    return report


def _format_comparison_report(
    results: Dict,
    orig_name: str,
    abl_name: str,
    ref_name: Optional[str],
) -> str:
    """Format comparison results into a markdown report."""
    lines = [
        "# Model Unfetter — Comparison Report",
        "",
        "## Models",
        f"- **Original:** {orig_name}",
        f"- **Ablated:** {abl_name}",
    ]
    if ref_name:
        lines.append(f"- **Reference:** {ref_name}")

    # Refusal rate table
    if "refusal" in results.get("original", {}):
        lines.extend(["", "## Refusal Rate", ""])
        lines.append("| Model | Refusal Rate | Refused | Complied |")
        lines.append("|-------|-------------|---------|----------|")

        for model_key, label in [("original", "Original"), ("ablated", "Ablated"), ("reference", "Reference")]:
            if model_key in results and "refusal" in results[model_key]:
                r = results[model_key]["refusal"]
                lines.append(
                    f"| {label} | {r['refusal_rate']}% | "
                    f"{r['refusal_count']}/{r['total']} | "
                    f"{r['compliance_count']}/{r['total']} |"
                )

    # Helpfulness table
    if "helpfulness" in results.get("original", {}):
        lines.extend(["", "## Helpfulness", ""])
        lines.append("| Model | Score |")
        lines.append("|-------|-------|")

        for model_key, label in [("original", "Original"), ("ablated", "Ablated"), ("reference", "Reference")]:
            if model_key in results and "helpfulness" in results[model_key]:
                h = results[model_key]["helpfulness"]
                lines.append(f"| {label} | {h['helpfulness_score']}% |")

    # KL Divergence table
    if "kl" in results.get("ablated", {}):
        lines.extend(["", "## KL Divergence (vs Original)", ""])
        lines.append("| Model | Mean KL | Min KL | Max KL |")
        lines.append("|-------|---------|--------|--------|")

        for model_key, label in [("ablated", "Ablated"), ("reference", "Reference")]:
            if model_key in results and "kl" in results[model_key]:
                k = results[model_key]["kl"]
                lines.append(
                    f"| {label} | {k['mean_kl']} | {k['min_kl']} | {k['max_kl']} |"
                )

    lines.extend(["", "---", "*Generated by Model Unfetter*"])
    return "\n".join(lines)
