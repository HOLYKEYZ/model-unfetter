"""
CLI command implementations.

Orchestrates the full ablation pipeline: load model → compute vectors →
select layers → ablate → save → validate.
"""

import logging
import time
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def run_ablation(config: Dict) -> None:
    """
    Execute the full ablation pipeline.

    Steps:
    1. Detect hardware and select backend
    2. Load model with appropriate quantization
    3. Load prompt datasets
    4. Compute refusal vector(s)
    5. Select target layers
    6. Apply directional ablation
    7. Save ablated model
    8. Optionally validate
    """
    import torch

    logger.info("=" * 60)
    logger.info("Model Unfetter — Ablation Pipeline")
    logger.info("=" * 60)

    model_path = config["model_path"]
    output_path = config["output"]
    strength = config["strength"]
    layer_spec = config["layers"]

    # Step 1: Select backend
    logger.info("Step 1/7: Selecting backend...")
    from unfetter.backends.auto import select_backend, print_hardware_info
    print(print_hardware_info())

    backend = select_backend(
        backend_name=config["backend"],
        ram_limit_gb=config.get("ram_limit_gb"),
        vram_limit_gb=config.get("vram_limit_gb"),
        checkpoint_dir=config.get("checkpoint_dir"),
        checkpoint_every=config.get("checkpoint_every", 5),
    )
    logger.info(f"Using backend: {backend.name}")

    # Step 2: Load model
    logger.info(f"Step 2/7: Loading model '{model_path}'...")
    start = time.time()
    model, tokenizer = backend.load_model(model_path)
    logger.info(f"Model loaded in {time.time() - start:.1f}s")

    # Step 3: Detect model family
    logger.info("Step 3/7: Detecting model architecture...")
    from unfetter.models.registry import get_model_handler
    # Ensure GenericModel is available for fallback
    import unfetter.models.generic 
    handler = get_model_handler(model, model_path)
    summary = handler.get_summary()
    logger.info(f"Architecture: {summary['family']}, {summary['num_layers']} layers, "
                f"hidden_size={summary['hidden_size']}")

    # Step 4: Load prompts and compute refusal vector
    logger.info("Step 4/7: Computing refusal vector...")
    from unfetter.datasets.loader import load_prompts
    refusal_prompts, compliance_prompts = load_prompts(
        source=config.get("dataset_source", "builtin"),
        max_samples=config.get("max_samples", 100),
    )
    logger.info(f"Loaded {len(refusal_prompts)} refusal + {len(compliance_prompts)} compliance prompts")

    # Check cache first
    from unfetter.core.vectors import compute_refusal_vector, cache_vector, load_cached_vector
    cached = None
    if config.get("cache_vectors"):
        cached = load_cached_vector(
            summary["family"], model_path,
            layer=config.get("target_layer", -2),
        )

    if cached is not None:
        refusal_vector = cached
        logger.info("Using cached refusal vector")
    else:
        start = time.time()
        refusal_vector = compute_refusal_vector(
            model, tokenizer,
            refusal_prompts, compliance_prompts,
            target_layer=config.get("target_layer", -2),
            max_samples=config.get("max_samples", 100),
        )
        logger.info(f"Refusal vector computed in {time.time() - start:.1f}s")

        if config.get("cache_vectors"):
            cache_vector(
                refusal_vector, summary["family"], model_path,
                layer=config.get("target_layer", -2),
            )

    # Step 5: Select layers
    logger.info("Step 5/7: Selecting target layers...")
    from unfetter.core.layers import select_layers, auto_select_layers

    num_layers = summary["num_layers"]
    if layer_spec == "auto":
        layer_indices = auto_select_layers(num_layers)
    else:
        layer_indices = select_layers(num_layers, layer_spec)
    logger.info(f"Selected {len(layer_indices)} layers: {layer_indices}")

    # Step 6: Ablate
    logger.info(f"Step 6/7: Applying directional ablation (strength={strength})...")
    start = time.time()

    def progress(current, total):
        pct = (current / total) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.1f}% ({current}/{total} layers)", end="", flush=True)

    results = backend.ablate(
        model, tokenizer,
        refusal_vector, layer_indices,
        strength=strength,
        target_modules=handler.get_target_module_names(),
        progress_callback=progress,
    )
    print()  # newline after progress bar
    logger.info(f"Ablation complete in {time.time() - start:.1f}s")
    logger.info(f"Modified {results.get('total_modules_modified', len(results.get('layer_stats', {})))} modules")

    # Step 7: Save
    logger.info(f"Step 7/7: Saving ablated model to '{output_path}'...")
    backend.save_model(model, tokenizer, output_path, config.get("output_format", "safetensors"))
    logger.info(f"Model saved to {output_path}")

    # Optional validation
    if config.get("validate"):
        logger.info("Running post-ablation validation...")
        _run_validation(model, tokenizer, refusal_prompts, model_path)

    # Cleanup
    backend.cleanup()

    logger.info("=" * 60)
    logger.info("✅ Ablation pipeline complete!")
    logger.info(f"   Output: {output_path}")
    logger.info("=" * 60)


def _run_validation(model, tokenizer, refusal_prompts, model_name):
    """Run quick validation after ablation."""
    from unfetter.core.validation import test_refusal_rate, generate_validation_report

    test_prompts = refusal_prompts[:20]  # Quick test with subset
    results = test_refusal_rate(model, tokenizer, test_prompts)

    report = generate_validation_report(results, model_name=model_name)
    logger.info(f"\n{report}")


def run_resume(checkpoint_path: str) -> None:
    """Resume an interrupted ablation from checkpoint."""
    import json

    cp_path = Path(checkpoint_path)
    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(cp_path) as f:
        checkpoint = json.load(f)

    logger.info(f"Resuming from checkpoint: layer {checkpoint.get('last_completed_layer', 0) + 1}")
    logger.info("Re-running ablation with checkpoint...")

    config = {
        "model_path": checkpoint.get("model_path", ""),
        "backend": "cpu",
        "output": checkpoint.get("output_path", "./resumed_output"),
        "strength": checkpoint.get("strength", 1.0),
        "layers": ",".join(str(i) for i in checkpoint.get("layer_indices", [])),
        "targets": "refusal",
        "checkpoint_dir": str(cp_path.parent),
        "checkpoint_every": 5,
        "validate": False,
        "cache_vectors": True,
        "output_format": "safetensors",
        "dataset_source": "builtin",
        "max_samples": 100,
        "target_layer": -2,
    }

    run_ablation(config)


def run_compare(
    original_model: str,
    ablated_model: str,
    tests: List[str],
    output_path: Optional[str],
    max_prompts: int,
) -> None:
    """Compare original and ablated models."""
    from unfetter.benchmarks.compare import compare_models

    report = compare_models(original_model, ablated_model, tests, max_prompts)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
        logger.info(f"Comparison report saved to {output_path}")
    else:
        print(report)


def run_validate(
    model_path: str,
    tests: List[str],
    max_prompts: int,
    output_path: Optional[str],
) -> None:
    """Validate an ablated model's quality."""
    logger.info(f"Validating model: {model_path}")

    from unfetter.backends.auto import select_backend
    backend = select_backend("auto")
    model, tokenizer = backend.load_model(model_path)

    from unfetter.datasets.loader import load_prompts
    refusal_prompts, compliance_prompts = load_prompts("builtin", max_samples=max_prompts)

    from unfetter.core.validation import (
        test_refusal_rate, test_helpfulness, generate_validation_report,
    )

    refusal_results = None
    help_results = None

    if "refusal" in tests:
        refusal_results = test_refusal_rate(model, tokenizer, refusal_prompts[:max_prompts])

    if "helpfulness" in tests:
        help_results = test_helpfulness(model, tokenizer, compliance_prompts[:max_prompts])

    report = generate_validation_report(
        refusal_results or {"refusal_rate": "N/A", "refusal_count": 0, "total": 0},
        help_results,
        model_name=model_path,
    )

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
        logger.info(f"Validation report saved to {output_path}")
    else:
        print(report)

    backend.cleanup()
