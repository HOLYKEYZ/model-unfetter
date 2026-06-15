"""
Hyperparameter optimization for ablation (Heretic-style TPE search).

Uses Optuna to find the best combination of:
    - continuous layer direction_index
    - global ablation strength
    - weight-kernel peak position
    - per-component alphas (optional)

The caller must supply an ``evaluate`` callable that returns a scalar score
(higher = better). A typical score combines refusal-rate reduction and
perplexity / capability preservation.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSpace:
    """Defines the bounds for each tuned ablation hyperparameter."""

    direction_index_min: float = 0.0
    direction_index_max: float = 1.0
    strength_min: float = 0.3
    strength_max: float = 2.0
    max_weight_position_min: float = 0.0
    max_weight_position_max: float = 1.0
    min_weight_min: float = 0.0
    min_weight_max: float = 0.5
    min_weight_distance_min: float = 0.1
    min_weight_distance_max: float = 1.0

    # Per-module alpha ranges; keys are module names (e.g. "self_attn.o_proj")
    component_ranges: Optional[Dict[str, tuple]] = None


def _build_params(trial, space: OptimizationSpace, target_modules: List[str]) -> Dict[str, Any]:
    """Sample a parameter dict from the Optuna trial."""
    params: Dict[str, Any] = {
        "direction_index": trial.suggest_float(
            "direction_index",
            space.direction_index_min,
            space.direction_index_max,
        ),
        "strength": trial.suggest_float(
            "strength",
            space.strength_min,
            space.strength_max,
        ),
        "max_weight_position": trial.suggest_float(
            "max_weight_position",
            space.max_weight_position_min,
            space.max_weight_position_max,
        ),
        "min_weight": trial.suggest_float(
            "min_weight",
            space.min_weight_min,
            space.min_weight_max,
        ),
        "min_weight_distance": trial.suggest_float(
            "min_weight_distance",
            space.min_weight_distance_min,
            space.min_weight_distance_max,
        ),
    }

    if space.component_ranges:
        params["component_alphas"] = {}
        for module, (lo, hi) in space.component_ranges.items():
            params["component_alphas"][module] = trial.suggest_float(
                f"alpha_{module.replace('.', '_')}", lo, hi
            )
    else:
        # Default: tune the two main components independently
        for module in target_modules:
            params.setdefault("component_alphas", {})[module] = trial.suggest_float(
                f"alpha_{module.replace('.', '_')}", 0.0, 2.0
            )

    return params


def optimize_ablation_parameters(
    model: nn.Module,
    layer_vectors: List[torch.Tensor],
    layer_indices: List[int],
    evaluate: Callable[[nn.Module], float],
    space: Optional[OptimizationSpace] = None,
    target_modules: Optional[List[str]] = None,
    method: str = "directional",
    n_trials: int = 30,
    study_name: str = "unfetter_ablation",
    direction: str = "maximize",
    load_if_exists: bool = False,
    copy_before_trial: bool = True,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run TPE optimization to find the best ablation hyperparameters.

    Args:
        model: Transformer model to ablate. Must be on the target device.
        layer_vectors: Per-layer refusal vectors (one per layer).
        layer_indices: Which layers to modify during trials.
        evaluate: Callable(model) -> float. Should reset the model or only
            inspect it; the optimizer will mutate weights between trials.
        space: OptimizationSpace with parameter ranges.
        target_modules: Submodules to ablate per layer.
        method: Ablation variant.
        n_trials: Number of Optuna trials.
        study_name: Optuna study name.
        direction: "maximize" or "minimize" for the evaluation score.
        load_if_exists: Whether to resume an existing study with the same name.
        copy_before_trial: If True, deep-copies the model before each trial so
            trials start from the original weights. Disable for huge models to
            save memory (but then ``evaluate`` must restore weights itself).
        progress_callback: Optional callable(trial_number, n_trials).

    Returns:
        Dict with best_params, best_score, study summary, and n_trials.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "Optuna is required for ablation optimization. "
            "Install: pip install optuna"
        ) from e

    space = space or OptimizationSpace()
    if target_modules is None:
        target_modules = ["self_attn.o_proj", "mlp.down_proj"]

    # Save a snapshot of the original weights so we can restore between trials
    original_state = {k: v.detach().clone().cpu() for k, v in model.named_parameters()}

    def objective(trial):
        # Restore original weights before each trial
        if copy_before_trial:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in original_state:
                        param.copy_(original_state[name].to(param.device, param.dtype))

        params = _build_params(trial, space, target_modules)

        from unfetter.core.ablation import compute_ablation_weights, ablate_with_float_index

        num_layers = len(layer_vectors)
        layer_weights = compute_ablation_weights(
            num_layers=num_layers,
            layer_indices=layer_indices,
            max_weight=1.0,
            max_weight_position=params["max_weight_position"],
            min_weight=params["min_weight"],
            min_weight_distance=params["min_weight_distance"],
        )

        ablate_with_float_index(
            model,
            layer_vectors=layer_vectors,
            direction_index=params["direction_index"],
            layer_indices=layer_indices,
            strength=params["strength"],
            target_modules=target_modules,
            component_alphas=params.get("component_alphas"),
            method=method,
            layer_weights=layer_weights,
        )

        score = evaluate(model)

        if progress_callback:
            progress_callback(trial.number + 1, n_trials)

        return score

    storage = None
    if load_if_exists:
        storage = f"sqlite:///./{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(),
        storage=storage,
        load_if_exists=load_if_exists,
    )

    logger.info(f"Starting ablation hyperparameter search: {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    logger.info(
        f"Optimization complete. Best score={best.value:.6f} at trial {best.number}"
    )

    # Build best layer_weights and component_alphas to return
    best_layer_weights = None
    if "max_weight_position" in best.params:
        from unfetter.core.ablation import compute_ablation_weights
        best_layer_weights = compute_ablation_weights(
            num_layers=len(layer_vectors),
            layer_indices=layer_indices,
            max_weight=1.0,
            max_weight_position=best.params["max_weight_position"],
            min_weight=best.params.get("min_weight", 0.0),
            min_weight_distance=best.params.get("min_weight_distance", 0.5),
        )

    component_alphas = {
        k.replace("alpha_", "").replace("_", "."): v
        for k, v in best.params.items()
        if k.startswith("alpha_")
    }

    return {
        "best_params": best.params,
        "best_score": best.value,
        "best_trial": best.number,
        "n_trials": len(study.trials),
        "direction": direction,
        "component_alphas": component_alphas,
        "layer_weights": best_layer_weights,
    }
