"""
Dataset loading utilities.

Supports loading prompt pairs from:
- Built-in JSON file
- HuggingFace datasets (mlabonne/harmful_behaviors, mlabonne/harmless_alpaca)
- Custom CSV/JSON files
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to built-in refusal pairs
BUILTIN_PAIRS_PATH = Path(__file__).parent / "refusal_pairs.json"


def load_builtin_pairs() -> Dict[str, List[str]]:
    """
    Load the built-in refusal/compliance prompt pairs.

    Returns:
        Dict with "refusal" and "compliance" lists.
    """
    if not BUILTIN_PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"Built-in prompt pairs not found at {BUILTIN_PAIRS_PATH}"
        )

    with open(BUILTIN_PAIRS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    refusal = data.get("refusal", [])
    compliance = data.get("compliance", [])

    logger.info(
        f"Loaded built-in pairs: {len(refusal)} refusal, {len(compliance)} compliance"
    )

    return {"refusal": refusal, "compliance": compliance}


def load_hf_dataset(
    dataset_name: str,
    split: str = "train",
    column: str = "text",
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    Load prompts from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "mlabonne/harmful_behaviors").
        split: Dataset split (e.g., "train", "train[:400]").
        column: Column containing the prompts.
        max_samples: Maximum number of samples to load.

    Returns:
        List of prompt strings.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for HuggingFace dataset loading. "
            "Install: pip install datasets"
        )

    logger.info(f"Loading HuggingFace dataset: {dataset_name} (split={split})")

    ds = load_dataset(dataset_name, split=split)
    prompts = [str(row[column]) for row in ds if column in row]

    if max_samples:
        prompts = prompts[:max_samples]

    logger.info(f"Loaded {len(prompts)} prompts from {dataset_name}")
    return prompts


def load_custom_json(file_path: str) -> Dict[str, List[str]]:
    """
    Load prompts from a custom JSON file.

    Expected format:
    {
        "refusal": ["prompt1", "prompt2", ...],
        "compliance": ["prompt1", "prompt2", ...]
    }

    Args:
        file_path: Path to JSON file.

    Returns:
        Dict with prompt lists.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom prompt file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    refusal = data.get("refusal", [])
    compliance = data.get("compliance", [])

    logger.info(
        f"Loaded custom prompts: {len(refusal)} refusal, {len(compliance)} compliance"
    )
    return {"refusal": refusal, "compliance": compliance}


def load_custom_csv(
    file_path: str,
    prompt_column: str = "prompt",
    label_column: str = "label",
    refusal_label: str = "refusal",
    compliance_label: str = "compliance",
) -> Dict[str, List[str]]:
    """
    Load prompts from a CSV file with labeled prompt/label columns.

    Args:
        file_path: Path to CSV file.
        prompt_column: Column name for prompts.
        label_column: Column name for labels.
        refusal_label: Label value for refusal prompts.
        compliance_label: Label value for compliance prompts.

    Returns:
        Dict with prompt lists.
    """
    import csv

    refusal = []
    compliance = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get(prompt_column, "").strip()
            label = row.get(label_column, "").strip().lower()

            if not prompt:
                continue
            if label == refusal_label:
                refusal.append(prompt)
            elif label == compliance_label:
                compliance.append(prompt)

    logger.info(
        f"Loaded CSV prompts: {len(refusal)} refusal, {len(compliance)} compliance"
    )
    return {"refusal": refusal, "compliance": compliance}


def load_prompts(
    source: str = "builtin",
    max_samples: Optional[int] = None,
    custom_path: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Unified prompt loading interface.

    Args:
        source: "builtin", "hf" (HuggingFace), "custom_json", or "custom_csv".
        max_samples: Maximum samples per category.
        custom_path: Path for custom file sources.

    Returns:
        Tuple of (refusal_prompts, compliance_prompts).
    """
    if source == "builtin":
        data = load_builtin_pairs()

    elif source == "hf":
        refusal = load_hf_dataset(
            "mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
            max_samples=max_samples,
        )
        compliance = load_hf_dataset(
            "mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
            max_samples=max_samples,
        )
        data = {"refusal": refusal, "compliance": compliance}

    elif source == "custom_json":
        if not custom_path:
            raise ValueError("custom_path required for custom_json source")
        data = load_custom_json(custom_path)

    elif source == "custom_csv":
        if not custom_path:
            raise ValueError("custom_path required for custom_csv source")
        data = load_custom_csv(custom_path)

    else:
        raise ValueError(
            f"Unknown source: '{source}'. "
            f"Choose from: builtin, hf, custom_json, custom_csv"
        )

    refusal = data.get("refusal", [])
    compliance = data.get("compliance", [])

    if max_samples:
        refusal = refusal[:max_samples]
        compliance = compliance[:max_samples]

    return refusal, compliance
