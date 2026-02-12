"""
Configuration management for Model Unfetter.

Supports TOML config files (like Heretic's config.default.toml)
merged with CLI arguments. CLI args take precedence.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULTS = {
    # Model loading
    "quantization": "none",
    "device_map": "auto",
    "torch_dtype": "auto",
    "trust_remote_code": False,

    # Ablation parameters
    "strength": 1.0,
    "target_modules": ["self_attn.o_proj", "mlp.down_proj"],
    "layer_spec": "auto",

    # Ablation weight kernel
    "max_weight": 1.0,
    "max_weight_position": 0.5,
    "min_weight": 0.0,
    "min_weight_distance": 0.5,

    # Vector computation
    "target_layer": -2,
    "max_samples": 100,
    "batch_size": 8,
    "dataset_source": "builtin",

    # Generation
    "max_response_length": 100,
    "system_prompt": "You are a helpful assistant.",

    # Checkpointing
    "checkpoint_every": 5,
    "checkpoint_dir": "checkpoints",

    # Output
    "output_format": "safetensors",

    # Refusal detection markers
    "refusal_markers": [
        "sorry",
        "i can't", "i cant", "i cannot",
        "i won't", "i wont", "i will not",
        "i unable", "im unable", "i'm unable", "i am unable",
        "i an ai", "im an ai", "i'm an ai", "i am an ai",
        "as an ai", "ai assistant",
        "i designed to", "im designed to", "i'm designed to", "i am designed to",
        "i programmed to", "im programmed to", "i'm programmed to", "i am programmed to",
        "violat", "prohibit", "illegal", "harmful",
        "inappropriate", "unethical", "ethical boundaries",
    ],
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a TOML file.

    Args:
        config_path: Path to TOML config file. If None, uses defaults.

    Returns:
        Configuration dictionary.
    """
    config = DEFAULTS.copy()

    if config_path:
        path = Path(config_path)
        if path.exists():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    logger.warning(
                        "Neither tomllib nor tomli available. "
                        "Install tomli for TOML config support: pip install tomli"
                    )
                    return config

            with open(path, "rb") as f:
                file_config = tomllib.load(f)

            # Flatten nested config (e.g., [good_prompts] -> good_prompts.*)
            config = _merge_config(config, file_config)
            logger.info(f"Loaded config from {path}")
        else:
            logger.warning(f"Config file not found: {config_path}")

    return config


def merge_cli_args(config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge CLI arguments into config. CLI args take precedence.

    Only non-None CLI args override config values.

    Args:
        config: Base configuration dict.
        cli_args: CLI argument dict.

    Returns:
        Merged configuration.
    """
    merged = config.copy()
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    return merged


def _merge_config(base: Dict, override: Dict) -> Dict:
    """Deep merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to a JSON file (TOML writing requires extra deps)."""
    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Config saved to {path}")
