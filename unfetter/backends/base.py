"""
Abstract backend interface.

All backends (CPU, GPU, Distributed) implement this interface
to provide a consistent API for the ablation pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Backend(ABC):
    """
    Abstract base class for ablation backends.

    Each backend handles model loading, ablation, and saving
    optimized for a specific hardware configuration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "base"

    @abstractmethod
    def load_model(self, model_path: str):
        """
        Load a model optimized for this backend.

        Args:
            model_path: HuggingFace model name or local path.

        Returns:
            Tuple of (model, tokenizer).
        """
        pass

    @abstractmethod
    def ablate(
        self,
        model: nn.Module,
        tokenizer,
        refusal_vector: torch.Tensor,
        layer_indices: List[int],
        strength: float,
        target_modules: Optional[List[str]] = None,
        progress_callback=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run directional ablation using this backend's strategy.

        Args:
            model: Loaded model.
            tokenizer: Tokenizer.
            refusal_vector: Refusal direction to remove.
            layer_indices: Which layers to modify.
            strength: Ablation intensity (0.0-1.0).
            target_modules: Which submodules to modify.
            progress_callback: Optional progress callback.

        Returns:
            Dict with ablation stats.
        """
        pass

    @abstractmethod
    def save_model(
        self,
        model: nn.Module,
        tokenizer,
        output_path: str,
        output_format: str = "safetensors",
    ) -> None:
        """
        Save the ablated model to disk.

        Args:
            model: Ablated model.
            tokenizer: Tokenizer.
            output_path: Directory to save to.
            output_format: Format (safetensors, pytorch, gguf).
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Return backend information and capabilities."""
        return {
            "name": self.name,
            "config": self.config,
        }

    def cleanup(self):
        """Release any resources held by the backend."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
