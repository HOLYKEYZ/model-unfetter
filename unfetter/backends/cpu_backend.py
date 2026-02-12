"""
CPU backend for systems with limited or no GPU resources.

Optimized for 16GB RAM systems using:
- 4-bit NF4 quantization to reduce memory
- Layer-by-layer sequential processing
- Disk offloading for large models
- Periodic garbage collection
- Checkpoint support for resumability
"""

import gc
import json
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
import torch.nn as nn

from unfetter.backends.base import Backend

logger = logging.getLogger(__name__)


class CPUBackend(Backend):
    """
    CPU-only backend optimized for low-memory systems.

    Uses 4-bit quantization and sequential layer processing to
    handle models up to 70B parameters on 16GB RAM.
    """

    def __init__(
        self,
        ram_limit_gb: int = 16,
        checkpoint_every: int = 5,
        checkpoint_dir: Optional[str] = None,
    ):
        super().__init__({
            "ram_limit_gb": ram_limit_gb,
            "checkpoint_every": checkpoint_every,
        })
        self.name = "cpu"
        self.ram_limit = ram_limit_gb
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def load_model(self, model_path: str):
        """Load model with 4-bit quantization on CPU."""
        from unfetter.core.quantization import load_quantized_model

        logger.info(f"[CPU] Loading model: {model_path} (4-bit quantization)")

        model, tokenizer = load_quantized_model(
            model_path,
            quantization="4bit",
            device_map="cpu",
        )

        return model, tokenizer

    def ablate(
        self,
        model: nn.Module,
        tokenizer,
        refusal_vector: torch.Tensor,
        layer_indices: List[int],
        strength: float = 1.0,
        target_modules: Optional[List[str]] = None,
        progress_callback=None,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sequential layer-by-layer ablation optimized for CPU.

        Processes one layer at a time, with periodic checkpointing
        and aggressive memory cleanup.
        """
        from unfetter.core.ablation import ablate_layer

        if target_modules is None:
            target_modules = ["self_attn.o_proj", "mlp.down_proj"]

        # Determine starting point (for resume)
        start_idx = 0
        if checkpoint_path:
            cp = self._load_checkpoint(checkpoint_path)
            if cp:
                start_idx = cp.get("last_completed_layer", -1) + 1
                logger.info(f"[CPU] Resuming from layer {start_idx}")

        total = len(layer_indices)
        results = {"layer_stats": {}, "backend": "cpu"}

        logger.info(
            f"[CPU] Starting ablation: {total} layers, "
            f"strength={strength}, starting at index {start_idx}"
        )

        from unfetter.core.ablation import _get_model_layers
        layers = _get_model_layers(model)

        start_time = time.time()

        for i in range(start_idx, total):
            layer_idx = layer_indices[i]

            # Get the layer
            layer = layers[layer_idx]

            # Ablate
            stats = ablate_layer(
                layer, refusal_vector,
                strength=strength,
                target_modules=target_modules,
            )
            results["layer_stats"][layer_idx] = stats

            # Aggressive cleanup
            gc.collect()

            # Checkpoint
            if self.checkpoint_dir and (i + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(
                    layer_idx=i,
                    total=total,
                    layer_indices=layer_indices,
                    strength=strength,
                )
                logger.info(f"[CPU] Checkpoint saved at layer {i + 1}/{total}")

            # Progress
            if progress_callback:
                progress_callback(i + 1, total)

            elapsed = time.time() - start_time
            rate = (i + 1 - start_idx) / max(elapsed, 0.01)
            remaining = (total - i - 1) / max(rate, 0.001)
            logger.info(
                f"[CPU] Layer {i + 1}/{total} done "
                f"({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)"
            )

        results["total_time"] = round(time.time() - start_time, 2)
        results["layers_processed"] = total - start_idx

        logger.info(
            f"[CPU] Ablation complete: {results['layers_processed']} layers "
            f"in {results['total_time']}s"
        )

        return results

    def save_model(
        self,
        model: nn.Module,
        tokenizer,
        output_path: str,
        output_format: str = "safetensors",
    ) -> None:
        """Save the ablated model to disk."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[CPU] Saving model to {output_dir} (format: {output_format})")

        if output_format == "safetensors":
            model.save_pretrained(output_dir, safe_serialization=True)
        else:
            model.save_pretrained(output_dir, safe_serialization=False)

        tokenizer.save_pretrained(output_dir)
        logger.info(f"[CPU] Model saved to {output_dir}")

    def _save_checkpoint(
        self,
        layer_idx: int,
        total: int,
        layer_indices: List[int],
        strength: float,
    ) -> None:
        """Save checkpoint for resumability."""
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cp_file = self.checkpoint_dir / "cpu_checkpoint.json"

        checkpoint = {
            "last_completed_layer": layer_idx,
            "total_layers": total,
            "layer_indices": layer_indices,
            "strength": strength,
            "timestamp": time.time(),
        }

        with open(cp_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self, path: str) -> Optional[Dict]:
        """Load checkpoint from disk."""
        cp_file = Path(path)
        if not cp_file.exists():
            return None

        try:
            with open(cp_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def get_info(self) -> Dict[str, Any]:
        """Return CPU backend information."""
        import psutil

        return {
            "name": "cpu",
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
            "cpu_count": psutil.cpu_count(),
            "quantization": "4bit",
            "checkpoint_every": self.checkpoint_every,
        }
