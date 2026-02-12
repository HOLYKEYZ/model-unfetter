"""
GPU backend for CUDA-capable systems.

Optimized for consumer GPUs (8-16GB VRAM) and high-end GPUs (24GB+).
Auto-selects quantization based on available VRAM.
"""

import gc
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
import torch.nn as nn

from unfetter.backends.base import Backend

logger = logging.getLogger(__name__)


class GPUBackend(Backend):
    """
    GPU backend with CUDA-optimized ablation.

    Supports:
    - 8GB VRAM: 4-bit quantization + layer-by-layer
    - 16GB VRAM: 8-bit quantization + batch processing
    - 24GB+ VRAM: fp16 full model in memory
    """

    def __init__(
        self,
        vram_limit_gb: Optional[float] = None,
        device: str = "cuda",
        batch_layers: int = 4,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Use CPUBackend instead.")

        if vram_limit_gb is None:
            vram_limit_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)

        super().__init__({
            "vram_limit_gb": vram_limit_gb,
            "device": device,
            "batch_layers": batch_layers,
        })
        self.name = "gpu"
        self.vram_limit = vram_limit_gb
        self.device = device
        self.batch_layers = batch_layers

        # Select quantization based on VRAM
        if vram_limit_gb >= 24:
            self.quantization = "none"  # fp16, full model
            self.dtype = torch.float16
        elif vram_limit_gb >= 12:
            self.quantization = "8bit"
            self.dtype = torch.float16
        else:
            self.quantization = "4bit"
            self.dtype = torch.float16

        logger.info(
            f"[GPU] Initialized: {vram_limit_gb:.1f}GB VRAM, "
            f"quant={self.quantization}, dtype={self.dtype}"
        )

    def load_model(self, model_path: str):
        """Load model with optimal quantization for available VRAM."""
        from unfetter.core.quantization import load_quantized_model

        logger.info(
            f"[GPU] Loading model: {model_path} "
            f"(quant={self.quantization}, device={self.device})"
        )

        model, tokenizer = load_quantized_model(
            model_path,
            quantization=self.quantization,
            device_map=self.device,
            torch_dtype=self.dtype,
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
        **kwargs,
    ) -> Dict[str, Any]:
        """
        GPU-optimized ablation with batch layer processing.

        Takes advantage of GPU parallelism to process multiple
        layers simultaneously when VRAM allows.
        """
        from unfetter.core.ablation import ablate_layer, _get_model_layers

        if target_modules is None:
            target_modules = ["self_attn.o_proj", "mlp.down_proj"]

        # Move refusal vector to GPU
        refusal_vector = refusal_vector.to(self.device)

        layers = _get_model_layers(model)
        total = len(layer_indices)
        results = {"layer_stats": {}, "backend": "gpu"}

        logger.info(f"[GPU] Starting ablation: {total} layers, strength={strength}")
        start_time = time.time()

        for i, layer_idx in enumerate(layer_indices):
            layer = layers[layer_idx]

            # Ablate on GPU
            stats = ablate_layer(
                layer, refusal_vector,
                strength=strength,
                target_modules=target_modules,
            )
            results["layer_stats"][layer_idx] = stats

            # Progress
            if progress_callback:
                progress_callback(i + 1, total)

            # Periodic VRAM cleanup
            if (i + 1) % self.batch_layers == 0:
                torch.cuda.empty_cache()

        results["total_time"] = round(time.time() - start_time, 2)
        results["layers_processed"] = total

        logger.info(
            f"[GPU] Ablation complete: {total} layers in {results['total_time']}s"
        )

        return results

    def save_model(
        self,
        model: nn.Module,
        tokenizer,
        output_path: str,
        output_format: str = "safetensors",
    ) -> None:
        """Save the ablated model."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[GPU] Saving model to {output_dir}")

        # Move to CPU for saving (reduces VRAM usage during save)
        model_cpu = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        if output_format == "safetensors":
            model_cpu.save_pretrained(output_dir, safe_serialization=True)
        else:
            model_cpu.save_pretrained(output_dir, safe_serialization=False)

        tokenizer.save_pretrained(output_dir)

        # Move back to GPU
        model.to(self.device)

        logger.info(f"[GPU] Model saved to {output_dir}")

    def get_info(self) -> Dict[str, Any]:
        """Return GPU backend information."""
        gpu_props = torch.cuda.get_device_properties(0)
        return {
            "name": "gpu",
            "gpu_name": gpu_props.name,
            "vram_total_gb": round(gpu_props.total_mem / (1024 ** 3), 2),
            "vram_used_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 2),
            "vram_free_gb": round(
                (gpu_props.total_mem - torch.cuda.memory_allocated()) / (1024 ** 3), 2
            ),
            "quantization": self.quantization,
            "dtype": str(self.dtype),
            "cuda_version": torch.version.cuda or "N/A",
            "batch_layers": self.batch_layers,
        }

    def cleanup(self):
        """Release GPU resources."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
