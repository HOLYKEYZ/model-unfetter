"""
Distributed backend for multi-GPU and cloud processing.

Uses torch.distributed and device_map strategies for splitting
large models across multiple GPUs.
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


class DistributedBackend(Backend):
    """
    Multi-GPU backend using model parallelism.

    Distributes model layers across available GPUs using
    HuggingFace Accelerate's device_map="auto" strategy.
    Supports pipeline parallelism for ablation.
    """

    def __init__(
        self,
        device_map: str = "auto",
        max_memory: Optional[Dict[int, str]] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for distributed backend.")

        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            raise RuntimeError("No GPUs detected for distributed backend.")

        super().__init__({
            "device_map": device_map,
            "num_gpus": num_gpus,
        })
        self.name = "distributed"
        self.device_map = device_map
        self.max_memory = max_memory
        self.num_gpus = num_gpus

        logger.info(
            f"[Distributed] Initialized: {num_gpus} GPUs, device_map={device_map}"
        )

    def load_model(self, model_path: str):
        """Load model distributed across multiple GPUs."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"[Distributed] Loading model: {model_path} across {self.num_gpus} GPUs")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {
            "pretrained_model_name_or_path": model_path,
            "device_map": self.device_map,
            "torch_dtype": torch.float16,
        }

        if self.max_memory:
            kwargs["max_memory"] = self.max_memory

        model = AutoModelForCausalLM.from_pretrained(**kwargs)

        logger.info(f"[Distributed] Model loaded across devices")
        if hasattr(model, "hf_device_map"):
            devices_used = set(str(v) for v in model.hf_device_map.values())
            logger.info(f"[Distributed] Devices used: {devices_used}")

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
        Ablation across distributed model.

        Handles the fact that different layers may be on different
        devices by moving the refusal vector to match each layer's device.
        """
        from unfetter.core.ablation import ablate_layer, _get_model_layers

        if target_modules is None:
            target_modules = ["self_attn.o_proj", "mlp.down_proj"]

        layers = _get_model_layers(model)
        total = len(layer_indices)
        results = {"layer_stats": {}, "backend": "distributed"}

        logger.info(f"[Distributed] Starting ablation: {total} layers, strength={strength}")
        start_time = time.time()

        for i, layer_idx in enumerate(layer_indices):
            layer = layers[layer_idx]

            # Determine which device this layer is on
            layer_device = next(layer.parameters()).device

            # Move refusal vector to the layer's device
            rv = refusal_vector.to(layer_device)

            stats = ablate_layer(
                layer, rv,
                strength=strength,
                target_modules=target_modules,
            )
            results["layer_stats"][layer_idx] = stats

            if progress_callback:
                progress_callback(i + 1, total)

        results["total_time"] = round(time.time() - start_time, 2)
        results["layers_processed"] = total

        logger.info(
            f"[Distributed] Ablation complete: {total} layers in {results['total_time']}s"
        )
        return results

    def save_model(
        self,
        model: nn.Module,
        tokenizer,
        output_path: str,
        output_format: str = "safetensors",
    ) -> None:
        """Save the distributed model."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Distributed] Saving model to {output_dir}")

        if output_format == "safetensors":
            model.save_pretrained(output_dir, safe_serialization=True)
        else:
            model.save_pretrained(output_dir, safe_serialization=False)

        tokenizer.save_pretrained(output_dir)
        logger.info(f"[Distributed] Model saved to {output_dir}")

    def get_info(self) -> Dict[str, Any]:
        """Return distributed backend information."""
        gpus = []
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "vram_gb": round(props.total_mem / (1024 ** 3), 2),
            })

        return {
            "name": "distributed",
            "num_gpus": self.num_gpus,
            "gpus": gpus,
            "device_map": self.device_map,
            "total_vram_gb": round(sum(g["vram_gb"] for g in gpus), 2),
        }

    def cleanup(self):
        """Release all GPU resources."""
        gc.collect()
        for i in range(self.num_gpus):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
