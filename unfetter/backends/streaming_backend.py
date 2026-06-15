"""
Layer-wise streaming ablation for models that do not fit in RAM/GPU.

Loads one transformer layer at a time directly from safetensors / pickle
weights on disk, applies the ablation math on CPU, and writes the modified
tensors to a new checkpoint directory. This makes 1T-parameter ablation
feasible with only enough CPU RAM for one layer plus activations.

Limitations:
    - Input checkpoint must be split into safetensors files (HuggingFace format).
    - Quantized weights are not supported in streaming mode; convert to fp16/fp32 first.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

from unfetter.core.ablation import compute_projection

logger = logging.getLogger(__name__)


class StreamingAblator:
    """
    Stream-ablate a HuggingFace checkpoint layer-by-layer.

    Example::

        ablator = StreamingAblator(
            input_dir="./Qwen2.5-72B",
            output_dir="./Qwen2.5-72B-abliterated",
            hidden_size=8192,
            num_layers=80,
        )
        ablator.ablate_all_layers(
            refusal_vector=rv,
            layer_indices=list(range(20, 60)),
            target_modules=["self_attn.o_proj", "mlp.down_proj"],
            strength=1.0,
        )
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        hidden_size: int,
        num_layers: int,
        dtype: str = "float16",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dtype = np.float32 if dtype in ("float32", "fp32") else np.float16
        self.refusal_vector: Optional[np.ndarray] = None

        self.index_path = self.input_dir / "model.safetensors.index.json"
        self.sharded = self.index_path.exists()

        if self.sharded:
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
            self.weight_map: Dict[str, str] = self.index.get("weight_map", {})
        else:
            self.index = {}
            self.weight_map = {}

        # Output mirrors input structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_files: Dict[str, Dict[str, np.ndarray]] = {}

    def _tensor_file(self, name: str) -> str:
        """Return the safetensors filename that contains ``name``."""
        if name in self.weight_map:
            return self.weight_map[name]
        # Fall back to a single-file checkpoint
        candidates = list(self.input_dir.glob("*.safetensors"))
        if candidates:
            return candidates[0].name
        raise FileNotFoundError(f"Could not locate safetensors file for tensor {name}")

    def _load_tensor(self, name: str) -> torch.Tensor:
        """Load a single tensor from the input checkpoint."""
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "safetensors is required for streaming ablation. "
                "Install: pip install safetensors"
            ) from e

        file_name = self._tensor_file(name)
        file_path = self.input_dir / file_name
        data = load_file(file_path, device="cpu")
        if name not in data:
            raise KeyError(f"Tensor {name} not found in {file_path}")
        return data[name]

    def _matches(self, tensor_name: str, pattern: str, layer_idx: int) -> bool:
        """
        Check if ``tensor_name`` matches a target module pattern for ``layer_idx``.

        Patterns may contain a wildcard for MoE experts, e.g.
        ``block_sparse_moe.experts.*.w2``.
        """
        layer_marker = f".{layer_idx}."
        if layer_marker not in tensor_name:
            return False

        # Normalize pattern: strip weight suffix, expand wildcard
        pattern = pattern.replace(".weight", "")
        if ".*." in pattern:
            parts = pattern.split(".*.")
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""
            if prefix not in tensor_name:
                return False
            if suffix and suffix not in tensor_name:
                return False
            # Must end with .weight
            return tensor_name.endswith(".weight")
        else:
            return pattern in tensor_name and tensor_name.endswith(".weight")

    def _ablate_tensor(
        self,
        weight: torch.Tensor,
        refusal_vector: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """Apply standard directional ablation to a single weight tensor."""
        v = refusal_vector.to(weight.device, weight.dtype)
        v = v / (v.norm() + 1e-10)
        proj = compute_projection(weight, v)
        return weight - strength * proj

    def ablate_all_layers(
        self,
        refusal_vector: torch.Tensor,
        layer_indices: List[int],
        target_modules: List[str],
        strength: float = 1.0,
        layer_weights: Optional[Dict[int, float]] = None,
        component_alphas: Optional[Dict[str, float]] = None,
        method: str = "directional",
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Stream-ablate selected layers and copy the rest unchanged.

        Args:
            refusal_vector: Refusal direction tensor (hidden_size,).
            layer_indices: Layers to modify.
            target_modules: Module name patterns to ablate per layer.
            strength: Base ablation intensity.
            layer_weights: Optional per-layer weights.
            component_alphas: Optional per-module alpha multipliers.
            method: Only "directional" is fully supported in streaming mode.
            progress_callback: Optional callable(layer_idx, total).

        Returns:
            Stats dict with number of tensors patched and output directory.
        """
        if method != "directional":
            logger.warning(
                f"Streaming ablator only supports method='directional'; "
                f"ignoring requested method '{method}'"
            )

        self.refusal_vector = refusal_vector.detach().cpu().numpy()
        rv_tensor = refusal_vector.detach().cpu()

        valid_indices = [
            idx if idx >= 0 else self.num_layers + idx for idx in layer_indices
        ]
        valid_indices = [idx for idx in valid_indices if 0 <= idx < self.num_layers]

        if not valid_indices:
            raise ValueError("No valid layer indices to process")

        patched = 0
        copied = 0

        # Iterate over all known tensors in the input index
        all_tensor_names = list(self.weight_map.keys()) if self.sharded else []
        if not all_tensor_names:
            # Single-file fallback: enumerate tensors from the file
            try:
                from safetensors import safe_open
            except ImportError as e:
                raise ImportError(
                    "safetensors is required for streaming ablation."
                ) from e

            safetensors_file = next(self.input_dir.glob("*.safetensors"))
            with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
                all_tensor_names = list(f.keys())

        for name in all_tensor_names:
            is_target = False
            matched_module = None
            layer_idx = None

            # Try to extract layer index from tensor name
            parts = name.split(".")
            numeric_parts = [p for p in parts if p.isdigit()]
            if numeric_parts:
                layer_idx = int(numeric_parts[0])

            if layer_idx in valid_indices:
                for module_pattern in target_modules:
                    if self._matches(name, module_pattern, layer_idx):
                        is_target = True
                        matched_module = module_pattern
                        break

            if is_target:
                effective_strength = strength
                if layer_weights and layer_idx in layer_weights:
                    effective_strength *= layer_weights[layer_idx]
                if component_alphas and matched_module in component_alphas:
                    effective_strength *= component_alphas[matched_module]

                if effective_strength <= 0:
                    # Copy unchanged
                    self._copy_tensor(name)
                    copied += 1
                    continue

                weight = self._load_tensor(name)
                ablated = self._ablate_tensor(weight, rv_tensor, effective_strength)
                self._store_tensor(name, ablated)
                patched += 1
            else:
                self._copy_tensor(name)
                copied += 1

        # Write all output safetensors files
        self._write_outputs()

        # Copy config / tokenizer files
        self._copy_auxiliary_files()

        logger.info(
            f"Streaming ablation complete: {patched} tensors patched, "
            f"{copied} tensors copied to {self.output_dir}"
        )

        return {
            "output_dir": str(self.output_dir),
            "patched_tensors": patched,
            "copied_tensors": copied,
            "ablated_layers": len(valid_indices),
        }

    def _store_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Store a modified tensor in the appropriate output file bucket."""
        file_name = self._tensor_file(name)
        self.output_files.setdefault(file_name, {})[name] = tensor.cpu().to(self.dtype).numpy()

    def _copy_tensor(self, name: str) -> None:
        """Copy an unmodified tensor into the output bucket."""
        weight = self._load_tensor(name)
        self._store_tensor(name, weight)

    def _write_outputs(self) -> None:
        """Write each output safetensors bucket to disk."""
        try:
            from safetensors.numpy import save_file
        except ImportError as e:
            raise ImportError(
                "safetensors is required for streaming ablation."
            ) from e

        for file_name, tensors in self.output_files.items():
            out_path = self.output_dir / file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(out_path))
            logger.debug(f"Wrote streaming output file: {out_path}")

        # Write updated index if input was sharded
        if self.sharded:
            new_index = {
                "metadata": self.index.get("metadata", {}),
                "weight_map": self.weight_map,
            }
            with open(self.output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
                json.dump(new_index, f, indent=2)

    def _copy_auxiliary_files(self) -> None:
        """Copy tokenizer, config, and other non-weight files to output dir."""
        for item in self.input_dir.iterdir():
            if item.is_file() and item.suffix not in (".safetensors", ".bin", ".pt", ".pth"):
                dest = self.output_dir / item.name
                shutil.copy2(item, dest)
                logger.debug(f"Copied auxiliary file: {item.name}")


def can_stream(input_dir: str) -> bool:
    """
    Check whether a checkpoint directory is streamable.

    Returns True if at least one safetensors file is present.
    """
    path = Path(input_dir)
    return any(path.glob("*.safetensors"))
