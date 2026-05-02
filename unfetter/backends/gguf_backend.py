"""
GGUF Direct Backend

Allows direct manipulation of GGUF tensor weights on CPU without passing through PyTorch or safetensors.
This is critical for low RAM environments.
"""

import logging
import os
from typing import List, Dict, Any, Optional

try:
    import numpy as np
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
except ImportError:
    pass  # Handled at CLI level if gguf is missing

logger = logging.getLogger(__name__)

class GGUFAblator:
    def __init__(self, gguf_path: str, refusal_vector: np.ndarray):
        self.gguf_path = gguf_path
        self.refusal_vector = refusal_vector
        self.reader = GGUFReader(gguf_path)
        self.hidden_size = refusal_vector.shape[0]

    def _apply_ablation_math(self, weight: np.ndarray, alpha: float) -> np.ndarray:
        """
        Applies W' = W - alpha * (v @ (v^T @ W))
        Using numpy.
        """
        v = self.refusal_vector
        v = v / (np.linalg.norm(v) + 1e-10)
        
        v_col = v.reshape(-1, 1)
        
        if weight.shape[0] == self.hidden_size:
            proj = np.matmul(v_col.T, weight)
            return weight - alpha * np.matmul(v_col, proj)
        elif weight.shape[1] == self.hidden_size:
            proj = np.matmul(weight, v_col)
            return weight - alpha * np.matmul(proj, v_col.T)
        else:
            return weight # shape mismatch, do nothing

    def ablate(self, output_path: str, target_modules: List[str], target_layers: List[int], alpha: float):
        """
        Reads the GGUF file, modifies the targeted tensors, and writes to a new GGUF file.
        """
        logger.info(f"Starting direct GGUF ablation to {output_path}...")
        
        writer = GGUFWriter(output_path, self.reader.fields["general.architecture"].parts[0].tolist().decode('utf-8'))
        
        # Copy metadata
        for key, field in self.reader.fields.items():
            writer.add_custom_alignment(field.alignment)
            # writer.add_key_value(key, field.parts[0].tolist()) # Simplified for planning purposes
            
        patched = 0
        for tensor in self.reader.tensors:
            name = tensor.name
            data = tensor.data
            
            # Check if this tensor should be ablated
            is_target = False
            for mod in target_modules:
                if mod in name:
                    # check layer
                    parts = name.split(".")
                    try:
                        layer_idx = int([p for p in parts if p.isdigit()][0])
                        if layer_idx in target_layers:
                            is_target = True
                            break
                    except IndexError:
                        pass
            
            if is_target:
                # We can only apply this to FP32 or FP16 tensors easily with numpy
                # For quantized tensors, we'd need to dequantize -> ablate -> quantize
                # For now, we assume the input GGUF is f16 or f32
                if tensor.tensor_type in [GGMLQuantizationType.F32, GGMLQuantizationType.F16]:
                    logger.debug(f"Ablating GGUF tensor: {name}")
                    dtype = np.float32 if tensor.tensor_type == GGMLQuantizationType.F32 else np.float16
                    weight_array = data.view(dtype).reshape(tensor.shape)
                    
                    # Convert to float32 for math precision
                    weight_array_f32 = weight_array.astype(np.float32)
                    
                    # Apply math
                    ablated_weight = self._apply_ablation_math(weight_array_f32, alpha)
                    
                    # Convert back
                    data_to_write = ablated_weight.astype(dtype).tobytes()
                    writer.add_tensor(name, data_to_write, tensor.shape, tensor.tensor_type)
                    patched += 1
                else:
                    logger.warning(f"Tensor {name} is quantized ({tensor.tensor_type}). GGUF backend currently requires F16/F32 base model.")
                    writer.add_tensor(name, data, tensor.shape, tensor.tensor_type)
            else:
                # Copy directly
                writer.add_tensor(name, data, tensor.shape, tensor.tensor_type)
                
        logger.info(f"Writing GGUF... Patched {patched} tensors.")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        logger.info("GGUF ablation complete.")
