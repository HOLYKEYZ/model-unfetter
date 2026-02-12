"""
Generic model handler with heuristic detection.

Attempts to automatically identify layer structures and target modules
for models not explicitly in the registry.
"""

import logging
import torch.nn as nn
from typing import List, Dict, Any, Optional
from unfetter.models.base import TransformerModel

logger = logging.getLogger(__name__)

class GenericModel(TransformerModel):
    """
    Fallback handler for unknown model architectures.
    
    Uses heuristics to find:
    1. The list of transformer layers
    2. The attention output projection
    3. The MLP down projection
    """
    
    FAMILY = "generic"

    def __init__(self, model: nn.Module, model_name: str = ""):
        super().__init__(model, model_name)
        self._layer_path = None
        self._attn_module = None
        self._mlp_module = None
        self._detect_architecture()

    def _detect_architecture(self):
        """Scan model structure to find key components."""
        # 1. Find Layers
        # Look for the largest ModuleList
        max_len = 0
        best_path = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList):
                if len(module) > max_len:
                    max_len = len(module)
                    best_path = name
        
        if best_path:
            self._layer_path = best_path
            logger.info(f"[Generic] Detected layer list at: '{best_path}' ({max_len} layers)")
        else:
            logger.warning("[Generic] Could not auto-detect layer list! Ablation may fail.")
            return

        # 2. Inspect the first layer to find targets
        layers = self.get_layers()
        if len(layers) > 0:
            first_layer = layers[0]
            self._find_target_modules(first_layer)

    def _find_target_modules(self, layer: nn.Module):
        """
        Heuristic search for projection weights within a layer.
        
        We look for Linear layers.
        - Attention output: usually has 'o_proj', 'out_proj', 'dense' inside 'attn' block
        - MLP down: usually 'down_proj', 'w2', 'fc2' inside 'mlp' block
        """
        candidates = []
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                candidates.append((name, mod.weight.shape))
        
        # Heuristic 1: Attention Output
        # Common names: o_proj, out_proj, c_proj (GPT), dense
        # Often shape is [hidden, hidden]
        potential_attn = ["o_proj", "out_proj", "c_proj", "dense"]
        for name, shape in candidates:
            if any(x in name for x in potential_attn) and "attn" in name:
                 self._attn_module = name
                 break
        
        # Fallback: Look for any output projection in attention
        if not self._attn_module:
            for name, _ in candidates:
                if "attn" in name and ("out" in name or "o_" in name):
                    self._attn_module = name
                    break

        # Heuristic 2: MLP Down
        # Common names: down_proj, w2, c_fc, fc_out
        potential_mlp = ["down_proj", "w2", "c_proj", "fc_out"]
        for name, shape in candidates:
            if any(x in name for x in potential_mlp) and ("mlp" in name or "feed_forward" in name):
                 self._mlp_module = name
                 break

        logger.info(f"[Generic] Auto-detected targets: Attn='{self._attn_module}', MLP='{self._mlp_module}'")

    def get_layers(self) -> nn.ModuleList:
        if self._layer_path:
            # Navigate to the layer list
            obj = self.model
            for part in self._layer_path.split("."):
                obj = getattr(obj, part)
            return obj
        return super().get_layers() # Fallback to base implementation

    def get_target_module_names(self) -> List[str]:
        targets = []
        if self._attn_module:
            targets.append(self._attn_module)
        if self._mlp_module:
            targets.append(self._mlp_module)
        
        # If heuristics failed, fallback to defaults
        if not targets:
            logger.warning("[Generic] No targets detected, using defaults.")
            return super().get_target_module_names()
            
        return targets
