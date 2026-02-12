"""
Checkpoint management for resumable ablation.

Saves and loads ablation state so that long-running jobs
can be interrupted and resumed.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class Checkpoint:
    """Manages ablation checkpoints for resume support."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model_path: str,
        output_path: str,
        layer_indices: List[int],
        last_completed: int,
        strength: float,
        target_modules: List[str],
        vector_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save an ablation checkpoint.

        Args:
            model_path: Original model path.
            output_path: Output directory.
            layer_indices: Full list of target layer indices.
            last_completed: Index of last completed layer in layer_indices.
            strength: Ablation strength.
            target_modules: Target module names.
            vector_path: Path to cached refusal vector.
            metadata: Additional metadata.

        Returns:
            Path to saved checkpoint file.
        """
        checkpoint = {
            "version": "1.0",
            "timestamp": time.time(),
            "model_path": model_path,
            "output_path": output_path,
            "layer_indices": layer_indices,
            "last_completed": last_completed,
            "total_layers": len(layer_indices),
            "strength": strength,
            "target_modules": target_modules,
            "vector_path": vector_path,
            "metadata": metadata or {},
        }

        cp_file = self.checkpoint_dir / "unfetter_checkpoint.json"
        with open(cp_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(
            f"Checkpoint saved: layer {last_completed + 1}/{len(layer_indices)}"
        )
        return cp_file

    def load(self, path: Optional[str] = None) -> Optional[Dict]:
        """
        Load a checkpoint.

        Args:
            path: Specific checkpoint file. If None, looks in checkpoint_dir.

        Returns:
            Checkpoint data, or None if not found.
        """
        if path:
            cp_file = Path(path)
        else:
            cp_file = self.checkpoint_dir / "unfetter_checkpoint.json"

        if not cp_file.exists():
            logger.debug(f"No checkpoint found at {cp_file}")
            return None

        try:
            with open(cp_file) as f:
                data = json.load(f)
            logger.info(
                f"Loaded checkpoint: layer "
                f"{data.get('last_completed', 0) + 1}/{data.get('total_layers', '?')}"
            )
            return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def clear(self) -> None:
        """Remove checkpoint files."""
        cp_file = self.checkpoint_dir / "unfetter_checkpoint.json"
        if cp_file.exists():
            cp_file.unlink()
            logger.info("Checkpoint cleared")

    def exists(self, path: Optional[str] = None) -> bool:
        """Check if a checkpoint exists."""
        if path:
            return Path(path).exists()
        return (self.checkpoint_dir / "unfetter_checkpoint.json").exists()
