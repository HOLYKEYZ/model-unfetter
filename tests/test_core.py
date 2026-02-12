"""
Unit tests for unfetter core modules.

Tests ablation math, vector operations, layer selection,
and config management without requiring model downloads.
"""

import json
import pytest
import torch
import torch.nn as nn
from pathlib import Path


# --- Ablation Tests ---

class TestAblationMath:
    """Test the core ablation formula: W' = W - α(W·v̂)v̂ᵀ"""

    def test_compute_projection_shape(self):
        from unfetter.core.ablation import compute_projection
        weight = torch.randn(512, 768)
        direction = torch.randn(768)
        proj = compute_projection(weight, direction)
        assert proj.shape == weight.shape

    def test_compute_projection_zero_direction(self):
        from unfetter.core.ablation import compute_projection
        weight = torch.randn(512, 768)
        direction = torch.zeros(768)
        proj = compute_projection(weight, direction)
        # Zero direction should return zero projection
        assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-6)

    def test_ablate_weight_identity(self):
        """Ablation with strength=0 should not change weights."""
        from unfetter.core.ablation import ablate_weight
        weight = torch.randn(512, 768)
        direction = torch.randn(768)
        result = ablate_weight(weight, direction, strength=0.0)
        assert torch.allclose(result, weight, atol=1e-6)

    def test_ablate_weight_full_strength(self):
        """Full strength ablation should remove the projection."""
        from unfetter.core.ablation import ablate_weight
        weight = torch.randn(512, 768)
        direction = torch.randn(768)
        result = ablate_weight(weight, direction, strength=1.0)
        # Result should differ from original
        assert not torch.allclose(result, weight)

    def test_ablate_weight_removes_direction(self):
        """After ablation, weight should have no component along direction."""
        from unfetter.core.ablation import ablate_weight, compute_projection
        weight = torch.randn(512, 768)
        direction = torch.randn(768)
        direction = direction / direction.norm()
        result = ablate_weight(weight, direction, strength=1.0)
        # Project result onto direction should be ~zero
        remaining = result @ direction
        assert torch.allclose(remaining, torch.zeros(512), atol=1e-5)


# --- Layer Selection Tests ---

class TestLayerSelection:
    """Test layer specification parsing."""

    def test_select_all(self):
        from unfetter.core.layers import select_layers
        result = select_layers(32, "all")
        assert result == list(range(32))

    def test_select_auto(self):
        from unfetter.core.layers import select_layers
        result = select_layers(32, "auto")
        assert result == list(range(32))

    def test_select_comma_separated(self):
        from unfetter.core.layers import select_layers
        result = select_layers(32, "5,10,15,20")
        assert result == [5, 10, 15, 20]

    def test_select_slice(self):
        from unfetter.core.layers import select_layers
        result = select_layers(32, "20:25")
        assert result == [20, 21, 22, 23, 24]

    def test_select_negative_slice(self):
        from unfetter.core.layers import select_layers
        result = select_layers(32, "-8:-1")
        assert 24 in result

    def test_select_percentage(self):
        from unfetter.core.layers import select_layers
        result = select_layers(40, "30%")
        assert len(result) == 12  # 30% of 40

    def test_auto_select_layers(self):
        from unfetter.core.layers import auto_select_layers
        result = auto_select_layers(40, percentage=0.3)
        assert len(result) == 12
        assert result[0] == 28  # Start at 70% mark

    def test_invalid_spec_raises(self):
        from unfetter.core.layers import select_layers
        with pytest.raises(ValueError):
            select_layers(32, "invalid_spec!!!")


# --- Config Tests ---

class TestConfig:
    """Test configuration management."""

    def test_default_config(self):
        from unfetter.cli.config import DEFAULTS
        assert DEFAULTS["strength"] == 1.0
        assert "self_attn.o_proj" in DEFAULTS["target_modules"]
        assert DEFAULTS["dataset_source"] == "builtin"

    def test_merge_cli_args(self):
        from unfetter.cli.config import merge_cli_args
        config = {"strength": 1.0, "layers": "auto"}
        args = {"strength": 0.8, "layers": None}  # None = not specified
        merged = merge_cli_args(config, args)
        assert merged["strength"] == 0.8
        assert merged["layers"] == "auto"  # Unchanged


# --- Dataset Tests ---

class TestDataset:
    """Test dataset loading."""

    def test_builtin_pairs_exist(self):
        from unfetter.datasets.loader import BUILTIN_PAIRS_PATH
        assert BUILTIN_PAIRS_PATH.exists()

    def test_load_builtin_pairs(self):
        from unfetter.datasets.loader import load_builtin_pairs
        data = load_builtin_pairs()
        assert "refusal" in data
        assert "compliance" in data
        assert len(data["refusal"]) >= 100
        assert len(data["compliance"]) >= 100

    def test_load_prompts_builtin(self):
        from unfetter.datasets.loader import load_prompts
        refusal, compliance = load_prompts("builtin")
        assert len(refusal) > 0
        assert len(compliance) > 0

    def test_load_prompts_max_samples(self):
        from unfetter.datasets.loader import load_prompts
        refusal, compliance = load_prompts("builtin", max_samples=10)
        assert len(refusal) == 10
        assert len(compliance) == 10

    def test_generator(self):
        from unfetter.datasets.generator import generate_paired_dataset
        data = generate_paired_dataset(count_per_side=20, seed=42)
        assert len(data["refusal"]) == 20
        assert len(data["compliance"]) == 20


# --- Model Registry Tests ---

class TestModelRegistry:
    """Test model family detection."""

    def test_detect_llama_from_name(self):
        from unfetter.models.registry import detect_model_family
        assert detect_model_family("meta-llama/Llama-3.1-8B-Instruct") == "llama"

    def test_detect_mistral_from_name(self):
        from unfetter.models.registry import detect_model_family
        assert detect_model_family("mistralai/Mistral-7B-v0.1") == "mistral"

    def test_detect_gemma_from_name(self):
        from unfetter.models.registry import detect_model_family
        assert detect_model_family("google/gemma-2-9b-it") == "gemma"

    def test_detect_qwen_from_name(self):
        from unfetter.models.registry import detect_model_family
        assert detect_model_family("Qwen/Qwen2.5-7B-Instruct") == "qwen"

    def test_unknown_model_returns_generic(self):
        from unfetter.models.registry import detect_model_family
        result = detect_model_family("completely-unknown-model-xyz")
        assert result == "generic"

    def test_list_supported_families(self):
        from unfetter.models.registry import list_supported_families
        families = list_supported_families()
        assert "llama" in families
        assert "mistral" in families
        assert "gemma" in families


# --- Hardware Detection Tests ---

class TestHardware:
    """Test hardware detection utilities."""

    def test_detect_hardware(self):
        from unfetter.backends.auto import detect_hardware
        hw = detect_hardware()
        assert "cpu_count" in hw
        assert "ram_total_gb" in hw
        assert "has_cuda" in hw
        assert "recommended_backend" in hw

    def test_print_hardware_info(self):
        from unfetter.backends.auto import print_hardware_info
        info = print_hardware_info()
        assert "Model Unfetter" in info
        assert "RAM" in info or "ram" in info.lower()

    def test_get_device_info(self):
        from unfetter.utils.device import get_device_info
        info = get_device_info()
        assert "system" in info
        assert "cpu" in info
        assert "gpu" in info
        assert "torch" in info


# --- Checkpoint Tests ---

class TestCheckpoint:
    """Test checkpoint save/load."""

    def test_checkpoint_roundtrip(self, tmp_path):
        from unfetter.utils.checkpoint import Checkpoint
        cp = Checkpoint(str(tmp_path / "checkpoints"))
        cp.save(
            model_path="test-model",
            output_path="./output",
            layer_indices=[20, 21, 22, 23, 24],
            last_completed=2,
            strength=0.8,
            target_modules=["self_attn.o_proj"],
        )
        assert cp.exists()

        loaded = cp.load()
        assert loaded is not None
        assert loaded["model_path"] == "test-model"
        assert loaded["last_completed"] == 2
        assert loaded["strength"] == 0.8

    def test_checkpoint_clear(self, tmp_path):
        from unfetter.utils.checkpoint import Checkpoint
        cp = Checkpoint(str(tmp_path / "checkpoints"))
        cp.save("m", "o", [1, 2], 0, 1.0, ["x"])
        assert cp.exists()
        cp.clear()
        assert not cp.exists()


# --- Progress Bar Tests ---

class TestProgressBar:
    """Test CLI progress bar utility."""

    def test_progress_bar_creation(self):
        from unfetter.utils.logging import ProgressBar
        pb = ProgressBar(total=100, desc="Test")
        assert pb.total == 100
        assert pb.current == 0

    def test_progress_bar_update(self, capsys):
        from unfetter.utils.logging import ProgressBar
        pb = ProgressBar(total=10, desc="Test")
        pb.update(5)
        assert pb.current == 5
