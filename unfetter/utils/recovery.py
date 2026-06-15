"""
Post-ablation capability recovery via low-rank adaptation (LoRA).

After ablation, some model capabilities may degrade slightly. This module
provides a lightweight LoRA overlay that can be trained on harmless data to
recover helpfulness while keeping the refusal direction suppressed.

This mirrors the Obliteratus "Stage 8" recovery step, but keeps the
ablated base weights frozen and only updates small adapter matrices.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Low-rank adapter for a frozen Linear layer.

    Forward: h = base(x) + (alpha / r) * (x @ A^T @ B^T)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze base weights
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        adapter_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * adapter_out


def attach_lora_recovery(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> Dict[str, LoRALayer]:
    """
    Attach LoRA adapters to specified modules in ``model``.

    Args:
        model: Transformer model (already ablated).
        target_modules: Module name patterns, e.g. ["self_attn.o_proj", "mlp.down_proj"].
        r: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: Dropout applied to adapter inputs.

    Returns:
        Dict mapping full module names to their LoRA wrappers.
    """
    adapters: Dict[str, LoRALayer] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        matched = any(pattern in name for pattern in target_modules)
        if not matched:
            continue

        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)

        lora_layer = LoRALayer(module, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)
        adapters[name] = lora_layer
        logger.info(f"Attached LoRA adapter to {name} (r={r}, alpha={alpha})")

    logger.info(f"Attached {len(adapters)} LoRA adapters")
    return adapters


def train_lora_recovery(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    max_length: int = 512,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train the LoRA adapters on a harmless prompt corpus.

    This is a minimal SFT loop; for large-scale recovery, use the PEFT / TRL
    trainers instead and load the resulting adapters via this module.

    Args:
        model: Model with LoRA adapters attached.
        tokenizer: Tokenizer.
        prompts: Harmless training prompts.
        num_epochs: Training epochs.
        batch_size: Prompts per batch.
        learning_rate: AdamW learning rate.
        max_length: Maximum sequence length.
        device: Device override.

    Returns:
        Dict with final loss and epoch count.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    model.train()

    # Only train LoRA parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    total_loss = 0.0
    steps = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            total_loss += loss.item()
            steps += 1

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"LoRA recovery epoch {epoch + 1}/{num_epochs}: loss={avg_epoch_loss:.4f}")

    model.eval()
    final_loss = total_loss / max(steps, 1)
    return {"final_loss": final_loss, "epochs": num_epochs, "steps": steps}
