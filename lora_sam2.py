"""Inference-time LoRA injection and checkpoint loading helpers for SAM2.

Vendored (inference-only subset) from
https://github.com/adidbdir/sam2-lora-reproduce (paper4/lora_sam2.py). Do not
hand-edit without checking upstream for corresponding fixes.
"""

import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

DEFAULT_LORA_TARGET_MODULES = ("qkv", "proj")
LORA_PARAMETER_MARKERS = (".lora_A.", ".lora_B.")


class LoRALinear(nn.Module):
    """Run a frozen linear layer in parallel with a trainable low-rank update.

    The effective operation is ``W_orig @ x + scaling * B @ A @ x``. ``B`` is
    initialized to zero, so injection does not change the model at step zero.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 512,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(original_linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for parameter in self.original_linear.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the scaled low-rank update to the original linear output."""
        return self.original_linear(x) + self.lora_B(self.lora_A(x)) * self.scaling


def _normalize_target_modules(target_modules: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(target_modules)))
    if not normalized:
        raise ValueError("target_modules must contain at least one module name")
    return normalized


def _lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
        if any(marker in f".{name}" for marker in LORA_PARAMETER_MARKERS)
    }


def _applied_lora_config(model: nn.Module) -> tuple[int, float, tuple[str, ...]]:
    lora_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if not lora_modules:
        raise ValueError("No LoRALinear modules are applied to the model")

    ranks = {module.rank for _, module in lora_modules}
    alphas = {module.alpha for _, module in lora_modules}
    if len(ranks) != 1 or len(alphas) != 1:
        raise ValueError("All applied LoRALinear modules must use one rank and alpha")

    target_modules = _normalize_target_modules(
        [name.rsplit(".", maxsplit=1)[-1] for name, _ in lora_modules]
    )
    return ranks.pop(), alphas.pop(), target_modules


def apply_lora_to_sam2(
    model: nn.Module,
    rank: int = 512,
    alpha: float = 1.0,
    target_modules: Sequence[str] | None = None,
) -> tuple[nn.Module, list[nn.Parameter]]:
    """Inject LoRA into selected attention linears in every SAM2 Hiera block.

    Each new adapter is explicitly moved to the original linear layer's device
    and dtype. This matters when SAM2 was placed on an accelerator before LoRA
    injection, because assigning a new child module does not move it implicitly.
    """
    targets = _normalize_target_modules(
        DEFAULT_LORA_TARGET_MODULES if target_modules is None else target_modules
    )
    lora_parameters: list[nn.Parameter] = []

    for block in model.image_encoder.trunk.blocks:
        attention = block.attn
        for module_name in targets:
            if not hasattr(attention, module_name):
                continue

            original_linear = getattr(attention, module_name)
            if not isinstance(original_linear, nn.Linear):
                raise TypeError(
                    f"Expected attention.{module_name} to be nn.Linear, "
                    f"got {type(original_linear).__name__}"
                )
            lora_linear = LoRALinear(original_linear, rank=rank, alpha=alpha)
            lora_linear.to(
                device=original_linear.weight.device,
                dtype=original_linear.weight.dtype,
            )
            setattr(attention, module_name, lora_linear)
            lora_parameters.extend(lora_linear.lora_A.parameters())
            lora_parameters.extend(lora_linear.lora_B.parameters())

    if not lora_parameters:
        raise ValueError(f"No matching attention modules found for targets {targets}")
    return model, lora_parameters


def load_lora_weights(
    model: nn.Module,
    path: str | Path,
    device: str | torch.device | None = None,
    strict_config: bool = True,
) -> None:
    """Load LoRA-only weights and validate metadata and architecture coverage."""
    checkpoint: dict[str, Any] = torch.load(
        Path(path), map_location=device, weights_only=True
    )
    required_keys = {"lora_state_dict", "rank", "alpha", "target_modules"}
    missing_metadata = required_keys.difference(checkpoint)
    if missing_metadata:
        raise ValueError(
            f"Invalid LoRA checkpoint; missing keys: {sorted(missing_metadata)}"
        )

    applied_config = _applied_lora_config(model)
    checkpoint_config = (
        checkpoint["rank"],
        checkpoint["alpha"],
        _normalize_target_modules(checkpoint["target_modules"]),
    )
    if checkpoint_config != applied_config:
        message = (
            "LoRA checkpoint configuration mismatch: "
            f"checkpoint={checkpoint_config}, applied={applied_config}"
        )
        if strict_config:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)

    lora_state = checkpoint["lora_state_dict"]
    if not isinstance(lora_state, dict):
        raise ValueError("lora_state_dict must be a dictionary")

    model_lora_keys = set(_lora_state_dict(model))
    checkpoint_lora_keys = set(lora_state)
    unexpected_keys = checkpoint_lora_keys.difference(model_lora_keys)
    missing_keys = model_lora_keys.difference(checkpoint_lora_keys)
    if unexpected_keys:
        raise ValueError(
            "Checkpoint LoRA keys were not found in the model: "
            f"{sorted(unexpected_keys)}"
        )
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing model LoRA keys: {sorted(missing_keys)}"
        )

    incompatible = model.load_state_dict(lora_state, strict=False)
    unexpected_after_load = [
        key for key in incompatible.unexpected_keys if key in checkpoint_lora_keys
    ]
    if unexpected_after_load:
        raise ValueError(
            f"Failed to load checkpoint LoRA keys: {sorted(unexpected_after_load)}"
        )
