"""Model parameter counting utility."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParamStats:
    total: int
    trainable: int
    trainable_pct: float


def count_params(model: torch.nn.Module) -> ParamStats:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return ParamStats(total=total, trainable=trainable, trainable_pct=pct)
