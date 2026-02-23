from __future__ import annotations

import inspect
from typing import Iterable

import torch

from ydchat.config import OptimConfig


def create_adamw(params: Iterable[torch.nn.Parameter], cfg: OptimConfig, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, object] = {
        "lr": cfg.lr,
        "betas": (cfg.beta1, cfg.beta2),
        "eps": cfg.eps,
        "weight_decay": cfg.weight_decay,
    }
    signature = inspect.signature(torch.optim.AdamW.__init__)
    fused_available = "fused" in signature.parameters
    if cfg.fused and device.type == "cuda" and fused_available:
        kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)
