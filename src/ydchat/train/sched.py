from __future__ import annotations

import math

import torch


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if step >= total_steps:
        return min_lr_ratio

    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )
