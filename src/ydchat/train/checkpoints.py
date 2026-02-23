from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ydchat.config import YDChatConfig


def save_checkpoint(
    output_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    cfg: YDChatConfig,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": asdict(cfg),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    step_path = out / f"step_{step:08d}.pt"
    last_path = out / "last.pt"
    torch.save(payload, step_path)
    torch.save(payload, last_path)
    return step_path


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    return int(state.get("step", 0))


def find_last_checkpoint(output_dir: str | Path) -> Path | None:
    last = Path(output_dir) / "last.pt"
    return last if last.exists() else None
