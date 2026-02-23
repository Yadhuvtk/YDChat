from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ScalarLog:
    step: int
    name: str
    value: float


class TrainLogger:
    def __init__(self, enable_tensorboard: bool = False, log_dir: str | None = None) -> None:
        self.writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                self.writer = None

    def log(self, *, step: int, metrics: dict[str, float]) -> None:
        parts = [f"step={step}"]
        for key, value in metrics.items():
            parts.append(f"{key}={value:.4f}")
            if self.writer is not None:
                self.writer.add_scalar(key, value, step)
        print(" | ".join(parts), flush=True)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
