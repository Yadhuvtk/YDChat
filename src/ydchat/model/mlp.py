from __future__ import annotations

import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))
