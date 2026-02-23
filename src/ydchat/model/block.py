from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ydchat.model.attention import CausalSelfAttention
from ydchat.model.mlp import SwiGLU
from ydchat.model.rotary import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.norm1 = RMSNorm(d_model, eps=rms_norm_eps)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model, eps=rms_norm_eps)
        self.mlp = SwiGLU(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotary: RotaryEmbedding,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv = self.attn(self.norm1(x), rotary=rotary, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv
