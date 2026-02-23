from __future__ import annotations

import math

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: [batch, heads, seq, dim]
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1).unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def build_causal_mask(q_len: int, k_len: int, offset: int, device: torch.device) -> torch.Tensor:
    q_positions = torch.arange(offset, offset + q_len, device=device).unsqueeze(1)
    k_positions = torch.arange(k_len, device=device).unsqueeze(0)
    return k_positions <= q_positions
