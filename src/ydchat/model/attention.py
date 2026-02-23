from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from ydchat.model.rotary import RotaryEmbedding, apply_rotary_emb, build_causal_mask


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        rotary: RotaryEmbedding,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = x.shape

        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        past_len = 0
        if past_kv is not None:
            past_len = past_kv[0].size(2)

        cos, sin = rotary.get_cos_sin(
            q_len,
            device=x.device,
            dtype=q.dtype,
            offset=past_len,
        )
        q, k = apply_rotary_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        k_len = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = build_causal_mask(q_len, k_len, offset=past_len, device=x.device)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -torch.finfo(scores.dtype).max)

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        probs = self.attn_dropout(probs)

        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        out = self.resid_dropout(self.o_proj(out))

        new_kv = (k, v) if use_cache else None
        return out, new_kv
