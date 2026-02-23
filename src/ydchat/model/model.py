from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ydchat.config import ModelConfig
from ydchat.model.block import RMSNorm, TransformerBlock
from ydchat.model.embeddings import TokenEmbeddings
from ydchat.model.rotary import RotaryEmbedding


@dataclass
class YDChatOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class YDChatLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbeddings(cfg.vocab_size, cfg.d_model, dropout=cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    rms_norm_eps=cfg.rms_norm_eps,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(cfg.d_model // cfg.n_heads, base=cfg.rope_theta)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.token.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> YDChatOutput:
        x = self.embed(input_ids)

        new_past: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[i]
            x, layer_cache = block(x, rotary=self.rotary, past_kv=layer_past, use_cache=use_cache)
            if use_cache and layer_cache is not None and new_past is not None:
                new_past.append(layer_cache)

        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return YDChatOutput(logits=logits, loss=loss, past_key_values=new_past)
