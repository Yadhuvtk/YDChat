from __future__ import annotations

import torch
from torch import nn


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token(input_ids)
        return self.dropout(x)
