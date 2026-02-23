from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import IterableDataset


class PackedTokenDataset(IterableDataset):
    """Pack a token stream into fixed-length autoregressive training samples."""

    def __init__(self, token_stream: IterableDataset, seq_len: int) -> None:
        super().__init__()
        self.token_stream = token_stream
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        for token_id in self.token_stream:
            buffer.append(int(token_id))
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                del buffer[: self.seq_len]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }
