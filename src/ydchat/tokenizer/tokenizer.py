from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


class YDTokenizer:
    """Light wrapper over SentencePiece with explicit BOS/EOS/PAD helpers."""

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return int(self.sp.vocab_size())

    @property
    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    @property
    def pad_id(self) -> int:
        return int(self.sp.pad_id())

    @property
    def unk_id(self) -> int:
        return int(self.sp.unk_id())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def id_to_piece(self, idx: int) -> str:
        return self.sp.id_to_piece(int(idx))

    def piece_to_id(self, piece: str) -> int:
        return int(self.sp.piece_to_id(piece))


def load_tokenizer(model_path: str | Path) -> YDTokenizer:
    return YDTokenizer(model_path=model_path)
